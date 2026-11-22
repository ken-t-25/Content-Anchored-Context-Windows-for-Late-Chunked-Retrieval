"""Run an aligned seven-policy retrieval quality comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latechunk.encoding import Encoder
from latechunk.study_data import fingerprint, load_native, prepare_native, prepare_scifact
from latechunk.study_experiment import (
    CONTEXT_CAPS,
    NONINFERIORITY_MARGIN,
    StudyConfig,
    read_frozen_config,
    run_quality,
    validate_phase,
    write_frozen_config,
    write_new_json,
)

DEFAULT_MODEL = "jinaai/jina-embeddings-v2-small-en"
STUDY_OVERRIDES = ("context_cap", "overlap", "anchor_k", "local_radius", "batch_size", "query_batch_size")
ENCODER_OVERRIDES = ("model", "model_revision", "tokenizer_revision", "max_length")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--native", type=Path, help="Native chunk corpus and qrels JSON")
    source.add_argument("--scifact", type=Path, help="Local BEIR SciFact directory for document-level diagnostics")
    parser.add_argument("--phase", choices=["development", "held-out", "transfer", "exploratory"], default="exploratory")
    parser.add_argument("--out", type=Path, required=True, help="New results JSON path")
    parser.add_argument("--write-frozen-config", type=Path, help="Save selected development settings to a new JSON file")
    parser.add_argument("--frozen-config", type=Path, help="Apply development-selected settings to a held-out or transfer native split")
    parser.add_argument("--smoke", action="store_true", help="Label a deliberately small exploratory run")
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--chunk-size", type=int, help="SciFact fixed-token chunk size (default 256)")
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device")
    parser.add_argument("--context-cap", type=int, choices=CONTEXT_CAPS)
    parser.add_argument("--overlap", type=int, help="Number of content tokens shared by adjacent Long Late windows")
    parser.add_argument("--anchor-k", type=int)
    parser.add_argument("--local-radius", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--query-batch-size", type=int)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    if args.write_frozen_config and (args.write_frozen_config.exists() or args.write_frozen_config.resolve() == args.out.resolve()):
        parser.error("--write-frozen-config must give a separate new path")
    if args.phase != "exploratory" and (args.scifact or args.smoke or args.limit_docs is not None):
        parser.error("Formal phases require the full native split. SciFact and smoke/limited runs use exploratory phase.")
    if args.chunk_size is not None and not args.scifact:
        parser.error("--chunk-size applies only to SciFact fixed chunking")
    if args.phase != "development" and args.write_frozen_config:
        parser.error("--write-frozen-config requires the development phase")
    if args.phase in {"held-out", "transfer"}:
        if not args.frozen_config:
            parser.error("--frozen-config is required for held-out or transfer evaluation")
        if any(getattr(args, key) is not None for key in STUDY_OVERRIDES + ENCODER_OVERRIDES):
            parser.error("Held-out and transfer evaluation take their study and encoder settings from --frozen-config")
    elif args.frozen_config:
        parser.error("--frozen-config requires held-out or transfer evaluation")
    return args


def resolve_configuration(args: argparse.Namespace) -> tuple[StudyConfig, dict, dict | None]:
    if args.frozen_config:
        frozen = read_frozen_config(args.frozen_config)
        return StudyConfig(**frozen["study_config"]), frozen["encoder_settings"], frozen
    overrides = {key: getattr(args, key) for key in STUDY_OVERRIDES if getattr(args, key) is not None}
    return StudyConfig(**overrides), {
        "model": DEFAULT_MODEL if args.model is None else args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "max_length": 8192 if args.max_length is None else args.max_length,
    }, None


def main() -> None:
    args = parse_args()
    config, settings, frozen = resolve_configuration(args)
    config.validate(settings["max_length"])
    torch.manual_seed(0)
    encoder = Encoder(
        settings["model"], device=args.device, max_length=settings["max_length"],
        model_revision=settings["model_revision"], tokenizer_revision=settings["tokenizer_revision"],
    )
    if args.native:
        dataset = prepare_native(encoder, load_native(args.native), limit_docs=args.limit_docs)
    else:
        chunk_size = 256 if args.chunk_size is None else args.chunk_size
        dataset = prepare_scifact(encoder, args.scifact, chunk_size=chunk_size, limit_docs=args.limit_docs)
    validate_phase(args.phase, dataset, frozen, encoder)
    result = run_quality(encoder, dataset, config) | {
        "phase": args.phase,
        "scope": "smoke" if args.smoke or args.limit_docs is not None else args.phase,
        "encoder_settings": settings,
        "frozen_config_sha256": fingerprint(frozen) if frozen is not None else None,
        "retrieval_units": "native_stable_chunks" if args.native else "scifact_fixed_title_plus_body_tokens",
    }
    if args.scifact:
        result["config"]["chunk_size"] = chunk_size
    if args.phase == "held-out":
        paired = result["content_anchored_vs_long_late_source_paired_bootstrap"]["ndcg@10"]
        result["noninferiority"] = {
            "contrast": "content_anchored_minus_long_late_ndcg@10",
            "lower_bound_threshold": NONINFERIORITY_MARGIN,
            "meets_threshold": None if paired is None else paired["interval_95"][0] > NONINFERIORITY_MARGIN,
        }
    write_new_json(args.out, result)
    if args.write_frozen_config and result["status"] == "complete":
        write_frozen_config(args.write_frozen_config, dataset, config, settings, encoder)
    if result["status"] == "no_eligible_data":
        print("No eligible documents or queries. The output records exclusions and coverage.")
    for method, row in result["methods"].items():
        print(f"{method}: nDCG@10={row['ndcg@10']:.4f} Recall@10={row['recall@10']:.4f} encode={row['encode_seconds']:.2f}s")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
