"""Verify all seven policies on identical edits and measure exact refresh cost."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latechunk.anchored import AnchoredGroups  # noqa: E402
from latechunk.encoding import Encoder, encode_late  # noqa: E402
from latechunk.group_encoding import encode_group_inputs  # noqa: E402
from latechunk.positional_refresh import PositionalRefresh  # noqa: E402
from latechunk.sticky import RefreshStats, StickyChunk, StickyGroups  # noqa: E402
from latechunk.study_data import load_native, prepare_native, prepare_scifact  # noqa: E402
from latechunk.study_experiment import (  # noqa: E402
    CONTEXT_CAPS, StudyConfig, encoder_provenance, read_frozen_config, validate_phase, write_new_json,
)
from latechunk.study_refresh import (  # noqa: E402
    STATELESS_METHODS, StatelessRefresh, build_stateless_units, encode_study_long_late, study_prefix,
)
from scripts.pilot_anchored import _as_document, _as_sticky, _snapshots  # noqa: E402


def _verify(operation, stats, ids, current, rebuilt):
    if operation.startswith("noop") and any((stats.encoder_calls, stats.refreshed_groups,
                                              stats.refreshed_input_tokens, stats.rewritten_vectors,
                                              stats.removed_vectors)):
        raise AssertionError(f"{operation} refreshed unchanged inputs")
    if set(ids) != current.keys() or set(ids) != rebuilt.keys():
        raise AssertionError(f"{operation} changed the retrieval units")
    actual = torch.stack([current[key] for key in ids])
    expected = torch.stack([rebuilt[key] for key in ids])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return asdict(stats) | {"max_rebuild_error": (actual - expected).abs().max().item()}


def _rebuild_stateless(encoder, method, snapshot, edited, ids, prefix, args):
    if method == "late":
        vectors = encode_late(encoder, [edited], batch_size=args.batch_size)
        return dict(zip(ids, vectors, strict=True))
    units = build_stateless_units(encoder, snapshot, method=method, token_cap=args.context_cap,
                                   local_radius=args.local_radius, prefix=prefix)
    inputs = list(dict.fromkeys(unit for unit, _ in units.values()))
    vectors = dict(zip(inputs, encode_group_inputs(encoder, inputs, args.batch_size), strict=True))
    return {key: vectors[unit][index] for key, (unit, index) in units.items()}


def run_sequence(encoder, doc, sequence, args):
    replacement, inserted, snapshots = sequence
    settings = {"token_cap": args.context_cap, "batch_size": args.batch_size}
    positional_settings = {"window": args.context_cap, "overlap": args.overlap,
                           "batch_size": args.batch_size, "special_tokens": "wrap", "oversized_singleton": True}
    original = doc.chunks
    ids = tuple(chunk.chunk_id for chunk in original)
    stateless = {method: StatelessRefresh(encoder, method=method, local_radius=args.local_radius,
                                          **settings) for method in STATELESS_METHODS}
    baseline_prefix = study_prefix(doc)
    initial = {method: asdict(state.refresh(original, prefix=baseline_prefix)) for method, state in stateless.items()}
    positional = PositionalRefresh(encoder, **positional_settings)
    initial["long_late"] = asdict(positional.refresh(doc.chunked, ids))
    sticky = StickyGroups(encoder, _as_sticky(original), **settings)
    anchored = AnchoredGroups(encoder, original, anchor_k=args.anchor_k, **settings)
    initial["sticky"] = asdict(sticky.stats) | {"group_map": sticky.group_map}
    initial["content_anchored"] = asdict(anchored.stats) | {"group_map": anchored.group_map}
    sticky_edits = {
        "replace": lambda: sticky.replace(StickyChunk(replacement.chunk_id, replacement.token_ids)),
        "insert": lambda: sticky.insert(StickyChunk(inserted.chunk_id, inserted.token_ids),
                                       after_id=replacement.chunk_id),
        "delete": lambda: sticky.delete(inserted.chunk_id),
    }

    events, fresh_references = {}, {}
    for operation, snapshot in (*snapshots.items(), ("title_prefix", snapshots["delete"]),
                                ("noop_title", snapshots["delete"])):
        ids = tuple(chunk.chunk_id for chunk in snapshot)
        edited = _as_document(encoder.tokenizer, doc.doc_id, snapshot)
        prefix = study_prefix(replace(doc, chunks=snapshot))
        if operation in {"title_prefix", "noop_title"}:
            prefix = "Updated document title. " + prefix
        row = {}
        for method, state in stateless.items():
            stats = state.refresh(snapshot, prefix=prefix)
            reference = (method, snapshot, prefix if method == "naive_prefix" else "")
            if reference not in fresh_references:
                fresh_references[reference] = _rebuild_stateless(encoder, method, snapshot, edited, ids, prefix, args)
            row[method] = _verify(operation, stats, ids, state.vectors, fresh_references[reference])

        stats = positional.refresh(edited, ids)
        reference = ("long_late", snapshot)
        if reference not in fresh_references:
            fresh = encode_study_long_late(encoder, [replace(doc, chunks=snapshot, chunked=edited)],
                                           token_cap=args.context_cap, overlap=args.overlap, batch_size=args.batch_size)
            fresh_references[reference] = dict(zip(ids, fresh, strict=True))
        row["long_late"] = _verify(operation, stats, ids, positional.vectors, fresh_references[reference])

        stats = sticky_edits.get(operation, sticky.refresh)()
        if sticky.chunks != _as_sticky(snapshot):
            raise AssertionError(f"{operation} changed Sticky content or order")
        reference = ("sticky", snapshot, sticky.group_map)
        if reference not in fresh_references:
            fresh_references[reference] = StickyGroups(encoder, sticky.chunks, group_map=sticky.group_map, **settings)
        sticky_rebuild = fresh_references[reference]
        row["sticky"] = _verify(operation, stats, ids, sticky.vectors, sticky_rebuild.vectors) | {
            "group_map": sticky.group_map,
        }
        stats = anchored.refresh(snapshot)
        reference = ("content_anchored", snapshot)
        if reference not in fresh_references:
            fresh_references[reference] = AnchoredGroups(encoder, snapshot, anchor_k=args.anchor_k, **settings)
        anchored_rebuild = fresh_references[reference]
        if anchored.group_map != anchored_rebuild.group_map:
            raise AssertionError(f"{operation} changed the deterministic anchor partition")
        row["content_anchored"] = _verify(operation, stats, ids, anchored.vectors, anchored_rebuild.vectors) | {
            "group_map": anchored.group_map,
        }
        events[operation] = row
    return {"edited_chunk_id": replacement.chunk_id, "inserted_chunk_id": inserted.chunk_id,
            "original_target_tokens": next(len(chunk.token_ids) for chunk in original if chunk.chunk_id == replacement.chunk_id),
            "replacement_tokens": len(replacement.token_ids), "insertion_tokens": len(inserted.token_ids),
            "initial_build": initial, "events": events}


def _sequences(encoder, chunks, positions):
    by_position = {"early": 0, "middle": len(chunks) // 2, "late": len(chunks) - 1}
    return {position: _snapshots(encoder.tokenizer, chunks, by_position[position]) for position in positions}


def _summarize(documents, positions):
    measurements = {}
    for position in positions:
        runs = [doc[position]["events"] for doc in documents.values()]
        measurements[position] = {}
        for operation, methods in runs[0].items():
            measurements[position][operation] = {}
            for method in methods:
                rows = [run[operation][method] for run in runs]
                measurements[position][operation][method] = {
                    **{key: sum(row[key] for row in rows) for key in asdict(RefreshStats())},
                    "median_refresh_seconds": median(row["total_seconds"] for row in rows),
                    "max_rebuild_error": max(row["max_rebuild_error"] for row in rows),
                }
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--native", type=Path, help="Native chunk study manifest")
    sources.add_argument("--scifact", type=Path, help="Extracted BEIR SciFact directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frozen-config", type=Path, help="Use the exact MLDR held-out encoding configuration")
    parser.add_argument("--context-cap", type=int, choices=CONTEXT_CAPS)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--anchor-k", type=int)
    parser.add_argument("--local-radius", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--limit-docs", type=int, default=3)
    parser.add_argument("--min-chunks", type=int, default=4)
    parser.add_argument("--edit-position", choices=("early", "middle", "late", "all"), default="all")
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    if min(args.chunk_size, args.limit_docs, args.min_chunks) < 1:
        parser.error("Require positive limits and chunk size")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    study_keys = ("context_cap", "overlap", "anchor_k", "local_radius", "batch_size")
    model_defaults = {"model": "jinaai/jina-embeddings-v2-small-en", "max_length": 8192,
                      "model_revision": None, "tokenizer_revision": None}
    if args.frozen_config:
        if not args.native or any(getattr(args, key) is not None for key in (*study_keys, *model_defaults)):
            parser.error("Frozen edit runs require a native test manifest and take encoding settings from the frozen config")
        frozen = read_frozen_config(args.frozen_config)
        config = StudyConfig(**frozen["study_config"])
        model_settings = frozen["encoder_settings"]
    else:
        frozen = None
        config = StudyConfig(**{key: getattr(args, key) for key in study_keys if getattr(args, key) is not None})
        model_settings = {key: getattr(args, key) if getattr(args, key) is not None else default
                          for key, default in model_defaults.items()}
    config.validate(model_settings["max_length"])
    for key in study_keys:
        setattr(args, key, getattr(config, key))

    torch.manual_seed(0)
    encoder = Encoder(model_settings["model"], device=args.device, max_length=model_settings["max_length"],
                      model_revision=model_settings["model_revision"], tokenizer_revision=model_settings["tokenizer_revision"])
    dataset = (prepare_native(encoder, load_native(args.native)) if args.native else
               prepare_scifact(encoder, args.scifact, chunk_size=args.chunk_size))
    if frozen:
        validate_phase("held-out", dataset, frozen, encoder)
    positions = ("early", "middle", "late") if args.edit_position == "all" else (args.edit_position,)
    documents, excluded = {}, {}
    for doc in dataset.documents:
        if len(doc.chunks) < args.min_chunks:
            continue
        try:
            sequences = _sequences(encoder, doc.chunks, positions)
        except ValueError as exc:
            excluded[doc.doc_id] = str(exc)
            continue
        if any(len(_as_document(encoder.tokenizer, doc.doc_id, snapshot).input_ids) > encoder.max_length
               for _, _, snapshots in sequences.values() for snapshot in snapshots.values()):
            excluded[doc.doc_id] = "edited document exceeds the Full Late model limit"
            continue
        documents[doc.doc_id] = {}
        for position, sequence in sequences.items():
            documents[doc.doc_id][position] = run_sequence(encoder, doc, sequence, args)
            print(f"Verified {doc.doc_id} at position {position}", flush=True)
        print(f"Verified {len(documents)} / {args.limit_docs} documents", flush=True)
        if len(documents) == args.limit_docs:
            break
    if len(documents) != args.limit_docs:
        raise ValueError(f"Found {len(documents)} eligible documents, expected {args.limit_docs}")

    results = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        | model_settings | {"long_special_tokens": "wrap", "effective_device": str(encoder.device)},
        "encoder_provenance": encoder_provenance(encoder),
        "dataset": dataset.dataset,
        "split": dataset.split,
        "baseline_coverage": dataset.coverage,
        "dataset_fingerprints": dataset.fingerprints,
        "selection": "First eligible documents with all requested edit positions",
        "edit_exclusions": excluded,
        "timing": "Per-update wall time includes regroup, model inference, pooling and in-memory writes; fresh checks excluded",
        "totals": _summarize(documents, positions),
        "documents": documents,
    }
    write_new_json(args.out, results)
    print(f"Saved refresh pilot to {args.out}")


if __name__ == "__main__":
    main()
