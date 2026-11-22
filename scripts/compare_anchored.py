"""Exploratory shared Content-Anchored sweep against archived SciFact baselines."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latechunk.anchored import plan_content_anchored  # noqa: E402
from latechunk.chunking import fixed_token_chunks  # noqa: E402
from latechunk.data import doc_text, download_beir, load_beir  # noqa: E402
from latechunk.encoding import Encoder, pool_spans  # noqa: E402
from latechunk.retrieval import evaluate, search_docs  # noqa: E402
from latechunk.study_data import fingerprint  # noqa: E402
from latechunk.study_experiment import encoder_provenance  # noqa: E402

MODEL = "jinaai/jina-embeddings-v2-small-en"
BATCH_SIZE = 16
QUERY_BATCH_SIZE = 64
MAX_LENGTH = 8192
TOKEN_CAP = 1024
CHUNK_SIZES = (256, 128, 64)
ANCHOR_DIVISORS = (4, 8)
BASELINES = ("naive", "naive_prefix", "late", "long_late", "local_contextual", "sticky")
RunKey = tuple[int, int, bool]
Spans = tuple[tuple[int, int], ...]
Projection = tuple[RunKey, int, Spans]


@dataclass
class Variant:
    embeddings: torch.Tensor
    groups: int
    input_tokens: int = 0
    full_document_groups: int = 0
    full_document_chunks: int = 0
    chunks_with_exact_full_document_input: int = 0
    max_group_tokens: int = 0


def load_baselines() -> tuple[dict, dict[int, dict]]:
    root = ROOT / "benchmarks"
    ablation = root / "scifact-chunk-size-ablation" / "summary.json"
    reference = json.loads(ablation.read_text(encoding="utf-8"))
    rows = {}
    for size in CHUNK_SIZES:
        size_ref = reference["sizes"][str(size)]
        sources = [ablation.parent / name for name in size_ref["source_results"]]
        sources.append(root / "scifact-sticky" / f"cs{size}.json")
        methods = {}
        for path in sources:
            run = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "dataset": "scifact", "model": MODEL, "chunk_size": str(size), "no_title": "False",
                "device": "cpu", "batch_size": str(BATCH_SIZE), "max_length": str(MAX_LENGTH),
                "limit_docs": "None",
            }
            if "long_late" in run:
                expected.update(long_window="256", long_overlap="64", long_special_tokens="raw")
            if "local_contextual" in run:
                expected.update(local_radius="1", local_context_cap=str(TOKEN_CAP))
            if "sticky" in run:
                expected["sticky_token_cap"] = str(TOKEN_CAP)
            for name, value in expected.items():
                if run["config"].get(name) != value:
                    raise ValueError(f"Incompatible baseline {path.name} for {name}")
            for name in BASELINES:
                if name in run:
                    if run[name]["n_queries"] != reference["test_queries"]:
                        raise ValueError(f"Unexpected query count for {name} at chunk size {size}")
                    methods[name] = run[name]
        if set(methods) != set(BASELINES) or any(methods[m] != row for m, row in size_ref["methods"].items()):
            raise ValueError(f"Missing or inconsistent baseline metrics for chunk size {size}")
        rows[size] = {"chunks": size_ref["chunks"], "methods": methods,
                      "source_results": [path.resolve().relative_to(ROOT).as_posix() for path in sources]}
    return reference, rows


def plan_variants(encoder: Encoder, corpus: dict, comparison: dict[int, dict]) -> tuple[dict, dict, dict]:
    routes: dict[tuple[int, ...], list[Projection]] = {}
    variants: dict[RunKey, Variant] = {}
    owners: dict[int, list[int]] = {}
    titles = {doc_id: doc["title"] for doc_id, doc in corpus.items()}
    for size in CHUNK_SIZES:
        docs = [fixed_token_chunks(doc_id, doc_text(doc), encoder.tokenizer, size) for doc_id, doc in corpus.items()]
        chunk_owner = [index for index, doc in enumerate(docs) for _ in doc.spans]
        if len(chunk_owner) != comparison[size]["chunks"]:
            raise ValueError(f"Unexpected chunk count at size {size}")
        owners[size] = chunk_owner
        comparison[size]["chunk_fingerprint"] = fingerprint([
            (doc.doc_id, doc.input_ids, doc.spans, doc.texts) for doc in docs
        ])
        for anchor_k in ANCHOR_DIVISORS:
            for use_prefix in (False, True):
                key = (size, anchor_k, use_prefix)
                groups = plan_content_anchored(
                    encoder, docs, token_cap=TOKEN_CAP, anchor_k=anchor_k, prefixes=titles if use_prefix else None,
                )
                variant = Variant(
                    torch.empty(len(chunk_owner), encoder.model.config.hidden_size), len(groups),
                )
                row = 0
                for group in groups:
                    tokens, spans = group.input.token_ids, group.input.spans
                    count = len(group.chunk_ids)
                    if not count or count != len(spans):
                        raise ValueError(f"Group has inconsistent chunk IDs and spans in {key}")
                    last = row + count - 1
                    if last >= len(chunk_owner) or chunk_owner[row] != chunk_owner[last]:
                        raise ValueError(f"Group crosses retrieval document boundaries in {key}")
                    routes.setdefault(tokens, []).append((key, row, spans))
                    variant.input_tokens += len(tokens)
                    variant.max_group_tokens = max(variant.max_group_tokens, len(tokens))
                    doc = docs[chunk_owner[row]]
                    if count == len(doc.spans):
                        variant.full_document_groups += 1
                        variant.full_document_chunks += count
                        if tokens == tuple(doc.input_ids):
                            variant.chunks_with_exact_full_document_input += count
                    row += count
                if row != len(chunk_owner):
                    raise ValueError(f"Missing retrieval chunks in {key}")
                variants[key] = variant
                print(f"Planned size {size}, k={anchor_k}, prefix={use_prefix}: {len(groups):,} groups", flush=True)
    return routes, variants, owners


def encode_shared(encoder: Encoder, routes: dict, variants: dict[RunKey, Variant]) -> dict:
    sequences = sorted(routes, key=len)
    print(f"Encoding {len(sequences):,} distinct group inputs across {len(variants)} settings", flush=True)
    started = last_progress = time.perf_counter()
    encode_seconds = 0.0
    for start in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[start : start + BATCH_SIZE]
        encode_started = time.perf_counter()
        outputs = encoder.token_embeddings(batch)
        encode_seconds += time.perf_counter() - encode_started
        for tokens, token_vectors in zip(batch, outputs, strict=True):
            pooled = {}
            for key, first, spans in routes[tokens]:
                if spans not in pooled:
                    pooled[spans] = F.normalize(pool_spans(token_vectors, spans), dim=-1)
                variants[key].embeddings[first : first + len(spans)] = pooled[spans]
        done = start + len(batch)
        now = time.perf_counter()
        if now - last_progress >= 20 or done == len(sequences):
            print(f"Encoded {done:,}/{len(sequences):,} distinct inputs in {now - started:.1f}s", flush=True)
            last_progress = now
    return {
        "distinct_inputs": len(sequences),
        "encoded_input_tokens": sum(map(len, sequences)),
        "encoder_calls": (len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE,
        "encoder_seconds": encode_seconds,
        "encoding_and_pooling_seconds": time.perf_counter() - started,
        "standalone_variant_timing": False,
    }


def record_quality(
    variant: Variant,
    query_vectors: torch.Tensor,
    doc_ids: Sequence[str],
    query_ids: Sequence[str],
    qrels: dict,
    chunk_owner: list[int],
    baselines: dict,
) -> dict:
    _, top = search_docs(query_vectors, variant.embeddings, chunk_owner, len(doc_ids), top_k=100)
    run = {q: [doc_ids[j] for j in row.tolist()] for q, row in zip(query_ids, top, strict=True)}
    metrics = evaluate(run, qrels, k=10)
    if metrics["n_queries"] != len(query_ids):
        raise ValueError("Unexpected judged query count")
    return metrics | {
        "groups": variant.groups,
        "input_tokens_before_sharing": variant.input_tokens,
        "max_group_tokens": variant.max_group_tokens,
        "full_document_groups": variant.full_document_groups,
        "full_document_chunks": variant.full_document_chunks,
        "chunks_with_exact_full_document_input": variant.chunks_with_exact_full_document_input,
        "deltas_vs_baselines": {
            name: {metric: metrics[metric] - row[metric] for metric in ("ndcg@10", "recall@10")}
            for name, row in baselines.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("results/scifact-content-anchored/comparison.json"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")

    reference, comparison = load_baselines()
    corpus, queries, qrels = load_beir(download_beir("scifact", args.data_dir))
    if len(corpus) != reference["documents"] or len(queries) != reference["test_queries"]:
        raise ValueError("SciFact document or query counts do not match the archived benchmarks")
    torch.manual_seed(0)
    encoder = Encoder(MODEL, device=args.device, max_length=MAX_LENGTH)
    plan_started = time.perf_counter()
    routes, variants, owners = plan_variants(encoder, corpus, comparison)
    planning_seconds = time.perf_counter() - plan_started
    shared = encode_shared(encoder, routes, variants) | {
        "planning_seconds": planning_seconds,
        "groups_before_sharing": sum(variant.groups for variant in variants.values()),
        "input_tokens_before_sharing": sum(variant.input_tokens for variant in variants.values()),
    }
    del routes

    doc_ids, query_ids = list(corpus), list(queries)
    query_started = time.perf_counter()
    query_vectors = encoder.encode_texts([queries[q] for q in query_ids], batch_size=QUERY_BATCH_SIZE)
    query_seconds = time.perf_counter() - query_started
    for (size, anchor_k, use_prefix), variant in variants.items():
        methods = comparison[size]["methods"]
        name = f"content_anchored_k{anchor_k}" + ("_prefix" if use_prefix else "")
        baselines = {baseline: methods[baseline] for baseline in BASELINES}
        row = record_quality(variant, query_vectors, doc_ids, query_ids, qrels, owners[size], baselines)
        methods[name] = row
        print(f"{size:>3} {name}: nDCG@10={100 * row['ndcg@10']:.2f}, Recall@10={100 * row['recall@10']:.2f}", flush=True)

    result = {
        "config": {"dataset": "scifact", "documents": len(corpus), "queries": len(queries), "model": MODEL,
                   "device": str(encoder.device), "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH,
                   "token_cap": TOKEN_CAP, "anchor_divisors": ANCHOR_DIVISORS, "title_plus_body": True},
        "scope": "exploratory_archived_baseline_comparison",
        "evaluation_level": "document",
        "encoder": encoder_provenance(encoder),
        "fingerprints": {
            "corpus": fingerprint(list(corpus.items())),
            "queries": fingerprint(list(queries.items())),
            "qrels": fingerprint(qrels),
        },
        "actual_batch_sizes": {
            "shared_corpus": BATCH_SIZE, "queries": QUERY_BATCH_SIZE,
            "archived_corpus_by_method": {name: 64 if name in {"naive", "naive_prefix"} else BATCH_SIZE
                                          for name in BASELINES},
        },
        "shared_encoder": shared | {
            "scope": "shared_across_content_anchored_sweep_variants",
            "query_encode_seconds": query_seconds,
        },
        "sizes": comparison,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
