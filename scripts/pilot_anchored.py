"""Compare exact incremental refreshes on matched SciFact edits."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Mapping

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latechunk.anchored import AnchoredChunk, AnchoredGroups, anchored_chunks  # noqa: E402
from latechunk.chunking import ChunkedDoc, fixed_token_chunks  # noqa: E402
from latechunk.data import doc_text, download_beir, load_beir  # noqa: E402
from latechunk.encoding import Encoder, encode_late  # noqa: E402
from latechunk.positional_refresh import PositionalRefresh  # noqa: E402
from latechunk.sticky import RefreshStats, StickyChunk, StickyGroups  # noqa: E402


def _snapshots(tokenizer, original: tuple[AnchoredChunk, ...], index: int):
    # Match pilot_sticky.py: replace one token and insert a copy of the original target.
    target = original[index]
    first_token = target.token_ids[0]
    alternative = next((token for chunk in original for token in chunk.token_ids if token != first_token), None)
    if alternative is None:
        raise ValueError(f"Pilot document containing {target.chunk_id!r} has no alternative token")
    replacement_tokens = (alternative, *target.token_ids[1:])
    replacement = AnchoredChunk(
        target.chunk_id,
        tokenizer.decode(list(replacement_tokens), clean_up_tokenization_spaces=False),
        replacement_tokens,
    )
    inserted = AnchoredChunk(f"{target.chunk_id}:pilot-insert-1", target.text, target.token_ids)
    replaced = original[:index] + (replacement,) + original[index + 1 :]
    with_insert = replaced[: index + 1] + (inserted,) + replaced[index + 1 :]
    return replacement, inserted, {
        "noop_before": original,
        "replace": replaced,
        "insert": with_insert,
        "delete": replaced,
        "noop_after": replaced,
    }


def _as_document(tokenizer, doc_id: str, chunks: tuple[AnchoredChunk, ...]) -> ChunkedDoc:
    """Keep the retrieval boundaries and model token IDs fixed across all policies."""
    wrap = tokenizer.build_inputs_with_special_tokens
    leading_specials = wrap([-1]).index(-1)
    ids, spans = [], []
    for chunk in chunks:
        start = len(ids) + leading_specials
        ids.extend(chunk.token_ids)
        spans.append((start, len(ids) + leading_specials))
    return ChunkedDoc(doc_id, wrap(ids), spans, [chunk.text for chunk in chunks])


def _as_sticky(chunks: tuple[AnchoredChunk, ...]) -> tuple[StickyChunk, ...]:
    return tuple(StickyChunk(chunk.chunk_id, chunk.token_ids) for chunk in chunks)


def _check(
    operation: str,
    stats: RefreshStats,
    chunk_ids: tuple[str, ...],
    vectors: Mapping[str, torch.Tensor],
    rebuilt: Mapping[str, torch.Tensor],
) -> dict:
    if operation.startswith("noop"):
        assert not (
            stats.refreshed_groups or stats.refreshed_input_tokens or stats.encoder_calls
            or stats.rewritten_vectors or stats.removed_vectors
        ), f"{operation} triggered a refresh"
    assert vectors.keys() == rebuilt.keys() == set(chunk_ids)
    actual = torch.stack([vectors[chunk_id] for chunk_id in chunk_ids])
    expected = torch.stack([rebuilt[chunk_id] for chunk_id in chunk_ids])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return asdict(stats) | {"max_rebuild_error": (actual - expected).abs().max().item()}


def run_edits(encoder: Encoder, doc: ChunkedDoc, chunks: tuple[AnchoredChunk, ...], args: argparse.Namespace) -> dict:
    index = {"early": 0, "middle": len(chunks) // 2, "late": len(chunks) - 1}[args.edit_position]
    replacement, inserted, snapshots = _snapshots(encoder.tokenizer, chunks, index)
    groups_settings = {"token_cap": args.token_cap, "batch_size": args.batch_size}
    sticky = StickyGroups(encoder, _as_sticky(chunks), **groups_settings)
    anchored = {k: AnchoredGroups(encoder, chunks, anchor_k=k, **groups_settings) for k in args.anchor_k}
    positional = PositionalRefresh(
        encoder, window=args.long_window, overlap=args.long_overlap, batch_size=args.batch_size
    )
    original_ids = tuple(chunk.chunk_id for chunk in chunks)
    initial_positional = positional.refresh(doc, original_ids)
    initial = {
        "sticky": asdict(sticky.stats) | {"group_map": sticky.group_map},
        "long_late": asdict(initial_positional),
    }
    initial.update({f"content_anchored_k{k}": asdict(state.stats) | {"group_map": state.group_map}
                    for k, state in anchored.items()})
    sticky_edits = {
        "noop_before": sticky.refresh,
        "replace": lambda: sticky.replace(StickyChunk(replacement.chunk_id, replacement.token_ids)),
        "insert": lambda: sticky.insert(StickyChunk(inserted.chunk_id, inserted.token_ids),
                                       after_id=replacement.chunk_id),
        "delete": lambda: sticky.delete(inserted.chunk_id),
        "noop_after": sticky.refresh,
    }
    events = {}
    for operation, snapshot in snapshots.items():
        chunk_ids = tuple(chunk.chunk_id for chunk in snapshot)
        edited = _as_document(encoder.tokenizer, doc.doc_id, snapshot)
        if operation == "noop_before":
            assert edited.input_ids == doc.input_ids and edited.spans == doc.spans

        sticky_stats = sticky_edits[operation]()
        assert sticky.chunks == _as_sticky(snapshot)
        sticky_rebuild = StickyGroups(encoder, sticky.chunks, group_map=sticky.group_map, **groups_settings)
        row = {"sticky": _check(operation, sticky_stats, chunk_ids, sticky.vectors, sticky_rebuild.vectors)
               | {"group_map": sticky.group_map}}

        positional_stats = positional.refresh(edited, chunk_ids)
        positional_rebuild = encode_late(encoder, [edited], batch_size=args.batch_size,
                                         long_window=args.long_window, long_overlap=args.long_overlap)
        row["long_late"] = _check(operation, positional_stats, chunk_ids, positional.vectors,
                                  dict(zip(chunk_ids, positional_rebuild, strict=True)))
        for k, state in anchored.items():
            anchored_stats = state.refresh(snapshot)
            anchored_rebuild = AnchoredGroups(encoder, snapshot, anchor_k=k, **groups_settings)
            assert state.group_map == anchored_rebuild.group_map
            row[f"content_anchored_k{k}"] = _check(operation, anchored_stats, chunk_ids, state.vectors,
                                                  anchored_rebuild.vectors) | {"group_map": state.group_map}
        events[operation] = row
    return {"edited_chunk_id": replacement.chunk_id, "inserted_chunk_id": inserted.chunk_id,
            "initial_build": initial, "events": events}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/scifact-anchored-pilot.json"))
    parser.add_argument("--limit-docs", type=int, default=20)
    parser.add_argument("--min-chunks", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--token-cap", type=int, default=1024)
    parser.add_argument("--anchor-k", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--long-window", type=int, default=256)
    parser.add_argument("--long-overlap", type=int, default=64)
    parser.add_argument("--edit-position", choices=["early", "middle", "late"], default="early")
    parser.add_argument("--model", default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if min(args.limit_docs, args.min_chunks, args.chunk_size, args.token_cap, args.batch_size, *args.anchor_k) < 1:
        parser.error("Document limit, minimum chunk count, chunk size, token cap, batch size and k must be positive")
    if not 0 <= args.long_overlap < args.long_window <= args.max_length:
        parser.error("Expected 0 <= long overlap < long window <= model limit")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")

    torch.manual_seed(0)
    corpus, _, _ = load_beir(download_beir("scifact", args.data_dir))
    encoder = Encoder(args.model, device=args.device, max_length=args.max_length)
    documents = {}
    for doc_id, record in corpus.items():
        doc = fixed_token_chunks(doc_id, doc_text(record), encoder.tokenizer, args.chunk_size)
        if len(doc.spans) < args.min_chunks:
            continue
        documents[doc_id] = run_edits(encoder, doc, anchored_chunks(doc), args)
        if len(documents) % 5 == 0:
            print(f"Verified {len(documents)}/{args.limit_docs} pilot documents", flush=True)
        if len(documents) == args.limit_docs:
            break
    if len(documents) != args.limit_docs:
        raise ValueError(f"Found {len(documents)} eligible SciFact documents, expected {args.limit_docs}")

    totals = {}
    first = next(iter(documents.values()))
    for operation, methods in first["events"].items():
        totals[operation] = {}
        for method in methods:
            rows = [doc["events"][operation][method] for doc in documents.values()]
            totals[operation][method] = {
                **{name: sum(row[name] for row in rows) for name in asdict(RefreshStats())},
                "median_refresh_seconds": median(row["total_seconds"] for row in rows),
                "max_rebuild_error": max(row["max_rebuild_error"] for row in rows),
            }
    results = {
        "config": {**vars(args), "data_dir": str(args.data_dir), "out": str(args.out), "device": str(encoder.device),
                   "long_special_tokens": "raw"},
        "selection": f"First {args.limit_docs} SciFact documents with at least {args.min_chunks} title-plus-body chunks",
        "totals": totals,
        "documents": documents,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(results, output, indent=2)
        output.write("\n")
    print(f"Verified {len(documents)} documents against full rebuilds and saved {args.out}")


if __name__ == "__main__":
    main()
