"""Run deterministic chunk edits on SciFact and verify Sticky ID Groups refreshes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latechunk.chunking import fixed_token_chunks  # noqa: E402
from latechunk.data import doc_text, download_beir, load_beir  # noqa: E402
from latechunk.encoding import Encoder  # noqa: E402
from latechunk.sticky import RefreshStats, StickyChunk, StickyGroups, sticky_chunks  # noqa: E402


def run_edits(
    encoder: Encoder, chunks: tuple[StickyChunk, ...], *, token_cap: int, batch_size: int
) -> dict:
    state = StickyGroups(encoder, chunks, token_cap=token_cap, batch_size=batch_size)
    initial_stats = asdict(state.stats)
    initial_group_map = state.group_map
    events = {}

    def check_refresh(operation: str, stats: RefreshStats) -> None:
        if operation.startswith("noop"):
            assert not (
                stats.encoder_calls
                or stats.refreshed_groups
                or stats.refreshed_input_tokens
                or stats.rewritten_vectors
                or stats.removed_vectors
            )
        rebuild = StickyGroups(
            encoder, state.chunks, token_cap=token_cap, batch_size=batch_size, group_map=state.group_map
        )
        chunk_ids = {chunk.chunk_id for chunk in state.chunks}
        vectors, rebuilt_vectors = state.vectors, rebuild.vectors
        assert vectors.keys() == rebuilt_vectors.keys() == chunk_ids
        max_error = 0.0
        for chunk_id in chunk_ids:
            actual, expected = vectors[chunk_id], rebuilt_vectors[chunk_id]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            max_error = max(max_error, (actual - expected).abs().max().item())
        events[operation] = asdict(stats) | {
            "group_map": state.group_map,
            "max_rebuild_error": max_error,
        }

    anchor = chunks[0]
    first_token = anchor.token_ids[0]
    alternate = next((token for chunk in chunks for token in chunk.token_ids if token != first_token), None)
    if alternate is None:
        raise ValueError(f"Pilot document containing chunk {anchor.chunk_id!r} has no alternative token")
    replacement = StickyChunk(anchor.chunk_id, (alternate, *anchor.token_ids[1:]))
    inserted = StickyChunk(f"{anchor.chunk_id}:pilot-insert-1", anchor.token_ids)
    check_refresh("noop_before", state.refresh())
    check_refresh("replace", state.replace(replacement))
    check_refresh("insert", state.insert(inserted, after_id=anchor.chunk_id))
    check_refresh("delete", state.delete(inserted.chunk_id))
    check_refresh("noop_after", state.refresh())
    return {
        "initial_build": initial_stats,
        "initial_group_map": initial_group_map,
        "events": events,
        # The final chunk lengths and order match the initial snapshot after deleting the insertion.
        "partition_drifted": state.group_map != initial_group_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/sticky-pilot.json"))
    parser.add_argument("--limit-docs", type=int, default=20, help="Number of eligible documents in corpus order")
    parser.add_argument("--min-chunks", type=int, default=4, help="Minimum original chunks per pilot document")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--token-cap", type=int, default=1024)
    parser.add_argument("--model", default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if min(args.limit_docs, args.chunk_size, args.token_cap, args.max_length, args.batch_size) < 1:
        parser.error("Document limit, chunk size, token cap, model length and batch size must be positive")
    if args.min_chunks < 2:
        parser.error("Pilot documents must have at least two chunks")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")

    torch.manual_seed(0)
    corpus, _, _ = load_beir(download_beir("scifact", args.data_dir))
    encoder = Encoder(args.model, device=args.device, max_length=args.max_length)
    documents = {}
    for doc_id, doc in corpus.items():
        chunks = sticky_chunks(fixed_token_chunks(doc_id, doc_text(doc), encoder.tokenizer, args.chunk_size))
        if len(chunks) < args.min_chunks:
            continue
        documents[doc_id] = run_edits(encoder, chunks, token_cap=args.token_cap, batch_size=args.batch_size)
        if len(documents) == args.limit_docs:
            break
    if len(documents) != args.limit_docs:
        raise ValueError(f"Found {len(documents)} eligible SciFact documents, expected {args.limit_docs}")

    first = next(iter(documents.values()))
    totals = {}
    for operation in first["events"]:
        rows = [doc["events"][operation] for doc in documents.values()]
        totals[operation] = {name: sum(row[name] for row in rows) for name in first["initial_build"]}
        totals[operation]["max_rebuild_error"] = max(row["max_rebuild_error"] for row in rows)

    results = {
        "config": {**vars(args), "data_dir": str(args.data_dir), "out": str(args.out), "device": str(encoder.device)},
        "selection": f"First documents with at least {args.min_chunks} title-plus-body chunks, in SciFact corpus order",
        "totals": totals,
        "documents_with_partition_drift": sum(doc["partition_drifted"] for doc in documents.values()),
        "documents": documents,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")
    print(f"Verified {len(documents)} documents against saved-group-map rebuilds")
    print(json.dumps(totals, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
