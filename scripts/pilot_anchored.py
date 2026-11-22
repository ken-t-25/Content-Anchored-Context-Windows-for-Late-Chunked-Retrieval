"""Compare exact incremental refreshes on matched SciFact edits."""

from __future__ import annotations

import argparse
import json
import re
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
from latechunk.encoding import Encoder, encode_late, validate_long_window  # noqa: E402
from latechunk.positional_refresh import PositionalRefresh  # noqa: E402
from latechunk.sticky import RefreshStats, StickyChunk, StickyGroups  # noqa: E402


def _content_ids(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def _replacement(tokenizer, original: tuple[AnchoredChunk, ...], target: AnchoredChunk) -> AnchoredChunk:
    """Make a deterministic text edit and verify it has the same token length as the target."""
    words = dict.fromkeys(word for chunk in original for word in re.findall(r"\S+", chunk.text))
    candidates = (*list(words)[:32], "updated", "evidence", "study", "revised", "new")
    for word in list(re.finditer(r"\S+", target.text))[:8]:
        for candidate in candidates:
            text = target.text[:word.start()] + candidate + target.text[word.end():]
            if text == target.text:
                continue
            ids = _content_ids(tokenizer, text)
            if len(ids) == len(target.token_ids) and ids != target.token_ids:
                return AnchoredChunk(target.chunk_id, text, ids)

    # A chunk can start or end inside a word. If changing one word cannot
    # preserve its length, try a complete replacement made of words.
    for candidate in candidates:
        if len(_content_ids(tokenizer, candidate)) != 1:
            continue
        text = " ".join([candidate] * len(target.token_ids))
        ids = _content_ids(tokenizer, text)
        if len(ids) == len(target.token_ids) and text != target.text and ids != target.token_ids:
            return AnchoredChunk(target.chunk_id, text, ids)
    raise ValueError(f"Could not construct a same-length text edit for {target.chunk_id!r}")


def _snapshots(tokenizer, original: tuple[AnchoredChunk, ...], index: int):
    target = original[index]
    replacement = _replacement(tokenizer, original, target)
    existing_ids = {chunk.chunk_id for chunk in original}
    number = 1
    while f"{target.chunk_id}:pilot-insert-{number}" in existing_ids:
        number += 1
    inserted = AnchoredChunk(
        f"{target.chunk_id}:pilot-insert-{number}", target.text, _content_ids(tokenizer, target.text),
    )
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


def run_edits(
    encoder: Encoder, doc: ChunkedDoc, chunks: tuple[AnchoredChunk, ...], args: argparse.Namespace,
    *, prefix: str = "",
) -> dict:
    index = {"early": 0, "middle": len(chunks) // 2, "late": len(chunks) - 1}[args.edit_position]
    replacement, inserted, snapshots = _snapshots(encoder.tokenizer, chunks, index)
    groups_settings = {"token_cap": args.token_cap, "batch_size": args.batch_size}
    sticky = StickyGroups(encoder, _as_sticky(chunks), **groups_settings)
    anchored = {k: AnchoredGroups(encoder, chunks, anchor_k=k, **groups_settings) for k in args.anchor_k}
    long_special_tokens = getattr(args, "long_special_tokens", "raw")
    positional = PositionalRefresh(
        encoder, window=args.long_window, overlap=args.long_overlap, batch_size=args.batch_size,
        special_tokens=long_special_tokens,
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

        before_groups = sticky.group_map
        sticky_stats = sticky_edits[operation]()
        assert sticky.chunks == _as_sticky(snapshot)
        sticky_rebuild = StickyGroups(encoder, sticky.chunks, group_map=sticky.group_map, **groups_settings)
        row = {"sticky": _check(operation, sticky_stats, chunk_ids, sticky.vectors, sticky_rebuild.vectors)
               | {"group_map": sticky.group_map}}
        group_delta = len(sticky.group_map) - len(before_groups)
        row["sticky"]["split_groups"] = max(0, group_delta) if operation == "insert" else 0
        row["sticky"]["merged_groups"] = (
            max(0, -group_delta)
            if operation == "delete" and any(inserted.chunk_id in group and len(group) > 1 for group in before_groups)
            else 0
        )

        positional_stats = positional.refresh(edited, chunk_ids)
        positional_rebuild = encode_late(
            encoder, [edited], batch_size=args.batch_size, long_window=args.long_window,
            long_overlap=args.long_overlap, special_tokens=long_special_tokens,
        )
        row["long_late"] = _check(operation, positional_stats, chunk_ids, positional.vectors,
                                  dict(zip(chunk_ids, positional_rebuild, strict=True)))
        for k, state in anchored.items():
            anchored_stats = state.refresh(snapshot)
            anchored_rebuild = AnchoredGroups(encoder, snapshot, anchor_k=k, **groups_settings)
            assert state.group_map == anchored_rebuild.group_map
            row[f"content_anchored_k{k}"] = _check(operation, anchored_stats, chunk_ids, state.vectors,
                                                  anchored_rebuild.vectors) | {"group_map": state.group_map}
        events[operation] = row
    result = {
        "edited_chunk_id": replacement.chunk_id,
        "original_tokens": len(chunks[index].token_ids),
        "replacement_tokens": len(replacement.token_ids),
        "replacement_changed_token_positions": sum(
            before != after for before, after in zip(chunks[index].token_ids, replacement.token_ids, strict=True)
        ),
        "inserted_chunk_id": inserted.chunk_id,
        "inserted_tokens": len(inserted.token_ids),
        "initial_build": initial,
        "events": events,
    }
    if getattr(args, "check_prefix", False):
        settings = {"anchor_k": args.anchor_k[0], **groups_settings}
        state = AnchoredGroups(encoder, chunks, prefix=prefix, **settings)
        checks = {}
        revised = f"Revised {prefix}".strip()
        for operation, title in (("replace_prefix", revised), ("noop_prefix", revised), ("restore_prefix", prefix)):
            stats = state.refresh(chunks, prefix=title)
            rebuilt = AnchoredGroups(encoder, chunks, prefix=title, **settings)
            assert state.group_map == rebuilt.group_map
            checks[operation] = _check(operation, stats, original_ids, state.vectors, rebuilt.vectors)
        result["prefix_checks"] = checks
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/scifact-anchored-pilot-corrected.json"))
    parser.add_argument("--limit-docs", type=int, default=20)
    parser.add_argument("--min-chunks", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--token-cap", type=int, default=1024)
    parser.add_argument("--anchor-k", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--long-window", type=int, help="Defaults to --token-cap for a matched comparison")
    parser.add_argument("--long-overlap", type=int, default=64)
    parser.add_argument("--long-special-tokens", choices=["raw", "wrap"], default="raw")
    parser.add_argument("--check-prefix", action="store_true", help="Verify a separate Anchored title edit")
    parser.add_argument("--edit-position", choices=["early", "middle", "late"], default="early")
    parser.add_argument("--model", default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if min(args.limit_docs, args.min_chunks, args.chunk_size, args.token_cap, args.batch_size, *args.anchor_k) < 1:
        parser.error("Document limit, minimum chunk count, chunk size, token cap, batch size and k must be positive")
    if args.long_window is None:
        args.long_window = args.token_cap
    try:
        validate_long_window(args.long_window, args.long_overlap, args.max_length, args.long_special_tokens)
    except ValueError as exc:
        parser.error(str(exc))
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
        documents[doc_id] = run_edits(encoder, doc, anchored_chunks(doc), args, prefix=record.get("title", ""))
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
            if method == "sticky":
                for name in ("split_groups", "merged_groups"):
                    totals[operation][method][name] = sum(row[name] for row in rows)
    results = {
        "config": {**vars(args), "data_dir": str(args.data_dir), "out": str(args.out), "device": str(encoder.device),
                   "model_revision": encoder.model_revision, "tokenizer_revision": encoder.tokenizer_revision,
                   "code_revision": encoder.code_revision},
        "selection": f"First {args.limit_docs} SciFact documents with at least {args.min_chunks} title-plus-body chunks",
        "edit_rule": "Replace same-length source text, then insert and delete a retokenized copy of the original text",
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
