"""Export published ConTEB chunks and original query labels to a study manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _doc_id(chunk_id: str) -> str:
    doc_id, separator, suffix = chunk_id.rpartition("_")
    if not doc_id or not separator or not suffix.isdecimal():
        raise ValueError(f"Expected a published doc_id_numericSuffix chunk ID, got {chunk_id!r}")
    return doc_id


def manifest(document_rows, query_rows, *, dataset: str, split: str, source: dict) -> dict:
    documents, chunk_owners, queries = {}, {}, []
    for row in document_rows:
        chunk_id = row["chunk_id"]
        doc_id = _doc_id(chunk_id)
        if chunk_id in chunk_owners:
            raise ValueError(f"Duplicate published chunk ID {chunk_id!r}")
        chunk_owners[chunk_id] = doc_id
        book = documents.setdefault(doc_id, {"id": doc_id, "title": "", "chunks": []})
        book["chunks"].append({"id": chunk_id, "text": row["chunk"]})
    for index, row in enumerate(query_rows):
        chunk_id = row["chunk_id"]
        if chunk_id not in chunk_owners:
            raise ValueError(f"Published query row {index} references absent chunk {chunk_id!r}")
        queries.append({"id": f"{split}:{index}", "text": row["query"], "source_doc_id": chunk_owners[chunk_id],
                        "relevance": {chunk_id: 1}})
    if not documents or not queries:
        raise ValueError("ConTEB exports require documents and queries")
    return {"dataset": dataset, "split": split, "source": source, "documents": list(documents.values()), "queries": queries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="ConTEB Hugging Face dataset repository")
    parser.add_argument("--revision", default="main", help="Upstream published revision, resolved to a commit")
    parser.add_argument("--source-split", required=True, help="Split name in the upstream repository")
    parser.add_argument("--dataset", required=True, help="Common study dataset name for matching train and test")
    parser.add_argument("--split", required=True, help="Study split label such as train or test")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(args.repo, revision=args.revision).sha
    if not revision:
        raise ValueError(f"Could not resolve dataset revision {args.revision!r}")
    settings = {"split": args.source_split, "revision": revision}
    documents = load_dataset(args.repo, "documents", **settings)
    queries = load_dataset(args.repo, "queries", **settings)
    payload = manifest(documents, queries, dataset=args.dataset, split=args.split,
                       source={"repo": args.repo, "revision": revision, "split": args.source_split})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved {len(payload['documents'])} documents and {len(payload['queries'])} queries to {args.out}")


if __name__ == "__main__":
    main()
