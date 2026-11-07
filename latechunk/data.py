"""BEIR dataset loading (SciFact by default). Uses only the standard library."""

from __future__ import annotations

import csv
import json
import urllib.request
import zipfile
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def download_beir(name: str, data_dir: Path) -> Path:
    """Download and extract a BEIR dataset if it is not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / name
    if (target / "corpus.jsonl").exists():
        return target
    zip_path = data_dir / f"{name}.zip"
    if not zip_path.exists():
        print(f"Downloading {BEIR_URL.format(name=name)} -> {zip_path}")
        urllib.request.urlretrieve(BEIR_URL.format(name=name), zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # guard against path traversal in the archive
            if Path(member).is_absolute() or ".." in Path(member).parts:
                raise ValueError(f"Unsafe path in archive: {member}")
        zf.extractall(data_dir)
    return target


def load_beir(path: Path, split: str = "test"):
    """Return (corpus, queries, qrels).

    corpus:  {doc_id: {"title": str, "text": str}}
    queries: {query_id: str}   (only queries that appear in the split's qrels)
    qrels:   {query_id: {doc_id: int}}
    """
    corpus = {}
    with open(path / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus[str(d["_id"])] = {"title": d.get("title", "") or "", "text": d.get("text", "") or ""}

    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels" / f"{split}.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id corpus-id score
        for qid, did, score in reader:
            qrels.setdefault(str(qid), {})[str(did)] = int(score)

    queries = {}
    with open(path / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            if str(q["_id"]) in qrels:
                queries[str(q["_id"])] = q["text"]
    return corpus, queries, qrels


def doc_text(doc: dict, use_title: bool = True) -> str:
    """MTEB / BEIR convention: title + ' ' + text."""
    if use_title and doc["title"]:
        return f"{doc['title']} {doc['text']}".strip()
    return doc["text"].strip()
