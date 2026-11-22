"""Load retrieval units and apply eligibility once for all study policies."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .anchored import AnchoredChunk, anchored_chunks
from .chunking import ChunkedDoc, fixed_token_chunks
from .data import doc_text, load_beir
from .group_encoding import group_input


@dataclass(frozen=True)
class NativeDocument:
    doc_id: str
    title: str
    chunks: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class StudyQuery:
    query_id: str
    text: str
    source_doc_id: str | None
    relevance: dict[str, int]


@dataclass(frozen=True)
class NativeDataset:
    dataset: str
    split: str
    documents: tuple[NativeDocument, ...]
    queries: tuple[StudyQuery, ...]
    input_fingerprint: str


@dataclass(frozen=True)
class StudyDocument:
    doc_id: str
    title: str
    chunks: tuple[AnchoredChunk, ...]
    chunked: ChunkedDoc


@dataclass(frozen=True)
class StudyDataset:
    dataset: str
    split: str
    level: str
    documents: tuple[StudyDocument, ...]
    queries: tuple[StudyQuery, ...]
    coverage: dict
    fingerprints: dict[str, str]


def fingerprint(value) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _numeric_suffix(chunk_id: str) -> int:
    match = re.search(r"([0-9]+)$", chunk_id)
    if match is None:
        raise ValueError(f"Native chunk ID {chunk_id!r} must end in a numeric suffix")
    return int(match[1])


def _require_unique(values, label: str) -> set[str]:
    seen = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Each {label} must be a non-empty string")
        if value in seen:
            raise ValueError(f"Duplicate {label} {value!r}")
        seen.add(value)
    return seen


def load_native(path: Path) -> NativeDataset:
    """Read an exported, split-specific corpus of native chunks and graded qrels."""
    data = json.loads(path.read_text(encoding="utf-8"))
    documents = []
    for doc in data["documents"]:
        chunks = sorted(
            ((chunk["id"], chunk["text"]) for chunk in doc["chunks"]),
            key=lambda item: _numeric_suffix(item[0]),
        )
        _require_unique((str(_numeric_suffix(chunk_id)) for chunk_id, _ in chunks), "chunk order within document")
        documents.append(NativeDocument(doc["id"], doc.get("title", ""), tuple(chunks)))
    doc_ids = _require_unique((doc.doc_id for doc in documents), "document ID")
    chunk_ids = _require_unique((chunk_id for doc in documents for chunk_id, _ in doc.chunks), "chunk ID")
    queries = tuple(
        StudyQuery(query["id"], query["text"], query["source_doc_id"], query["relevance"])
        for query in data["queries"]
    )
    _require_unique((query.query_id for query in queries), "query ID")
    for query in queries:
        if query.source_doc_id not in doc_ids:
            raise ValueError(f"Query {query.query_id!r} has an unknown source document")
        unknown = query.relevance.keys() - chunk_ids
        if unknown:
            raise ValueError(f"Query {query.query_id!r} has unknown chunk qrels: {sorted(unknown)}")
        if any(type(grade) is not int or grade < 0 for grade in query.relevance.values()):
            raise ValueError(f"Query {query.query_id!r} requires nonnegative integer relevance grades")
    return NativeDataset(data["dataset"], data["split"], tuple(documents), queries, fingerprint(data))


def _token_ids(tokenizer, text: str, *, special_tokens: bool) -> list[int]:
    return tokenizer(text, add_special_tokens=special_tokens, truncation=False, verbose=False)["input_ids"]


def _candidate_limit(documents, limit_docs: int | None):
    if limit_docs is not None and limit_docs < 1:
        raise ValueError("limit_docs must be positive")
    return documents[:limit_docs]


def _assemble(
    encoder,
    *,
    dataset: str,
    split: str,
    level: str,
    limit_docs: int | None,
    document_count: int,
    candidates: dict[str, int],
    documents: list[StudyDocument],
    queries: tuple[StudyQuery, ...],
    excluded_documents: dict,
    input_fingerprint: str,
) -> StudyDataset:
    doc_ids = {doc.doc_id for doc in documents}
    unit_ids = doc_ids if level == "document" else {chunk.chunk_id for doc in documents for chunk in doc.chunks}
    included_queries, excluded_queries = [], {}
    for query in queries:
        rels = {unit: grade for unit, grade in query.relevance.items() if unit in unit_ids}
        if query.source_doc_id is not None and query.source_doc_id not in doc_ids:
            reason = "source_document_outside_eligible_corpus"
        elif not any(grade > 0 for grade in rels.values()):
            reason = "no_positive_eligible_relevance"
        elif not 0 < len(_token_ids(encoder.tokenizer, query.text, special_tokens=True)) <= encoder.max_length:
            reason = "query_input_outside_model_length"
        else:
            included_queries.append(StudyQuery(query.query_id, query.text, query.source_doc_id, rels))
            continue
        excluded_queries[query.query_id] = reason
    included_query_tuple = tuple(included_queries)
    coverage = {
        "document_limit": limit_docs,
        "source_documents": document_count,
        "candidate_documents": len(candidates),
        "eligible_documents": len(documents),
        "candidate_chunks": sum(candidates.values()),
        "eligible_chunks": sum(len(doc.chunks) for doc in documents),
        "excluded_documents": excluded_documents,
        "source_queries": len(queries),
        "eligible_queries": len(included_query_tuple),
        "eligible_query_source_documents": len({q.source_doc_id for q in included_query_tuple if q.source_doc_id}),
        "excluded_queries": excluded_queries,
        "eligible_positive_qrels": sum(sum(grade > 0 for grade in q.relevance.values()) for q in included_query_tuple),
        "source_positive_qrels": sum(sum(grade > 0 for grade in q.relevance.values()) for q in queries),
    }
    fingerprints = {
        "source": input_fingerprint,
        "corpus": fingerprint([
            (doc.doc_id, doc.title, doc.chunked.input_ids) for doc in documents
        ]),
        "chunks": fingerprint([
            (doc.doc_id, chunk.chunk_id, chunk.text, span, chunk.token_ids)
            for doc in documents for chunk, span in zip(doc.chunks, doc.chunked.spans, strict=True)
        ]),
        "queries": fingerprint([(q.query_id, q.text, q.source_doc_id) for q in included_query_tuple]),
        "qrels": fingerprint([(q.query_id, q.relevance) for q in included_query_tuple]),
    }
    return StudyDataset(dataset, split, level, tuple(documents), included_query_tuple, coverage, fingerprints)


def _overflow(chunked: ChunkedDoc, model_limit: int) -> dict:
    return {
        "reason": "full_late_document_exceeds_model_limit",
        "wrapped_tokens": len(chunked.input_ids),
        "model_limit": model_limit,
    }


def prepare_native(encoder, native: NativeDataset, *, limit_docs: int | None = None) -> StudyDataset:
    """Retain native IDs, independently tokenize text, and wrap each whole document once."""
    candidates, documents, excluded = {}, [], {}
    for doc in _candidate_limit(native.documents, limit_docs):
        candidates[doc.doc_id] = len(doc.chunks)
        tokens = [_token_ids(encoder.tokenizer, text, special_tokens=False) for _, text in doc.chunks]
        if not tokens or any(not ids for ids in tokens):
            excluded[doc.doc_id] = {"reason": "document_contains_no_chunks_or_empty_chunk"}
            continue
        group = group_input(encoder.tokenizer, tokens)
        chunked = ChunkedDoc(doc.doc_id, list(group.token_ids), list(group.spans), [text for _, text in doc.chunks])
        if len(chunked.input_ids) > encoder.max_length:
            excluded[doc.doc_id] = _overflow(chunked, encoder.max_length)
            continue
        chunks = tuple(
            AnchoredChunk(chunk_id, text, tuple(ids)) for (chunk_id, text), ids in zip(doc.chunks, tokens, strict=True)
        )
        documents.append(StudyDocument(doc.doc_id, doc.title, chunks, chunked))
    return _assemble(
        encoder, dataset=native.dataset, split=native.split, level="chunk", limit_docs=limit_docs,
        document_count=len(native.documents), candidates=candidates, documents=documents,
        queries=native.queries, excluded_documents=excluded, input_fingerprint=native.input_fingerprint,
    )


def prepare_scifact(
    encoder, path: Path, *, chunk_size: int = 256, limit_docs: int | None = None,
) -> StudyDataset:
    """Keep original title-plus-body tokenization and BEIR document-level qrels."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    corpus, query_texts, qrels = load_beir(path)
    candidates, documents, excluded = {}, [], {}
    for doc_id in _candidate_limit(tuple(corpus), limit_docs):
        source = corpus[doc_id]
        chunked = fixed_token_chunks(doc_id, doc_text(source), encoder.tokenizer, chunk_size)
        candidates[doc_id] = len(chunked.spans)
        if not chunked.spans:
            excluded[doc_id] = {"reason": "document_contains_no_chunks"}
        elif len(chunked.input_ids) > encoder.max_length:
            excluded[doc_id] = _overflow(chunked, encoder.max_length)
        else:
            documents.append(StudyDocument(doc_id, source["title"], anchored_chunks(chunked), chunked))
    queries = tuple(StudyQuery(query_id, text, None, qrels[query_id]) for query_id, text in query_texts.items())
    return _assemble(
        encoder, dataset="scifact", split="test", level="document", limit_docs=limit_docs,
        document_count=len(corpus), candidates=candidates, documents=documents,
        queries=queries, excluded_documents=excluded,
        input_fingerprint=fingerprint((list(corpus.items()), list(query_texts.items()), qrels)),
    )
