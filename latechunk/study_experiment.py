"""Aligned seven-policy retrieval quality and reproducible study configuration."""

from __future__ import annotations

import json
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from .anchored import encode_content_anchored
from .encoding import encode_late
from .group_encoding import encode_group_inputs
from .retrieval import ndcg_at_k, recall_at_k, search_docs
from .sticky import encode_sticky
from .study_data import StudyDataset, fingerprint

POLICIES = ("naive", "naive_prefix", "late", "long_late", "local_contextual", "sticky", "content_anchored")
CONTEXT_CAPS = (1024, 2048, 4096)
LONG_SPECIAL_TOKENS = "wrap"
CONFIG_SCHEMA = "latechunk.study.config.v1"
NONINFERIORITY_MARGIN = -0.02


@dataclass(frozen=True)
class StudyConfig:
    context_cap: int = 1024
    overlap: int = 64
    anchor_k: int = 4
    local_radius: int = 1
    batch_size: int = 16
    query_batch_size: int = 64
    k: int = 10

    def validate(self, max_length: int) -> None:
        if self.context_cap not in CONTEXT_CAPS:
            raise ValueError(f"context_cap must be one of {CONTEXT_CAPS}")
        if min(self.anchor_k, self.batch_size, self.query_batch_size, self.k) < 1 or self.local_radius < 0:
            raise ValueError("Study settings require positive anchor_k, batch sizes, and k, and nonnegative local_radius")
        from .encoding import validate_long_window

        validate_long_window(self.context_cap, self.overlap, max_length, special_tokens=LONG_SPECIAL_TOKENS)


def write_new_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def runtime_provenance(encoder) -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "device": str(encoder.device), "python": platform.python_version(), "torch": str(torch.__version__),
        "transformers": version("transformers"), "project_revision": revision,
    }


def encoder_provenance(encoder) -> dict:
    tokenizer = encoder.tokenizer
    return {
        "model": getattr(encoder, "model_name", None),
        "model_revision": getattr(encoder, "model_revision", None),
        "code_revision": getattr(encoder, "code_revision", None),
        "tokenizer": tokenizer.name_or_path,
        "tokenizer_revision": getattr(encoder, "tokenizer_revision", None),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_sha256": fingerprint(tokenizer.backend_tokenizer.to_str()) if getattr(tokenizer, "is_fast", False) else None,
        "max_length": encoder.max_length,
    }


def read_frozen_config(path: Path) -> dict:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Unsupported frozen study configuration")
    return frozen


def validate_phase(phase: str, dataset: StudyDataset, frozen: dict | None = None, encoder=None) -> None:
    if phase == "exploratory":
        if frozen is not None:
            raise ValueError("Frozen settings are used for held-out or transfer evaluation")
        return
    if dataset.level != "chunk":
        raise ValueError("Formal development and held-out phases require native chunk qrels")
    split = dataset.split.lower()
    if phase == "development":
        if frozen is not None or split not in {"dev", "development", "train", "validation", "val"}:
            raise ValueError("Development selection requires a native development/training split and no frozen config")
        return
    if phase not in {"held-out", "transfer"}:
        raise ValueError(f"Unknown study phase {phase!r}")
    if not frozen or split not in {"test", "held-out", "held_out"}:
        raise ValueError(f"{phase} evaluation requires a native test split and a frozen development configuration")
    if phase == "held-out" and dataset.dataset != frozen["dataset"]:
        raise ValueError("Frozen config dataset must match the held-out dataset")
    if phase == "transfer" and dataset.dataset == frozen["dataset"]:
        raise ValueError("Transfer evaluation requires a different dataset from the frozen development dataset")
    held_out_queries = {fingerprint(query.text) for query in dataset.queries}
    if held_out_queries.intersection(frozen["development_query_text_hashes"]):
        raise ValueError("Held-out queries overlap the effective development queries")
    if encoder is not None and encoder_provenance(encoder) != frozen["encoder_provenance"]:
        raise ValueError("Held-out encoder provenance differs from the frozen development encoder")


def write_frozen_config(path: Path, dataset: StudyDataset, config: StudyConfig, encoder_settings: dict, encoder) -> None:
    validate_phase("development", dataset)
    if not dataset.documents or not dataset.queries:
        raise ValueError("Cannot freeze configuration from development data without eligible documents and queries")
    write_new_json(path, {
        "schema": CONFIG_SCHEMA,
        "dataset": dataset.dataset,
        "development_split": dataset.split,
        "development_fingerprints": dataset.fingerprints,
        "development_query_text_hashes": sorted({fingerprint(query.text) for query in dataset.queries}),
        "study_config": asdict(config),
        "encoder_settings": encoder_settings,
        "encoder_provenance": encoder_provenance(encoder),
    })


def _encode_grouped(encoder, dataset: StudyDataset, method: str, config: StudyConfig) -> torch.Tensor:
    from .study_refresh import build_stateless_units, study_prefix

    locations = []
    for doc in dataset.documents:
        units = build_stateless_units(
            encoder, doc.chunks, method=method, token_cap=config.context_cap,
            local_radius=config.local_radius, prefix=study_prefix(doc) if method == "naive_prefix" else "",
        )
        locations.extend(units[chunk.chunk_id] for chunk in doc.chunks)
    inputs = list(dict.fromkeys(group for group, _ in locations))
    vectors = dict(zip(inputs, encode_group_inputs(encoder, inputs, config.batch_size), strict=True))
    return torch.stack([vectors[group][row] for group, row in locations])


def encode_policy(encoder, dataset: StudyDataset, method: str, config: StudyConfig) -> torch.Tensor:
    docs = [doc.chunked for doc in dataset.documents]
    if method in {"naive", "naive_prefix", "local_contextual"}:
        return _encode_grouped(encoder, dataset, method, config)
    if method == "late":
        return encode_late(encoder, docs, batch_size=config.batch_size)
    if method == "long_late":
        from .study_refresh import encode_study_long_late

        return encode_study_long_late(
            encoder, dataset.documents, token_cap=config.context_cap, overlap=config.overlap,
            batch_size=config.batch_size,
        )
    if method == "sticky":
        return encode_sticky(encoder, docs, batch_size=config.batch_size, token_cap=config.context_cap)
    if method == "content_anchored":
        return encode_content_anchored(
            encoder, docs, batch_size=config.batch_size, token_cap=config.context_cap, anchor_k=config.anchor_k,
        )
    raise ValueError(f"Unknown study policy {method!r}")


def _rank(queries: torch.Tensor, chunks: torch.Tensor, dataset: StudyDataset, k: int) -> list[list[str]]:
    if dataset.level == "document":
        owners = [index for index, doc in enumerate(dataset.documents) for _ in doc.chunks]
        _, top = search_docs(queries, chunks, owners, len(dataset.documents), top_k=k)
        ids = [doc.doc_id for doc in dataset.documents]
        return [[ids[index] for index in row] for row in top.tolist()]
    ids = [chunk.chunk_id for doc in dataset.documents for chunk in doc.chunks]
    rankings = []
    for batch in queries.split(32):
        similarities = batch @ chunks.T
        # Stable sorting resolves exact ties by the common retrieval order.
        top = torch.argsort(similarities, descending=True, stable=True)[:, :k]
        rankings.extend([ids[index] for index in row] for row in top.tolist())
    return rankings


def _metric_rows(rankings: list[list[str]], dataset: StudyDataset, k: int) -> dict:
    return {
        query.query_id: {
            "source_doc_id": query.source_doc_id, "ranking": ranking,
            f"ndcg@{k}": ndcg_at_k(ranking, query.relevance, k),
            f"recall@{k}": recall_at_k(ranking, query.relevance, k),
        }
        for query, ranking in zip(dataset.queries, rankings, strict=True)
    }


def source_paired_bootstrap(
    treatment: dict, baseline: dict, metric: str, *, replicates: int = 2000, seed: int = 0,
) -> dict | None:
    """Resample paired source-document clusters and aggregate over sampled queries."""
    if replicates < 1 or treatment.keys() != baseline.keys():
        raise ValueError("Bootstrap requires positive replicates and identical query IDs")
    groups: dict[str, list[float]] = {}
    for query_id, row in treatment.items():
        source = row["source_doc_id"]
        if source is None or source != baseline[query_id]["source_doc_id"]:
            return None
        groups.setdefault(source, []).append(row[metric] - baseline[query_id][metric])
    if len(groups) < 2:
        return None
    totals = [(sum(values), len(values)) for values in groups.values()]
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sample = rng.choices(totals, k=len(totals))
        draws.append(sum(total for total, _ in sample) / sum(count for _, count in sample))
    draws.sort()
    return {
        "estimate": sum(total for total, _ in totals) / sum(count for _, count in totals),
        "interval_95": [draws[int(0.025 * (replicates - 1))], draws[int(0.975 * (replicates - 1))]],
        "sources": len(totals), "replicates": replicates, "seed": seed,
    }


def run_quality(encoder, dataset: StudyDataset, config: StudyConfig) -> dict:
    config.validate(encoder.max_length)
    result = {
        "config": asdict(config) | {
            "long_late_special_tokens": LONG_SPECIAL_TOKENS,
            "target_pooling": "mean_of_original_chunk_span_then_l2",
            "context_cap_counts_special_tokens": True,
            "oversize_chunk_policy": "whole_chunk_singleton_at_matched_cap_when_model_allows",
            "independent_prefix": "document_title_or_first_chunk_lead_up_to_64_tokens_in_prompt_only",
        },
        "dataset": dataset.dataset,
        "split": dataset.split,
        "evaluation_level": dataset.level,
        "coverage": dataset.coverage,
        "fingerprints": dataset.fingerprints,
        "encoder": encoder_provenance(encoder),
        "runtime": runtime_provenance(encoder),
    }
    metrics = (f"ndcg@{config.k}", f"recall@{config.k}")
    if not dataset.documents or not dataset.queries:
        return result | {
            "status": "no_eligible_data",
            "timing": None,
            "methods": {},
            "content_anchored_vs_long_late_source_paired_bootstrap": dict.fromkeys(metrics),
        }
    expected_rows = dataset.coverage["eligible_chunks"]
    started = perf_counter()
    query_vectors = encoder.encode_texts([q.text for q in dataset.queries], batch_size=config.query_batch_size)
    query_seconds = perf_counter() - started
    methods = {}
    for method in POLICIES:
        started = perf_counter()
        chunk_vectors = encode_policy(encoder, dataset, method, config)
        encode_seconds = perf_counter() - started
        if chunk_vectors.shape[0] != expected_rows:
            raise ValueError(f"{method} returned {chunk_vectors.shape[0]} vectors for {expected_rows} retrieval chunks")
        started = perf_counter()
        rows = _metric_rows(_rank(query_vectors, chunk_vectors, dataset, config.k), dataset, config.k)
        methods[method] = {
            **{metric: mean(row[metric] for row in rows.values()) for metric in metrics},
            "encode_seconds": encode_seconds,
            "retrieve_evaluate_seconds": perf_counter() - started,
            "batch_size": config.batch_size,
            "chunks": expected_rows,
            "chunk_fingerprint": dataset.fingerprints["chunks"],
            "per_query": rows,
        }
        del chunk_vectors
    for row in methods.values():
        row["deltas"] = {
            baseline: {metric: row[metric] - methods[baseline][metric] for metric in metrics}
            for baseline in ("naive", "late", "long_late")
        }
    anchored = methods["content_anchored"]["per_query"]
    positional = methods["long_late"]["per_query"]
    paired = {metric: source_paired_bootstrap(anchored, positional, metric) for metric in metrics}
    return result | {
        "status": "complete",
        "timing": {
            "method_scope": "standalone_corpus_planning_encoding_and_pooling_no_cross_policy_cache",
            "shared_query_encode_seconds": query_seconds,
            "shared_query_batch_size": config.query_batch_size,
        },
        "methods": methods,
        "content_anchored_vs_long_late_source_paired_bootstrap": paired,
    }
