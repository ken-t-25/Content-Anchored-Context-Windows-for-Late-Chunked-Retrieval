from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers import BertTokenizerFast

from latechunk.encoding import Encoder
from latechunk.study_data import load_native, prepare_native
from latechunk.study_experiment import (
    POLICIES,
    StudyConfig,
    read_frozen_config,
    run_quality,
    source_paired_bootstrap,
    validate_phase,
    write_frozen_config,
)
from scripts import reproduce_late_chunking as legacy
from scripts.run_study import parse_args


@pytest.fixture
def study_encoder(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\ngamma\ntitle\n", encoding="utf-8")
    encoder = Encoder.__new__(Encoder)
    encoder.tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    encoder.max_length = 1536
    encoder.device = "cpu"
    encoder.model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
    encoder.model_name = "test"
    encoder.model_revision = "test-model-rev"
    encoder.code_revision = None
    encoder.tokenizer_revision = "test-tokenizer-rev"
    calls = []

    def embeddings(inputs):
        calls.append(tuple(tuple(sequence) for sequence in inputs))
        return [
            torch.tensor([[float(token), float(position + 1), float(len(sequence))]
                          for position, token in enumerate(sequence)])
            for sequence in inputs
        ]

    encoder.token_embeddings = embeddings
    return encoder, calls


@pytest.fixture
def manifest(tmp_path):
    data = {
        "dataset": "native-test",
        "split": "dev",
        "documents": [
            {"id": "A", "title": "title", "chunks": [
                {"id": "A:20", "text": "gamma"},
                {"id": "A:2", "text": "alpha alpha"},
            ]},
            {"id": "B", "chunks": [{"id": "B:1", "text": "beta " * 1535}]},
            {"id": "C", "chunks": [{"id": "C:0", "text": "gamma beta"}]},
        ],
        "queries": [
            {"id": "q1", "text": "alpha", "source_doc_id": "A", "relevance": {"A:2": 2, "B:1": 1}},
            {"id": "q2", "text": "beta", "source_doc_id": "B", "relevance": {"B:1": 1}},
            {"id": "q3", "text": "gamma", "source_doc_id": "C", "relevance": {"C:0": 1}},
        ],
    }

    def write(name, **overrides):
        path = tmp_path / name
        path.write_text(json.dumps(data | overrides), encoding="utf-8")
        return path

    return data, write


def test_native_preserves_stable_order_and_filters_coverage_once(study_encoder, manifest):
    encoder, _ = study_encoder
    _, write = manifest
    dataset = prepare_native(encoder, load_native(write("dev.json")))
    first, second = dataset.documents
    assert first.doc_id == "A"
    assert second.doc_id == "C"
    assert [chunk.chunk_id for chunk in first.chunks] == ["A:2", "A:20"]
    assert first.chunked.spans == [(1, 3), (3, 4)]
    assert first.chunked.input_ids == [
        encoder.tokenizer.cls_token_id, 5, 5, 7, encoder.tokenizer.sep_token_id,
    ]
    assert [q.query_id for q in dataset.queries] == ["q1", "q3"]
    assert dataset.queries[0].relevance == {"A:2": 2}
    assert dataset.coverage["candidate_chunks"] == 4
    assert dataset.coverage["eligible_chunks"] == 3
    assert dataset.coverage["excluded_documents"]["B"] == {
        "reason": "full_late_document_exceeds_model_limit",
        "wrapped_tokens": 1537,
        "model_limit": 1536,
    }
    assert dataset.coverage["excluded_queries"] == {"q2": "source_document_outside_eligible_corpus"}
    assert dataset.coverage["eligible_positive_qrels"] == 2


def test_native_qrels_reject_unknown_chunk_id(manifest):
    data, write = manifest
    queries = [data["queries"][0] | {"relevance": {"missing:0": 1}}]
    with pytest.raises(ValueError, match="unknown chunk qrels"):
        load_native(write("unknown.json", queries=queries))


def test_development_config_freezes_provenance_and_rejects_query_leakage(study_encoder, manifest, tmp_path):
    encoder, _ = study_encoder
    data, write = manifest
    dev = prepare_native(encoder, load_native(write("dev.json")))
    path = tmp_path / "frozen.json"
    config = StudyConfig(context_cap=1024, overlap=48, anchor_k=8)
    settings = {"model": "test", "model_revision": "test-model-rev", "tokenizer_revision": "test-tokenizer-rev",
                "max_length": 1536}
    write_frozen_config(path, dev, config, settings, encoder)
    with pytest.raises(FileExistsError):
        write_frozen_config(path, dev, config, settings, encoder)
    frozen = read_frozen_config(path)
    assert StudyConfig(**frozen["study_config"]) == config
    assert frozen["encoder_settings"] == settings
    test = prepare_native(encoder, load_native(write("test.json", split="test")))
    with pytest.raises(ValueError, match="overlap"):
        validate_phase("held-out", test, frozen, encoder)
    true_test = replace(test, queries=tuple(replace(query, text=query.text + " beta") for query in test.queries))
    validate_phase("held-out", true_test, frozen, encoder)
    transfer = replace(true_test, dataset="native-transfer")
    validate_phase("transfer", transfer, frozen, encoder)
    with pytest.raises(ValueError, match="different dataset"):
        validate_phase("transfer", true_test, frozen, encoder)
    encoder.model_revision = "changed"
    with pytest.raises(ValueError, match="provenance"):
        validate_phase("held-out", true_test, frozen, encoder)
    with pytest.raises(ValueError, match="development/training split"):
        validate_phase("development", true_test)


def test_cli_rejects_tuning_on_held_out_and_requires_native_for_development(tmp_path):
    flags = ["--out", str(tmp_path / "test.json")]
    with pytest.raises(SystemExit):
        parse_args(["--native", "test.json", "--phase", "held-out", "--frozen-config", "dev.json",
                    "--anchor-k", "8", *flags])
    with pytest.raises(SystemExit):
        parse_args(["--scifact", "data/scifact", "--phase", "development", *flags])


def test_paired_bootstrap_preserves_all_queries_within_each_sampled_document():
    treatment = {"a": {"source_doc_id": "ab", "metric": 1}, "b": {"source_doc_id": "ab", "metric": 0},
                 "c": {"source_doc_id": "c", "metric": 1}}
    baseline = {query: row | {"metric": 0} for query, row in treatment.items()}
    comparison = source_paired_bootstrap(treatment, baseline, "metric", replicates=128)
    assert comparison["estimate"] == pytest.approx(2 / 3)
    assert comparison["interval_95"] == [0.5, 1.0]
    assert comparison["sources"] == 2
    assert source_paired_bootstrap({"a": treatment["a"]}, {"a": baseline["a"]}, "metric") is None


def test_all_seven_methods_rank_identical_native_ids(study_encoder, manifest):
    encoder, calls = study_encoder
    data, write = manifest
    docs = [data["documents"][0], data["documents"][2] | {"chunks": [
        {"id": "C:0", "text": "alpha " * 1050},
        {"id": "C:1", "text": "gamma " * 300},
    ]}]
    queries = [data["queries"][0] | {"relevance": {"A:2": 2}}, data["queries"][2]]
    dataset = prepare_native(encoder, load_native(write("seven.json", documents=docs, queries=queries)))
    config = StudyConfig(batch_size=1, query_batch_size=2)
    result = run_quality(encoder, dataset, config)
    assert tuple(result["methods"]) == POLICIES
    assert result["coverage"]["eligible_chunks"] == 4
    assert len(calls[0]) == 2
    assert all(len(batch) == 1 for batch in calls[1:])
    expected = {"A:2", "A:20", "C:0", "C:1"}
    for row in result["methods"].values():
        assert row["chunks"] == 4
        assert row["chunk_fingerprint"] == result["fingerprints"]["chunks"]
        assert set(row["per_query"]) == {"q1", "q3"}
        assert all(set(query["ranking"]) == expected for query in row["per_query"].values())
    assert result["content_anchored_vs_long_late_source_paired_bootstrap"]["ndcg@10"]["sources"] == 2


def test_legacy_cli_records_actual_batches_and_filters_limited_qrels(study_encoder, tmp_path, monkeypatch):
    encoder, calls = study_encoder
    corpus = {"A": {"title": "title", "text": "alpha beta"}, "B": {"title": "", "text": "gamma"}}
    queries = {"q1": "alpha", "q2": "beta", "q3": "gamma"}
    qrels = {"q1": {"A": 1, "B": 1}, "q2": {"A": 1}, "q3": {"B": 1}}
    monkeypatch.setattr(legacy, "Encoder", lambda *args, **kwargs: encoder)
    monkeypatch.setattr(legacy, "download_beir", lambda *args: tmp_path)
    monkeypatch.setattr(legacy, "load_beir", lambda path: (corpus, queries, qrels))
    monkeypatch.setattr(sys, "argv", [
        "legacy", "--methods", "naive", "late", "--chunk-size", "1", "--skip-sanity", "--limit-docs", "1",
        "--batch-size", "1", "--independent-batch-size", "2", "--query-batch-size", "2",
        "--out-dir", str(tmp_path),
    ])
    legacy.main()
    result = json.loads((tmp_path / "repro_scifact_cs1.json").read_text(encoding="utf-8"))
    assert result["actual_batch_sizes"] == {"queries": 2, "corpus_by_method": {"naive": 2, "late": 1}}
    assert [len(batch) for batch in calls] == [2, 2, 1, 1]
    for method in ("naive", "late"):
        assert result[method]["n_queries"] == 2
        assert result[method]["recall@10"] == 1


def test_no_shared_eligible_documents_still_report_coverage(study_encoder, manifest):
    encoder, calls = study_encoder
    data, write = manifest
    dataset = prepare_native(encoder, load_native(write(
        "only-overlength.json", documents=data["documents"][1:2], queries=data["queries"][1:2],
    )))
    result = run_quality(encoder, dataset, StudyConfig())
    assert result["status"] == "no_eligible_data"
    assert result["methods"] == {}
    assert result["coverage"]["eligible_documents"] == 0
    assert result["coverage"]["excluded_documents"]["B"]["wrapped_tokens"] == 1537
    assert calls == []
