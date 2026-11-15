"""Model-free tests for Algorithm 2 stitching, doc-level max aggregation, and metrics."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latechunk.chunking import ChunkedDoc  # noqa: E402
from latechunk.encoding import (  # noqa: E402
    Encoder,
    encode_naive,
    encode_naive_prefix,
    long_late_token_embeddings,
    macro_windows,
)
from latechunk.retrieval import ndcg_at_k, recall_at_k, search_docs  # noqa: E402
from scripts import reproduce_late_chunking as repro  # noqa: E402

CLS, SEP = 101, 102


def contextless(seqs):
    """Fake encoder: each output row = [token id, position within window]."""
    return [torch.tensor([[float(t), float(p)] for p, t in enumerate(s)]) for s in seqs]


@pytest.fixture
def prefix_encoder(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ntitle\nalpha\nbeta\n", encoding="utf-8")
    encoder = Encoder.__new__(Encoder)
    encoder.tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    encoder.max_length = 128
    encoder.model = SimpleNamespace(config=SimpleNamespace(hidden_size=2))
    encoder.token_embeddings = contextless
    return encoder


def test_missing_prefix_matches_naive(prefix_encoder):
    docs = [ChunkedDoc("doc", [], [], ["alpha beta", "alpha"])]
    torch.testing.assert_close(encode_naive_prefix(prefix_encoder, docs, {}), encode_naive(prefix_encoder, docs))


@pytest.mark.parametrize("max_length,expected_prefix_len", [(128, 64), (5, 1), (4, 0)])
def test_prefix_reserves_room_for_chunk_and_special_tokens(prefix_encoder, max_length, expected_prefix_len):
    prefix_encoder.max_length = max_length
    docs = [ChunkedDoc("doc", [], [], ["alpha beta"])]
    vectors = encode_naive_prefix(prefix_encoder, docs, {"doc": "title " * 80})
    tok = prefix_encoder.tokenizer
    expected_input = "title " * expected_prefix_len + "alpha beta"
    expected_ids = tok(expected_input)["input_ids"]
    expected = torch.nn.functional.normalize(contextless([expected_ids])[0].mean(dim=0, keepdim=True), dim=-1)
    torch.testing.assert_close(vectors, expected)


def test_oversized_chunk_raises_before_encoding(prefix_encoder):
    prefix_encoder.max_length = 3
    docs = [ChunkedDoc("doc", [], [], ["alpha beta"])]
    with pytest.raises(ValueError, match="Chunk in document doc exceeds max_length=3 before prefixing"):
        encode_naive_prefix(prefix_encoder, docs, {"doc": "title"})


def test_four_method_comparison_shares_original_chunks(prefix_encoder, monkeypatch, tmp_path):
    corpus = {
        "a": {"title": "title", "text": "alpha " * 256 + "beta"},
        "b": {"title": "beta", "text": "alpha"},
    }
    methods = ["naive", "late", "long_late", "naive_prefix"]
    prefix_encoder.max_length = 8192
    prefix_encoder.device = "cpu"
    monkeypatch.setattr(repro, "Encoder", lambda *args, **kwargs: prefix_encoder)
    monkeypatch.setattr(repro, "download_beir", lambda *args: tmp_path)
    monkeypatch.setattr(repro, "load_beir", lambda path: (corpus, {"q": "alpha"}, {"q": {"a": 1}}))
    monkeypatch.setattr(sys, "argv", ["repro", "--methods", *methods, "--skip-sanity", "--out-dir", str(tmp_path)])

    expected = [
        ("a", 0, (1, 257), "title" + " alpha" * 255),
        ("a", 1, (257, 259), "alpha beta"),
        ("b", 0, (1, 3), "beta alpha"),
    ]
    seen = []

    def snapshot(docs):
        return [
            (doc.doc_id, index, span, text)
            for doc in docs
            for index, (span, text) in enumerate(zip(doc.spans, doc.texts, strict=True))
        ]

    def record(name, encode):
        def wrapped(encoder, docs, *args, **kwargs):
            seen.append("long_late" if "long_window" in kwargs else name)
            assert snapshot(docs) == expected
            vectors = encode(encoder, docs, *args, **kwargs)
            assert snapshot(docs) == expected
            assert vectors.shape[0] == len(expected)
            return vectors

        return wrapped

    monkeypatch.setattr(repro, "encode_naive", record("naive", repro.encode_naive))
    monkeypatch.setattr(repro, "encode_late", record("late", repro.encode_late))
    monkeypatch.setattr(repro, "encode_naive_prefix", record("naive_prefix", repro.encode_naive_prefix))
    repro.main()
    assert seen == methods


def test_prefix_duplicates_existing_title_in_first_chunk(prefix_encoder, monkeypatch):
    tok = prefix_encoder.tokenizer
    docs = [repro.fixed_token_chunks("a", "title alpha beta", tok, chunk_size=2)]
    received = []

    def capture(ids, batch_size):
        received.extend(ids)
        return torch.empty(len(ids), 2)

    monkeypatch.setattr(prefix_encoder, "encode_token_ids", capture)
    encode_naive_prefix(prefix_encoder, docs, {"a": "title"})
    assert received == tok(["title title alpha", "title beta"])["input_ids"]


@pytest.mark.parametrize("n,window,overlap", [(10, 6, 2), (9, 6, 2), (6, 6, 2), (3, 6, 2), (100, 16, 0), (101, 16, 15)])
def test_windows_partition(n, window, overlap):
    owned = []
    for s, e, k in macro_windows(n, window, overlap):
        assert e - s <= window
        owned.extend(range(s + k, e))
    assert owned == list(range(n))  # every position owned exactly once, in order


@pytest.mark.parametrize("mode", ["raw", "wrap"])
@pytest.mark.parametrize("n,window,overlap", [(50, 16, 4), (17, 16, 4), (12, 16, 4)])
def test_long_late_stitching(mode, n, window, overlap):
    ids = [CLS, *range(1000, 1000 + n - 2), SEP]
    emb = long_late_token_embeddings(ids, contextless, window, overlap, mode, CLS, SEP)
    assert emb.shape == (n, 2)
    content = slice(1, n - 1)
    # token identity is preserved at every content position
    assert emb[content, 0].tolist() == [float(t) for t in ids[1:-1]]
    if mode == "raw":
        wins = macro_windows(n, window, overlap)
        for s, e, k in wins:  # each owned position came from its owning window
            assert emb[s + k : e, 1].tolist() == [float(p) for p in range(k, e - s)]
    else:
        # wrapped windows: content never sits at window position 0 ([CLS])
        assert (emb[content, 1] >= 1).all()


def test_long_late_equals_full_when_fits():
    ids = [CLS, *range(1000, 1020), SEP]
    full = contextless([ids])[0]
    assert torch.equal(long_late_token_embeddings(ids, contextless, 64, 8), full)


def test_max_aggregation():
    chunk_emb = torch.eye(3)
    chunk_doc = [0, 0, 1]
    q = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    s, i = search_docs(q, chunk_emb, chunk_doc, n_docs=2, top_k=2)
    assert i[:, 0].tolist() == [0, 1]
    assert s[:, 0].tolist() == [1.0, 1.0]


def test_metrics():
    rels = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], rels) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "a"], rels) == pytest.approx((1 / 1.584962500721156) / (1 + 1 / 1.584962500721156))
    assert recall_at_k(["c", "a"], rels) == 0.5
