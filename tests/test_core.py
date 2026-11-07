"""Model-free tests for Algorithm 2 stitching, doc-level max aggregation, and metrics."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latechunk.encoding import long_late_token_embeddings, macro_windows  # noqa: E402
from latechunk.retrieval import ndcg_at_k, recall_at_k, search_docs  # noqa: E402

CLS, SEP = 101, 102


def contextless(seqs):
    """Fake encoder: each output row = [token id, position within window]."""
    return [torch.tensor([[float(t), float(p)] for p, t in enumerate(s)]) for s in seqs]


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
