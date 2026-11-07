"""Exact chunk search, chunk -> document max-aggregation, and nDCG@k / Recall@k."""

from __future__ import annotations

import math
from typing import Sequence

import torch


def search_docs(
    q_emb: torch.Tensor,
    chunk_emb: torch.Tensor,
    chunk_doc: Sequence[int],
    n_docs: int,
    top_k: int = 100,
    q_batch: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each document by its best chunk (max cosine; inputs are L2-normalized).

    Returns (scores, doc_indices), each (Q, top_k), sorted descending.
    """
    owner = torch.as_tensor(chunk_doc, dtype=torch.long)
    all_s, all_i = [], []
    for b in range(0, q_emb.shape[0], q_batch):
        sims = q_emb[b : b + q_batch] @ chunk_emb.T  # (q, C)
        doc_s = torch.full((sims.shape[0], n_docs), float("-inf"))
        doc_s.scatter_reduce_(1, owner.expand_as(sims), sims, reduce="amax", include_self=True)
        s, i = doc_s.topk(min(top_k, n_docs), dim=1)
        all_s.append(s)
        all_i.append(i)
    return torch.cat(all_s), torch.cat(all_i)


def ndcg_at_k(ranked: Sequence[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(r + 2) for r, d in enumerate(ranked[:k]))
    ideal = sorted((v for v in rels.values() if v > 0), reverse=True)[:k]
    idcg = sum(v / math.log2(r + 2) for r, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], rels: dict[str, int], k: int = 10) -> float:
    relevant = {d for d, v in rels.items() if v > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def evaluate(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 10) -> dict[str, float]:
    qids = [q for q in qrels if any(v > 0 for v in qrels[q].values())]
    return {
        f"ndcg@{k}": sum(ndcg_at_k(run.get(q, []), qrels[q], k) for q in qids) / len(qids),
        f"recall@{k}": sum(recall_at_k(run.get(q, []), qrels[q], k) for q in qids) / len(qids),
        "n_queries": len(qids),
    }
