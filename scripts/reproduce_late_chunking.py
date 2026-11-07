"""Section 5.1 reproduction: independent (naive) vs late chunking on BEIR SciFact
with jina-embeddings-v2-small-en and 256-token fixed chunks (Late Chunking paper,
Sec. 4.1), plus Long Late Chunking (Algorithm 2).

Example:
    python scripts/reproduce_late_chunking.py
    python scripts/reproduce_late_chunking.py --methods naive late long_late --long-window 256 --long-overlap 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latechunk.chunking import fixed_token_chunks  # noqa: E402
from latechunk.data import doc_text, download_beir, load_beir  # noqa: E402
from latechunk.encoding import Encoder, encode_late, encode_naive, long_late_token_embeddings  # noqa: E402
from latechunk.retrieval import evaluate, search_docs  # noqa: E402

# Reference values (nDCG@10 x100) for jina-embeddings-v2-small-en, fixed 256-token chunks,
# as reported in the jina-ai/late-chunking README. Check them against Table 2 of the paper.
REFERENCE = {"scifact": {"naive": 64.20, "late": 66.10}}


def sanity_checks(enc: Encoder, docs, n: int = 50) -> dict:
    """Checks for token spans, special tokens, normalization, and Algorithm 2."""
    tok = enc.tokenizer
    sample = [d for d in docs if d.spans][:n]
    report = {}

    # 1. spans cover exactly the content tokens, contiguously, with no special tokens
    for d in sample:
        assert d.input_ids[0] == tok.cls_token_id and d.input_ids[-1] == tok.sep_token_id
        assert d.spans[0][0] == 1 and d.spans[-1][1] == len(d.input_ids) - 1
        assert all(a[1] == b[0] for a, b in zip(d.spans, d.spans[1:]))
    report["spans_ok"] = True

    # 2. how often re-tokenizing a chunk's text (naive path) differs from its span tokens
    mism = total = 0
    for d in sample:
        for (s, e), t in zip(d.spans, d.texts):
            total += 1
            mism += tok(t, add_special_tokens=False)["input_ids"] != d.input_ids[s:e]
    report["naive_retokenization_mismatch_rate"] = mism / max(total, 1)

    # 3. Long Late Chunking with window >= doc length must equal single-pass late chunking
    embed = enc.embed_fn()
    max_diff = 0.0
    for d in sample[:10]:
        full = enc.token_embeddings([d.input_ids])[0]
        llc = long_late_token_embeddings(d.input_ids, embed, len(d.input_ids) + 10, 0)
        max_diff = max(max_diff, (full - llc).abs().max().item())
    report["llc_equals_late_when_fits_maxabs"] = max_diff
    assert max_diff < 1e-4, max_diff

    # 4. batching/padding must not change outputs (ALiBi + attention mask)
    by_len = sorted(sample, key=lambda d: len(d.input_ids))
    pair = [by_len[0], by_len[-1]]  # maximize padding on the shorter one
    batched = enc.token_embeddings([d.input_ids for d in pair])
    alone = [enc.token_embeddings([d.input_ids])[0] for d in pair]
    report["padding_invariance_maxabs"] = max((a - b).abs().max().item() for a, b in zip(batched, alone))

    # 5. normalization
    v = encode_late(enc, sample[:5])
    report["late_norms_range"] = [v.norm(dim=-1).min().item(), v.norm(dim=-1).max().item()]
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--model", default="jinaai/jina-embeddings-v2-small-en")
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--methods", nargs="+", default=["naive", "late", "long_late"],
                   choices=["naive", "late", "long_late"])
    p.add_argument("--long-window", type=int, default=256, help="Algorithm 2 macro-window size (tokens)")
    p.add_argument("--long-overlap", type=int, default=64, help="Algorithm 2 overlap (tokens)")
    p.add_argument("--long-special-tokens", choices=["raw", "wrap"], default="raw")
    p.add_argument("--no-title", action="store_true", help="Do not prepend the BEIR title to the document text")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit-docs", type=int, default=None, help="Debug: only use the first N documents")
    p.add_argument("--device", default=None)
    p.add_argument("--skip-sanity", action="store_true")
    args = p.parse_args()

    torch.manual_seed(0)
    corpus, queries, qrels = load_beir(download_beir(args.dataset, args.data_dir))
    doc_ids = list(corpus)[: args.limit_docs]
    print(f"{args.dataset}: {len(doc_ids)} docs, {len(queries)} test queries")

    enc = Encoder(args.model, device=args.device, max_length=args.max_length)
    print(f"Model {args.model} on {enc.device}")

    t0 = time.time()
    docs = [fixed_token_chunks(i, doc_text(corpus[i], not args.no_title), enc.tokenizer, args.chunk_size)
            for i in doc_ids]
    chunk_doc = [j for j, d in enumerate(docs) for _ in d.spans]
    lens = sorted(len(d.input_ids) for d in docs)
    print(f"Chunked in {time.time() - t0:.1f}s: {len(chunk_doc)} chunks; "
          f"doc tokens median={lens[len(lens) // 2]} max={lens[-1]}; "
          f"docs with no chunks={sum(not d.spans for d in docs)}")

    results = {"config": {k: str(v) for k, v in vars(args).items()}}
    if not args.skip_sanity:
        results["sanity"] = sanity_checks(enc, docs)
        print("Sanity:", json.dumps(results["sanity"], indent=2))

    qids = list(queries)
    q_emb = enc.encode_texts([queries[q] for q in qids], batch_size=64)

    for method in args.methods:
        t0 = time.time()
        if method == "naive":
            c_emb = encode_naive(enc, docs, batch_size=64)
        elif method == "late":
            c_emb = encode_late(enc, docs, batch_size=args.batch_size)
        else:
            c_emb = encode_late(enc, docs, batch_size=args.batch_size, long_window=args.long_window,
                                long_overlap=args.long_overlap, special_tokens=args.long_special_tokens)
        enc_time = time.time() - t0
        assert c_emb.shape[0] == len(chunk_doc)

        _, top = search_docs(q_emb, c_emb, chunk_doc, len(docs), top_k=100)
        run = {q: [doc_ids[j] for j in row.tolist()] for q, row in zip(qids, top)}
        m = evaluate(run, qrels, k=10)
        m["encode_seconds"] = enc_time
        ref = REFERENCE.get(args.dataset, {}).get(method)
        results[method] = m
        print(f"[{method:9s}] nDCG@10={100 * m['ndcg@10']:.2f}  Recall@10={100 * m['recall@10']:.2f}"
              + (f"  (reference nDCG@10={ref:.2f})" if ref is not None else "")
              + f"  encode={enc_time:.0f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"repro_{args.dataset}_cs{args.chunk_size}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
