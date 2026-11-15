# Content-Anchored Context Windows for Late-Chunked Retrieval

## 5.1 Reproduction: independent vs. late chunking (Late Chunking paper, Sec. 4.1)

Setup: `jina-embeddings-v2-small-en`, fixed 256-token chunks, BEIR SciFact (test split), exact search,
document score = max cosine over its chunks, nDCG@10 / Recall@10.

Requires `transformers<5`: the model's remote code (`jinaai/jina-bert-implementation`, pinned in
`latechunk/encoding.py`) imports a helper that was removed in transformers 5.

```bash
conda create -n latechunk python=3.11 -y && conda activate latechunk
pip install -r requirements.txt
python scripts/reproduce_late_chunking.py                 # naive, late, long_late (window 256, overlap 64)
python scripts/reproduce_late_chunking.py --limit-docs 200 --methods naive late   # quick smoke run
pytest tests                                              # model-free tests (Algorithm 2 stitching, metrics)
```

SciFact is downloaded to `data/` and results are written to `results/repro_scifact_cs256.json`.
Reference nDCG@10 (jina-ai/late-chunking README; check against Table 2 of the paper): naive 64.20, late 66.10.

### Implementation notes
- **Same retrieval units.** Each document (`title + " " + text`) is tokenized once, with special tokens.
  Chunks are spans of 256 content tokens. Naive chunking encodes the character substring of each span on its own.
  Late chunking mean-pools the same span from a single document pass.
- **Special tokens.** Spans never include `[CLS]`/`[SEP]`. Naive/query embeddings use the model's standard
  mask-weighted mean pooling, which includes them, as jina-v2's `encode` does. All vectors are L2-normalized.
- **Long Late Chunking (Algorithm 2)** (`latechunk/encoding.py`): windows of `--long-window` tokens with step
  `window - overlap`. Ownership rule: an overlapping token belongs to the earlier window, so each later window
  drops its first `overlap` outputs. `--long-special-tokens raw` slices the full `[CLS] … [SEP]` sequence, like the
  reference code; `wrap` gives each window its own `[CLS]`/`[SEP]`. SciFact abstracts fit in 8192 tokens, so
  the default `long_late` run uses a small window (256) to exercise the algorithm.
- **Sanity checks** (run before the evaluation): span contiguity, coverage, and special-token exclusion; the rate at
  which naive re-tokenization differs from the span tokens; Long LC with window ≥ length equals single-pass late
  chunking; padding invariance; vector norms.

## Independent + prefix

The four-method SciFact comparison uses the original title-plus-body tokenization and 256-token retrieval
chunks. `naive_prefix` prepends up to 64 title tokens to each existing chunk at encoding time. This
intentionally repeats the title in the first chunk: the prefix and the title already in that chunk are both
included in its input. `naive_prefix` uses the same mean pooling and L2 normalization as `naive`.

```bash
python scripts/reproduce_late_chunking.py --methods naive late long_late naive_prefix --device cpu --out-dir results/scifact-four-methods
```

### SciFact test results

All four methods used the same 5,183 documents, 9,356 retrieval chunks, and 300 queries.
Encoding time is for one full-corpus CPU run. See the [recorded configuration and metrics](benchmarks/scifact-four-methods.json).

| Method | nDCG@10 (×100) | Recall@10 | Encoding time (s) |
|---|---:|---:|---:|
| `naive` | 64.18 | 76.59% | 421.4 |
| `late` | 65.78 | 76.76% | 578.6 |
| `long_late` | 65.04 | 76.59% | 798.6 |
| `naive_prefix` | 64.53 | 76.51% | 522.4 |

| `naive_prefix` relative to | Δ nDCG@10 points (×100) | Δ Recall@10 percentage points | Δ encoding time (s) |
|---|---:|---:|---:|
| `naive` | +0.35 | −0.08 | +101.0 |
| `late` | −1.25 | −0.25 | −56.1 |

The separate `--no-title --methods naive naive_prefix --out-dir results/prefix` comparison uses body-only
retrieval chunks and adds the title from metadata only in `naive_prefix`.
