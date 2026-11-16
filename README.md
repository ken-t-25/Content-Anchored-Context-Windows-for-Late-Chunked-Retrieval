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

The four original results appear in the 256-token rows of the [chunk-size ablation](#scifact-chunk-size-ablation).
See the [recorded configuration and metrics](benchmarks/scifact-four-methods.json) for that run.

| `naive_prefix` relative to | Δ nDCG@10 points (×100) | Δ Recall@10 percentage points | Δ encoding time (s) |
|---|---:|---:|---:|
| `naive` | +0.35 | −0.08 | +101.0 |
| `late` | −1.25 | −0.25 | −56.1 |

The separate `--no-title --methods naive naive_prefix --out-dir results/prefix` comparison uses body-only
retrieval chunks and adds the title from metadata only in `naive_prefix`.

## Local Contextual

The default window includes the target retrieval chunk and up to one adjacent chunk on each side, in
document order. The model encodes the window jointly, then mean-pools and normalizes only the target
tokens. Targets in the same document that share identical window bounds reuse one encoder pass.
The 1,024-token context cap includes special tokens. Neighbours are added whole, nearest first and
left first on ties. A target longer than the cap is encoded alone if it fits the model limit.

```bash
python scripts/reproduce_late_chunking.py --methods local_contextual --device cpu --out-dir results/scifact-local-contextual
```

## SciFact chunk-size ablation

Each size uses title-plus-body chunking on the same 5,183 documents and 300 test queries. All five methods
share the retrieval chunks at a given size. Long Late Chunking uses 256-token windows with 64-token overlap.
Local Contextual uses one neighbor per side and a 1,024-token cap at every size. The 256-token rows reuse the
earlier runs. Encoding times are from individual full-corpus CPU runs.

| Chunk size (tokens) | Retrieval chunks | Local target windows covering the whole document | Local − Late nDCG@10 (points) |
|---:|---:|---:|---:|
| 256 | 9,356 | 89.07% (8,333 / 9,356) | −0.14 |
| 128 | 16,193 | 30.92% (5,007 / 16,193) | −0.50 |
| 64 | 29,726 | 2.34% (697 / 29,726) | +1.82 |

Coverage counts one window per target chunk, even when multiple targets share an encoder pass. At 64 tokens,
Local Contextual scored highest on both retrieval metrics and took about twice as long to encode the corpus
as full Late Chunking. The ablation changes both retrieval granularity and the context available per target.

| Chunk size (tokens) | Method | nDCG@10 (×100) | Recall@10 | Encoding time (s) |
|---:|---|---:|---:|---:|
| 256 | `naive` | 64.18 | 76.59% | 421.4 |
| 256 | `naive_prefix` | 64.53 | 76.51% | 522.4 |
| 256 | `late` | 65.78 | 76.76% | 578.6 |
| 256 | `long_late` | 65.04 | 76.59% | 798.6 |
| 256 | `local_contextual` | 65.64 | 77.26% | 686.2 |
| 128 | `naive` | 65.21 | 78.81% | 388.6 |
| 128 | `naive_prefix` | 65.55 | 78.03% | 494.2 |
| 128 | `late` | 66.12 | 78.53% | 578.7 |
| 128 | `long_late` | 65.24 | 75.87% | 798.5 |
| 128 | `local_contextual` | 65.62 | 77.87% | 1,174.2 |
| 64 | `naive` | 66.49 | 77.89% | 425.0 |
| 64 | `naive_prefix` | 67.02 | 80.06% | 591.2 |
| 64 | `late` | 66.05 | 78.59% | 588.7 |
| 64 | `long_late` | 65.91 | 76.70% | 801.5 |
| 64 | `local_contextual` | 67.87 | 80.26% | 1,212.8 |

The [ablation summary](benchmarks/scifact-chunk-size-ablation/summary.json) contains exact scores,
encoding times, and coverage counts. Raw runs are available for [128 tokens](benchmarks/scifact-chunk-size-ablation/cs128.json),
[64 tokens](benchmarks/scifact-chunk-size-ablation/cs64.json), and
[256-token Local Contextual](benchmarks/scifact-chunk-size-ablation/cs256-local-contextual.json).

```bash
for size in 128 64; do
  python scripts/reproduce_late_chunking.py --methods naive late long_late naive_prefix local_contextual \
    --chunk-size "$size" --local-radius 1 --local-context-cap 1024 --device cpu \
    --out-dir "results/scifact-ablation-cs$size"
done
```
