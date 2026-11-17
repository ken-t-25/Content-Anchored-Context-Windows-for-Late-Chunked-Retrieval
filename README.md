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

## Sticky ID Groups

Sticky groups maintain an ordered map of stable chunk IDs. The initial build packs whole chunks in
document order, counting special tokens against the group cap. Each group is encoded jointly and
mean-pooled back to its original chunks. A chunk above the group cap runs alone if it fits the model.

An insertion joins its predecessor's group, or the first group for an insertion at the start. Replacements
and insertions split only the affected group if it exceeds the cap. After a deletion, a surviving group
below half of its content-token budget can merge with one adjacent group that fits, trying the left first.
An empty group is removed. Vectors are reused when the group's ordered IDs and token inputs match.
Restoring the saved group map preserves the partition reached through edits.

### Full SciFact retrieval

These runs use the initial grouping at a 1,024-token cap and reuse the original title-plus-body pipeline,
5,183 documents, and 300 queries. Scores are compared with `late` at the same chunk size. Encoding times
come from individual full-corpus CPU runs.

| Chunk size | Sticky nDCG@10 (×100) | Recall@10 | Δ nDCG@10 vs. `late` (points) | Encoding time (s) | Chunks with full-document input |
|---:|---:|---:|---:|---:|---:|
| 256 | 65.7761 | 76.76% | −0.0076 | 953.4 | 99.31% (9,291 / 9,356) |
| 128 | 65.9687 | 78.20% | −0.1541 | 917.5 | 99.23% (16,068 / 16,193) |
| 64 | 66.0524 | 78.59% | 0.0000 | 810.8 | 99.18% (29,481 / 29,726) |

The cap splits the same 11 documents at each size. More than 99% of chunks receive the full-document
input used by `late`. Raw results are available for [256 tokens](benchmarks/scifact-sticky/cs256.json),
[128 tokens](benchmarks/scifact-sticky/cs128.json), and [64 tokens](benchmarks/scifact-sticky/cs64.json).

```bash
for size in 256 128 64; do
  python scripts/reproduce_late_chunking.py --methods sticky --chunk-size "$size" \
    --sticky-token-cap 1024 --device cpu --out-dir "results/scifact-sticky-cs$size"
done
```

### Edit pilot

The pilot uses 256-token chunking and selects the first 20 SciFact documents with at least four chunks.
It applies a replacement, an insertion, and deletion of the inserted chunk. Each refresh matched a full re-encode using the saved
group map, with a maximum absolute vector difference of `6.9e-8`. Every insertion split its group, and all
20 documents kept different group boundaries after deleting the inserted chunk. No-op refreshes made zero
encoder calls.

| Edit | Refreshed tokens, Sticky / rebuild | Rewritten vectors, Sticky / rebuild | Median CPU refresh per document (s) |
|---|---:|---:|---:|
| Replace one token | 17,034 / 18,551 | 77 / 84 | 1.43 |
| Insert one chunk | 22,194 / 23,711 | 97 / 104 | 1.34 |
| Delete the inserted chunk | 10,280 / 18,591 | 40 / 84 | 0.70 |

The [pilot record](benchmarks/scifact-sticky/edit-pilot.json) includes per-document group maps, encoder call
counts, and regrouping, encoding, and vector write times.

```bash
python scripts/pilot_sticky.py --limit-docs 20 --min-chunks 4 --token-cap 1024 --device cpu \
  --out results/sticky-pilot/cs256-cap1024-min4.json
```
