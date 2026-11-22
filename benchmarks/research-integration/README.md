# Research integration verification

These runs used the integrated seven-policy pipeline at `3267abc` and the cached `jinaai/jina-embeddings-v2-small-en` model on CPU. The final unit suite passed 130 tests. The dataset, encoder revisions, and timing details are in the individual JSON files.

- [SciFact quality smoke](scifact-quality-smoke.json) evaluates 64 real documents, 115 shared retrieval chunks, and 3 queries with eligible relevance labels. It verifies the common retrieval and scoring pipeline on a small subset.
- [SciFact matched-cap edit pilot](scifact-matched-edits.json) measures synthetic edits on two real SciFact documents. It uses 256-token retrieval chunks, a 1,024-token context cap, and 256-token Long Late overlap. Content-Anchored refreshed 7,320 tokens across replacement, insertion, and deletion. Long Late refreshed 8,936. Both numbers are totals across the two document sequences. Timing measures each refresh separately from the rebuild used to verify its vectors. The pilot measures refresh cost only.
- [Synthetic native verification](synthetic-verification/) contains the authored test manifests and the outputs used to exercise configuration freezing, held-out evaluation, transfer evaluation, and edit refresh. Every file in this directory is a pipeline fixture. Its scores and noninferiority fields are test output on synthetic examples.

The full ConTEB experiment sequence is documented in [the study protocol](../../STUDY.md).
