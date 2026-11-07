"""Independent (naive) chunk encoding, Late Chunking, and Long Late Chunking (Algorithm 2)."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .chunking import ChunkedDoc

# Pinned revision of jinaai/jina-bert-implementation (the remote modeling code that
# jina-embeddings-v2-*-en loads), so every run executes the same model code.
JINA_BERT_CODE_REVISION = "f3ec4cf7de7e561007f27c9efc7148b0bd713f81"

# embed_fn: list of token-id lists -> list of (len_i, d) token-embedding tensors
EmbedFn = Callable[[Sequence[Sequence[int]]], list[torch.Tensor]]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Encoder:
    """Thin wrapper around a HF encoder that exposes token-level outputs."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_length: int = 8192,
        code_revision: str | None = JINA_BERT_CODE_REVISION,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or pick_device()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModel.from_pretrained(model_name, trust_remote_code=True, code_revision=code_revision)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def token_embeddings(self, batch_ids: Sequence[Sequence[int]]) -> list[torch.Tensor]:
        """Run the encoder on already-tokenized sequences (no tokens are added or removed)."""
        lens = [len(x) for x in batch_ids]
        if max(lens) > self.max_length:
            raise ValueError(f"Sequence of {max(lens)} tokens exceeds max_length={self.max_length}")
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(batch_ids), max(lens)), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(batch_ids):
            ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device))[0]
        out = out.float().cpu()
        return [out[i, :n] for i, n in enumerate(lens)]

    def embed_fn(self, batch_size: int = 16) -> EmbedFn:
        def fn(seqs):
            res = []
            for i in range(0, len(seqs), batch_size):
                res.extend(self.token_embeddings(seqs[i : i + batch_size]))
            return res

        return fn

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        """Standard independent encoding: tokenize with special tokens, mean-pool over
        the attention mask (incl. [CLS]/[SEP], as jina-embeddings-v2 `encode` does), L2-normalize."""
        enc = self.tokenizer(list(texts), add_special_tokens=True, truncation=True, max_length=self.max_length)
        ids = enc["input_ids"]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))  # length bucketing
        out = torch.empty(len(ids), self.model.config.hidden_size)
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            toks = self.token_embeddings([ids[i] for i in idx])
            for i, t in zip(idx, toks):
                out[i] = t.mean(dim=0)
        return F.normalize(out, dim=-1)


# ---------------------------------------------------------------------------
# Long Late Chunking (Algorithm 2)
# ---------------------------------------------------------------------------

def macro_windows(n: int, window: int, overlap: int) -> list[tuple[int, int, int]]:
    """Overlapping windows over n tokens.

    Returns (start, end, keep_from): token embeddings for positions
    [start + keep_from, end) are taken from this window. Ownership rule: an
    overlapping token belongs to the EARLIER window, i.e. every window after
    the first drops its first `overlap` outputs. Each position is owned exactly once.
    """
    if not 0 <= overlap < window:
        raise ValueError("Need 0 <= overlap < window")
    step = window - overlap
    wins, start = [], 0
    while True:
        end = min(start + window, n)
        wins.append((start, end, 0 if start == 0 else overlap))
        if end == n:
            return wins
        start += step


def long_late_token_embeddings(
    input_ids: Sequence[int],
    embed_fn: EmbedFn,
    window: int,
    overlap: int,
    special_tokens: str = "raw",
    cls_id: int | None = None,
    sep_id: int | None = None,
) -> torch.Tensor:
    """Token embeddings for a sequence possibly longer than the encoder window.

    special_tokens:
      "raw"  – slice the full [CLS] ... [SEP] sequence directly into windows
               (what the reference implementation does: only the first window
               sees [CLS], only the last sees [SEP]).
      "wrap" – window the content tokens only and wrap every window in its own
               [CLS] ... [SEP]; `window` then counts the two special tokens.
    Output has one row per position of `input_ids`.
    """
    n = len(input_ids)
    if special_tokens == "raw":
        wins = macro_windows(n, window, overlap)
        outs = embed_fn([list(input_ids[s:e]) for s, e, _ in wins])
        pieces = [o[k:] for (s, e, k), o in zip(wins, outs)]
    elif special_tokens == "wrap":
        if input_ids[0] != cls_id or input_ids[-1] != sep_id:
            raise ValueError("'wrap' mode expects input_ids = [CLS] ... [SEP]")
        content = list(input_ids[1:-1])
        wins = macro_windows(len(content), window - 2, overlap)
        outs = embed_fn([[cls_id, *content[s:e], sep_id] for s, e, _ in wins])
        inner = [o[1:-1][k:] for (s, e, k), o in zip(wins, outs)]
        d = inner[0].shape[-1]
        # special positions are never pooled; fill with zeros to keep indices aligned
        pieces = [torch.zeros(1, d), *inner, torch.zeros(1, d)]
    else:
        raise ValueError(special_tokens)
    emb = torch.cat(pieces, dim=0)
    assert emb.shape[0] == n, (emb.shape, n)
    return emb


# ---------------------------------------------------------------------------
# Chunk-level encoders
# ---------------------------------------------------------------------------

def pool_spans(token_emb: torch.Tensor, spans: Sequence[tuple[int, int]]) -> torch.Tensor:
    """Mean-pool token embeddings over each chunk span (spans exclude special tokens)."""
    return torch.stack([token_emb[s:e].mean(dim=0) for s, e in spans])


def encode_naive(encoder: Encoder, docs: Sequence[ChunkedDoc], batch_size: int = 64) -> torch.Tensor:
    texts = [t for d in docs for t in d.texts]
    return encoder.encode_texts(texts, batch_size=batch_size)


def encode_late(
    encoder: Encoder,
    docs: Sequence[ChunkedDoc],
    batch_size: int = 16,
    long_window: int | None = None,
    long_overlap: int = 0,
    special_tokens: str = "raw",
) -> torch.Tensor:
    """Late chunking. If long_window is None, each document is encoded in one pass
    (falls back to Long Late Chunking with window = encoder.max_length for
    documents that do not fit). Otherwise Long Late Chunking with the given window."""
    embed = encoder.embed_fn(batch_size)
    window = long_window or encoder.max_length
    tok = encoder.tokenizer
    out: list[torch.Tensor | None] = [None] * len(docs)

    # documents that fit in one window: batched single pass (length-bucketed)
    single = [i for i, d in enumerate(docs) if len(d.input_ids) <= window and d.spans]
    single.sort(key=lambda i: len(docs[i].input_ids))
    for b in range(0, len(single), batch_size):
        idx = single[b : b + batch_size]
        for i, t in zip(idx, encoder.token_embeddings([docs[i].input_ids for i in idx])):
            out[i] = pool_spans(t, docs[i].spans)

    for i, d in enumerate(docs):
        if out[i] is None and d.spans:
            t = long_late_token_embeddings(
                d.input_ids, embed, window, long_overlap, special_tokens, tok.cls_token_id, tok.sep_token_id
            )
            out[i] = pool_spans(t, d.spans)

    return F.normalize(torch.cat([o for o in out if o is not None]), dim=-1)
