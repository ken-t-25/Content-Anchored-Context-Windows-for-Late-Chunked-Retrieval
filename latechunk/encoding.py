"""Independent and contextual chunk encoders."""

from __future__ import annotations

import re
from numbers import Integral
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .chunking import ChunkedDoc

# Pinned revision of jinaai/jina-bert-implementation (remote modeling code for jina-embeddings-v2-*-en).
JINA_BERT_CODE_REVISION = "f3ec4cf7de7e561007f27c9efc7148b0bd713f81"
MAX_PREFIX_TOKENS = 64

# embed_fn: list of token-id lists -> list of (len_i, d) token-embedding tensors
EmbedFn = Callable[[Sequence[Sequence[int]]], list[torch.Tensor]]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")


def _validate_inputs(inputs: Sequence[Sequence[int]], max_length: int) -> None:
    for ids in inputs:
        if len(ids) == 0:
            raise ValueError("Encoder inputs must contain at least one token")
        if len(ids) > max_length:
            raise ValueError(f"Sequence of {len(ids)} tokens exceeds max_length={max_length}")


def _resolved_tokenizer_revision(tokenizer, model_name: str, revision: str | None) -> str | None:
    from huggingface_hub import try_to_load_from_cache

    commit = getattr(tokenizer, "_commit_hash", None) or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if commit or Path(model_name).is_dir():
        return commit or revision
    for filename in ("tokenizer_config.json", "tokenizer.json", "config.json"):
        cached = try_to_load_from_cache(model_name, filename, revision=revision)
        if isinstance(cached, str):
            parent = Path(cached).parent
            if parent.parent.name == "snapshots":
                return parent.name
    return revision


class Encoder:
    """Thin wrapper around a HF encoder that exposes token-level outputs."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_length: int = 8192,
        code_revision: str | None = JINA_BERT_CODE_REVISION,
        *,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
    ):
        from transformers import AutoModel, AutoTokenizer

        if not isinstance(max_length, Integral) or isinstance(max_length, bool) or max_length <= 0:
            raise ValueError(f"max_length must be a positive integer, got {max_length!r}")

        self.device = device or pick_device()
        self.model_name = model_name
        tokenizer_revision = model_revision if tokenizer_revision is None else tokenizer_revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=tokenizer_revision)
        self.model = (
            AutoModel.from_pretrained(
                model_name, revision=model_revision, trust_remote_code=True, code_revision=code_revision
            )
            .to(self.device)
            .eval()
        )
        self.max_length = max_length
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(model_limit, Integral) and model_limit > 0:
            self.max_length = min(max_length, model_limit)
        self.model_revision = getattr(self.model.config, "_commit_hash", None) or model_revision
        self.tokenizer_revision = _resolved_tokenizer_revision(self.tokenizer, model_name, tokenizer_revision)
        self.code_revision = code_revision
        model_module = type(self.model).__module__
        if model_module.startswith("transformers_modules."):
            for segment in reversed(model_module.split(".")):
                if re.fullmatch(r"[0-9a-f]{40}", segment):
                    self.code_revision = segment
                    break

    @torch.inference_mode()
    def token_embeddings(self, batch_ids: Sequence[Sequence[int]]) -> list[torch.Tensor]:
        """Run the encoder on already-tokenized sequences (no tokens are added or removed)."""
        _validate_inputs(batch_ids, self.max_length)
        if len(batch_ids) == 0:
            return []
        lens = [len(x) for x in batch_ids]
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
        _validate_batch_size(batch_size)

        def fn(seqs):
            _validate_inputs(seqs, self.max_length)
            res = []
            for i in range(0, len(seqs), batch_size):
                res.extend(self.token_embeddings(seqs[i : i + batch_size]))
            return res

        return fn

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        """Independently encode text with specials, mean-pool all attended tokens, and normalize."""
        _validate_batch_size(batch_size)
        inputs = list(texts)
        if not inputs:
            return self.encode_token_ids([], batch_size=batch_size)
        enc = self.tokenizer(inputs, add_special_tokens=True, truncation=False, verbose=False)
        return self.encode_token_ids(enc["input_ids"], batch_size=batch_size)

    def encode_token_ids(self, ids: Sequence[Sequence[int]], batch_size: int = 64) -> torch.Tensor:
        """Mean-pool and L2-normalize inputs that already include any special tokens."""
        _validate_batch_size(batch_size)
        _validate_inputs(ids, self.max_length)
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))  # length bucketing
        out = torch.empty(len(ids), self.model.config.hidden_size)
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            toks = self.token_embeddings([ids[i] for i in idx])
            for i, t in zip(idx, toks, strict=True):
                out[i] = t.mean(dim=0)
        return F.normalize(out, dim=-1)


# ---------------------------------------------------------------------------
# Long Late Chunking (Algorithm 2)
# ---------------------------------------------------------------------------

def validate_long_window(window: int, overlap: int, max_length: int, special_tokens: str = "raw") -> None:
    """Validate the wrapped model-input limit and the overlap in windowed tokens."""
    if special_tokens not in ("raw", "wrap"):
        raise ValueError(f"Invalid special_tokens={special_tokens!r}. Expected 'raw' or 'wrap'")
    if not isinstance(window, Integral) or isinstance(window, bool) or not 0 < window <= max_length:
        raise ValueError(f"Long Late window must be a positive integer <= max_length={max_length}, got {window!r}")
    usable_window = window - 2 if special_tokens == "wrap" else window
    if not isinstance(overlap, Integral) or isinstance(overlap, bool) or not 0 <= overlap < usable_window:
        raise ValueError(
            f"Long Late {special_tokens} mode requires 0 <= overlap < {usable_window} usable tokens, got {overlap!r}"
        )


def macro_windows(n: int, window: int, overlap: int) -> list[tuple[int, int, int]]:
    """Overlapping windows over n tokens.

    Returns (start, end, keep_from): token embeddings for positions
    [start + keep_from, end) are taken from this window. Ownership rule: an
    overlapping token belongs to the earlier window. Each position is owned exactly once.
    """
    validate_long_window(window, overlap, window)
    if n < 0:
        raise ValueError(f"Token count must be nonnegative, got {n}")
    step = window - overlap
    wins, start = [], 0
    while True:
        end = min(start + window, n)
        wins.append((start, end, 0 if start == 0 else overlap))
        if end == n:
            return wins
        start += step


def _validate_wrap_input(input_ids: Sequence[int], cls_id: int | None, sep_id: int | None) -> None:
    if cls_id is None or sep_id is None or len(input_ids) < 2 or input_ids[0] != cls_id or input_ids[-1] != sep_id:
        raise ValueError("'wrap' mode expects input_ids = [CLS] ... [SEP]")


def plan_long_windows(
    input_ids: Sequence[int],
    window: int,
    overlap: int,
    special_tokens: str = "raw",
    cls_id: int | None = None,
    sep_id: int | None = None,
) -> list[tuple[int, int, int, int, list[int]]]:
    """Plan windows as (doc_start, doc_end, owned_start, window_offset, input_ids).

    A document position p maps to p - doc_start + window_offset in the input.
    In wrap mode only content positions are owned, since each window has new specials.
    """
    validate_long_window(window, overlap, window, special_tokens)
    if len(input_ids) == 0:
        raise ValueError("Long Late requires a nonempty sequence")
    if special_tokens == "raw":
        return [
            (start, end, start + keep_from, 0, list(input_ids[start:end]))
            for start, end, keep_from in macro_windows(len(input_ids), window, overlap)
        ]

    _validate_wrap_input(input_ids, cls_id, sep_id)
    content = input_ids[1:-1]
    return [
        (start + 1, end + 1, start + keep_from + 1, 1, [cls_id, *content[start:end], sep_id])
        for start, end, keep_from in macro_windows(len(content), window - 2, overlap)
    ]


def long_late_token_embeddings(
    input_ids: Sequence[int],
    embed_fn: EmbedFn,
    window: int,
    overlap: int,
    special_tokens: str = "raw",
    cls_id: int | None = None,
    sep_id: int | None = None,
) -> torch.Tensor:
    """Stitch one embedding per document position from overlapping windows."""
    windows = plan_long_windows(input_ids, window, overlap, special_tokens, cls_id, sep_id)
    embeddings = embed_fn([ids for _, _, _, _, ids in windows])
    pieces = []
    for (start, end, owned, offset, ids), rows in zip(windows, embeddings, strict=True):
        if rows.ndim != 2 or rows.shape[0] != len(ids):
            raise ValueError(f"Expected token embeddings for {len(ids)} input positions, got shape {tuple(rows.shape)}")
        pieces.append(rows[owned - start + offset:end - start + offset])

    if special_tokens == "wrap":
        # Preserve the document offsets for the original special positions.
        special = embeddings[0].new_zeros((1, embeddings[0].shape[-1]))
        pieces = [special, *pieces, special]
    return torch.cat(pieces, dim=0)


# ---------------------------------------------------------------------------
# Chunk-level encoders
# ---------------------------------------------------------------------------

def _valid_span(start: int, end: int, first: int, last: int) -> bool:
    return isinstance(start, Integral) and isinstance(end, Integral) and first <= start < end <= last


def _template_special_counts(tokenizer) -> tuple[int, int]:
    probe = [-1, -2]
    wrapped = tokenizer.build_inputs_with_special_tokens(probe)
    if wrapped.count(probe[0]) != 1:
        raise ValueError("Expected contiguous content in the tokenizer single-sequence template")
    leading = wrapped.index(probe[0])
    if wrapped[leading:leading + len(probe)] != probe or wrapped.count(probe[1]) != 1:
        raise ValueError("Expected contiguous content in the tokenizer single-sequence template")
    return leading, len(wrapped) - leading - len(probe)


def _validate_doc_spans(doc: ChunkedDoc, leading: int, trailing: int) -> None:
    previous_end = leading
    content_end = len(doc.input_ids) - trailing
    for start, end in doc.spans:
        if not _valid_span(start, end, previous_end, content_end):
            raise ValueError(f"Invalid pooling span ({start}, {end}) in document {doc.doc_id}")
        previous_end = end


def pool_spans(token_emb: torch.Tensor, spans: Sequence[tuple[int, int]]) -> torch.Tensor:
    """Mean-pool token embeddings over each chunk span."""
    if token_emb.ndim != 2:
        raise ValueError(f"Expected a two-dimensional token tensor, got shape {tuple(token_emb.shape)}")
    pooled = []
    for start, end in spans:
        if not _valid_span(start, end, 0, len(token_emb)):
            raise ValueError(f"Invalid pooling span ({start}, {end}) for {len(token_emb)} token embeddings")
        pooled.append(token_emb[start:end].mean(dim=0))
    return torch.stack(pooled) if pooled else token_emb[:0]


def encode_naive(encoder: Encoder, docs: Sequence[ChunkedDoc], batch_size: int = 64) -> torch.Tensor:
    _validate_batch_size(batch_size)
    texts = [t for d in docs for t in d.texts]
    return encoder.encode_texts(texts, batch_size=batch_size)


def encode_naive_prefix(
    encoder: Encoder,
    docs: Sequence[ChunkedDoc],
    prefixes: Mapping[str, str],
    batch_size: int = 64,
) -> torch.Tensor:
    """Independently encode each chunk with up to 64 tokens of document metadata."""
    _validate_batch_size(batch_size)
    tokenizer = encoder.tokenizer
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    inputs = []
    for doc in docs:
        if not doc.texts:
            continue
        prefix = prefixes.get(doc.doc_id, "")
        prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        prefix_ids = prefix_ids[:MAX_PREFIX_TOKENS]
        chunks = tokenizer(doc.texts, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        for chunk_ids in chunks:
            available = encoder.max_length - len(chunk_ids) - special_count
            if available < 0:
                raise ValueError(f"Chunk in document {doc.doc_id} exceeds max_length={encoder.max_length} before prefixing")
            inputs.append(tokenizer.build_inputs_with_special_tokens(prefix_ids[:available] + chunk_ids))
    return encoder.encode_token_ids(inputs, batch_size=batch_size)


def _local_window_bounds(
    spans: Sequence[tuple[int, int]], target: int, radius: int, content_budget: int
) -> tuple[int, int]:
    left = right = target
    start, end = spans[target]
    # Prefer closer neighbours, breaking ties to the left. Keep the window contiguous.
    for distance in range(1, min(radius, len(spans) - 1) + 1):
        previous, following = target - distance, target + distance
        expanded = False
        if previous >= 0 and previous == left - 1 and end - spans[previous][0] <= content_budget:
            left, start = previous, spans[previous][0]
            expanded = True
        if following < len(spans) and following == right + 1 and spans[following][1] - start <= content_budget:
            right, end = following, spans[following][1]
            expanded = True
        if not expanded:
            break
    return start, end


def encode_local_contextual(
    encoder: Encoder,
    docs: Sequence[ChunkedDoc],
    batch_size: int = 16,
    *,
    radius: int = 1,
    context_cap: int = 1024,
) -> torch.Tensor:
    """Use whole adjacent chunks as context and pool only each target span."""
    _validate_batch_size(batch_size)
    if not isinstance(radius, Integral) or isinstance(radius, bool) or radius < 0:
        raise ValueError(f"Local contextual radius must be a nonnegative integer, got {radius!r}")
    if not isinstance(context_cap, Integral) or isinstance(context_cap, bool) or context_cap <= 0:
        raise ValueError(f"Local contextual context_cap must be a positive integer, got {context_cap!r}")

    tokenizer = encoder.tokenizer
    leading_specials, trailing_specials = _template_special_counts(tokenizer)
    special_count = leading_specials + trailing_specials
    content_budget = min(context_cap, encoder.max_length) - special_count
    windows: list[list[int]] = []
    targets: list[list[tuple[int, tuple[int, int]]]] = []
    next_chunk = 0
    for doc in docs:
        _validate_doc_spans(doc, leading_specials, trailing_specials)
        window_indices: dict[tuple[int, int], int] = {}
        for target, (start, end) in enumerate(doc.spans):
            if end - start + special_count > encoder.max_length:
                raise ValueError(f"Chunk in document {doc.doc_id} exceeds max_length={encoder.max_length}")
            # The target remains whole even when it alone exceeds the context budget.
            window_start, window_end = _local_window_bounds(doc.spans, target, radius, content_budget)
            bounds = (window_start, window_end)
            if bounds not in window_indices:
                content = doc.input_ids[window_start:window_end]
                wrapped = tokenizer.build_inputs_with_special_tokens(content)
                if (
                    len(wrapped) != len(content) + special_count
                    or wrapped[leading_specials:leading_specials + len(content)] != content
                ):
                    raise ValueError("Expected contiguous content in the tokenizer single-sequence template")
                window_indices[bounds] = len(windows)
                windows.append(wrapped)
                targets.append([])
            offset = leading_specials - window_start
            targets[window_indices[bounds]].append((next_chunk, (start + offset, end + offset)))
            next_chunk += 1

    order = sorted(range(len(windows)), key=lambda index: len(windows[index]))
    out = torch.empty(next_chunk, encoder.model.config.hidden_size)
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        for index, token_emb in zip(batch, encoder.token_embeddings([windows[i] for i in batch]), strict=True):
            chunk_indices, spans = zip(*targets[index])
            out[list(chunk_indices)] = pool_spans(token_emb, spans)
    return F.normalize(out, dim=-1)


def encode_late(
    encoder: Encoder,
    docs: Sequence[ChunkedDoc],
    batch_size: int = 16,
    long_window: int | None = None,
    long_overlap: int = 0,
    special_tokens: str = "raw",
) -> torch.Tensor:
    """Encode each full document once, or explicitly use Long Late Chunking."""
    _validate_batch_size(batch_size)
    window = encoder.max_length if long_window is None else long_window
    validate_long_window(window, long_overlap, encoder.max_length, special_tokens)
    tok = encoder.tokenizer
    leading, trailing = _template_special_counts(tok)
    cls_id = getattr(tok, "cls_token_id", None)
    sep_id = getattr(tok, "sep_token_id", None)
    if special_tokens == "wrap" and (leading, trailing) != (1, 1):
        raise ValueError("'wrap' mode requires a single [CLS] prefix and a single [SEP] suffix")

    for doc in docs:
        if not doc.spans:
            continue
        _validate_doc_spans(doc, leading, trailing)
        if long_window is None and len(doc.input_ids) > window:
            raise ValueError(
                f"Full Late document {doc.doc_id} has {len(doc.input_ids)} tokens and exceeds max_length={window}"
            )
        if special_tokens == "wrap":
            _validate_wrap_input(doc.input_ids, cls_id, sep_id)

    out: list[torch.Tensor | None] = [None] * len(docs)
    # Batch documents which fit in one forward sequence.
    single = [i for i, d in enumerate(docs) if len(d.input_ids) <= window and d.spans]
    single.sort(key=lambda i: len(docs[i].input_ids))
    for b in range(0, len(single), batch_size):
        idx = single[b : b + batch_size]
        for i, t in zip(idx, encoder.token_embeddings([docs[i].input_ids for i in idx]), strict=True):
            out[i] = pool_spans(t, docs[i].spans)

    if long_window is not None:
        embed = encoder.embed_fn(batch_size)
        for i, d in enumerate(docs):
            if out[i] is None and d.spans:
                t = long_late_token_embeddings(
                    d.input_ids, embed, window, long_overlap, special_tokens, cls_id, sep_id
                )
                out[i] = pool_spans(t, d.spans)

    pooled = [rows for rows in out if rows is not None]
    if not pooled:
        return torch.empty(0, encoder.model.config.hidden_size)
    return F.normalize(torch.cat(pooled), dim=-1)
