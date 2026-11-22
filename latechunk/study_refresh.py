"""Aligned full builds and exact updates for the stateless study policies."""

from __future__ import annotations

from itertools import groupby
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

import torch
import torch.nn.functional as F

from .anchored import AnchoredChunk
from .chunking import ChunkedDoc
from .encoding import (
    MAX_PREFIX_TOKENS, Encoder, _local_window_bounds, _template_special_counts, _validate_batch_size,
    encode_late, validate_long_window,
)
from .group_encoding import GroupInput, encoder_cache_identity, group_input, refresh_token_inputs
from .sticky import RefreshStats

if TYPE_CHECKING:
    from .study_data import StudyDocument


STATELESS_METHODS = ("naive", "naive_prefix", "late", "local_contextual")
_Location = tuple[tuple[int, ...], tuple[int, int]]


def study_prefix(document: StudyDocument) -> str:
    """Use a native title when supplied, or the document's lead chunk."""
    return document.title or (document.chunks[0].text if document.chunks else "")


def long_late_parts(doc: ChunkedDoc, cap: int, leading: int, trailing: int) -> list[tuple[int, ChunkedDoc, bool]]:
    """Split only at oversized chunks, preserving coordinates for exact refresh."""
    oversized = [end - start + leading + trailing > cap for start, end in doc.spans]
    if not any(oversized):
        return [(0, doc, False)]
    prefix = doc.input_ids[:leading]
    suffix = doc.input_ids[len(doc.input_ids) - trailing:]
    parts = []
    for _, indices in groupby(range(len(doc.spans)), key=lambda i: i if oversized[i] else None):
        indices = list(indices)
        first, last = indices[0], indices[-1]
        start, end = doc.spans[first][0], doc.spans[last][1]
        offset = start - leading
        spans = [(s - offset, e - offset) for s, e in doc.spans[first:last + 1]]
        ids = [*prefix, *doc.input_ids[start:end], *suffix]
        parts.append((offset, ChunkedDoc(doc.doc_id, ids, spans, doc.texts[first:last + 1]), oversized[first]))
    return parts


def encode_study_long_late(
    encoder: Encoder, docs: Sequence[StudyDocument], *, token_cap: int, overlap: int, batch_size: int
) -> torch.Tensor:
    """Wrapped Long Late with singleton treatment for model-eligible oversized chunks."""
    validate_long_window(token_cap, overlap, encoder.max_length, "wrap")
    settings = {"batch_size": batch_size, "special_tokens": "wrap"}
    ordered = []
    regular, singleton = [], []
    for document in docs:
        for _, part, oversized in long_late_parts(document.chunked, token_cap, 1, 1):
            ordered.append((oversized, len(part.spans)))
            (singleton if oversized else regular).append(part)
    ordinary = encode_late(encoder, regular, long_window=token_cap, long_overlap=overlap, **settings)
    if not singleton:
        return ordinary
    singleton_rows = iter(encode_late(encoder, singleton, **settings))
    ordinary_rows = iter(ordinary)
    return torch.stack([next(singleton_rows if oversized else ordinary_rows)
                        for oversized, count in ordered for _ in range(count)])


def build_stateless_units(
    encoder: Encoder,
    chunks: Sequence[AnchoredChunk],
    *,
    method: str,
    token_cap: int = 1024,
    local_radius: int = 1,
    prefix: str = "",
) -> dict[str, tuple[GroupInput, int]]:
    """Map stable retrieval IDs to wrapped encoder inputs and target pooling rows."""
    if method not in STATELESS_METHODS:
        raise ValueError(f"Unknown stateless policy {method!r}")
    if token_cap <= 0 or local_radius < 0:
        raise ValueError("token_cap must be positive and local_radius must be nonnegative")

    tokenizer = encoder.tokenizer
    special_count = sum(_template_special_counts(tokenizer))
    model_budget = encoder.max_length - special_count
    budget = min(token_cap, encoder.max_length) - special_count
    ids = [chunk.chunk_id for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Expected unique retrieval chunk IDs")
    if budget <= 0:
        raise ValueError("The context cap must allow at least one content token")
    if not chunks:
        return {}
    tokens = [chunk.token_ids for chunk in chunks]
    if any(len(ids) > model_budget for ids in tokens):
        raise ValueError("A retrieval chunk exceeds the encoder model limit")

    if method == "late":
        if sum(map(len, tokens)) > model_budget:
            raise ValueError("Full Late document exceeds the encoder model limit")
        unit = group_input(tokenizer, tokens)
        return {chunk_id: (unit, index) for index, chunk_id in enumerate(ids)}

    if method in ("naive", "naive_prefix"):
        prefix_ids = []
        if method == "naive_prefix" and prefix:
            prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"][:MAX_PREFIX_TOKENS]
        return {
            chunk_id: (group_input(tokenizer, [body], prefix_ids[:max(0, budget - len(body))]), 0)
            for chunk_id, body in zip(ids, tokens, strict=True)
        }

    # Body offsets give the existing neighborhood selector the same chunk boundaries.
    spans = []
    offset = 0
    for body in tokens:
        spans.append((offset, offset + len(body)))
        offset += len(body)
    start_index = {start: i for i, (start, _) in enumerate(spans)}
    end_index = {end: i + 1 for i, (_, end) in enumerate(spans)}
    windows = {}
    units = {}
    for i, chunk_id in enumerate(ids):
        bounds = _local_window_bounds(spans, i, local_radius, budget)
        left, right = start_index[bounds[0]], end_index[bounds[1]]
        if bounds not in windows:
            windows[bounds] = group_input(tokenizer, tokens[left:right])
        units[chunk_id] = windows[bounds], i - left
    return units


class StatelessRefresh:
    """Keep only the current exact encoder outputs for one document and policy."""

    def __init__(
        self,
        encoder: Encoder,
        *,
        method: str,
        token_cap: int = 1024,
        local_radius: int = 1,
        batch_size: int = 16,
    ) -> None:
        _validate_batch_size(batch_size)
        self._encoder = encoder
        self._settings = {"method": method, "token_cap": token_cap, "local_radius": local_radius}
        self._batch_size = batch_size
        self._identity = encoder_cache_identity(encoder)
        self._outputs: dict[tuple[int, ...], torch.Tensor] = {}
        self._locations: dict[str, _Location] = {}
        self._vectors: dict[str, torch.Tensor] = {}

    @property
    def vectors(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(self._vectors)

    def refresh(self, chunks: Sequence[AnchoredChunk], *, prefix: str = "") -> RefreshStats:
        started = perf_counter()
        identity = encoder_cache_identity(self._encoder)
        same_encoder = identity == self._identity
        units = build_stateless_units(self._encoder, chunks, prefix=prefix, **self._settings)
        locations = {chunk_id: (unit.token_ids, unit.spans[row]) for chunk_id, (unit, row) in units.items()}
        inputs = [unit.token_ids for unit, _ in units.values()]
        rewritten = {key: value for key, value in locations.items()
                     if not same_encoder or self._locations.get(key) != value}
        removed = self._locations.keys() - locations.keys()

        encode_started = perf_counter()
        cache, pending = refresh_token_inputs(
            self._encoder, inputs, self._batch_size, self._outputs if same_encoder else None
        )
        write_started = perf_counter()
        fresh = {key: F.normalize(cache[ids][start:end].mean(dim=0), dim=0)
                 for key, (ids, (start, end)) in rewritten.items()}
        vectors = {key: vector for key, vector in self._vectors.items() if key not in removed}
        vectors.update(fresh)
        self._identity, self._outputs, self._locations, self._vectors = identity, cache, locations, vectors
        finished = perf_counter()
        return RefreshStats(
            refreshed_groups=len(pending),
            refreshed_input_tokens=sum(map(len, pending)),
            encoder_calls=(len(pending) + self._batch_size - 1) // self._batch_size,
            rewritten_vectors=len(fresh),
            removed_vectors=len(removed),
            regroup_seconds=encode_started - started,
            encode_seconds=write_started - encode_started,
            write_seconds=finished - write_started,
            total_seconds=finished - started,
        )
