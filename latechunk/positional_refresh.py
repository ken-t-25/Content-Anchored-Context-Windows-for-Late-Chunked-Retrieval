"""Exact document refresh for Long Late Chunking windows."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import torch
import torch.nn.functional as F

from .chunking import ChunkedDoc
from .encoding import Encoder, plan_long_windows, validate_long_window
from .group_encoding import encoder_cache_identity, refresh_token_inputs
from .sticky import RefreshStats


WindowInput = tuple[int, ...]
ChunkDependencies = tuple[tuple[WindowInput, int, int], ...]


@dataclass(frozen=True)
class _Window:
    start: int
    end: int
    owned_start: int
    offset: int
    input_ids: WindowInput


class PositionalRefresh:
    """Reuse window outputs for a document with stable retrieval chunk IDs."""

    def __init__(
        self, encoder: Encoder, *, window: int = 256, overlap: int = 64, batch_size: int = 16,
        special_tokens: str = "raw",
    ):
        validate_long_window(window, overlap, encoder.max_length, special_tokens)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._encoder = encoder
        self._window = window
        self._overlap = overlap
        self._batch_size = batch_size
        self._special_tokens = special_tokens
        self._identity = None
        self._outputs: dict[WindowInput, torch.Tensor] = {}
        self._dependencies: dict[str, ChunkDependencies] = {}
        self.vectors: dict[str, torch.Tensor] = {}

    def _plan(self, input_ids: Sequence[int]) -> list[_Window]:
        tokenizer = self._encoder.tokenizer
        return [
            _Window(start, end, owned, offset, tuple(ids))
            for start, end, owned, offset, ids in plan_long_windows(
                input_ids, self._window, self._overlap, self._special_tokens,
                getattr(tokenizer, "cls_token_id", None), getattr(tokenizer, "sep_token_id", None),
            )
        ]

    def refresh(self, doc: ChunkedDoc, chunk_ids: Sequence[str]) -> RefreshStats:
        """Update only windows and chunk vectors with changed inputs or ownership."""
        started = perf_counter()
        if len(chunk_ids) != len(doc.spans) or len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Expected one unique ID per retrieval span")
        if self._window > self._encoder.max_length:
            raise ValueError("The positional window exceeds the encoder model limit")
        identity = encoder_cache_identity(self._encoder)
        same_encoder = identity == self._identity
        windows = self._plan(doc.input_ids) if chunk_ids else []

        # Record both each owning window's input and the exact local slice to pool.
        # The first window owns its whole output, and subsequent windows drop their overlap.
        dependencies = {}
        first_window = 0
        for chunk_id, (start, end) in zip(chunk_ids, doc.spans, strict=True):
            if not windows[0].start <= start < end <= windows[-1].end:
                raise ValueError(f"Retrieval span for {chunk_id!r} lies outside the positional window coverage")
            while first_window and windows[first_window].owned_start > start:
                first_window -= 1
            while windows[first_window].end <= start:
                first_window += 1
            pieces = []
            for index in range(first_window, len(windows)):
                window = windows[index]
                if window.owned_start >= end:
                    break
                local_start = max(start, window.owned_start) - window.start + window.offset
                local_end = min(end, window.end) - window.start + window.offset
                pieces.append((window.input_ids, local_start, local_end))
            dependencies[chunk_id] = tuple(pieces)

        changed = [
            chunk_id for chunk_id, pieces in dependencies.items()
            if not same_encoder or pieces != self._dependencies.get(chunk_id)
        ]
        removed = self.vectors.keys() - dependencies.keys()
        encode_started = perf_counter()
        batch_size = self._batch_size
        outputs, pending = refresh_token_inputs(
            self._encoder, [window.input_ids for window in windows], batch_size,
            self._outputs if same_encoder else None,
        )
        write_started = perf_counter()

        vectors = self.vectors.copy()
        for chunk_id in removed:
            del vectors[chunk_id]
        for chunk_id in changed:
            slices = [outputs[ids][start:end] for ids, start, end in dependencies[chunk_id]]
            vectors[chunk_id] = F.normalize(torch.cat(slices).mean(dim=0), dim=0)
        self._identity = identity
        self._outputs, self._dependencies, self.vectors = outputs, dependencies, vectors
        finished = perf_counter()
        return RefreshStats(
            refreshed_groups=len(pending),
            refreshed_input_tokens=sum(map(len, pending)),
            encoder_calls=(len(pending) + batch_size - 1) // batch_size,
            rewritten_vectors=len(changed),
            removed_vectors=len(removed),
            regroup_seconds=encode_started - started,
            encode_seconds=write_started - encode_started,
            write_seconds=finished - write_started,
            total_seconds=finished - started,
        )
