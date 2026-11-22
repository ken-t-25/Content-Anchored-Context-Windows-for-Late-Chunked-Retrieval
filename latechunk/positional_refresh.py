"""Exact document refresh for raw Long Late Chunking windows."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import torch
import torch.nn.functional as F

from .chunking import ChunkedDoc
from .encoding import Encoder, macro_windows
from .sticky import RefreshStats


WindowInput = tuple[int, ...]
ChunkDependencies = tuple[tuple[WindowInput, int, int], ...]


@dataclass(frozen=True)
class _Window:
    start: int
    end: int
    owned_start: int
    input_ids: WindowInput


class PositionalRefresh:
    """Reuse raw window outputs for a document with a fixed encoder and chunk IDs."""

    def __init__(self, encoder: Encoder, *, window: int = 256, overlap: int = 64, batch_size: int = 16):
        if not 0 <= overlap < window <= encoder.max_length or batch_size <= 0:
            raise ValueError("Expected 0 <= overlap < window <= model limit and a positive batch size")
        self._encoder = encoder
        self._window = window
        self._overlap = overlap
        self._batch_size = batch_size
        self._outputs: dict[WindowInput, torch.Tensor] = {}
        self._dependencies: dict[str, ChunkDependencies] = {}
        self.vectors: dict[str, torch.Tensor] = {}

    def refresh(self, doc: ChunkedDoc, chunk_ids: Sequence[str]) -> RefreshStats:
        """Update only windows and chunk vectors with changed inputs or ownership."""
        started = perf_counter()
        if len(chunk_ids) != len(doc.spans) or len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Expected one unique ID per retrieval span")
        windows = [
            _Window(start, end, start + overlap, tuple(doc.input_ids[start:end]))
            for start, end, overlap in macro_windows(len(doc.input_ids), self._window, self._overlap)
        ] if chunk_ids else []

        # Record both each owning window's input and the exact local slice to pool.
        # The first window owns its whole output, and subsequent windows drop their overlap.
        dependencies = {}
        first_window = 0
        for chunk_id, (start, end) in zip(chunk_ids, doc.spans, strict=True):
            while windows[first_window].end <= start:
                first_window += 1
            pieces = []
            for index in range(first_window, len(windows)):
                window = windows[index]
                if window.owned_start >= end:
                    break
                pieces.append((window.input_ids, max(start, window.owned_start) - window.start,
                               min(end, window.end) - window.start))
            dependencies[chunk_id] = tuple(pieces)

        live = dict.fromkeys(window.input_ids for window in windows)
        pending = [input_ids for input_ids in live if input_ids not in self._outputs]
        changed = [chunk_id for chunk_id, pieces in dependencies.items()
                   if pieces != self._dependencies.get(chunk_id)]
        removed = self.vectors.keys() - dependencies.keys()
        refreshed_input_tokens = sum(map(len, pending))
        encode_started = perf_counter()

        batch_size = self._batch_size
        encoded = self._encoder.embed_fn(batch_size)(pending)
        outputs = {ids: self._outputs[ids] for ids in live if ids in self._outputs}
        outputs.update(zip(pending, encoded, strict=True))
        updated = {}
        for chunk_id in changed:
            slices = [outputs[ids][start:end] for ids, start, end in dependencies[chunk_id]]
            updated[chunk_id] = F.normalize(torch.cat(slices).mean(dim=0), dim=0)
        write_started = perf_counter()

        vectors = self.vectors.copy()
        for chunk_id in removed:
            del vectors[chunk_id]
        vectors.update(updated)
        self._outputs, self._dependencies, self.vectors = outputs, dependencies, vectors
        finished = perf_counter()
        return RefreshStats(
            refreshed_groups=len(pending),
            refreshed_input_tokens=refreshed_input_tokens,
            encoder_calls=(len(pending) + batch_size - 1) // batch_size,
            rewritten_vectors=len(changed),
            removed_vectors=len(removed),
            regroup_seconds=encode_started - started,
            encode_seconds=write_started - encode_started,
            write_seconds=finished - write_started,
            total_seconds=finished - started,
        )
