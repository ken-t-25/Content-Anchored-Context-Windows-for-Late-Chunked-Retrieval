"""Stateful editing and encoding for Sticky ID Groups."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from .chunking import ChunkedDoc
from .encoding import Encoder
from .group_encoding import encode_group_inputs, group_input


@dataclass(frozen=True)
class StickyChunk:
    chunk_id: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if not self.token_ids:
            raise ValueError("Sticky chunks must contain at least one token")


Group = tuple[StickyChunk, ...]


@dataclass(frozen=True)
class RefreshStats:
    refreshed_groups: int = 0
    refreshed_input_tokens: int = 0
    encoder_calls: int = 0
    rewritten_vectors: int = 0
    removed_vectors: int = 0
    regroup_seconds: float = 0.0
    encode_seconds: float = 0.0
    write_seconds: float = 0.0
    total_seconds: float = 0.0


def sticky_chunks(doc: ChunkedDoc) -> tuple[StickyChunk, ...]:
    """Assign initial IDs to existing retrieval spans before any edits."""
    return tuple(
        StickyChunk(f"{doc.doc_id}:{index}", tuple(doc.input_ids[start:end]))
        for index, (start, end) in enumerate(doc.spans)
    )


def _budgets(encoder: Encoder, token_cap: int, batch_size: int) -> tuple[int, int, int]:
    if token_cap <= 0 or batch_size <= 0:
        raise ValueError("Sticky grouping requires a positive token_cap and batch_size")
    special_count = encoder.tokenizer.num_special_tokens_to_add(pair=False)
    return min(token_cap, encoder.max_length) - special_count, encoder.max_length - special_count, special_count


def _content_length(group: Group) -> int:
    return sum(len(chunk.token_ids) for chunk in group)


def _pack(chunks: Sequence[StickyChunk], budget: int, model_budget: int) -> list[Group]:
    groups = []
    start = used = 0
    for index, chunk in enumerate(chunks):
        length = len(chunk.token_ids)
        if length > model_budget:
            raise ValueError(f"Chunk {chunk.chunk_id!r} exceeds the encoder model limit")
        # A chunk that exceeds the group cap can still occupy a group on its own.
        if index > start and used + length > budget:
            groups.append(tuple(chunks[start:index]))
            start, used = index, 0
        used += length
    if start < len(chunks):
        groups.append(tuple(chunks[start:]))
    return groups


def _encode_groups(encoder: Encoder, groups: Sequence[Group], batch_size: int) -> list[torch.Tensor]:
    inputs = [group_input(encoder.tokenizer, [chunk.token_ids for chunk in group]) for group in groups]
    return encode_group_inputs(encoder, inputs, batch_size)


def encode_sticky(
    encoder: Encoder, docs: Sequence[ChunkedDoc], batch_size: int = 16, *, token_cap: int = 1024
) -> torch.Tensor:
    """Encode fresh document groups in the original retrieval chunk order."""
    budget, model_budget, _ = _budgets(encoder, token_cap, batch_size)
    groups = [group for doc in docs for group in _pack(sticky_chunks(doc), budget, model_budget)]
    encoded = _encode_groups(encoder, groups, batch_size)
    if not encoded:
        return torch.empty(0, encoder.model.config.hidden_size)
    return torch.cat(encoded)


class StickyGroups:
    """Mutable groups and vectors for one document, tied to a fixed encoder and tokenizer."""

    def __init__(
        self,
        encoder: Encoder,
        chunks: Sequence[StickyChunk],
        *,
        token_cap: int = 1024,
        batch_size: int = 16,
        group_map: Sequence[Sequence[str]] | None = None,
    ) -> None:
        started = perf_counter()
        self._encoder = encoder
        self._batch_size = batch_size
        self._budget, self._model_budget, self._special_count = _budgets(encoder, token_cap, batch_size)
        self._groups: tuple[Group, ...] = ()
        self._vectors: dict[str, torch.Tensor] = {}
        chunks = tuple(chunks)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(by_id) != len(chunks):
            raise ValueError("Duplicate chunk ID")

        if group_map is None:
            groups = _pack(chunks, self._budget, self._model_budget)
        else:
            ids = tuple(tuple(group) for group in group_map)
            if any(not group for group in ids) or tuple(chunk_id for group in ids for chunk_id in group) != tuple(by_id):
                raise ValueError("The group map must partition the chunks in document order")
            groups = [tuple(by_id[chunk_id] for chunk_id in group) for group in ids]
            if any(len(_pack(group, self._budget, self._model_budget)) != 1 for group in groups):
                raise ValueError("A stored group exceeds its token budget")
        self._commit(groups, started)

    @property
    def chunks(self) -> tuple[StickyChunk, ...]:
        return tuple(chunk for group in self._groups for chunk in group)

    @property
    def group_map(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(chunk.chunk_id for chunk in group) for group in self._groups)

    @property
    def vectors(self) -> Mapping[str, torch.Tensor]:
        """The current chunk vectors. Treat the tensor values as read-only."""
        return MappingProxyType(self._vectors)

    def _find(self, chunk_id: str) -> tuple[int, int]:
        for group_index, group in enumerate(self._groups):
            for index, chunk in enumerate(group):
                if chunk.chunk_id == chunk_id:
                    return group_index, index
        raise KeyError(chunk_id)

    def _put(self, group_index: int, members: Group, started: float) -> RefreshStats:
        packed = _pack(members, self._budget, self._model_budget)
        groups = self._groups[:group_index] + tuple(packed) + self._groups[group_index + 1 :]
        return self._commit(groups, started)

    def replace(self, chunk: StickyChunk) -> RefreshStats:
        started = perf_counter()
        group_index, index = self._find(chunk.chunk_id)
        group = self._groups[group_index]
        return self._put(group_index, group[:index] + (chunk,) + group[index + 1 :], started)

    def insert(self, chunk: StickyChunk, *, after_id: str | None) -> RefreshStats:
        started = perf_counter()
        if chunk.chunk_id in self._vectors:
            raise ValueError(f"Duplicate chunk ID {chunk.chunk_id!r}")
        if after_id is None:
            group = self._groups[0] if self._groups else ()
            return self._put(0, (chunk,) + group, started)
        group_index, index = self._find(after_id)
        group = self._groups[group_index]
        return self._put(group_index, group[: index + 1] + (chunk,) + group[index + 1 :], started)

    def delete(self, chunk_id: str) -> RefreshStats:
        started = perf_counter()
        group_index, index = self._find(chunk_id)
        group = self._groups[group_index]
        kept = group[:index] + group[index + 1 :]
        groups = list(self._groups)
        if not kept:
            del groups[group_index]
        else:
            groups[group_index] = kept
            # Merge below half capacity, preferring the left adjacent group.
            if 2 * _content_length(kept) < self._budget:
                for neighbour in (group_index - 1, group_index + 1):
                    if 0 <= neighbour < len(groups) and _content_length(kept + groups[neighbour]) <= self._budget:
                        left = min(group_index, neighbour)
                        groups[left : left + 2] = [groups[left] + groups[left + 1]]
                        break
        return self._commit(groups, started)

    def refresh(self) -> RefreshStats:
        """Refresh with no edits. The current group inputs and vectors still agree."""
        self.stats = RefreshStats()
        return self.stats

    def _commit(self, groups: Sequence[Group], started: float) -> RefreshStats:
        # Equal groups have the same ordered IDs, full token input and pooling spans.
        previous = set(self._groups)
        pending = [group for group in groups if group not in previous]
        removed = self._vectors.keys() - {chunk.chunk_id for group in groups for chunk in group}
        input_tokens = sum(_content_length(group) + self._special_count for group in pending)
        encode_started = perf_counter()
        encoded = _encode_groups(self._encoder, pending, self._batch_size)
        write_started = perf_counter()

        vectors = self._vectors.copy()
        for chunk_id in removed:
            del vectors[chunk_id]
        for group, rows in zip(pending, encoded, strict=True):
            for chunk, row in zip(group, rows, strict=True):
                vectors[chunk.chunk_id] = row
        self._groups, self._vectors = tuple(groups), vectors

        finished = perf_counter()
        self.stats = RefreshStats(
            refreshed_groups=len(pending),
            refreshed_input_tokens=input_tokens,
            encoder_calls=(len(pending) + self._batch_size - 1) // self._batch_size,
            rewritten_vectors=sum(len(group) for group in pending),
            removed_vectors=len(removed),
            regroup_seconds=encode_started - started,
            encode_seconds=write_started - encode_started,
            write_seconds=finished - write_started,
            total_seconds=finished - started,
        )
        return self.stats
