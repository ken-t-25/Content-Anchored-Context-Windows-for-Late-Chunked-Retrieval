"""Content-Anchored grouping and exact incremental chunk vector refresh."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from time import perf_counter
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .chunking import ChunkedDoc
from .encoding import MAX_PREFIX_TOKENS, Encoder
from .group_encoding import GroupInput, encode_group_inputs, encoder_cache_identity, group_input, refresh_token_inputs
from .sticky import RefreshStats


@dataclass(frozen=True)
class AnchoredChunk:
    chunk_id: str
    text: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if not self.token_ids:
            raise ValueError("Anchored chunks must contain at least one token")


@dataclass(frozen=True)
class AnchoredGroup:
    chunk_ids: tuple[str, ...]
    input: GroupInput


def is_anchor(text: str, anchor_k: int) -> bool:
    if anchor_k <= 0:
        raise ValueError("anchor_k must be positive")
    digest = blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % anchor_k == 0


def anchored_chunks(doc: ChunkedDoc) -> tuple[AnchoredChunk, ...]:
    return tuple(
        AnchoredChunk(f"{doc.doc_id}:{index}", text, tuple(doc.input_ids[start:end]))
        for index, ((start, end), text) in enumerate(zip(doc.spans, doc.texts, strict=True))
    )


def _plan(
    encoder: Encoder, chunks: Sequence[AnchoredChunk], token_cap: int, anchor_k: int, prefix: str
) -> list[AnchoredGroup]:
    if token_cap <= 0 or anchor_k <= 0:
        raise ValueError("token_cap and anchor_k must be positive")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk ID")
    tok = encoder.tokenizer
    specials = tok.num_special_tokens_to_add(pair=False)
    model_budget = encoder.max_length - specials
    budget = min(token_cap, encoder.max_length) - specials
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"][:MAX_PREFIX_TOKENS] if prefix else []
    body_budget = budget - len(prefix_ids)
    groups = []
    members: list[AnchoredChunk] = []
    used = 0

    def close_group() -> None:
        nonlocal used
        prefix_within_cap = prefix_ids[:max(0, budget - used)]
        inputs = group_input(tok, [chunk.token_ids for chunk in members], prefix_within_cap)
        groups.append(AnchoredGroup(tuple(chunk.chunk_id for chunk in members), inputs))
        members.clear()
        used = 0

    for chunk in chunks:
        length = len(chunk.token_ids)
        if length > model_budget:
            raise ValueError(f"Chunk {chunk.chunk_id!r} exceeds the encoder model limit")
        if members and used + length > body_budget:
            close_group()
        members.append(chunk)
        used += length
        if is_anchor(chunk.text, anchor_k):
            close_group()
    if members:
        close_group()
    return groups


def plan_content_anchored(
    encoder: Encoder,
    docs: Sequence[ChunkedDoc],
    *,
    token_cap: int = 1024,
    anchor_k: int = 4,
    prefixes: Mapping[str, str] | None = None,
) -> list[AnchoredGroup]:
    """Plan groups in document and retrieval chunk order."""
    prefixes = prefixes or {}
    return [
        group for doc in docs
        for group in _plan(encoder, anchored_chunks(doc), token_cap, anchor_k, prefixes.get(doc.doc_id, ""))
    ]


def encode_content_anchored(
    encoder: Encoder,
    docs: Sequence[ChunkedDoc],
    batch_size: int = 16,
    *,
    token_cap: int = 1024,
    anchor_k: int = 4,
    prefixes: Mapping[str, str] | None = None,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    groups = plan_content_anchored(encoder, docs, token_cap=token_cap, anchor_k=anchor_k, prefixes=prefixes)
    if not groups:
        return torch.empty(0, encoder.model.config.hidden_size)
    return torch.cat(encode_group_inputs(encoder, [group.input for group in groups], batch_size))


class AnchoredGroups:
    """One document's cache, scoped to the current encoder and tokenizer."""

    def __init__(
        self,
        encoder: Encoder,
        chunks: Sequence[AnchoredChunk],
        *,
        token_cap: int = 1024,
        anchor_k: int = 4,
        batch_size: int = 16,
        prefix: str = "",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._encoder = encoder
        self._token_cap = token_cap
        self._anchor_k = anchor_k
        self._batch_size = batch_size
        self._prefix = prefix
        self._identity = None
        self._outputs: dict[tuple[int, ...], torch.Tensor] = {}
        self._locations: dict[str, tuple[tuple[int, ...], tuple[int, int]]] = {}
        self._vectors: dict[str, torch.Tensor] = {}
        self.refresh(chunks)

    @property
    def group_map(self) -> tuple[tuple[str, ...], ...]:
        return self._group_map

    @property
    def vectors(self) -> Mapping[str, torch.Tensor]:
        """The current chunk vectors. Treat the tensor values as read-only."""
        return MappingProxyType(self._vectors)

    def refresh(self, chunks: Sequence[AnchoredChunk], *, prefix: str | None = None) -> RefreshStats:
        """Derive the next partition from content and reuse exact input matches."""
        started = perf_counter()
        prefix = self._prefix if prefix is None else prefix
        groups = _plan(self._encoder, chunks, self._token_cap, self._anchor_k, prefix)
        identity = encoder_cache_identity(self._encoder)
        same_encoder = identity == self._identity
        locations = {
            chunk_id: (group.input.token_ids, span)
            for group in groups for chunk_id, span in zip(group.chunk_ids, group.input.spans, strict=True)
        }
        rewritten = [
            chunk_id for chunk_id, address in locations.items()
            if not same_encoder or self._locations.get(chunk_id) != address
        ]
        removed = self._locations.keys() - locations.keys()

        encode_started = perf_counter()
        outputs, pending = refresh_token_inputs(
            self._encoder, [group.input.token_ids for group in groups], self._batch_size,
            self._outputs if same_encoder else None,
        )
        write_started = perf_counter()
        vectors = self._vectors.copy()
        for chunk_id in removed:
            del vectors[chunk_id]
        for chunk_id in rewritten:
            ids, (start, end) = locations[chunk_id]
            vectors[chunk_id] = F.normalize(outputs[ids][start:end].mean(dim=0), dim=0)
        self._identity, self._prefix = identity, prefix
        self._outputs, self._locations, self._vectors = outputs, locations, vectors
        self._group_map = tuple(group.chunk_ids for group in groups)

        finished = perf_counter()
        self.stats = RefreshStats(
            refreshed_groups=len(pending),
            refreshed_input_tokens=sum(map(len, pending)),
            encoder_calls=(len(pending) + self._batch_size - 1) // self._batch_size,
            rewritten_vectors=len(rewritten),
            removed_vectors=len(removed),
            regroup_seconds=encode_started - started,
            encode_seconds=write_started - encode_started,
            write_seconds=finished - write_started,
            total_seconds=finished - started,
        )
        return self.stats
