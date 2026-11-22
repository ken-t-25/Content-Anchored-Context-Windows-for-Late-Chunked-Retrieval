"""Encode groups of existing retrieval chunks and pool their original spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from .encoding import Encoder, pool_spans


@dataclass(frozen=True)
class GroupInput:
    token_ids: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]


def group_input(tokenizer, chunks: Sequence[Sequence[int]], prefix: Sequence[int] = ()) -> GroupInput:
    wrap = tokenizer.build_inputs_with_special_tokens
    # A sentinel locates the leading special tokens in the single-input template.
    offset = wrap([-1]).index(-1)
    content = list(prefix)
    spans = []
    for tokens in chunks:
        start = offset + len(content)
        content.extend(tokens)
        spans.append((start, offset + len(content)))
    return GroupInput(tuple(wrap(content)), tuple(spans))


@dataclass(frozen=True, eq=False)
class _ObjectIdentity:
    value: object

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ObjectIdentity) and self.value is other.value

    def __hash__(self) -> int:
        return id(self.value)


def encoder_cache_identity(encoder: Encoder) -> tuple:
    """Identify the current encoder and tokenizer namespace for in-memory caches."""
    model = getattr(encoder, "model", None)
    tokenizer = getattr(encoder, "tokenizer", None)
    tokenizer_config = getattr(tokenizer, "init_kwargs", {}) or {}
    return (
        _ObjectIdentity(encoder),
        _ObjectIdentity(model),
        _ObjectIdentity(tokenizer),
        getattr(encoder, "model_name", None),
        getattr(encoder, "model_revision", None),
        getattr(encoder, "tokenizer_revision", None),
        getattr(encoder, "code_revision", None),
        getattr(getattr(model, "config", None), "_commit_hash", None),
        tokenizer_config.get("_commit_hash"),
        tokenizer_config.get("revision"),
        getattr(tokenizer, "pad_token_id", None),
        getattr(encoder, "device", None),
        getattr(model, "dtype", None),
        getattr(encoder, "max_length", None),
    )


def _token_batches(
    encoder: Encoder, inputs: Sequence[tuple[int, ...]], batch_size: int
) -> Iterator[tuple[tuple[int, ...], torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered = sorted(inputs, key=len)
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        yield from zip(batch, encoder.token_embeddings(batch), strict=True)


def refresh_token_inputs(
    encoder: Encoder,
    inputs: Sequence[tuple[int, ...]],
    batch_size: int,
    previous: Mapping[tuple[int, ...], torch.Tensor] | None = None,
) -> tuple[dict[tuple[int, ...], torch.Tensor], tuple[tuple[int, ...], ...]]:
    """Return live token outputs and the unique input rows encoded in this refresh."""
    previous = previous or {}
    live = dict.fromkeys(inputs)
    outputs = {ids: previous[ids] for ids in live if ids in previous}
    missing = tuple(ids for ids in live if ids not in previous)
    outputs.update(_token_batches(encoder, missing, batch_size))
    return outputs, missing


def encode_group_inputs(encoder: Encoder, inputs: Sequence[GroupInput], batch_size: int) -> list[torch.Tensor]:
    layouts: dict[tuple[int, ...], dict[GroupInput, None]] = {}
    for item in inputs:
        layouts.setdefault(item.token_ids, {})[item] = None
    pooled: dict[GroupInput, torch.Tensor] = {}
    # Keep only pooled rows after each batch, so corpus builds need not store raw outputs.
    for ids, token_vectors in _token_batches(encoder, list(layouts), batch_size):
        for item in layouts[ids]:
            pooled[item] = F.normalize(pool_spans(token_vectors, item.spans), dim=-1)
    return [pooled[item] for item in inputs]
