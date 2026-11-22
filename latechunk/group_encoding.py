"""Encode groups of existing retrieval chunks and pool their original spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def encode_group_inputs(encoder: Encoder, inputs: Sequence[GroupInput], batch_size: int) -> list[torch.Tensor]:
    order = sorted(range(len(inputs)), key=lambda index: len(inputs[index].token_ids))
    encoded = {}
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        outputs = encoder.token_embeddings([inputs[index].token_ids for index in batch])
        for index, token_vectors in zip(batch, outputs, strict=True):
            encoded[index] = F.normalize(pool_spans(token_vectors, inputs[index].spans), dim=-1)
    return [encoded[index] for index in range(len(inputs))]
