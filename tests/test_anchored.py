"""Model-free checks for content-anchored grouping and exact refresh."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latechunk.anchored import (  # noqa: E402
    AnchoredChunk,
    AnchoredGroups,
    anchored_chunks,
    encode_content_anchored,
    is_anchor,
)
from latechunk.chunking import ChunkedDoc  # noqa: E402
from latechunk.encoding import Encoder  # noqa: E402


class AnchoredTestTokenizer:
    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, **kwargs):
        ids = [int(token) for token in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if add_special_tokens:
            ids = self.build_inputs_with_special_tokens(ids)
        return {"input_ids": ids}

    def encode(self, text, **kwargs):
        return self(text, **kwargs)["input_ids"]

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [101, *ids, 102]


@pytest.fixture
def anchored_encoder():
    encoder = Encoder.__new__(Encoder)
    encoder.tokenizer = AnchoredTestTokenizer()
    encoder.max_length = 128
    encoder.model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
    batches = []

    def token_embeddings(inputs):
        batches.append(tuple(map(tuple, inputs)))
        return [
            torch.tensor([[token, position, sum(ids)] for position, token in enumerate(ids)], dtype=torch.float32)
            for ids in inputs
        ]

    encoder.token_embeddings = token_embeddings
    return encoder, batches


@pytest.mark.parametrize(
    "text,k4,k8",
    [("A", False, False), ("D", True, True), ("\u951a\u70b99", True, False)],
)
def test_anchor_decision_uses_stable_utf8_hash(text, k4, k8):
    assert is_anchor(text, 4) is k4
    assert is_anchor(text, 8) is k8


def test_insertion_resynchronizes_after_next_unchanged_anchor(anchored_encoder):
    encoder, batches = anchored_encoder
    chunks = (
        AnchoredChunk("A", "A", (10, 11)),
        AnchoredChunk("B", "B", (20, 21)),
        AnchoredChunk("C", "D", (30, 31)),  # Anchor ends the first segment.
        AnchoredChunk("D", "C", (40, 41)),
        AnchoredChunk("E", "F", (50, 51)),  # Anchor ends the second segment.
        AnchoredChunk("F", "E", (60, 61)),
    )
    state = AnchoredGroups(encoder, chunks, token_cap=8)
    assert state.group_map == (("A", "B", "C"), ("D", "E"), ("F",))
    before = {chunk_id: state.vectors[chunk_id].clone() for chunk_id in ("D", "E", "F")}
    batches.clear()

    edited = chunks[:1] + (AnchoredChunk("X", "X", (90, 91)),) + chunks[1:]
    stats = state.refresh(edited)

    assert state.group_map == (("A", "X", "B"), ("C",), ("D", "E"), ("F",))
    assert {ids for batch in batches for ids in batch} == {
        (101, 10, 11, 90, 91, 20, 21, 102),
        (101, 30, 31, 102),
    }
    assert (stats.refreshed_groups, stats.refreshed_input_tokens, stats.rewritten_vectors) == (2, 12, 4)
    for chunk_id, vector in before.items():
        torch.testing.assert_close(state.vectors[chunk_id], vector)

    rebuilt = AnchoredGroups(encoder, edited, token_cap=8)
    assert rebuilt.group_map == state.group_map
    for chunk_id, vector in state.vectors.items():
        torch.testing.assert_close(rebuilt.vectors[chunk_id], vector)

    batches.clear()
    noop = state.refresh(edited)
    assert noop.encoder_calls == noop.refreshed_input_tokens == noop.rewritten_vectors == 0
    assert not batches


def test_deleting_anchor_changes_only_groups_before_next_boundary(anchored_encoder):
    encoder, batches = anchored_encoder
    chunks = (
        AnchoredChunk("A", "A", (10,)),
        AnchoredChunk("B", "D", (20,)),
        AnchoredChunk("C", "C", (30,)),
        AnchoredChunk("D", "F", (40,)),
        AnchoredChunk("E", "E", (50,)),
    )
    state = AnchoredGroups(encoder, chunks, token_cap=10)
    assert state.group_map == (("A", "B"), ("C", "D"), ("E",))
    batches.clear()

    stats = state.refresh(chunks[:1] + chunks[2:])

    assert state.group_map == (("A", "C", "D"), ("E",))
    assert set(state.vectors) == {"A", "C", "D", "E"}
    assert batches == [((101, 10, 30, 40, 102),)]
    assert stats.removed_vectors == 1
    assert stats.rewritten_vectors == 3


def test_prefix_influences_context_without_entering_pooling_span(anchored_encoder):
    encoder, batches = anchored_encoder
    chunks = (AnchoredChunk("A", "A", (11, 13)), AnchoredChunk("B", "B", (21,)))
    prefix = tuple(range(300, 365))
    state = AnchoredGroups(encoder, chunks, anchor_k=1, token_cap=100, prefix=" ".join(map(str, prefix)))
    input_a = (101, *prefix[:64], 11, 13, 102)
    input_b = (101, *prefix[:64], 21, 102)
    assert {ids for batch in batches for ids in batch} == {input_a, input_b}
    expected_a = F.normalize(torch.tensor([12.0, 65.5, sum(input_a)]), dim=0)
    torch.testing.assert_close(state.vectors["A"], expected_a)

    batches.clear()
    # The discarded suffix of a prefix cannot change an encoder input.
    clipped = state.refresh(chunks, prefix=" ".join(map(str, (*prefix[:64], 900))))
    assert clipped.encoder_calls == clipped.rewritten_vectors == 0
    assert not batches

    changed = state.refresh(chunks, prefix=" ".join(map(str, (900, *prefix[1:]))))
    assert changed.refreshed_groups == changed.rewritten_vectors == 2
    assert {ids[1] for batch in batches for ids in batch} == {900}
    assert not torch.allclose(state.vectors["A"], expected_a)


def test_cap_counts_special_tokens_and_clips_prefix_before_chunk(anchored_encoder):
    encoder, batches = anchored_encoder
    chunks = (AnchoredChunk("A", "A", (11, 13)), AnchoredChunk("B", "B", (21, 23)))
    state = AnchoredGroups(encoder, chunks, token_cap=5, prefix="200 201 202")
    assert state.group_map == (("A",), ("B",))
    assert {ids for batch in batches for ids in batch} == {(101, 200, 11, 13, 102), (101, 200, 21, 23, 102)}


def test_chunk_that_exceeds_group_cap_is_preserved_within_model_limit(anchored_encoder):
    encoder, batches = anchored_encoder
    encoder.max_length = 7
    chunks = (
        AnchoredChunk("A", "A", (11,) * 5),
        AnchoredChunk("B", "B", (21, 22)),
        AnchoredChunk("C", "C", (31, 32)),
    )
    state = AnchoredGroups(encoder, chunks, token_cap=6)
    assert state.group_map == (("A",), ("B", "C"))
    assert {ids for batch in batches for ids in batch} == {(101, 11, 11, 11, 11, 11, 102), (101, 21, 22, 31, 32, 102)}

    batches.clear()
    with pytest.raises(ValueError):
        state.refresh((AnchoredChunk("A", "A", (11,) * 6), *chunks[1:]))
    assert batches == []
    assert state.group_map == (("A",), ("B", "C"))


def test_new_chunk_ids_reuse_exact_input_while_changed_pooling_refreshes_vectors(anchored_encoder):
    encoder, batches = anchored_encoder
    state = AnchoredGroups(encoder, [AnchoredChunk("A", "A", (11, 13))])
    pooled = state.vectors["A"].clone()
    batches.clear()

    rename = state.refresh([AnchoredChunk("B", "B", (11, 13))])
    assert rename.encoder_calls == 0
    assert not batches
    assert set(state.vectors) == {"B"}
    torch.testing.assert_close(state.vectors["B"], pooled)

    # The raw token input stays the same, but two independent spans must be pooled.
    state.refresh([AnchoredChunk("B", "B", (11,)), AnchoredChunk("C", "C", (13,))])
    assert state.group_map == (("B", "C"),)
    torch.testing.assert_close(state.vectors["B"], F.normalize(torch.tensor([11.0, 1.0, 227.0]), dim=0))
    torch.testing.assert_close(state.vectors["C"], F.normalize(torch.tensor([13.0, 2.0, 227.0]), dim=0))


def test_static_document_encoding_keeps_retrieval_chunk_order(anchored_encoder):
    encoder, batches = anchored_encoder
    docs = [
        ChunkedDoc("first", [101, 11, 12, 21, 102], [(1, 3), (3, 4)], ["A", "D"]),
        ChunkedDoc("second", [101, 31, 102], [(1, 2)], ["E"]),
    ]
    first, second = anchored_chunks(docs[0])
    assert (first.text, first.token_ids, second.text, second.token_ids) == ("A", (11, 12), "D", (21,))

    result = encode_content_anchored(encoder, docs, batch_size=2, prefixes={"first": "300"})

    first_input = (101, 300, 11, 12, 21, 102)
    second_input = (101, 31, 102)
    assert {ids for batch in batches for ids in batch} == {first_input, second_input}
    expected = torch.tensor([[11.5, 2.5, sum(first_input)], [21, 4, sum(first_input)], [31, 1, sum(second_input)]])
    torch.testing.assert_close(result, F.normalize(expected, dim=-1))
