"""Model-free checks for Sticky ID Groups and exact incremental refresh."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latechunk.encoding import Encoder  # noqa: E402
from latechunk.sticky import StickyChunk, StickyGroups  # noqa: E402


class StickyTestTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [self.cls_token_id, *ids, self.sep_token_id]


@pytest.fixture
def sticky_encoder():
    encoder = Encoder.__new__(Encoder)
    encoder.tokenizer = StickyTestTokenizer()
    encoder.max_length = 32
    encoder.model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
    batches = []

    def token_embeddings(inputs):
        batches.append(tuple(map(tuple, inputs)))
        # The position and the full group both affect each output row.
        return [
            torch.tensor([[token, position, sum(ids)] for position, token in enumerate(ids)], dtype=torch.float32)
            for ids in inputs
        ]

    encoder.token_embeddings = token_embeddings
    return encoder, batches


def six_chunks():
    return [StickyChunk(name, (token, token + 1)) for name, token in zip("ABCDEF", range(10, 70, 10))]


def test_initial_groups_pool_each_span_and_batch_encoder_calls(sticky_encoder):
    encoder, batches = sticky_encoder
    sticky = StickyGroups(encoder, six_chunks(), token_cap=8)

    assert sticky.group_map == (("A", "B", "C"), ("D", "E", "F"))
    inputs = {ids for batch in batches for ids in batch}
    first_input = (101, 10, 11, 20, 21, 30, 31, 102)
    assert inputs == {first_input, (101, 40, 41, 50, 51, 60, 61, 102)}
    expected = F.normalize(torch.tensor([20.5, 3.5, float(sum(first_input))]), dim=0)
    torch.testing.assert_close(sticky.vectors["B"], expected)
    assert sticky.stats.refreshed_groups == 2
    assert sticky.stats.refreshed_input_tokens == 16
    assert sticky.stats.encoder_calls == len(batches) == 1
    assert sticky.stats.rewritten_vectors == 6


def test_replacement_refreshes_the_changed_group_and_no_op_reuses_it(sticky_encoder):
    encoder, batches = sticky_encoder
    chunks = six_chunks()
    sticky = StickyGroups(encoder, chunks, token_cap=8)
    original_a = sticky.vectors["A"].clone()
    original_d = sticky.vectors["D"].clone()
    batches.clear()

    replacement = StickyChunk("B", (70, 71))
    stats = sticky.replace(replacement)

    assert sticky.group_map == (("A", "B", "C"), ("D", "E", "F"))
    assert sticky.chunks[1] == replacement
    assert batches == [((101, 10, 11, 70, 71, 30, 31, 102),)]
    assert (stats.refreshed_groups, stats.refreshed_input_tokens, stats.rewritten_vectors) == (1, 8, 3)
    assert not torch.allclose(sticky.vectors["A"], original_a)
    torch.testing.assert_close(sticky.vectors["D"], original_d)

    batches.clear()
    for stats in (sticky.refresh(), sticky.replace(replacement)):
        assert stats.encoder_calls == stats.refreshed_groups == stats.rewritten_vectors == 0
        assert stats.refreshed_input_tokens == stats.removed_vectors == 0
    assert batches == []


def test_insertion_splits_locally_and_matches_rebuild_with_saved_groups(sticky_encoder):
    encoder, batches = sticky_encoder
    sticky = StickyGroups(encoder, six_chunks(), token_cap=8)
    original_d = sticky.vectors["D"].clone()
    batches.clear()

    stats = sticky.insert(StickyChunk("X", (80, 81)), after_id="B")

    assert sticky.group_map == (("A", "B", "X"), ("C",), ("D", "E", "F"))
    assert tuple(chunk.chunk_id for chunk in sticky.chunks) == tuple("ABXCDEF")
    assert {ids for batch in batches for ids in batch} == {
        (101, 10, 11, 20, 21, 80, 81, 102),
        (101, 30, 31, 102),
    }
    assert (stats.refreshed_groups, stats.refreshed_input_tokens, stats.rewritten_vectors) == (2, 12, 4)
    assert stats.encoder_calls == len(batches) == 1
    torch.testing.assert_close(sticky.vectors["D"], original_d)

    rebuilt = StickyGroups(encoder, sticky.chunks, token_cap=8, group_map=sticky.group_map)
    assert rebuilt.group_map == sticky.group_map
    for chunk_id, vector in sticky.vectors.items():
        torch.testing.assert_close(rebuilt.vectors[chunk_id], vector)

    freshly_grouped = StickyGroups(encoder, sticky.chunks, token_cap=8)
    assert freshly_grouped.group_map == (("A", "B", "X"), ("C", "D", "E"), ("F",))


@pytest.mark.parametrize(
    "left_size,expected",
    [(1, (("A", "B"), ("D",))), (5, (("A",), ("B", "D")))],
)
def test_deletion_merges_small_group_with_a_fitting_neighbor(sticky_encoder, left_size, expected):
    encoder, batches = sticky_encoder
    chunks = [StickyChunk("A", (11,) * left_size), StickyChunk("B", (21, 22)),
              StickyChunk("C", (31, 32)), StickyChunk("D", (41,))]
    sticky = StickyGroups(encoder, chunks, token_cap=8, group_map=(("A",), ("B", "C"), ("D",)))
    batches.clear()

    stats = sticky.delete("C")

    assert sticky.group_map == expected
    assert set(sticky.vectors) == {"A", "B", "D"}
    assert (stats.refreshed_groups, stats.refreshed_input_tokens, stats.rewritten_vectors) == (1, 5, 2)
    assert stats.removed_vectors == 1
    assert stats.encoder_calls == len(batches) == 1


def test_delete_singleton_drops_vector_without_encoding_and_can_reinsert(sticky_encoder):
    encoder, batches = sticky_encoder
    sticky = StickyGroups(encoder, [StickyChunk("A", (11,))], token_cap=8)
    batches.clear()

    stats = sticky.delete("A")

    assert sticky.group_map == ()
    assert not sticky.vectors
    assert (stats.removed_vectors, stats.rewritten_vectors, stats.encoder_calls) == (1, 0, 0)
    assert batches == []

    stats = sticky.insert(StickyChunk("B", (21,)), after_id=None)

    assert sticky.group_map == (("B",),)
    assert set(sticky.vectors) == {"B"}
    assert (stats.refreshed_groups, stats.rewritten_vectors, stats.encoder_calls) == (1, 1, 1)


def test_context_and_model_limits_include_special_tokens(sticky_encoder):
    encoder, batches = sticky_encoder
    chunks = [StickyChunk("A", (11,) * 5), StickyChunk("B", (21, 22)), StickyChunk("C", (31, 32))]
    encoder.max_length = 7
    sticky = StickyGroups(encoder, chunks, token_cap=6)

    assert sticky.group_map == (("A",), ("B", "C"))
    assert sorted(len(ids) for batch in batches for ids in batch) == [6, 7]

    with pytest.raises(ValueError):
        sticky.replace(StickyChunk("A", (11,) * 6))
    assert sticky.chunks == tuple(chunks)
    assert sum(len(batch) for batch in batches) == 2


@pytest.mark.parametrize("group_map", [(("B", "A"),), (("A",),), (("A", "A"),)])
def test_saved_map_must_be_an_ordered_partition(sticky_encoder, group_map):
    encoder, batches = sticky_encoder
    with pytest.raises(ValueError):
        StickyGroups(encoder, six_chunks()[:2], token_cap=8, group_map=group_map)
    assert batches == []


def test_rejects_duplicate_ids_and_missing_insertion_anchors(sticky_encoder):
    encoder, batches = sticky_encoder
    chunks = six_chunks()[:2]
    with pytest.raises(ValueError):
        StickyGroups(encoder, [*chunks, StickyChunk("A", (50,))], token_cap=8)
    assert batches == []

    sticky = StickyGroups(encoder, chunks, token_cap=8)
    batches.clear()
    with pytest.raises(ValueError):
        sticky.insert(StickyChunk("A", (50,)), after_id="B")
    with pytest.raises((KeyError, ValueError)):
        sticky.insert(StickyChunk("X", (50,)), after_id="missing")

    assert sticky.chunks == tuple(chunks)
    assert batches == []


def test_failed_encoder_leaves_state_and_vectors_unchanged(sticky_encoder, monkeypatch):
    encoder, _ = sticky_encoder
    chunks = six_chunks()
    sticky = StickyGroups(encoder, chunks, token_cap=8)
    before = {chunk_id: vector.clone() for chunk_id, vector in sticky.vectors.items()}

    def fail(inputs):
        raise RuntimeError("simulated encoder failure")

    monkeypatch.setattr(encoder, "token_embeddings", fail)
    with pytest.raises(RuntimeError, match="simulated encoder failure"):
        sticky.insert(StickyChunk("X", (80, 81)), after_id="B")

    assert sticky.group_map == (("A", "B", "C"), ("D", "E", "F"))
    assert sticky.chunks == tuple(chunks)
    assert set(sticky.vectors) == set(before)
    for chunk_id, vector in sticky.vectors.items():
        torch.testing.assert_close(vector, before[chunk_id])
