"""Regression checks for exact incremental inference and real text edits."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from latechunk.anchored import AnchoredChunk, AnchoredGroups
from latechunk.encoding import Encoder, encode_late
from latechunk.group_encoding import encode_group_inputs, group_input
from latechunk.positional_refresh import PositionalRefresh
from latechunk.sticky import StickyChunk, StickyGroups
from scripts.pilot_anchored import _as_document, _snapshots, run_edits


class TestTokenizer:
    __test__ = False
    cls_token_id, sep_token_id, pad_token_id = 101, 102, 0

    def build_inputs_with_special_tokens(self, ids):
        return [self.cls_token_id, *ids, self.sep_token_id]

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, *, add_special_tokens, truncation=False):
        assert not truncation
        ids = []
        for word in text.lower().split():
            if word == "suffix":
                ids.extend([40, 41])
            else:
                ids.append(sum(ord(letter) for letter in word) % 79 + 1)
        return {"input_ids": self.build_inputs_with_special_tokens(ids) if add_special_tokens else ids}


@pytest.fixture
def harness():
    encoder = Encoder.__new__(Encoder)
    encoder.max_length = 32
    encoder.model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
    encoder.tokenizer = TestTokenizer()
    calls = []

    def token_embeddings(inputs):
        calls.append(tuple(map(tuple, inputs)))
        shift = getattr(encoder, "test_shift", 0)
        return [
            torch.tensor([[token + shift, position, sum(ids)] for position, token in enumerate(ids)],
                         dtype=torch.float32)
            for ids in inputs
        ]

    encoder.token_embeddings = token_embeddings
    return encoder, calls


def _counts(stats):
    return stats.refreshed_groups, stats.refreshed_input_tokens, stats.encoder_calls, stats.rewritten_vectors


def _assert_rebuild(actual, expected):
    assert actual.keys() == expected.keys()
    for chunk_id, vector in actual.items():
        torch.testing.assert_close(vector, expected[chunk_id])


def test_group_encoding_reuses_one_inference_for_distinct_spans_in_the_same_call(harness):
    encoder, calls = harness
    single = group_input(encoder.tokenizer, [[11, 12]])
    split = group_input(encoder.tokenizer, [[11], [12]])
    whole, parts, repeated = encode_group_inputs(encoder, [single, split, single], batch_size=1)
    assert calls == [((101, 11, 12, 102),)]
    assert parts.shape == (2, 3)
    torch.testing.assert_close(whole, repeated)
    assert not torch.equal(whole[0], parts[0])


def test_anchored_reuses_input_but_updates_only_changed_spans(harness):
    encoder, calls = harness
    a = AnchoredChunk("A", "alpha", (11,))
    b = AnchoredChunk("B", "alpha", (12, 13))
    c = AnchoredChunk("C", "alpha", (14,))
    initial = (a, b, c)
    state = AnchoredGroups(encoder, initial, anchor_k=2**64)
    edited = (a, AnchoredChunk("B", "alpha", (12,)), AnchoredChunk("C", "alpha", (13, 14)))
    calls.clear()

    stats = state.refresh(edited)
    assert _counts(stats) == (0, 0, 0, 2)
    assert not calls
    rebuilt = AnchoredGroups(encoder, edited, anchor_k=2**64)
    _assert_rebuild(state.vectors, rebuilt.vectors)


def test_sticky_edit_sequence_splits_then_merges_and_matches_saved_map_harness(harness):
    encoder, calls = harness
    chunks = (StickyChunk("A", (11, 12)), StickyChunk("B", (21, 22)), StickyChunk("C", (31, 32)))
    state = StickyGroups(encoder, chunks, token_cap=8)
    assert state.group_map == (("A", "B", "C"),)
    inserted = StickyChunk("inserted", (41, 42, 43))

    split = state.insert(inserted, after_id="A")
    assert state.group_map == (("A", "inserted"), ("B", "C"))
    assert _counts(split) == (2, 13, 1, 4)
    _assert_rebuild(state.vectors, StickyGroups(encoder, state.chunks, token_cap=8,
                                                group_map=state.group_map).vectors)

    merge = state.delete(inserted.chunk_id)
    assert state.group_map == (("A", "B", "C"),)
    assert state.chunks == chunks
    assert _counts(merge) == (1, 8, 1, 3)
    assert merge.removed_vectors == 1
    _assert_rebuild(state.vectors, StickyGroups(encoder, state.chunks, token_cap=8,
                                                group_map=state.group_map).vectors)

    calls.clear()
    assert _counts(state.refresh()) == (0, 0, 0, 0)
    assert not calls


def test_sticky_reuses_equal_inputs_across_new_stable_ids(harness):
    encoder, calls = harness
    state = StickyGroups(encoder, [StickyChunk("A", (11, 12))], token_cap=4)
    stats = state.insert(StickyChunk("B", (11, 12)), after_id="A")
    assert _counts(stats) == (0, 0, 0, 1)
    assert len(calls) == 1
    assert state.group_map == (("A",), ("B",))
    torch.testing.assert_close(state.vectors["A"], state.vectors["B"])


@pytest.mark.parametrize("policy", ["anchored", "sticky", "positional"])
def test_exact_cache_is_invalidated_by_object_replacement_and_revisions(harness, policy):
    encoder, calls = harness
    chunks = (AnchoredChunk("A", "first", (11, 12)), AnchoredChunk("B", "second", (21, 22)))
    doc = _as_document(encoder.tokenizer, "D", chunks)
    if policy == "anchored":
        state = AnchoredGroups(encoder, chunks, anchor_k=2**64)
        def refresh():
            return state.refresh(chunks)
    elif policy == "sticky":
        state = StickyGroups(encoder, [StickyChunk(chunk.chunk_id, chunk.token_ids) for chunk in chunks])
        refresh = state.refresh
    else:
        state = PositionalRefresh(encoder, window=8, overlap=1)
        def refresh():
            return state.refresh(doc, ("A", "B"))
        refresh()

    assert _counts(refresh()) == (0, 0, 0, 0)
    changes = {
        "model_revision": "updated-model",
        "tokenizer_revision": "updated-tokenizer",
        "code_revision": "updated-code",
        "model": SimpleNamespace(config=SimpleNamespace(hidden_size=3)),
        "tokenizer": TestTokenizer(),
    }
    for attribute, value in changes.items():
        before = state.vectors["A"].clone()
        encoder.test_shift = getattr(encoder, "test_shift", 0) + 5
        setattr(encoder, attribute, value)
        calls.clear()
        assert _counts(refresh()) == (1, 6, 1, 2)
        assert len(calls) == 1
        assert not torch.equal(state.vectors["A"], before)
        assert _counts(refresh()) == (0, 0, 0, 0)
        assert len(calls) == 1


@pytest.mark.parametrize("special_tokens", ["raw", "wrap"])
def test_positional_refresh_matches_long_late_for_edits_crossing_window_boundaries(harness, special_tokens):
    encoder, calls = harness
    initial = (
        AnchoredChunk("A", "A", (11, 12, 13)),
        AnchoredChunk("B", "B", (21, 22, 23, 24)),
        AnchoredChunk("C", "C", (31, 32, 33, 34)),
    )
    edited = (
        initial[0], AnchoredChunk("X", "X", (41, 42)),
        AnchoredChunk("B", "B revised", (23, 24, 25, 26)), initial[2],
    )
    state = PositionalRefresh(encoder, window=7, overlap=2, special_tokens=special_tokens)
    for chunks in (initial, edited, initial, initial[:1]):
        doc = _as_document(encoder.tokenizer, "D", chunks)
        ids = tuple(chunk.chunk_id for chunk in chunks)
        state.refresh(doc, ids)
        expected = encode_late(encoder, [doc], long_window=7, long_overlap=2, special_tokens=special_tokens)
        _assert_rebuild(state.vectors, dict(zip(ids, expected, strict=True)))
        calls.clear()
        assert _counts(state.refresh(doc, ids)) == (0, 0, 0, 0)
        assert not calls


def test_pilot_retokenizes_insertions_and_replacements_but_keeps_retained_ids(harness):
    encoder, _ = harness
    tokenizer = encoder.tokenizer
    original = (
        AnchoredChunk("A", "begin", (10,)),
        AnchoredChunk("B", "suffix item", (42, 43)),
        AnchoredChunk("C", "end", (30,)),
    )
    replacement, inserted, snapshots = _snapshots(tokenizer, original, 1)
    assert replacement.chunk_id == "B"
    assert replacement.text != original[1].text
    assert len(replacement.token_ids) == len(original[1].token_ids)
    assert replacement.token_ids != original[1].token_ids
    assert inserted.text == original[1].text
    assert len(inserted.token_ids) == 3  # Standalone text expands at the original chunk boundary.
    for chunk in (replacement, inserted):
        assert chunk.token_ids == tuple(tokenizer(chunk.text, add_special_tokens=False)["input_ids"])
    assert snapshots["insert"] == (original[0], replacement, inserted, original[2])
    assert snapshots["delete"] == snapshots["replace"]
    edited = _as_document(tokenizer, "D", snapshots["insert"])
    for chunk, (start, end) in zip(snapshots["insert"], edited.spans, strict=True):
        assert tuple(edited.input_ids[start:end]) == chunk.token_ids

    args = Namespace(edit_position="middle", token_cap=7, batch_size=2, anchor_k=[4],
                     long_window=7, long_overlap=2, long_special_tokens="wrap", check_prefix=True)
    report = run_edits(encoder, _as_document(tokenizer, "D", original), original, args, prefix="Old title")
    assert report["replacement_tokens"] == report["original_tokens"] == 2
    assert report["inserted_tokens"] == 3
    assert report["prefix_checks"]["replace_prefix"]["encoder_calls"] > 0
    assert report["prefix_checks"]["noop_prefix"]["encoder_calls"] == 0
    for event, methods in report["events"].items():
        for stats in methods.values():
            assert stats["max_rebuild_error"] <= 1e-6
            if event.startswith("noop"):
                assert stats["encoder_calls"] == stats["rewritten_vectors"] == 0
