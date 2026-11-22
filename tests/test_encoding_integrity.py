"""Model-free regression tests for encoding integrity."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from latechunk.chunking import ChunkedDoc
from latechunk.encoding import (
    Encoder,
    encode_late,
    encode_local_contextual,
    encode_naive,
    encode_naive_prefix,
    plan_long_windows,
    pool_spans,
)


class SyntheticTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [self.cls_token_id, *ids, self.sep_token_id]

    def __call__(self, texts, *, add_special_tokens, truncation, verbose):
        assert truncation is False
        is_single = isinstance(texts, str)
        inputs = [list(map(int, text.split())) for text in ([texts] if is_single else texts)]
        if add_special_tokens:
            inputs = [self.build_inputs_with_special_tokens(ids) for ids in inputs]
        return {"input_ids": inputs[0] if is_single else inputs}


class TwoLeadingSpecials(SyntheticTokenizer):
    def build_inputs_with_special_tokens(self, ids):
        return [self.cls_token_id, 103, *ids, self.sep_token_id]


def document(doc_id, chunks):
    ids = [101, *(token for chunk in chunks for token in chunk), 102]
    spans = []
    start = 1
    for chunk in chunks:
        end = start + len(chunk)
        spans.append((start, end))
        start = end
    return ChunkedDoc(doc_id, ids, spans, [" ".join(map(str, chunk)) for chunk in chunks])


@pytest.fixture
def encoder_and_inputs():
    encoder = Encoder.__new__(Encoder)
    encoder.max_length = 8
    encoder.model = SimpleNamespace(config=SimpleNamespace(hidden_size=2))
    encoder.tokenizer = SyntheticTokenizer()
    seen = []

    def token_embeddings(inputs):
        seen.extend(inputs)
        return [torch.tensor([[token, pos] for pos, token in enumerate(ids)], dtype=torch.float32) for ids in inputs]

    encoder.token_embeddings = token_embeddings
    return encoder, seen


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
@pytest.mark.parametrize(
    "encode",
    [
        lambda e, size: e.embed_fn(size),
        lambda e, size: e.encode_texts(["1"], size),
        lambda e, size: e.encode_token_ids([[1]], size),
        lambda e, size: encode_naive(e, [], size),
        lambda e, size: encode_naive_prefix(e, [], {}, size),
        lambda e, size: encode_late(e, [], size),
        lambda e, size: encode_late(e, [], size, long_window=6),
        lambda e, size: encode_local_contextual(e, [], size),
    ],
)
def test_invalid_batch_sizes_fail_before_any_inference(encoder_and_inputs, batch_size, encode):
    encoder, inputs = encoder_and_inputs
    with pytest.raises(ValueError, match="batch_size"):
        encode(encoder, batch_size)
    assert inputs == []


def test_query_and_independent_encoding_reject_instead_of_truncating(encoder_and_inputs):
    encoder, inputs = encoder_and_inputs
    oversized = "1 2 3 4 5 6 7"
    with pytest.raises(ValueError, match="9 tokens exceeds max_length=8"):
        encoder.encode_texts(["1", oversized], batch_size=1)
    with pytest.raises(ValueError, match="9 tokens exceeds max_length=8"):
        encode_naive(encoder, [ChunkedDoc("a", [], [], [oversized])])
    assert inputs == []
    encoder.encode_texts(["1 2 3 4 5 6"])
    assert inputs == [[101, 1, 2, 3, 4, 5, 6, 102]]


def test_empty_encoder_input_fails_before_inference(encoder_and_inputs):
    encoder, inputs = encoder_and_inputs
    with pytest.raises(ValueError, match="at least one token"):
        encoder.encode_token_ids([[1], []], batch_size=1)
    assert inputs == []
    assert encoder.encode_texts([]).shape == (0, 2)
    assert Encoder.token_embeddings(encoder, []) == []


def test_full_late_overflow_is_not_replaced_with_long_late(encoder_and_inputs):
    encoder, inputs = encoder_and_inputs
    docs = [document("short", [[1]]), document("long", [[11, 12, 13], [14, 15, 16, 17]])]
    with pytest.raises(ValueError, match="Full Late document long.*exceeds max_length=8"):
        encode_late(encoder, docs, batch_size=1)
    assert inputs == []

    vectors = encode_late(encoder, docs, batch_size=1, long_window=6, long_overlap=2)
    assert vectors.shape == (3, 2)
    assert inputs == [[101, 1, 102], [101, 11, 12, 13, 14, 15], [14, 15, 16, 17, 102]]


@pytest.mark.parametrize(
    "options,pattern",
    [
        ({"long_window": 0}, "window"),
        ({"long_window": 9}, "max_length"),
        ({"long_window": 4, "long_overlap": -1}, "overlap"),
        ({"long_window": 4, "long_overlap": 4}, "overlap"),
        ({"long_window": 3, "long_overlap": 1, "special_tokens": "wrap"}, "overlap"),
        ({"long_window": 2, "special_tokens": "wrap"}, "usable tokens"),
        ({"long_window": 4, "special_tokens": "invalid"}, "special_tokens"),
    ],
)
def test_invalid_long_settings_fail_even_if_all_docs_fit(encoder_and_inputs, options, pattern):
    encoder, inputs = encoder_and_inputs
    with pytest.raises(ValueError, match=pattern):
        encode_late(encoder, [document("a", [[1]])], **options)
    assert inputs == []


def test_wrap_requires_template_tokens_even_for_a_fitting_doc(encoder_and_inputs):
    encoder, inputs = encoder_and_inputs
    invalid = ChunkedDoc("wrong-wrapper", [0, 1, 102], [(1, 2)], ["1"])
    with pytest.raises(ValueError, match="expects input_ids"):
        encode_late(encoder, [invalid], long_window=6, special_tokens="wrap")
    assert inputs == []


@pytest.mark.parametrize("encode", [encode_late, encode_local_contextual])
@pytest.mark.parametrize("spans", [[(1, 1)], [(0, 2)], [(2, 4)], [(1, 3), (2, 3)]])
def test_invalid_doc_spans_fail_before_contextual_encoding(encoder_and_inputs, encode, spans):
    encoder, inputs = encoder_and_inputs
    valid = document("valid", [[1, 2]])
    invalid = ChunkedDoc("invalid", valid.input_ids, spans, ["1"])
    with pytest.raises(ValueError, match="Invalid pooling span"):
        encode(encoder, [valid, invalid], batch_size=1)
    assert inputs == []


@pytest.mark.parametrize("span", [(-1, 1), (0, 0), (0, 3), (0.5, 1)])
def test_pooling_rejects_empty_or_out_of_bounds_spans(span):
    tensor = torch.ones((2, 3))
    with pytest.raises(ValueError, match="Invalid pooling span"):
        pool_spans(tensor, [span])
    assert pool_spans(tensor, []).shape == (0, 3)


def test_local_uses_special_template_offset_with_partial_target_spans(encoder_and_inputs):
    encoder, inputs = encoder_and_inputs

    encoder.tokenizer = TwoLeadingSpecials()
    doc = ChunkedDoc("partial", [101, 103, 40, 41, 5, 6, 7, 102], [(4, 6), (6, 7)], ["5 6", "7"])
    vectors = encode_local_contextual(encoder, [doc])
    assert inputs == [[101, 103, 5, 6, 7, 102]]
    torch.testing.assert_close(vectors, F.normalize(torch.tensor([[5.5, 2.5], [7.0, 4.0]]), dim=-1))


def test_prefix_budget_counts_actual_specials(encoder_and_inputs):
    encoder, inputs = encoder_and_inputs
    encoder.tokenizer = TwoLeadingSpecials()
    doc = ChunkedDoc("a", [101, 103, 4, 5, 6, 7, 102], [(2, 6)], ["4 5 6 7"])
    encode_naive_prefix(encoder, [doc], {"a": "8 9 10"})
    assert inputs == [[101, 103, 8, 4, 5, 6, 7, 102]]


def test_long_window_plans_have_exact_document_ownership():
    ids = [101, 1, 2, 3, 4, 5, 102]
    assert plan_long_windows(ids, 5, 1) == [
        (0, 5, 0, 0, [101, 1, 2, 3, 4]),
        (4, 7, 5, 0, [4, 5, 102]),
    ]
    assert plan_long_windows(ids, 5, 1, "wrap", 101, 102) == [
        (1, 4, 1, 1, [101, 1, 2, 3, 102]),
        (3, 6, 4, 1, [101, 3, 4, 5, 102]),
    ]


@pytest.mark.parametrize("tokenizer_revision", [None, "different-tokenizer"])
def test_encoder_records_loaded_revisions_and_enforces_model_limit(monkeypatch, tmp_path, tokenizer_revision):
    from transformers import AutoModel, AutoTokenizer

    requested_model = "model-branch"
    expected_tokenizer_input = requested_model if tokenizer_revision is None else tokenizer_revision
    model_commit, tokenizer_commit, code_commit = ("a" * 40, "b" * 40, "c" * 40)
    model_name = "org/model"

    class Model:
        config = SimpleNamespace(_commit_hash=model_commit, max_position_embeddings=64)

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

    def load_model(name, *, revision, trust_remote_code, code_revision):
        assert (name, revision, trust_remote_code, code_revision) == (model_name, requested_model, True, "code-branch")
        return Model()

    def load_tokenizer(name, *, revision):
        assert (name, revision) == (model_name, expected_tokenizer_input)
        return SimpleNamespace(init_kwargs={})

    def cached_tokenizer(name, filename, *, revision):
        assert (name, filename, revision) == (model_name, "tokenizer_config.json", expected_tokenizer_input)
        return str(tmp_path / "snapshots" / tokenizer_commit / filename)

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", cached_tokenizer)
    monkeypatch.setattr(AutoModel, "from_pretrained", load_model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(Model, "__module__", f"transformers_modules.remote.{code_commit}.modeling")
    encoder = Encoder(
        model_name,
        "cpu",
        128,
        "code-branch",
        model_revision=requested_model,
        tokenizer_revision=tokenizer_revision,
    )
    assert (encoder.model_name, encoder.model_revision, encoder.tokenizer_revision, encoder.code_revision) == (
        model_name, model_commit, tokenizer_commit, code_commit
    )
    assert encoder.max_length == 64
