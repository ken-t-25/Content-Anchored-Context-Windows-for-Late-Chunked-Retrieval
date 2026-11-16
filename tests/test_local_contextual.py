import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latechunk.chunking import ChunkedDoc  # noqa: E402
from latechunk.encoding import Encoder, encode_late, encode_local_contextual  # noqa: E402


class LocalTestTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [self.cls_token_id, *ids, self.sep_token_id]


def make_doc(doc_id, chunks):
    tokenizer = LocalTestTokenizer()
    tokens = [tokenizer.cls_token_id]
    spans = []
    for chunk in chunks:
        start = len(tokens)
        tokens.extend(chunk)
        spans.append((start, len(tokens)))
    tokens.append(tokenizer.sep_token_id)
    texts = [" ".join(map(str, chunk)) for chunk in chunks]
    return ChunkedDoc(doc_id, tokens, spans, texts)


@pytest.fixture
def local_encoder():
    encoder = Encoder.__new__(Encoder)
    encoder.tokenizer = LocalTestTokenizer()
    encoder.max_length = 32
    encoder.model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
    encoded_inputs = []

    def token_embeddings(batch_ids):
        # Include position and window context so both influence the target embedding.
        encoded_inputs.extend(map(tuple, batch_ids))
        return [
            torch.tensor([[token, pos, sum(ids)] for pos, token in enumerate(ids)], dtype=torch.float32)
            for ids in batch_ids
        ]

    encoder.token_embeddings = token_embeddings
    return encoder, encoded_inputs


def test_local_windows_pool_only_the_target_in_original_order(local_encoder):
    encoder, encoded_inputs = local_encoder
    docs = [make_doc("first", [[11, 12], [21], [31, 32], [41]]), make_doc("second", [[51]])]

    result = encode_local_contextual(encoder, docs, batch_size=2)

    windows = [
        (101, 11, 12, 21, 102),
        (101, 11, 12, 21, 31, 32, 102),
        (101, 21, 31, 32, 41, 102),
        (101, 31, 32, 41, 102),
        (101, 51, 102),
    ]
    assert set(encoded_inputs) == set(windows)
    expected = torch.tensor([
        [11.5, 1.5, sum(windows[0])],
        [21.0, 3.0, sum(windows[1])],
        [31.5, 2.5, sum(windows[2])],
        [41.0, 3.0, sum(windows[3])],
        [51.0, 1.0, sum(windows[4])],
    ])
    torch.testing.assert_close(result, F.normalize(expected, dim=-1))


def test_local_window_covering_document_matches_late(local_encoder):
    encoder, encoded_inputs = local_encoder
    docs = [make_doc("doc", [[11, 12], [21], [31, 32]])]

    result = encode_local_contextual(encoder, docs, radius=2)

    assert encoded_inputs == [tuple(docs[0].input_ids)]
    torch.testing.assert_close(result, encode_late(encoder, docs))


@pytest.mark.parametrize(
    "context_cap,target_position,window",
    [(5, 3, (101, 11, 12, 21, 102)), (4, 1, (101, 21, 31, 102))],
)
def test_local_cap_counts_special_tokens_and_keeps_whole_neighbors(
    local_encoder, context_cap, target_position, window
):
    encoder, encoded_inputs = local_encoder
    docs = [make_doc("doc", [[11, 12], [21], [31]])]

    result = encode_local_contextual(encoder, docs, context_cap=context_cap)

    expected = torch.tensor([21.0, target_position, sum(window)])
    torch.testing.assert_close(result[1], F.normalize(expected, dim=0))
    assert window in encoded_inputs
    assert all(len(ids) <= context_cap for ids in encoded_inputs)


def test_local_target_above_context_cap_is_encoded_alone(local_encoder):
    encoder, encoded_inputs = local_encoder
    docs = [make_doc("doc", [[11, 12, 13, 14], [21]])]

    encode_local_contextual(encoder, docs, context_cap=4)

    assert set(encoded_inputs) == {(101, 11, 12, 13, 14, 102), (101, 21, 102)}


def test_local_target_above_model_limit_raises(local_encoder):
    encoder, _ = local_encoder
    encoder.max_length = 5
    docs = [make_doc("doc", [[11, 12, 13, 14]])]

    with pytest.raises(ValueError):
        encode_local_contextual(encoder, docs)
