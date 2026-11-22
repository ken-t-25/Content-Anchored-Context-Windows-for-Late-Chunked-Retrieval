"""Regression checks at the native data and matched positional policy boundaries."""

from types import SimpleNamespace

import torch

from latechunk.chunking import ChunkedDoc
from latechunk.encoding import Encoder, encode_late
from latechunk.positional_refresh import PositionalRefresh
from latechunk.study_refresh import encode_study_long_late
from scripts.prepare_conteb import manifest


def test_export_groups_by_published_id_and_preserves_qrels():
    chunks = [{"chunk_id": "doc_a_10", "chunk": "Later paragraph"},
              {"chunk_id": "doc_a_2", "chunk": "Earlier paragraph"}]
    queries = [{"chunk_id": "doc_a_10", "query": "What changed?"}]
    data = manifest(chunks, queries, dataset="MLDR", split="test", source={"revision": "abc"})
    assert data["documents"] == [{"id": "doc_a", "title": "", "chunks": [
        {"id": "doc_a_10", "text": "Later paragraph"}, {"id": "doc_a_2", "text": "Earlier paragraph"},
    ]}]
    assert data["queries"][0]["source_doc_id"] == "doc_a"
    assert data["queries"][0]["relevance"] == {"doc_a_10": 1}


def test_matched_positional_singleton_pools_full_chunk_and_reuses_other_inputs():
    encoder = Encoder.__new__(Encoder)
    encoder.max_length = 32
    encoder.model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
    encoder.tokenizer = SimpleNamespace(cls_token_id=101, sep_token_id=102,
                                        build_inputs_with_special_tokens=lambda ids: [101, *ids, 102])
    calls = []

    def token_embeddings(batch):
        calls.extend(tuple(ids) for ids in batch)
        return [torch.tensor([[token, offset, sum(ids)] for offset, token in enumerate(ids)], dtype=torch.float32)
                for ids in batch]

    encoder.token_embeddings = token_embeddings
    cap = 6
    left = ChunkedDoc("a", [101, 1, 2, 3, 4, 5, 102], [(1, 4), (4, 6)], ["a", "b"])
    wide = ChunkedDoc("a", [101, 6, 7, 8, 9, 10, 102], [(1, 6)], ["c"])
    right = ChunkedDoc("a", [101, 11, 12, 102], [(1, 3)], ["d"])
    joined = ChunkedDoc("a", [101, *range(1, 13), 102], [(1, 4), (4, 6), (6, 11), (11, 13)], ["a", "b", "c", "d"])
    ids = ("a", "b", "c", "d")
    expected = torch.cat((encode_late(encoder, [left], long_window=cap, long_overlap=1, special_tokens="wrap"),
                          encode_late(encoder, [wide, right])))
    calls.clear()
    study = encode_study_long_late(encoder, [SimpleNamespace(chunked=joined)], token_cap=cap, overlap=1, batch_size=2)
    assert [item for item in calls if len(item) > cap] == [tuple(wide.input_ids)]
    torch.testing.assert_close(study, expected)

    state = PositionalRefresh(encoder, window=cap, overlap=1, batch_size=2, special_tokens="wrap",
                              oversized_singleton=True)
    state.refresh(joined, ids)
    torch.testing.assert_close(torch.stack([state.vectors[key] for key in ids]), expected)
    assert state.refresh(joined, ids).encoder_calls == 0
    changed_ids = joined.input_ids.copy()
    changed_ids[7] = 77
    changed = ChunkedDoc(joined.doc_id, changed_ids, joined.spans, joined.texts)
    stats = state.refresh(changed, ids)
    assert (stats.encoder_calls, stats.refreshed_input_tokens, stats.rewritten_vectors) == (1, 7, 1)
    rebuilt = encode_study_long_late(encoder, [SimpleNamespace(chunked=changed)], token_cap=cap, overlap=1, batch_size=2)
    torch.testing.assert_close(torch.stack([state.vectors[key] for key in ids]), rebuilt)
