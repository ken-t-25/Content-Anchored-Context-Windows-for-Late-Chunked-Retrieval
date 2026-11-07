"""Fixed-size token chunking that yields BOTH token spans (for late chunking)
and the corresponding character substrings (for independent / naive chunking),
so that both methods index exactly the same retrieval units."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkedDoc:
    doc_id: str
    input_ids: list[int]                # full document tokenization, WITH special tokens ([CLS] ... [SEP])
    spans: list[tuple[int, int]]        # chunk token spans into input_ids, end-exclusive, content tokens only
    texts: list[str]                    # chunk text = text[char_start(first token) : char_end(last token)]


def fixed_token_chunks(doc_id: str, text: str, tokenizer, chunk_size: int = 256) -> ChunkedDoc:
    """Split a document into consecutive spans of `chunk_size` content tokens.

    Spans never include special tokens. Because the document is tokenized with
    special tokens, span indices are already shifted by the leading [CLS]; they
    can be used directly to index the encoder's token outputs.
    """
    enc = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=False,
        verbose=False,
    )
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    special = enc["special_tokens_mask"]
    content = [i for i, s in enumerate(special) if s == 0]
    if content and content != list(range(content[0], content[-1] + 1)):
        raise ValueError(f"Non-contiguous content tokens in doc {doc_id}")

    spans, texts = [], []
    for k in range(0, len(content), chunk_size):
        start = content[k]
        end = content[min(k + chunk_size, len(content)) - 1] + 1
        spans.append((start, end))
        texts.append(text[offsets[start][0] : offsets[end - 1][1]])
    return ChunkedDoc(doc_id, ids, spans, texts)
