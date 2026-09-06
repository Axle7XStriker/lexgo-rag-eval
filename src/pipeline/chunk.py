"""Fixed-window token chunker for P1 baseline.

500-token target with 50-token overlap over the whole document, tokenized
via `tiktoken cl100k_base`. Chunks carry their originating `page_start` /
`page_end` so citations can point at a page range.

Design notes worth remembering:
  - Voyage does not publish a public tokenizer. cl100k_base (OpenAI's) is
    the de facto lingua franca — fast, offline, and stable. The 500-token
    target is *approximate* vs. Voyage's internal count.
  - Pages join with `hashing.PAGE_JOIN` into one token stream. The
    separator's tokens are attributed to whichever page's boundary they
    straddle — cheap and honest enough for citations.
  - `PIPELINE_TAG` is the exact string written to `chunks.pipeline` so all
    P1 rows are queryable with a single WHERE clause. Do not typo it.
  - The final chunk is kept even if shorter than `target_tokens` — dropping
    the tail would silently lose the last few paragraphs of every doc.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from src.pipeline.extract import ExtractedDoc
from src.pipeline.hashing import PAGE_JOIN, sha256_utf8

PIPELINE_TAG = "p1_fixed_500_50"
DEFAULT_TARGET_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50
DEFAULT_ENCODING = "cl100k_base"


@dataclass(frozen=True)
class Chunk:
    """One fixed-window chunk destined for `chunks` (pre-embedding form)."""

    text: str
    num_tokens: int
    chunk_index: int
    page_start: int
    page_end: int
    content_hash: str


def _encode_pages(
    doc: ExtractedDoc,
    encoder: tiktoken.Encoding,
) -> tuple[list[int], list[int]]:
    """Tokenize each page into one flat token stream, recording per-page start indices.

    Returns `(page_starts, tokens)`:
      - `page_starts[i]` = the index in `tokens` where page `i+1` begins.
      - `tokens` = concatenated tokens for all pages, with `PAGE_JOIN` tokens
        inserted between pages so a chunk that straddles pages reports both.

    `_page_of(token_index, page_starts)` maps an absolute token index in
    `tokens` back to a 1-indexed page number.
    """
    starts: list[int] = []
    tokens: list[int] = []
    join_tokens = encoder.encode(PAGE_JOIN) if doc.pages else []
    for idx, page in enumerate(doc.pages):
        starts.append(len(tokens))
        tokens.extend(encoder.encode(page.text))
        if idx < len(doc.pages) - 1:
            # Attribute the join tokens to the current page's tail; the next
            # page's `starts[i+1]` will point at the first token of its own
            # text, so a chunk that straddles the boundary reports both pages.
            tokens.extend(join_tokens)
    return starts, tokens


def _page_of(token_index: int, page_starts: list[int]) -> int:
    """1-indexed page number containing `token_index` in the joined stream.

    Linear scan (not bisect) — page counts per doc are small (≤ hundreds)
    and this keeps the code obvious.
    """
    page = 1
    for p, start in enumerate(page_starts, start=1):
        if start <= token_index:
            page = p
        else:
            break
    return page


def chunk_fixed(
    doc: ExtractedDoc,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[Chunk]:
    """Split `doc` into fixed-window chunks with per-chunk page ranges.

    Windows advance by `target_tokens - overlap_tokens`. The last chunk is
    kept even if shorter than the target. Empty documents return an empty
    list (caller decides whether that's a fatal error).

    Raises:
      ValueError — `overlap_tokens >= target_tokens` (would loop forever).
    """
    if overlap_tokens >= target_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be strictly less than "
            f"target_tokens ({target_tokens})"
        )
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be > 0, got {target_tokens}")

    # Whitespace-only inputs would tokenize to whitespace tokens and produce
    # a spurious first chunk of literal whitespace. Guard here so callers get
    # an unambiguous empty list rather than an ingested whitespace chunk.
    if not any(p.text.strip() for p in doc.pages):
        return []

    encoder = tiktoken.get_encoding(encoding_name)
    page_starts, tokens = _encode_pages(doc, encoder)
    total = len(tokens)
    if total == 0:
        return []

    step = target_tokens - overlap_tokens
    chunks: list[Chunk] = []
    chunk_index = 0
    start = 0
    while start < total:
        end = min(start + target_tokens, total)
        window = tokens[start:end]
        text = encoder.decode(window)
        page_start = _page_of(start, page_starts)
        page_end = _page_of(end - 1, page_starts)
        chunks.append(
            Chunk(
                text=text,
                num_tokens=len(window),
                chunk_index=chunk_index,
                page_start=page_start,
                page_end=page_end,
                content_hash=sha256_utf8(text),
            )
        )
        chunk_index += 1
        if end >= total:
            break
        start += step
    return chunks
