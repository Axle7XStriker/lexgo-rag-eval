"""Fixed-window token chunker."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from src.pipeline.extract import ExtractedDoc
from src.pipeline.hashing import PAGE_JOIN, sha256_utf8
from src.pipeline.pipeline_config import ChunkerConfig


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
    """
    starts: list[int] = []
    tokens: list[int] = []
    join_tokens = encoder.encode(PAGE_JOIN) if doc.pages else []
    for idx, page in enumerate(doc.pages):
        starts.append(len(tokens))
        tokens.extend(encoder.encode(page.text))
        if idx < len(doc.pages) - 1:
            tokens.extend(join_tokens)
    return starts, tokens


def _page_of(token_index: int, page_starts: list[int]) -> int:
    """1-indexed page number containing `token_index` in the joined stream."""
    page = 1
    for p, start in enumerate(page_starts, start=1):
        if start <= token_index:
            page = p
        else:
            break
    return page


def chunk_fixed(doc: ExtractedDoc, config: ChunkerConfig) -> list[Chunk]:
    """Split `doc` into fixed-window chunks per `config`.

    Windows advance by `target_tokens - overlap_tokens`. The last chunk is
    kept even if shorter than the target. Empty documents return an empty
    list (caller decides whether that's a fatal error).

    Raises:
      ValueError — `config.algorithm != "fixed"`, or the fixed knobs are
        missing/inconsistent (e.g. `overlap_tokens >= target_tokens`).
    """
    if config.algorithm != "fixed":
        raise ValueError(
            f"chunk_fixed requires ChunkerConfig(algorithm='fixed'), got {config.algorithm!r}"
        )
    if config.target_tokens is None or config.overlap_tokens is None:
        raise ValueError(
            "chunk_fixed requires target_tokens and overlap_tokens to be set "
            f"on the ChunkerConfig; got target_tokens={config.target_tokens}, "
            f"overlap_tokens={config.overlap_tokens}"
        )
    target_tokens = config.target_tokens
    overlap_tokens = config.overlap_tokens
    if overlap_tokens >= target_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be strictly less than "
            f"target_tokens ({target_tokens})"
        )
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be > 0, got {target_tokens}")

    if not any(p.text.strip() for p in doc.pages):
        return []

    encoder = tiktoken.get_encoding(config.encoding)
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
