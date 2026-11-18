"""Split documents where adjacent sentence embeddings diverge.

The chunker emits the same `Chunk` shape and token accounting as the fixed
chunker. Page boundaries use one newline so wrapped sentences remain intact;
content hashing continues to use the canonical two-newline page separator.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

import tiktoken

from src.observability import get_logger
from src.pipeline.chunk import Chunk
from src.pipeline.extract import ExtractedDoc
from src.pipeline.hashing import sha256_utf8
from src.pipeline.pipeline_config import ChunkerConfig

_logger = get_logger("chunk_semantic")


class _Embedder(Protocol):
    """Structural type for the embedder dependency."""

    def embed_documents(
        self,
        texts: list[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]: ...


# Sentence end or a blank line. A lowercase continuation stays intact. "Dr." still splits.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z(\[\d])"
    r"|\n\s*\n"
)


@dataclass(frozen=True)
class _Sentence:
    """One sentence with the 1-indexed page range it spans."""

    text: str
    page_start: int
    page_end: int


_SENTENCE_PAGE_JOIN = "\n"


def _split_sentences(doc: ExtractedDoc) -> list[_Sentence]:
    """Split the whole doc's text into sentences, tagging each with its page range.
    Filters empty and whitespace-only results. Preserves reading order.
    """
    if not doc.pages:
        return []

    stripped = [p.text.rstrip() for p in doc.pages]  # Else a blank line splits each page.

    offsets: list[int] = []
    running = 0
    for i, text in enumerate(stripped):
        if i > 0:
            running += len(_SENTENCE_PAGE_JOIN)
        offsets.append(running)
        running += len(text)
    joined = _SENTENCE_PAGE_JOIN.join(stripped)

    def page_at(offset: int) -> int:
        """1-indexed PDF page number containing character `offset` in `joined`."""
        for i in range(len(offsets) - 1, -1, -1):
            if offsets[i] <= offset:
                return doc.pages[i].page_number
        return doc.pages[0].page_number

    def emit(raw: str, base_offset: int) -> _Sentence | None:
        text = raw.strip()
        if not text:
            return None
        leading_ws = len(raw) - len(raw.lstrip())
        abs_start = base_offset + leading_ws
        abs_end = abs_start + len(text) - 1
        return _Sentence(
            text=text,
            page_start=page_at(abs_start),
            page_end=page_at(abs_end),
        )

    out: list[_Sentence] = []
    cursor = 0
    for match in _SENTENCE_SPLIT_RE.finditer(joined):
        sentence = emit(joined[cursor : match.start()], cursor)
        if sentence is not None:
            out.append(sentence)
        cursor = match.end()
    tail = emit(joined[cursor:], cursor)
    if tail is not None:
        out.append(tail)
    return out


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance = 1 - cosine similarity. Zero-vector safe."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def _percentile(values: list[float], p: float) -> float:
    """Return an observed distance at the nearest-rank percentile."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile p={p} out of range [0, 100]")
    sorted_vals = sorted(values)
    idx = math.ceil(p / 100.0 * len(sorted_vals)) - 1  # Nearest-rank index.
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return sorted_vals[idx]


def _group_by_cuts(sentences: list[_Sentence], cut_after: set[int]) -> list[list[_Sentence]]:
    """Group sentences into runs, cutting after each index in `cut_after`.

    `i in cut_after` means a chunk boundary sits between sentences
    `i` and `i+1`. Returns a list of runs, each a list of sentences.
    Empty input → empty list.
    """
    if not sentences:
        return []
    groups: list[list[_Sentence]] = []
    current: list[_Sentence] = []
    for i, s in enumerate(sentences):
        current.append(s)
        if i in cut_after:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _tokens_of(text: str, encoder: tiktoken.Encoding) -> int:
    """Token count under the pipeline's tokenizer (cl100k_base)."""
    return len(encoder.encode(text))


def _merge_undersized(
    groups: list[list[_Sentence]],
    *,
    distances: list[float],
    encoder: tiktoken.Encoding,
    min_tokens: int,
) -> list[list[_Sentence]]:
    """Merge every group under `min_tokens` into a neighbor."""
    if len(groups) <= 1:
        return groups

    def group_tokens(g: list[_Sentence]) -> int:
        return _tokens_of(" ".join(s.text for s in g), encoder)

    def start_indices(gs: list[list[_Sentence]]) -> list[int]:
        idx: list[int] = []
        running = 0
        for g in gs:
            idx.append(running)
            running += len(g)
        return idx

    working = [list(g) for g in groups]
    # Bounded iteration: each pass either merges (reducing len by 1) or
    # terminates. Loop invariant: `changed=True` iff a merge fired this pass.
    while True:
        if len(working) <= 1:
            break
        starts = start_indices(working)
        changed = False
        for i, g in enumerate(working):
            if group_tokens(g) >= min_tokens:
                continue
            # Pick neighbor with lower boundary distance.
            left_dist = distances[starts[i] - 1] if i > 0 else math.inf
            right_dist = distances[starts[i] + len(g) - 1] if i < len(working) - 1 else math.inf
            if left_dist <= right_dist and i > 0:
                merged = working[i - 1] + g
                working = [*working[: i - 1], merged, *working[i + 1 :]]
            elif i < len(working) - 1:
                merged = g + working[i + 1]
                working = [*working[:i], merged, *working[i + 2 :]]
            else:
                break
            changed = True
            break
        if not changed:
            break
    return working


def _split_oversized(
    groups: list[list[_Sentence]],
    *,
    distances: list[float],
    encoder: tiktoken.Encoding,
    max_tokens: int,
) -> list[list[_Sentence]]:
    """Split every group over `max_tokens` at its strongest internal boundary.

    Strategy:
      - Find the internal sentence boundary with the highest distance
        (best "next cut point" per the semantic signal that percentile
        thresholding didn't already reach) and split there.
      - Recurse on each half.
      - A single sentence longer than `max_tokens` cannot be split
        semantically — fall back to a fixed token-window split.
    """

    def group_tokens(g: list[_Sentence]) -> int:
        return _tokens_of(" ".join(s.text for s in g), encoder)

    def start_index(g: list[_Sentence], all_sentences: list[_Sentence]) -> int:
        # Sentences are frozen dataclasses shared by identity — find the
        # first occurrence by object identity, not equality (two sentences
        # with the same text on different pages must not collide).
        for i, s in enumerate(all_sentences):
            if s is g[0]:
                return i
        raise AssertionError("group[0] not found in flat sentence list")

    all_sentences: list[_Sentence] = [s for g in groups for s in g]

    def split_one(g: list[_Sentence]) -> list[list[_Sentence]]:
        if group_tokens(g) <= max_tokens:
            return [g]
        if len(g) == 1:
            return _fixed_split_single_sentence(  # No semantic cut exists inside one sentence.
                g[0], encoder=encoder, max_tokens=max_tokens
            )
        start = start_index(g, all_sentences)
        internal_dists = distances[start : start + len(g) - 1]
        cut = max(range(len(internal_dists)), key=lambda i: internal_dists[i])
        left = g[: cut + 1]
        right = g[cut + 1 :]
        return split_one(left) + split_one(right)

    out: list[list[_Sentence]] = []
    for g in groups:
        out.extend(split_one(g))
    return out


def _fixed_split_single_sentence(
    sentence: _Sentence,
    *,
    encoder: tiktoken.Encoding,
    max_tokens: int,
) -> list[list[_Sentence]]:
    """Fallback: a single sentence too long for `max_tokens`."""
    tokens = encoder.encode(sentence.text)
    if not tokens:
        return []
    pieces: list[list[_Sentence]] = []
    for start in range(0, len(tokens), max_tokens):
        window = tokens[start : start + max_tokens]
        piece_text = encoder.decode(window)
        pieces.append(
            [
                _Sentence(
                    text=piece_text,
                    page_start=sentence.page_start,
                    page_end=sentence.page_end,
                )
            ]
        )
    return pieces


def _finalize(
    groups: list[list[_Sentence]],
    *,
    encoder: tiktoken.Encoding,
) -> list[Chunk]:
    """Turn sentence groups into `Chunk` records with page ranges + hashes."""
    chunks: list[Chunk] = []
    for idx, group in enumerate(groups):
        if not group:
            continue
        text = " ".join(s.text for s in group)
        num_tokens = _tokens_of(text, encoder)
        chunks.append(
            Chunk(
                text=text,
                num_tokens=num_tokens,
                chunk_index=idx,
                page_start=min(s.page_start for s in group),
                page_end=max(s.page_end for s in group),
                content_hash=sha256_utf8(text),
            )
        )
    return chunks


def chunk_semantic(
    doc: ExtractedDoc,
    *,
    embedder: _Embedder,
    config: ChunkerConfig,
    run_id: str | None = None,
) -> list[Chunk]:
    """Split `doc` into semantically coherent chunks via sentence-embedding distance.

    Recipe (Kamradt et al.):
      1. Split each page into sentences (regex).
      2. Batch-embed every sentence via the same Voyage model used at
         retrieval time.
      3. Compute cosine distance between each consecutive pair.
      4. Cut after any pair whose distance ≥ the Nth percentile
         (`config.percentile_threshold`, default 95) of the doc's
         distance distribution.
      5. Merge chunks under `config.min_tokens` into semantic neighbors.
      6. Split chunks over `config.max_tokens` at the strongest internal
         boundary; single-sentence monsters fall back to fixed-window.
      7. Emit `Chunk` records with 1-indexed page ranges + sha256.

    Raises:
      ValueError — `config.algorithm != "semantic"`, or a required knob is
        missing (`percentile_threshold`, `min_tokens`, `max_tokens`).
    """
    if config.algorithm != "semantic":
        raise ValueError(
            f"chunk_semantic requires ChunkerConfig(algorithm='semantic'), got {config.algorithm!r}"
        )
    if (
        config.percentile_threshold is None
        or config.min_tokens is None
        or config.max_tokens is None
    ):
        raise ValueError(
            "chunk_semantic requires percentile_threshold, min_tokens, and "
            f"max_tokens on the ChunkerConfig; got "
            f"percentile_threshold={config.percentile_threshold}, "
            f"min_tokens={config.min_tokens}, max_tokens={config.max_tokens}"
        )
    if config.min_tokens > config.max_tokens:
        raise ValueError(
            f"min_tokens ({config.min_tokens}) must be ≤ max_tokens ({config.max_tokens})"
        )

    if not any(p.text.strip() for p in doc.pages):
        return []

    sentences = _split_sentences(doc)
    if not sentences:
        return []

    encoder = tiktoken.get_encoding(config.encoding)

    if len(sentences) == 1:
        groups = _split_oversized(
            [sentences],
            distances=[],
            encoder=encoder,
            max_tokens=config.max_tokens,
        )
        return _finalize(groups, encoder=encoder)

    embeddings = embedder.embed_documents([s.text for s in sentences], run_id=run_id)
    if len(embeddings) != len(sentences):
        raise RuntimeError(
            f"embedder returned {len(embeddings)} vectors for {len(sentences)} "
            "sentences; refusing to align mismatched inputs"
        )

    distances = [
        _cosine_distance(embeddings[i], embeddings[i + 1]) for i in range(len(sentences) - 1)
    ]

    threshold = _percentile(distances, config.percentile_threshold)
    cut_after = {i for i, d in enumerate(distances) if d >= threshold}

    _logger.info(
        "semantic_chunk_stats",
        n_sentences=len(sentences),
        n_cuts=len(cut_after),
        percentile_threshold=config.percentile_threshold,
        threshold_distance=round(threshold, 4),
        min_distance=round(min(distances), 4),
        max_distance=round(max(distances), 4),
    )

    groups = _group_by_cuts(sentences, cut_after)
    groups = _merge_undersized(
        groups,
        distances=distances,
        encoder=encoder,
        min_tokens=config.min_tokens,
    )
    groups = _split_oversized(
        groups,
        distances=distances,
        encoder=encoder,
        max_tokens=config.max_tokens,
    )
    return _finalize(groups, encoder=encoder)
