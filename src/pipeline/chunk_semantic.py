"""Embedding-based semantic chunker (a.k.a. "semantic splitting").

The technique — split a document into sentences, embed each, and cut where
consecutive-sentence embedding distance jumps past a per-document
percentile threshold — was popularized by **Greg Kamradt** in
["The 5 Levels of Text Splitting for Retrieval"][kamradt] (2024). It's a
community-adopted heuristic, NOT a peer-reviewed paper. Reference
implementations that follow the same recipe:

  - LangChain — `SemanticChunker` in `langchain_experimental.text_splitter`
    ([source](https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/text_splitter.py))
  - LlamaIndex — `SemanticSplitterNodeParser`
    ([docs](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/semantic_splitter/))

We reimplement rather than depending on either library because both add a
large dependency tree (LangChain in particular), our corpus is small
enough that our own sentence splitter suffices, and having the algorithm
inline keeps every knob visible and versioned in `pipeline_config.py` —
the eval numbers must be fully our code to be defensible.

"Semantic chunking" is distinct from adjacent published techniques:
  - **Late chunking** (Günther et al., 2024, [arXiv:2409.04701]) — embeds
    the full doc first, then pools per chunk. Different mechanism.
  - **Cross-segment classifier** approaches (Lukasik et al., 2020,
    [ACL 2020]) — train a supervised model to predict segment boundaries.
    Different mechanism.

[kamradt]: https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
[arXiv:2409.04701]: https://arxiv.org/abs/2409.04701
[ACL 2020]: https://aclanthology.org/2020.acl-main.380/

Design notes worth remembering:
  - Sentence splitter is regex-based, no external NLP dep. Splits on
    `.!?` followed by whitespace + capital/digit/bracket, and on blank
    lines. Handles OCW course notes adequately; edge cases (formulas,
    footnote glue) surface as short "sentences" that the min-token
    merge step absorbs.
  - Emits the same `Chunk` dataclass P1 emits — DB rows are shape-
    compatible, only the `pipeline` tag differs.
  - Uses the same `cl100k_base` tokenizer as `chunk_fixed` for `num_tokens`
    accounting, so P1 vs P2 token counts are directly comparable.
  - Cost: one `embed_documents` batch per ~64 sentences at chunk time
    (Voyage-3-large ~ $0.13/1M input tokens). For our corpus (~10 docs,
    thousands of sentences) that's a one-time ~$1-3 spend, logged in
    `logs/llm_calls.jsonl` under the ingest `run_id`.
  - Empty/whitespace-only doc → `[]` (matches `chunk_fixed`).
  - Single-sentence doc → one chunk, no embed call needed (short-circuit).
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
    """Structural type for the embedder dependency.

    Only `embed_documents` is called; matches VoyageEmbedder's signature
    exactly. Declared as a Protocol so tests can inject fakes without
    subclassing.
    """

    def embed_documents(
        self,
        texts: list[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]: ...


# Split on `.!?` followed by whitespace and a capital / digit / opening
# bracket — or on a blank line (paragraph break). The blank-line branch is
# the paragraph-boundary signal in OCW notes where sentences don't always
# end with `.`. False positives from abbreviations ("Dr.", "e.g.") produce
# short pseudo-sentences that the min-token merge step reabsorbs; false
# negatives just leave two sentences fused, absorbed by the max-token
# split step. Neither corrupts the output.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z(\[\d])"
    r"|\n\s*\n"
)


@dataclass(frozen=True)
class _Sentence:
    """One sentence with its 1-indexed originating page."""

    text: str
    page: int


def _split_sentences(doc: ExtractedDoc) -> list[_Sentence]:
    """Split every page into sentences, tagging each with its page number.

    Filters empty and whitespace-only results. Preserves reading order:
    all page-1 sentences come first, then page-2, etc.
    """
    out: list[_Sentence] = []
    for page in doc.pages:
        for raw in _SENTENCE_SPLIT_RE.split(page.text):
            sentence = raw.strip()
            if sentence:
                out.append(_Sentence(text=sentence, page=page.page_number))
    return out


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance = 1 - cosine similarity. Zero-vector safe.

    Pure Python — no numpy dep, and the arithmetic is fine at our scale
    (a few thousand ~1024-dim vectors per doc, one-time per ingest).
    """
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        # Degenerate embedding (all zeros) — treat as maximum distance
        # rather than dividing by zero. Shouldn't happen with Voyage but
        # cheap to be defensive.
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. `p` in [0, 100]. `values` must be non-empty.

    Chosen over linear-interpolation percentiles (numpy.percentile,
    statistics.quantiles) because it always returns an actual distance
    from the input distribution — cleaner threshold semantics for the
    log messages when a run's cut behavior is being debugged.
    """
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile p={p} out of range [0, 100]")
    sorted_vals = sorted(values)
    # Nearest-rank formula: index = ceil(p/100 * n) - 1, clamped to [0, n-1].
    idx = math.ceil(p / 100.0 * len(sorted_vals)) - 1
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return sorted_vals[idx]


def _group_by_cuts(sentences: list[_Sentence], cut_after: set[int]) -> list[list[_Sentence]]:
    """Group sentences into runs, cutting after each index in `cut_after`.

    `cut_after[i] = True` means a chunk boundary sits between sentences
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
    """Merge every group under `min_tokens` into a neighbor.

    Merge direction: prefer the side with the LOWER cross-boundary distance
    (semantically closer). Ties and edge cases (first/last group) fall back
    to the sole available neighbor. Iterates until every group meets the
    threshold or only one group remains.

    `distances[i]` is the semantic distance between the original sentence i
    and i+1 — used to pick the closer neighbor. Distances aren't re-derived
    after a merge because the merged group's cross-boundary distances (to
    its new left/right neighbors) are already in the same `distances` list;
    only intra-group distances become stale, which we don't reference here.
    """
    if len(groups) <= 1:
        return groups

    def group_tokens(g: list[_Sentence]) -> int:
        return _tokens_of(" ".join(s.text for s in g), encoder)

    # Track each group's starting sentence index in the original list so
    # we can look up cross-boundary distances. Distance BEFORE group i is
    # `distances[start_index_of_group_i - 1]`; AFTER is
    # `distances[start_index_of_group_i + len(group_i) - 1]`.
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
                # Sole undersized group and no neighbors — nothing to merge
                # with. Keep the tail (better a short chunk than a lost
                # one) and stop.
                break
            changed = True
            break  # restart the outer pass so `starts`/indices rebuild
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

    def start_index(g: list[_Sentence], all_flat: list[_Sentence]) -> int:
        # Sentences are frozen dataclasses shared by identity — find the
        # first occurrence by object identity, not equality (two sentences
        # with the same text on different pages must not collide).
        for i, s in enumerate(all_flat):
            if s is g[0]:
                return i
        raise AssertionError("group[0] not found in flat sentence list")

    all_flat: list[_Sentence] = [s for g in groups for s in g]

    def split_one(g: list[_Sentence]) -> list[list[_Sentence]]:
        if group_tokens(g) <= max_tokens:
            return [g]
        if len(g) == 1:
            # A single monster sentence: no semantic boundary exists.
            # Fall back to a fixed token-window split, preserving the page
            # tag on every fragment.
            return _fixed_split_single_sentence(g[0], encoder=encoder, max_tokens=max_tokens)
        # Find the internal boundary with the highest distance.
        start = start_index(g, all_flat)
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
    """Fallback: a single sentence too long for `max_tokens`.

    Tokenizes and splits into contiguous `max_tokens`-sized windows,
    reconstituting each as a `_Sentence` on the SAME page (the source
    sentence was on one page by construction). No overlap.
    """
    tokens = encoder.encode(sentence.text)
    if not tokens:
        return []
    pieces: list[list[_Sentence]] = []
    for start in range(0, len(tokens), max_tokens):
        window = tokens[start : start + max_tokens]
        piece_text = encoder.decode(window)
        pieces.append([_Sentence(text=piece_text, page=sentence.page)])
    return pieces


def _finalize(
    groups: list[list[_Sentence]],
    *,
    encoder: tiktoken.Encoding,
) -> list[Chunk]:
    """Turn sentence groups into `Chunk` records with page ranges + hashes.

    Joins group sentences with a single space (matches the sentence-boundary
    convention we split on). Emits `num_tokens` from cl100k_base and
    `content_hash` = sha256 of the joined text so dedup + idempotency
    stay identical to `chunk_fixed`.
    """
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
                page_start=min(s.page for s in group),
                page_end=max(s.page for s in group),
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

    # Whitespace-only inputs would produce spurious empty chunks; match
    # `chunk_fixed`'s early return.
    if not any(p.text.strip() for p in doc.pages):
        return []

    sentences = _split_sentences(doc)
    if not sentences:
        return []

    encoder = tiktoken.get_encoding(config.encoding)

    # Single-sentence doc: no distances to compute, no embedding call
    # needed. Emit one chunk (respecting max_tokens if the sentence is huge).
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
