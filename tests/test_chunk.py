"""Fixed-window chunker tests. Fully offline (uses tiktoken cl100k_base)."""

from __future__ import annotations

import itertools

import pytest
import tiktoken

from src.pipeline.chunk import (
    DEFAULT_ENCODING,
    PIPELINE_TAG,
    chunk_fixed,
)
from src.pipeline.extract import ExtractedDoc, PageText


def _doc(pages: list[str]) -> ExtractedDoc:
    """Build a synthetic ExtractedDoc — bypasses PyMuPDF."""
    return ExtractedDoc(
        pages=[PageText(page_number=i + 1, text=t) for i, t in enumerate(pages)],
        num_pages=len(pages),
        title="test",
        content_hash="stub",
    )


def _long_page(word: str, n_words: int) -> str:
    return " ".join([word] * n_words)


class TestChunkFixedShape:
    """Chunk sizes, overlap, and final-chunk retention."""

    def test_pipeline_tag_matches_plan(self) -> None:
        # Load-bearing constant: every P1 chunk row in the DB carries this string.
        assert PIPELINE_TAG == "p1_fixed_500_50"

    def test_empty_doc_returns_empty_list(self) -> None:
        assert chunk_fixed(_doc([])) == []

    def test_all_whitespace_returns_empty_list(self) -> None:
        assert chunk_fixed(_doc(["   ", "\n\n"])) == []

    def test_chunk_sizes_bounded(self) -> None:
        # ~3000 tokens of repeated content → several full windows + a tail.
        doc = _doc([_long_page("alpha", 3000)])
        chunks = chunk_fixed(doc)
        assert len(chunks) > 1
        assert all(c.num_tokens <= 500 for c in chunks)
        # Every full chunk except possibly the last is at the target.
        assert all(c.num_tokens == 500 for c in chunks[:-1])

    def test_overlap_is_honored(self) -> None:
        # Consecutive chunks share `overlap_tokens` at the join in token space.
        # We assert this by re-tokenizing and comparing the tail of chunk N
        # against the head of chunk N+1.
        doc = _doc([_long_page("gamma", 2500)])
        chunks = chunk_fixed(doc, target_tokens=500, overlap_tokens=50)
        encoder = tiktoken.get_encoding(DEFAULT_ENCODING)
        for a, b in itertools.pairwise(chunks):
            a_tail = encoder.encode(a.text)[-50:]
            b_head = encoder.encode(b.text)[:50]
            assert a_tail == b_head, (
                f"overlap mismatch between chunk {a.chunk_index} and {b.chunk_index}"
            )

    def test_tail_chunk_kept_even_if_short(self) -> None:
        # Total tokens not a multiple of `step` — the final chunk is shorter
        # than target_tokens and MUST NOT be dropped.
        doc = _doc([_long_page("delta", 601)])  # 601 words → ~601 tokens
        chunks = chunk_fixed(doc, target_tokens=500, overlap_tokens=50)
        assert chunks[-1].num_tokens < 500
        assert chunks[-1].num_tokens > 0

    def test_chunk_indices_are_dense_and_monotonic(self) -> None:
        doc = _doc([_long_page("epsilon", 2000)])
        chunks = chunk_fixed(doc)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_content_hash_uniqueness(self) -> None:
        # Distinct chunk TEXT → distinct sha256. Repeating one word would give
        # periodic windows with identical decoded text (and correctly identical
        # hashes), so vary the content per token instead.
        text = " ".join(f"w{i}" for i in range(3000))
        doc = _doc([text])
        chunks = chunk_fixed(doc)
        hashes = [c.content_hash for c in chunks]
        assert len(hashes) == len(set(hashes)), "duplicate content_hash across chunks"


class TestChunkFixedPageRanges:
    """`page_start` / `page_end` are monotonic and honest across page breaks."""

    def test_single_page_chunks_share_page(self) -> None:
        doc = _doc([_long_page("eta", 300)])
        chunks = chunk_fixed(doc)
        assert all(c.page_start == 1 and c.page_end == 1 for c in chunks)

    def test_multi_page_chunk_reports_range(self) -> None:
        # Two ~400-token pages → any single 500-token chunk straddles them.
        doc = _doc([_long_page("theta", 400), _long_page("iota", 400)])
        chunks = chunk_fixed(doc, target_tokens=500, overlap_tokens=50)
        straddlers = [c for c in chunks if c.page_start != c.page_end]
        assert straddlers, "expected at least one chunk spanning both pages"
        assert all(c.page_start == 1 and c.page_end == 2 for c in straddlers)

    def test_page_start_monotonic(self) -> None:
        doc = _doc([_long_page(f"w{i}", 300) for i in range(6)])
        chunks = chunk_fixed(doc)
        assert chunks[0].page_start == 1
        # page_start advances monotonically as we walk chunks.
        starts = [c.page_start for c in chunks]
        assert starts == sorted(starts)


class TestChunkFixedValidation:
    """Parameter validation — bad settings surface early."""

    def test_overlap_ge_target_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap_tokens"):
            chunk_fixed(_doc(["x"]), target_tokens=500, overlap_tokens=500)

    def test_target_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="target_tokens"):
            chunk_fixed(_doc(["x"]), target_tokens=0)
