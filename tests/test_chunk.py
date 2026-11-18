"""Fixed-window chunker tests. Fully offline (uses tiktoken cl100k_base).

Covers:
    - Chunk sizes, overlap, and final-chunk retention
    - Page attribution: single-page, multi-page span, blank middle page
    - Parameter validation
"""

from __future__ import annotations

import itertools

import pytest
import tiktoken

from src.pipeline.chunk import chunk_fixed
from src.pipeline.extract import ExtractedDoc, PageText
from src.pipeline.pipeline_config import ChunkerConfig


def _cfg(
    target_tokens: int = 500,
    overlap_tokens: int = 50,
) -> ChunkerConfig:
    """Fixed-chunker config for unit tests."""
    return ChunkerConfig(
        algorithm="fixed",
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )


def _doc(pages: list[str]) -> ExtractedDoc:
    return ExtractedDoc(
        pages=[PageText(page_number=i + 1, text=t) for i, t in enumerate(pages)],
        num_pages=len(pages),
        title="test",
        content_hash="stub",
    )


def _page_of_repeated_word(word: str, n_words: int) -> str:
    return " ".join([word] * n_words)


class TestChunkFixedShape:
    def test_empty_doc_returns_empty_list(self) -> None:
        assert chunk_fixed(_doc([]), _cfg()) == []

    def test_all_whitespace_returns_empty_list(self) -> None:
        assert chunk_fixed(_doc(["   ", "\n\n"]), _cfg()) == []

    def test_chunk_sizes_bounded(self) -> None:
        cfg = _cfg(400, 40)
        doc = _doc([_page_of_repeated_word("alpha", 3000)])
        chunks = chunk_fixed(doc, cfg)
        assert len(chunks) > 1
        assert all(c.num_tokens == 400 for c in chunks[:-1])
        assert chunks[-1].num_tokens < 400

    def test_overlap_is_honored(self) -> None:
        cfg = _cfg(500, 50)
        doc = _doc([_page_of_repeated_word("gamma", 2500)])
        chunks = chunk_fixed(doc, cfg)
        encoder = tiktoken.get_encoding(cfg.encoding)
        for a, b in itertools.pairwise(chunks):
            assert encoder.encode(a.text)[-50:] == encoder.encode(b.text)[:50], (
                f"overlap mismatch between chunk {a.chunk_index} and {b.chunk_index}"
            )

    def test_tail_chunk_kept_even_if_short(self) -> None:
        doc = _doc([_page_of_repeated_word("delta", 601)])
        chunks = chunk_fixed(doc, _cfg(500, 50))
        assert chunks[-1].num_tokens < 500
        assert chunks[-1].num_tokens > 0

    def test_chunk_indices_are_dense_and_monotonic(self) -> None:
        doc = _doc([_page_of_repeated_word("epsilon", 2000)])
        chunks = chunk_fixed(doc, _cfg())
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_content_hash_uniqueness(self) -> None:
        text = " ".join(f"w{i}" for i in range(3000))
        doc = _doc([text])
        chunks = chunk_fixed(doc, _cfg())
        hashes = [c.content_hash for c in chunks]
        assert len(hashes) == len(set(hashes)), "duplicate content_hash across chunks"


class TestChunkFixedPageRanges:
    def test_single_page_chunks_share_page(self) -> None:
        doc = _doc([_page_of_repeated_word("eta", 300)])
        chunks = chunk_fixed(doc, _cfg())
        assert all(c.page_start == 1 and c.page_end == 1 for c in chunks)

    def test_multi_page_chunk_reports_range(self) -> None:
        doc = _doc([_page_of_repeated_word("theta", 400), _page_of_repeated_word("iota", 400)])
        chunks = chunk_fixed(doc, _cfg(500, 50))
        straddlers = [c for c in chunks if c.page_start != c.page_end]
        assert straddlers, "expected at least one chunk spanning multiple pages"
        for c in straddlers:
            assert c.page_start < c.page_end

    def test_page_start_monotonic(self) -> None:
        doc = _doc([_page_of_repeated_word(f"w{i}", 300) for i in range(6)])
        chunks = chunk_fixed(doc, _cfg())
        assert chunks[0].page_start == 1
        starts = [c.page_start for c in chunks]
        assert starts == sorted(starts)

    def test_middle_blank_page_preserves_page_numbering(self) -> None:
        doc = _doc([_page_of_repeated_word("alpha", 300), "", _page_of_repeated_word("gamma", 300)])
        chunks = chunk_fixed(doc, _cfg())
        assert chunks, "expected at least one chunk"
        assert chunks[0].page_start == 1
        assert chunks[-1].page_end == 3


class TestChunkFixedValidation:
    def test_overlap_ge_target_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap_tokens"):
            chunk_fixed(_doc(["x"]), _cfg(500, 500))

    def test_target_zero_raises(self) -> None:
        # target=0 with overlap=0 would trip the earlier overlap-ge-target
        # guard (0 >= 0). Use a strictly-smaller negative overlap so control
        # flow reaches the `target_tokens <= 0` branch this test names, and
        # bind `match` to that branch's unique wording rather than the loose
        # substring "target_tokens" (which also appears in the overlap-check
        # error).
        with pytest.raises(ValueError, match=r"target_tokens must be > 0"):
            chunk_fixed(_doc(["x"]), _cfg(0, -1))

    def test_wrong_algorithm_raises(self) -> None:
        semantic_cfg = ChunkerConfig(
            algorithm="semantic",
            percentile_threshold=95.0,
            min_tokens=200,
            max_tokens=750,
        )
        with pytest.raises(ValueError, match="algorithm='fixed'"):
            chunk_fixed(_doc(["x"]), semantic_cfg)

    def test_missing_fixed_knobs_raises(self) -> None:
        cfg = ChunkerConfig(algorithm="fixed")
        with pytest.raises(ValueError, match="target_tokens and overlap_tokens"):
            chunk_fixed(_doc(["x"]), cfg)
