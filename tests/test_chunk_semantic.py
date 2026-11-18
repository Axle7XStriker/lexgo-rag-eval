"""Semantic chunker tests. Fully offline — fake embedder returns canned vectors.

Covers:
  - Sentence splitter edge cases (periods, question marks, ellipses,
    paragraph breaks, bullet-like content).
  - Boundary math: canned distances that make the 95th-percentile cut
    land at a KNOWN sentence boundary → assert chunks land where expected.
  - Bound enforcement: undersized-merge, oversized-split, single-oversize
    sentence fallback.
  - Page attribution: single-page, multi-page span, blank middle page.
  - Empty / whitespace / single-sentence early returns (no embed call).
  - Deterministic content hashes.
  - Signature guard: fake embedder matches VoyageEmbedder.embed_documents.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import pytest

from src.pipeline.chunk_semantic import (
    _cosine_distance,
    _percentile,
    _split_sentences,
    chunk_semantic,
)
from src.pipeline.embed import VoyageEmbedder
from src.pipeline.extract import ExtractedDoc, PageText
from src.pipeline.pipeline_config import ChunkerConfig


def _doc(pages: list[str]) -> ExtractedDoc:
    """Build a synthetic ExtractedDoc — bypasses PyMuPDF."""
    return ExtractedDoc(
        pages=[PageText(page_number=i + 1, text=t) for i, t in enumerate(pages)],
        num_pages=len(pages),
        title="test",
        content_hash="stub",
    )


def _cfg(
    *,
    percentile_threshold: float = 95.0,
    min_tokens: int = 10,
    max_tokens: int = 200,
    target_tokens: int = 100,
) -> ChunkerConfig:
    """Semantic-chunker config with small token bounds tuned for synthetic docs."""
    return ChunkerConfig(
        algorithm="semantic",
        target_tokens=target_tokens,
        percentile_threshold=percentile_threshold,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )


@dataclass
class _FakeEmbedder:
    """Duck-types VoyageEmbedder — returns canned vectors keyed by input text."""

    # maps a sentence-prefix substring → vector
    vectors_by_prefix: dict[str, list[float]] = field(default_factory=dict)
    # used for sentences with no matching prefix
    default_vector: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    calls: list[dict] = field(default_factory=list)

    def embed_documents(
        self,
        texts: list[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]:
        self.calls.append({"n": len(texts), "run_id": run_id})
        out: list[list[float]] = []
        for text in texts:
            hit = None
            for prefix, vec in self.vectors_by_prefix.items():
                if text.startswith(prefix):
                    hit = vec
                    break
            out.append(hit if hit is not None else list(self.default_vector))
        return out


class TestSplitSentences:
    def test_period_question_exclamation(self) -> None:
        doc = _doc(["First. Second? Third! Fourth."])
        sents = _split_sentences(doc)
        assert [s.text for s in sents] == ["First.", "Second?", "Third!", "Fourth."]

    def test_paragraph_break_splits(self) -> None:
        doc = _doc(["Alpha paragraph without period\n\nBeta paragraph"])
        sents = _split_sentences(doc)
        assert [s.text for s in sents] == [
            "Alpha paragraph without period",
            "Beta paragraph",
        ]

    def test_ellipsis_not_split_mid_sentence(self) -> None:
        doc = _doc(["I was thinking... maybe we shouldn't. Second sentence."])
        sents = _split_sentences(doc)
        assert [s.text for s in sents] == [
            "I was thinking... maybe we shouldn't.",
            "Second sentence.",
        ]

    def test_whitespace_only_page_yields_no_sentences(self) -> None:
        doc = _doc(["   \n\n  "])
        assert _split_sentences(doc) == []

    def test_page_tag_preserved(self) -> None:
        doc = _doc(["First. Second.", "Third. Fourth."])
        sents = _split_sentences(doc)
        assert [(s.text, s.page_start, s.page_end) for s in sents] == [
            ("First.", 1, 1),
            ("Second.", 1, 1),
            ("Third.", 2, 2),
            ("Fourth.", 2, 2),
        ]

    def test_blank_middle_page_preserves_page_numbers(self) -> None:
        doc = _doc(["Alpha.", "", "Gamma."])
        sents = _split_sentences(doc)
        assert [(s.text, s.page_start, s.page_end) for s in sents] == [
            ("Alpha.", 1, 1),
            ("Gamma.", 3, 3),
        ]

    def test_sentence_wrapping_across_pages_is_not_fragmented(self) -> None:
        doc = _doc(
            [
                "Prior sentence. The quicksort algorithm has average-",
                "case complexity O(n log n). Following sentence.",
            ]
        )
        sents = _split_sentences(doc)
        assert len(sents) == 3
        assert sents[0].text == "Prior sentence."
        assert sents[2].text == "Following sentence."
        assert "average-" in sents[1].text
        assert "case complexity O(n log n)." in sents[1].text
        assert [(s.page_start, s.page_end) for s in sents] == [(1, 1), (1, 2), (2, 2)]

    def test_page_boundary_between_sentences_does_not_over_split(self) -> None:
        # A trailing blank line at the bottom of page 1 plus a leading blank
        # line at the top of page 2 must not reconstruct `\n\s*\n` and trigger
        # a spurious paragraph-break split at every page boundary — page-end
        # rstrip in `_split_sentences` prevents that.
        doc = _doc(["First. Second.\n\n", "\n\nThird. Fourth."])
        sents = _split_sentences(doc)
        assert [s.text for s in sents] == ["First.", "Second.", "Third.", "Fourth."]


class TestPercentile:
    def test_nearest_rank_at_boundary(self) -> None:
        assert _percentile(list(range(10)), 95.0) == 9

    def test_median(self) -> None:
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50.0) == 3.0

    def test_zero_percentile_returns_min(self) -> None:
        assert _percentile([5.0, 2.0, 9.0, 1.0], 0.0) == 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _percentile([], 50.0)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _percentile([1.0], 150.0)


class TestCosineDistance:
    def test_identical_vectors_zero(self) -> None:
        assert _cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_vectors_one(self) -> None:
        assert _cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_opposite_vectors_two(self) -> None:
        assert _cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)

    def test_zero_vector_returns_max_distance(self) -> None:
        assert _cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0


class TestChunkerHappyPath:
    def test_cuts_land_at_expected_boundary(self) -> None:
        pages = [
            "Alpha sentence about dogs. Beta about dogs too. Gamma dogs again. "
            "Delta about photosynthesis. Epsilon photosynthesis too."
        ]
        embedder = _FakeEmbedder(
            vectors_by_prefix={
                "Alpha": [1.0, 0.0],
                "Beta": [1.0, 0.0],
                "Gamma": [1.0, 0.0],
                "Delta": [0.0, 1.0],
                "Epsilon": [0.0, 1.0],
            },
        )
        chunks = chunk_semantic(
            _doc(pages),
            embedder=embedder,
            config=_cfg(min_tokens=1, max_tokens=999),
        )
        assert [chunk.text for chunk in chunks] == [
            "Alpha sentence about dogs. Beta about dogs too. Gamma dogs again.",
            "Delta about photosynthesis. Epsilon photosynthesis too.",
        ]

    def test_embedder_called_once_with_sentence_count_and_run_id(self) -> None:
        embedder = _FakeEmbedder()
        chunk_semantic(
            _doc(["Only one. And another."]),
            embedder=embedder,
            config=_cfg(min_tokens=1),
            run_id="run_abc",
        )
        assert embedder.calls == [{"n": 2, "run_id": "run_abc"}]

    def test_content_hashes_deterministic(self) -> None:
        embedder = _FakeEmbedder()
        pages = ["First. Second. Third. Fourth."]
        a = chunk_semantic(_doc(pages), embedder=embedder, config=_cfg(min_tokens=1))
        b = chunk_semantic(_doc(pages), embedder=embedder, config=_cfg(min_tokens=1))
        assert [c.content_hash for c in a] == [c.content_hash for c in b]

    def test_chunk_indices_are_dense_and_monotonic(self) -> None:
        embedder = _FakeEmbedder()
        chunks = chunk_semantic(
            _doc(["A. B. C. D. E. F."]),
            embedder=embedder,
            config=_cfg(min_tokens=1),
        )
        assert [c.chunk_index for c in chunks] == list(range(6))


class TestEarlyReturns:
    def test_empty_doc_returns_empty(self) -> None:
        embedder = _FakeEmbedder()
        assert chunk_semantic(_doc([]), embedder=embedder, config=_cfg()) == []
        assert embedder.calls == []

    def test_whitespace_only_returns_empty(self) -> None:
        embedder = _FakeEmbedder()
        assert chunk_semantic(_doc(["   ", "\n"]), embedder=embedder, config=_cfg()) == []
        assert embedder.calls == []

    def test_single_sentence_no_embed_call(self) -> None:
        embedder = _FakeEmbedder()
        chunks = chunk_semantic(
            _doc(["Just one sentence."]),
            embedder=embedder,
            config=_cfg(),
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Just one sentence."
        assert embedder.calls == []


class TestPageRanges:
    def test_single_page_chunks_share_page(self) -> None:
        embedder = _FakeEmbedder()
        chunks = chunk_semantic(
            _doc(["First. Second. Third."]),
            embedder=embedder,
            config=_cfg(min_tokens=1),
        )
        assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1)] * 3

    def test_multi_page_chunk_reports_page_range(self) -> None:
        embedder = _FakeEmbedder()
        chunks = chunk_semantic(
            _doc(["Short.", "Also short.", "Third short."]),
            embedder=embedder,
            config=_cfg(min_tokens=100, percentile_threshold=95.0),
        )
        assert len(chunks) == 1
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 3

    def test_blank_middle_page_preserves_range(self) -> None:
        embedder = _FakeEmbedder()
        chunks = chunk_semantic(
            _doc(["Alpha content here.", "", "Gamma content here."]),
            embedder=embedder,
            config=_cfg(min_tokens=100, percentile_threshold=95.0),
        )
        assert len(chunks) == 1
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 3


class TestBoundEnforcement:
    def test_undersized_groups_merge(self) -> None:
        embedder = _FakeEmbedder(
            vectors_by_prefix={
                "Alpha": [1.0, 0.0],
                "Beta": [1.0, 0.0],
                "Gamma": [0.2, 0.9797959],
                "Delta": [-0.9797959, 0.2],
                "Epsilon": [-0.9797959, 0.2],
            },
        )
        chunks = chunk_semantic(
            _doc(["Alpha one. Beta two. Gamma three. Delta four. Epsilon five."]),
            embedder=embedder,
            config=_cfg(min_tokens=5, max_tokens=999, percentile_threshold=75.0),
        )
        assert [chunk.text for chunk in chunks] == [
            "Alpha one. Beta two. Gamma three.",
            "Delta four. Epsilon five.",
        ]

    def test_oversized_group_splits(self) -> None:
        embedder = _FakeEmbedder(
            vectors_by_prefix={
                "Alpha": [0.0, -1.0],
                "Beta": [1.0, 0.0],
                "Gamma": [1.0, 0.0],
                "Delta": [1.0, 0.0],
                "Epsilon": [0.5, 0.8660254],
                "Zeta": [0.5, 0.8660254],
            },
        )
        chunks = chunk_semantic(
            _doc(["Alpha one. Beta two. Gamma three. Delta four. Epsilon five. Zeta six."]),
            embedder=embedder,
            config=_cfg(min_tokens=1, max_tokens=9, percentile_threshold=100.0),
        )
        assert [chunk.text for chunk in chunks] == [
            "Alpha one.",
            "Beta two. Gamma three. Delta four.",
            "Epsilon five. Zeta six.",
        ]
        assert all(chunk.num_tokens <= 9 for chunk in chunks)

    def test_single_oversized_sentence_falls_back_to_fixed_split(self) -> None:
        embedder = _FakeEmbedder()
        long_sentence = "word " * 200 + "end."
        chunks = chunk_semantic(
            _doc([long_sentence]),
            embedder=embedder,
            config=_cfg(min_tokens=1, max_tokens=50, percentile_threshold=95.0),
        )
        assert len(chunks) > 1
        assert chunks[-1].num_tokens <= 50
        assert all(c.num_tokens == 50 for c in chunks[:-1])
        assert "".join(c.text for c in chunks) == long_sentence
        assert embedder.calls == []
        assert all(c.page_start == 1 and c.page_end == 1 for c in chunks)


class TestConfigValidation:
    def test_wrong_algorithm_raises(self) -> None:
        embedder = _FakeEmbedder()
        cfg = ChunkerConfig(algorithm="fixed", target_tokens=500, overlap_tokens=50)
        with pytest.raises(ValueError, match=r"algorithm='semantic'"):
            chunk_semantic(_doc(["A."]), embedder=embedder, config=cfg)

    @pytest.mark.parametrize(
        ("cfg", "missing"),
        [
            (
                ChunkerConfig(algorithm="semantic", min_tokens=10, max_tokens=500),
                "percentile_threshold",
            ),
            (
                ChunkerConfig(
                    algorithm="semantic",
                    percentile_threshold=95.0,
                    max_tokens=500,
                ),
                "min_tokens",
            ),
            (
                ChunkerConfig(
                    algorithm="semantic",
                    percentile_threshold=95.0,
                    min_tokens=10,
                ),
                "max_tokens",
            ),
        ],
    )
    def test_missing_knobs_raises(self, cfg: ChunkerConfig, missing: str) -> None:
        embedder = _FakeEmbedder()
        with pytest.raises(ValueError, match=rf"{missing}=None"):
            chunk_semantic(_doc(["A."]), embedder=embedder, config=cfg)

    def test_min_gt_max_raises(self) -> None:
        embedder = _FakeEmbedder()
        cfg = ChunkerConfig(
            algorithm="semantic",
            percentile_threshold=95.0,
            min_tokens=500,
            max_tokens=100,
        )
        with pytest.raises(ValueError, match=r"min_tokens.*max_tokens"):
            chunk_semantic(_doc(["A."]), embedder=embedder, config=cfg)


def test_fake_embedder_signature_matches_real() -> None:
    real = inspect.signature(VoyageEmbedder.embed_documents)
    fake = inspect.signature(_FakeEmbedder.embed_documents)
    assert fake == real
