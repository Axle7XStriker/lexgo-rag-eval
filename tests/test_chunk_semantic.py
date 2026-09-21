"""Semantic chunker tests. Fully offline — fake embedder returns canned vectors.

Mirrors `tests/test_chunk.py`'s style: synthetic `ExtractedDoc`s and a
`_FakeEmbedder` injected in place of Voyage. Covers:

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

# ── Fixtures ──────────────────────────────────────────────────────────


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
    """Duck-types VoyageEmbedder — returns canned vectors keyed by input text.

    `vectors_by_prefix` maps a sentence-prefix substring → vector. The
    first prefix that matches a sentence wins. Sentences with no matching
    prefix get `default_vector`.
    """

    vectors_by_prefix: dict[str, list[float]] = field(default_factory=dict)
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


# ── Sentence splitter ────────────────────────────────────────────────


class TestSplitSentences:
    def test_period_question_exclamation(self) -> None:
        doc = _doc(["First. Second? Third! Fourth."])
        sents = _split_sentences(doc)
        assert [s.text for s in sents] == ["First.", "Second?", "Third!", "Fourth."]

    def test_paragraph_break_splits(self) -> None:
        # A blank line is a hard split even without terminal punctuation.
        doc = _doc(["Alpha paragraph without period\n\nBeta paragraph"])
        sents = _split_sentences(doc)
        assert len(sents) == 2
        assert sents[0].text.startswith("Alpha")
        assert sents[1].text.startswith("Beta")

    def test_ellipsis_not_split_mid_sentence(self) -> None:
        # `...` inside a sentence followed by lowercase should not split.
        # (Regex requires capital/digit/bracket after the whitespace.)
        doc = _doc(["I was thinking... maybe we shouldn't. Second sentence."])
        sents = _split_sentences(doc)
        assert len(sents) == 2
        assert "thinking" in sents[0].text
        assert sents[1].text == "Second sentence."

    def test_whitespace_only_page_yields_no_sentences(self) -> None:
        doc = _doc(["   \n\n  "])
        assert _split_sentences(doc) == []

    def test_page_tag_preserved(self) -> None:
        doc = _doc(["First. Second.", "Third. Fourth."])
        sents = _split_sentences(doc)
        assert [(s.text, s.page) for s in sents] == [
            ("First.", 1),
            ("Second.", 1),
            ("Third.", 2),
            ("Fourth.", 2),
        ]

    def test_blank_middle_page_preserves_page_numbers(self) -> None:
        # Mirrors test_extract's blank-page invariant: sentences on page 3
        # must still report page 3 even if page 2 is empty.
        doc = _doc(["Alpha.", "", "Gamma."])
        sents = _split_sentences(doc)
        assert [(s.text, s.page) for s in sents] == [("Alpha.", 1), ("Gamma.", 3)]


# ── Percentile helper ────────────────────────────────────────────────


class TestPercentile:
    def test_nearest_rank_at_boundary(self) -> None:
        # 10 values → 95th percentile = value at index ceil(0.95*10)-1 = 9 (max).
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


# ── Cosine distance ──────────────────────────────────────────────────


class TestCosineDistance:
    def test_identical_vectors_zero(self) -> None:
        assert _cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_vectors_one(self) -> None:
        assert _cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_opposite_vectors_two(self) -> None:
        assert _cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)

    def test_zero_vector_returns_max_distance(self) -> None:
        # Guard against div-by-zero on a degenerate embedding.
        assert _cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0


# ── Chunker: happy path + boundary placement ─────────────────────────


class TestChunkerHappyPath:
    def test_cuts_land_at_expected_boundary(self) -> None:
        # 5 sentences, first 3 point one way, last 2 point orthogonally.
        # Consecutive distances will be roughly [0, 0, 1, 0] — the single
        # `1.0` (max) lands ≥ 95th percentile, so exactly one cut fires,
        # between sentences 3 and 4. Bounds are permissive (min=1, max=999)
        # so the cut is preserved.
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
        assert len(chunks) == 2
        assert "Alpha" in chunks[0].text and "Gamma" in chunks[0].text
        assert "Delta" in chunks[1].text and "Epsilon" in chunks[1].text

    def test_run_id_threaded_to_embedder(self) -> None:
        embedder = _FakeEmbedder()
        chunk_semantic(
            _doc(["Only one. And another."]),
            embedder=embedder,
            config=_cfg(min_tokens=1),
            run_id="run_abc",
        )
        assert embedder.calls[0]["run_id"] == "run_abc"

    def test_content_hashes_deterministic(self) -> None:
        # Same input + same fake vectors → identical Chunk.content_hash.
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
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ── Early returns ────────────────────────────────────────────────────


class TestEarlyReturns:
    def test_empty_doc_returns_empty(self) -> None:
        embedder = _FakeEmbedder()
        assert chunk_semantic(_doc([]), embedder=embedder, config=_cfg()) == []
        assert embedder.calls == []  # no wasted API call

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
        # No embed call — the single-sentence path short-circuits.
        assert embedder.calls == []


# ── Page attribution ─────────────────────────────────────────────────


class TestPageRanges:
    def test_single_page_chunks_share_page(self) -> None:
        embedder = _FakeEmbedder()
        chunks = chunk_semantic(
            _doc(["First. Second. Third."]),
            embedder=embedder,
            config=_cfg(min_tokens=1),
        )
        assert all(c.page_start == 1 and c.page_end == 1 for c in chunks)

    def test_multi_page_chunk_reports_range(self) -> None:
        # Two pages, same theme (identical embeddings) so no cut → one
        # chunk spanning both pages.
        embedder = _FakeEmbedder()  # every sentence gets default_vector
        chunks = chunk_semantic(
            _doc(["First sentence.", "Second sentence."]),
            embedder=embedder,
            config=_cfg(min_tokens=1, percentile_threshold=99.0),
        )
        # With identical vectors all distances = 0, so 99th percentile = 0
        # and every distance >= threshold → ALL cuts fire. But since only
        # 1 pair exists (2 sentences), exactly 1 cut → 2 chunks.
        # Let's use a doc where a cut is IMPOSSIBLE to prove page span.
        # Actually with 2 sentences even with a cut we get 2 chunks
        # each on a distinct page. Let's build a case that MUST span.
        # 3 pages, 3 sentences, identical vectors → distances=[0,0].
        # percentile=95: threshold = 0, ALL >= 0 → 2 cuts → 3 chunks.
        # Use percentile=100.001? Out of range. Use very high threshold
        # via an unreachable value? _percentile clamps ≤ 100.
        # Simplest: prove multi-page span via `_merge_undersized` forcing
        # a merge across pages when min_tokens is high enough.
        chunks = chunk_semantic(
            _doc(["Short.", "Also short.", "Third short."]),
            embedder=embedder,
            config=_cfg(min_tokens=100, percentile_threshold=95.0),
        )
        # min_tokens=100 forces all groups to merge → one chunk spanning
        # pages 1-3.
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
        # Single merged chunk covers pages 1-3 even though page 2 is blank.
        assert len(chunks) == 1
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 3


# ── Bound enforcement ────────────────────────────────────────────────


class TestBoundEnforcement:
    def test_undersized_groups_merge(self) -> None:
        # Every sentence stands alone (5 sentences, all cuts fire under
        # a low threshold), but min_tokens=100 forces merges. End state:
        # far fewer than 5 groups. This confirms the merge pass runs.
        embedder = _FakeEmbedder()  # all identical → threshold=0 → every cut
        chunks = chunk_semantic(
            _doc(["A short. B short. C short. D short. E short."]),
            embedder=embedder,
            config=_cfg(min_tokens=100, max_tokens=999, percentile_threshold=1.0),
        )
        assert len(chunks) < 5
        # All merged content preserved.
        joined = " ".join(c.text for c in chunks)
        for token in ("A short", "B short", "C short", "D short", "E short"):
            assert token in joined

    def test_oversized_group_splits(self) -> None:
        # One big group with an obvious internal boundary (identical vecs
        # on left, orthogonal vec in the middle). max_tokens=5 forces the
        # split to fire even though no percentile-cut fires.
        embedder = _FakeEmbedder(
            vectors_by_prefix={
                "Alpha": [1.0, 0.0],
                "Beta": [1.0, 0.0],
                "Gamma": [0.0, 1.0],
                "Delta": [0.0, 1.0],
            },
        )
        # Bump percentile_threshold to 100 so NO percentile cut fires
        # (every dist < threshold). Only the max_tokens split does work.
        chunks = chunk_semantic(
            _doc(["Alpha one. Beta two. Gamma three. Delta four."]),
            embedder=embedder,
            config=_cfg(min_tokens=1, max_tokens=5, percentile_threshold=100.0),
        )
        assert len(chunks) >= 2
        # Splits happen at the strongest internal boundary; the Alpha/Beta
        # pair should end up together and separate from Gamma/Delta.
        first_text = chunks[0].text
        assert "Alpha" in first_text
        # Gamma should be in a later chunk (post-split).
        gamma_chunk = next(c for c in chunks if "Gamma" in c.text)
        assert "Alpha" not in gamma_chunk.text

    def test_single_oversized_sentence_falls_back_to_fixed_split(self) -> None:
        # A single sentence longer than max_tokens has no semantic
        # boundary to split on — the fallback fixed-window split must fire.
        embedder = _FakeEmbedder()
        long_sentence = "word " * 200 + "end."  # ~200 tokens, max_tokens=50
        chunks = chunk_semantic(
            _doc([long_sentence]),
            embedder=embedder,
            config=_cfg(min_tokens=1, max_tokens=50, percentile_threshold=95.0),
        )
        assert len(chunks) > 1  # split into windows
        assert all(c.num_tokens <= 50 for c in chunks)
        # All chunks share the sentence's originating page.
        assert all(c.page_start == 1 and c.page_end == 1 for c in chunks)


# ── Validation ───────────────────────────────────────────────────────


class TestConfigValidation:
    def test_wrong_algorithm_raises(self) -> None:
        embedder = _FakeEmbedder()
        cfg = ChunkerConfig(algorithm="fixed", target_tokens=500, overlap_tokens=50)
        with pytest.raises(ValueError, match=r"algorithm='semantic'"):
            chunk_semantic(_doc(["A."]), embedder=embedder, config=cfg)

    def test_missing_knobs_raises(self) -> None:
        embedder = _FakeEmbedder()
        # Missing min_tokens.
        cfg = ChunkerConfig(
            algorithm="semantic",
            percentile_threshold=95.0,
            max_tokens=500,
        )
        with pytest.raises(ValueError, match="min_tokens"):
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


# ── Structural guard ─────────────────────────────────────────────────


def test_fake_embedder_signature_matches_real() -> None:
    """Same guard as test_ingest.py — chunker's fake embedder must match Voyage's."""
    real = inspect.signature(VoyageEmbedder.embed_documents)
    fake = inspect.signature(_FakeEmbedder.embed_documents)
    assert list(real.parameters.keys()) == list(fake.parameters.keys())
