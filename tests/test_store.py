"""Store-layer tests for RRF fusion and hybrid-search orchestration.

The BM25 SQL path (`lexical_search`) is best exercised against a real Postgres —
that's a manual smoke via `make db-up && make ingest --pipeline p3 && make eval
--pipeline p3`. Fusion math and hybrid branch orchestration are tested offline.
"""

from __future__ import annotations

import pytest

from src.pipeline.store import RetrievedChunk, VectorStore, _rrf_fuse


def _chunk(chunk_id: int, *, score: float = 0.0, doc_path: str | None = None) -> RetrievedChunk:
    """Build a RetrievedChunk with unique `chunk_id`. All other fields are
    filler — RRF only reads chunk_id + provenance, ignores incoming score."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id * 10,
        doc_path=doc_path or f"fixture/doc{chunk_id}.pdf",
        source_id="A1",
        pipeline="test_p3_hybrid",
        chunk_index=chunk_id - 1,
        text=f"chunk-{chunk_id}-body",
        page_start=chunk_id,
        page_end=chunk_id,
        score=score,
    )


class TestRRFFuseMath:
    def test_disjoint_lists_ranked_by_position(self) -> None:
        """No chunk in both lists: fused rank = per-source rank; sums are strictly
        smaller for later positions. Verifies the base RRF formula per source."""
        dense = [_chunk(1), _chunk(2), _chunk(3)]
        lexical = [_chunk(101), _chunk(102), _chunk(103)]
        # Interleaved by rank: rank-1 from each ties, then rank-2 from each, ...
        # Tie-break is chunk_id ASC, so dense (1) < lexical (101).
        fused = _rrf_fuse(dense, lexical, rrf_k=60, top_k=6)
        assert [c.chunk_id for c in fused] == [1, 101, 2, 102, 3, 103]
        # Verify the actual score at rank 1 = 1/(60+1) ≈ 0.01639.
        assert abs(fused[0].score - 1.0 / 61) < 1e-9
        assert abs(fused[1].score - 1.0 / 61) < 1e-9
        # Rank 2 in each: 1/62 each.
        assert abs(fused[2].score - 1.0 / 62) < 1e-9

    def test_chunk_in_both_lists_sums_scores(self) -> None:
        """A chunk ranked in both sources gets the sum of its per-source
        reciprocals — the whole point of RRF."""
        dense = [_chunk(1), _chunk(2)]
        # Same chunk_id=2 appears in both; expected fused = 1/(60+2) + 1/(60+1).
        lexical = [_chunk(2), _chunk(99)]
        fused = _rrf_fuse(dense, lexical, rrf_k=60, top_k=4)
        by_id = {c.chunk_id: c for c in fused}
        assert abs(by_id[2].score - (1.0 / 62 + 1.0 / 61)) < 1e-9
        # chunk 2 has the highest fused score → ranks first.
        assert fused[0].chunk_id == 2

    def test_tie_break_by_chunk_id_ascending(self) -> None:
        """Same fused score → chunk_id ASC. Determinism matters — two eval
        runs must return the same ordering for the same DB state.

        Constructed tie:
          chunk 5: 1/61 (dense@1) + 1/62 (lexical@2)
          chunk 3: 1/62 (dense@2) + 1/61 (lexical@1)
        → identical fused scores; tie-break by chunk_id ASC → [3, 5].
        """
        dense_tie = [_chunk(5), _chunk(3)]
        lexical_tie = [_chunk(3), _chunk(5)]
        fused_tie = _rrf_fuse(dense_tie, lexical_tie, rrf_k=60, top_k=2)
        assert [c.chunk_id for c in fused_tie] == [3, 5]

    def test_top_k_truncates(self) -> None:
        """`top_k` caps output; higher-scoring chunks survive."""
        dense = [_chunk(i) for i in range(1, 11)]
        lexical: list[RetrievedChunk] = []
        fused = _rrf_fuse(dense, lexical, rrf_k=60, top_k=3)
        assert [c.chunk_id for c in fused] == [1, 2, 3]

    @pytest.mark.parametrize(
        ("rrf_k", "top_k", "invalid_name"),
        [
            (0, 1, "rrf_k"),
            (-1, 1, "rrf_k"),
            (60, 0, "top_k"),
            (60, -1, "top_k"),
        ],
    )
    def test_non_positive_parameters_raise(self, rrf_k: int, top_k: int, invalid_name: str) -> None:
        with pytest.raises(ValueError, match=rf"{invalid_name} must be positive"):
            _rrf_fuse([_chunk(1)], [], rrf_k=rrf_k, top_k=top_k)

    def test_empty_both_returns_empty(self) -> None:
        """Neither source returned anything → empty fusion; no crash."""
        assert _rrf_fuse([], [], rrf_k=60, top_k=10) == []

    def test_empty_dense_only_lexical_survives(self) -> None:
        """Dense empty + lexical populated → lexical result set, in order."""
        lexical = [_chunk(1), _chunk(2)]
        fused = _rrf_fuse([], lexical, rrf_k=60, top_k=5)
        assert [c.chunk_id for c in fused] == [1, 2]
        assert abs(fused[0].score - 1.0 / 61) < 1e-9

    def test_empty_lexical_only_dense_survives(self) -> None:
        """Lexical empty + dense populated → dense result set, in order."""
        dense = [_chunk(1), _chunk(2)]
        fused = _rrf_fuse(dense, [], rrf_k=60, top_k=5)
        assert [c.chunk_id for c in fused] == [1, 2]
        assert abs(fused[0].score - 1.0 / 61) < 1e-9

    def test_dense_provenance_wins_on_overlap(self) -> None:
        """When a chunk appears in both lists, the dense list's `RetrievedChunk`
        deterministically supplies the fused result's non-score fields."""
        dense_chunk = _chunk(42, doc_path="fixture/from_dense.pdf")
        lexical_chunk = _chunk(42, doc_path="fixture/from_lexical.pdf")
        fused = _rrf_fuse([dense_chunk], [lexical_chunk], rrf_k=60, top_k=1)
        assert fused[0].doc_path == "fixture/from_dense.pdf"

    def test_rrf_k_smaller_sharpens_rank_gap(self) -> None:
        """Smaller `rrf_k` widens the score gap between rank 1 and rank 2.

        Sanity-check that the knob has the documented effect — a future
        refactor that swaps in a fixed rrf_k constant would break this.
        """
        dense = [_chunk(1), _chunk(2)]
        gap_60 = _rrf_fuse(dense, [], rrf_k=60, top_k=2)
        gap_10 = _rrf_fuse(dense, [], rrf_k=10, top_k=2)
        # Score gap between rank 1 and rank 2, per rrf_k:
        # rrf_k=60 → 1/61 - 1/62 ≈ 0.000264
        # rrf_k=10 → 1/11 - 1/12 ≈ 0.007576
        assert (gap_10[0].score - gap_10[1].score) > (gap_60[0].score - gap_60[1].score)

    def test_returned_chunks_carry_fused_score_not_source_score(self) -> None:
        """Ensures the RRF score replaces the incoming (cosine / ts_rank_cd)
        score — otherwise the demo UI + eval reporter would surface the
        wrong number."""
        dense = [_chunk(1, score=0.99)]  # incoming cosine — must be overwritten
        fused = _rrf_fuse(dense, [], rrf_k=60, top_k=1)
        assert fused[0].score != 0.99
        assert abs(fused[0].score - 1.0 / 61) < 1e-9


class TestHybridSearch:
    def test_runs_both_branches_and_fuses_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = VectorStore("unused")
        query_embedding = [0.1, 0.2]
        calls: list[tuple] = []

        def dense_search(pipeline: str, embedding: list[float], k: int) -> list[RetrievedChunk]:
            calls.append(("dense", pipeline, embedding, k))
            return [_chunk(1), _chunk(2)]

        def lexical_search(pipeline: str, query_text: str, k: int) -> list[RetrievedChunk]:
            calls.append(("lexical", pipeline, query_text, k))
            return [_chunk(2), _chunk(3)]

        monkeypatch.setattr(store, "dense_search", dense_search)
        monkeypatch.setattr(store, "lexical_search", lexical_search)

        fused = store.hybrid_search(
            "test_pipeline",
            query_embedding,
            "what is merge sort?",
            k=2,
            rrf_k=60,
        )

        assert calls == [
            ("dense", "test_pipeline", query_embedding, 2),
            ("lexical", "test_pipeline", "what is merge sort?", 2),
        ]
        assert [chunk.chunk_id for chunk in fused] == [2, 1]

    @pytest.mark.parametrize(("k", "rrf_k"), [(0, 60), (2, 0)])
    def test_invalid_parameters_fail_before_search(
        self, monkeypatch: pytest.MonkeyPatch, k: int, rrf_k: int
    ) -> None:
        store = VectorStore("unused")
        calls: list[str] = []

        monkeypatch.setattr(
            store,
            "dense_search",
            lambda *_args, **_kwargs: calls.append("dense"),
        )
        monkeypatch.setattr(
            store,
            "lexical_search",
            lambda *_args, **_kwargs: calls.append("lexical"),
        )

        with pytest.raises(ValueError, match="must be positive"):
            store.hybrid_search("test_pipeline", [0.1], "query", k=k, rrf_k=rrf_k)

        assert calls == []
