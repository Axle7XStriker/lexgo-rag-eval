"""Store-layer tests.

Two layers:
  - Pure RRF math + offline hybrid-search orchestration (fakes stand in for
    the two retriever branches).
  - Real-Postgres tests for the BM25 SQL path (`lexical_search`) + the full
    `hybrid_search` end-to-end — the FTS SQL string, GIN index, generated
    `text_tsv` column, and `websearch_to_tsquery` behavior are all things a
    monkeypatch would silently green-light while shipping broken SQL. Skipped
    automatically when the local Postgres isn't up (see tests/conftest.py).
"""

from __future__ import annotations

import pytest

from src.pipeline.store import (
    EMBEDDING_DIM,
    ChunkRow,
    DocumentRow,
    RetrievedChunk,
    VectorStore,
    _rrf_fuse,
)


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
        """Both branches ran, were pulled at the widened pool size, and RRF-fused."""
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

        # Widened per-source pool: max(k*2, 20) = max(4, 20) = 20 for k=2.
        # Pulling > k per source so RRF has more fusion opportunities before
        # truncation to final k; see hybrid_search's comment for the rationale.
        assert calls == [
            ("dense", "test_pipeline", query_embedding, 20),
            ("lexical", "test_pipeline", "what is merge sort?", 20),
        ]
        # Final result truncated to k=2; chunk_id=2 appears in both branches
        # so it ranks first (sum of two reciprocals).
        assert [chunk.chunk_id for chunk in fused] == [2, 1]

    @pytest.mark.parametrize(
        ("k", "expected_pool"),
        [
            (2, 20),  # floor dominates
            (10, 20),  # k*2 == floor — same value
            (15, 30),  # k*2 dominates
        ],
    )
    def test_per_source_pool_formula(
        self, monkeypatch: pytest.MonkeyPatch, k: int, expected_pool: int
    ) -> None:
        """Pool = max(k*2, 20) — guards the widened fetch against regression.

        Smaller `k` than the floor still gets a meaningful candidate pool
        (otherwise k=1 would pull a single candidate per source and never
        exercise fusion); larger `k` scales linearly.
        """
        store = VectorStore("unused")
        pool_calls: list[int] = []

        def dense_search(pipeline: str, embedding: list[float], k: int) -> list[RetrievedChunk]:
            pool_calls.append(k)
            return []

        def lexical_search(pipeline: str, query_text: str, k: int) -> list[RetrievedChunk]:
            pool_calls.append(k)
            return []

        monkeypatch.setattr(store, "dense_search", dense_search)
        monkeypatch.setattr(store, "lexical_search", lexical_search)

        store.hybrid_search("test_pipeline", [0.1], "query", k=k, rrf_k=60)
        assert pool_calls == [expected_pool, expected_pool]

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


# ── Real-Postgres FTS + hybrid tests ──────────────────────────────────
#
# Everything below talks to a live Postgres via the `clean_store` fixture
# (tests/conftest.py). Skipped automatically if the DB isn't reachable.
# Seeds a small deterministic corpus and asserts the SQL paths behave on
# a real `tsvector` + GIN index + `websearch_to_tsquery` round-trip.


_PIPELINE_TAG = "test_p3_hybrid_real"


def _seed_text_chunk(
    store: VectorStore,
    *,
    document_id: int,
    chunk_index: int,
    text: str,
    embedding: list[float] | None = None,
) -> None:
    """Insert one chunk with a known text and a deterministic embedding.

    Embedding defaults to a one-hot-ish vector keyed off `chunk_index` so
    dense retrieval has a reproducible ordering; lexical tests don't read it.
    """
    if embedding is None:
        embedding = [0.0] * EMBEDDING_DIM
        embedding[chunk_index % EMBEDDING_DIM] = 1.0
    store.upsert_chunks(
        document_id,
        [
            ChunkRow(
                pipeline=_PIPELINE_TAG,
                chunk_index=chunk_index,
                text=text,
                num_tokens=max(1, len(text.split())),
                page_start=chunk_index + 1,
                page_end=chunk_index + 1,
                content_hash=f"testhash_{chunk_index}",
                embedding=embedding,
            )
        ],
    )


@pytest.fixture
def seeded_store(clean_store: VectorStore) -> VectorStore:
    """Four-chunk corpus with distinct lexical signatures.

    Keyword plan:
      - chunk 0: "quicksort" — hit by `quicksort` queries, not by `database`.
      - chunk 1: "hash table" + "collision" — hit by `hash` / `collision`.
      - chunk 2: "database" + "transaction" — hit by `database` / `transaction`.
      - chunk 3: "quicksort" + "pivot" — overlaps chunk 0 on `quicksort`.
    """
    doc_id = clean_store.upsert_document(
        DocumentRow(
            source_id="A1",
            doc_path="test/fts_fixture.pdf",
            title="FTS fixture",
            num_pages=4,
            content_hash="doc_fixture_hash",
        )
    )
    _seed_text_chunk(
        clean_store,
        document_id=doc_id,
        chunk_index=0,
        text="Quicksort is a comparison-based sorting algorithm.",
    )
    _seed_text_chunk(
        clean_store,
        document_id=doc_id,
        chunk_index=1,
        text="A hash table resolves a collision via open addressing or chaining.",
    )
    _seed_text_chunk(
        clean_store,
        document_id=doc_id,
        chunk_index=2,
        text="A database transaction preserves atomicity across multiple writes.",
    )
    _seed_text_chunk(
        clean_store,
        document_id=doc_id,
        chunk_index=3,
        text="Quicksort with a median-of-medians pivot is worst-case linearithmic.",
    )
    clean_store.conn.commit()
    return clean_store


class TestLexicalSearchReal:
    def test_single_term_match_returns_chunk(self, seeded_store: VectorStore) -> None:
        """A term that appears in exactly one chunk returns that chunk."""
        hits = seeded_store.lexical_search(_PIPELINE_TAG, "collision", k=5)
        assert len(hits) == 1
        assert "collision" in hits[0].text
        assert hits[0].source_id == "A1"
        assert hits[0].score > 0

    def test_multi_term_match_ranks_chunks(self, seeded_store: VectorStore) -> None:
        """A shared term returns every matching chunk ranked with a positive score."""
        hits = seeded_store.lexical_search(_PIPELINE_TAG, "quicksort", k=5)
        chunk_indexes = sorted(h.chunk_index for h in hits)
        assert chunk_indexes == [0, 3]
        assert all(h.score > 0 for h in hits)

    def test_no_match_returns_empty_list(self, seeded_store: VectorStore) -> None:
        """A term absent from every chunk returns [], not an error."""
        hits = seeded_store.lexical_search(_PIPELINE_TAG, "zebra", k=5)
        assert hits == []

    def test_operator_input_does_not_raise(self, seeded_store: VectorStore) -> None:
        """websearch_to_tsquery must swallow operator-looking input gracefully.

        plainto_tsquery would also not raise here; the point is to lock in
        the parser we picked and prove quoted-phrase + OR syntax reach FTS.
        """
        # Quoted phrase: matches chunk 1 as an exact-ish phrase lookup.
        quoted = seeded_store.lexical_search(_PIPELINE_TAG, '"hash table"', k=5)
        assert any("hash table" in h.text for h in quoted)
        # Explicit OR: broadens recall across two chunks.
        or_query = seeded_store.lexical_search(_PIPELINE_TAG, "quicksort OR transaction", k=5)
        or_indexes = {hit.chunk_index for hit in or_query}
        assert 2 in or_indexes
        assert or_indexes & {0, 3}

    def test_pipeline_filter_isolates_rows(self, seeded_store: VectorStore) -> None:
        """A query against an unknown pipeline tag returns nothing even when the
        lexeme matches rows under a different tag — the WHERE clause composes."""
        hits = seeded_store.lexical_search("nonexistent_pipeline_tag", "quicksort", k=5)
        assert hits == []


class TestHybridSearchReal:
    def test_fuses_dense_and_lexical_end_to_end(self, seeded_store: VectorStore) -> None:
        """Hybrid returns a non-empty fused list over a real DB; chunks come from
        both branches when they disagree."""
        # Query embedding aligned with chunk 2's one-hot vector so dense ranks
        # chunk 2 near the top; lexical hits chunk 0 and chunk 3 on 'quicksort'.
        # Fused list must include chunks from both branches.
        query_embedding = [0.0] * EMBEDDING_DIM
        query_embedding[2] = 1.0

        fused = seeded_store.hybrid_search(
            _PIPELINE_TAG,
            query_embedding,
            query_text="quicksort",
            k=5,
            rrf_k=60,
        )
        chunk_indexes = {h.chunk_index for h in fused}
        assert 2 in chunk_indexes  # Dense match: one-hot embedding aligned to chunk 2.
        assert chunk_indexes & {0, 3}  # Lexical matches: "quicksort" occurs in chunks 0 and 3.
        # Scores are RRF values — positive floats, bounded by 2/(rrf_k+1).
        assert all(h.score > 0 for h in fused)
        assert all(h.score <= 2.0 / (60 + 1) + 1e-9 for h in fused)

    def test_hybrid_filters_by_pipeline_tag(self, seeded_store: VectorStore) -> None:
        """An unknown pipeline tag returns empty from both branches → empty fusion."""
        fused = seeded_store.hybrid_search(
            "nonexistent_pipeline_tag",
            [0.1] * EMBEDDING_DIM,
            query_text="quicksort",
            k=5,
            rrf_k=60,
        )
        assert fused == []
