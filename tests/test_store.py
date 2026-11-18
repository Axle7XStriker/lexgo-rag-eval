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
        chunk_set_id="test_fixed_500_50",
        chunk_index=chunk_id - 1,
        text=f"chunk-{chunk_id}-body",
        page_start=chunk_id,
        page_end=chunk_id,
        score=score,
    )


class TestRRFFuseMath:
    def test_disjoint_lists_ranked_by_position(self) -> None:
        dense = [_chunk(1), _chunk(2), _chunk(3)]
        lexical = [_chunk(101), _chunk(102), _chunk(103)]
        fused = _rrf_fuse(dense, lexical, rrf_k=60, top_k=6)
        assert [c.chunk_id for c in fused] == [1, 101, 2, 102, 3, 103]
        assert abs(fused[0].score - 1.0 / 61) < 1e-9
        assert abs(fused[1].score - 1.0 / 61) < 1e-9
        assert abs(fused[2].score - 1.0 / 62) < 1e-9

    def test_chunk_in_both_lists_sums_scores(self) -> None:
        dense = [_chunk(1), _chunk(2)]
        lexical = [_chunk(2), _chunk(99)]
        fused = _rrf_fuse(dense, lexical, rrf_k=60, top_k=4)
        by_id = {c.chunk_id: c for c in fused}
        assert abs(by_id[2].score - (1.0 / 62 + 1.0 / 61)) < 1e-9
        assert fused[0].chunk_id == 2

    def test_tie_break_by_chunk_id_ascending(self) -> None:
        dense_tie = [_chunk(5), _chunk(3)]
        lexical_tie = [_chunk(3), _chunk(5)]
        fused_tie = _rrf_fuse(dense_tie, lexical_tie, rrf_k=60, top_k=2)
        assert [c.chunk_id for c in fused_tie] == [3, 5]

    def test_top_k_truncates(self) -> None:
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
        assert _rrf_fuse([], [], rrf_k=60, top_k=10) == []

    def test_empty_dense_only_lexical_survives(self) -> None:
        lexical = [_chunk(1), _chunk(2)]
        fused = _rrf_fuse([], lexical, rrf_k=60, top_k=5)
        assert [c.chunk_id for c in fused] == [1, 2]
        assert abs(fused[0].score - 1.0 / 61) < 1e-9

    def test_empty_lexical_only_dense_survives(self) -> None:
        dense = [_chunk(1), _chunk(2)]
        fused = _rrf_fuse(dense, [], rrf_k=60, top_k=5)
        assert [c.chunk_id for c in fused] == [1, 2]
        assert abs(fused[0].score - 1.0 / 61) < 1e-9

    def test_dense_provenance_wins_on_overlap(self) -> None:
        dense_chunk = _chunk(42, doc_path="fixture/from_dense.pdf")
        lexical_chunk = _chunk(42, doc_path="fixture/from_lexical.pdf")
        fused = _rrf_fuse([dense_chunk], [lexical_chunk], rrf_k=60, top_k=1)
        assert fused[0].doc_path == "fixture/from_dense.pdf"

    def test_rrf_k_smaller_sharpens_rank_gap(self) -> None:
        dense = [_chunk(1), _chunk(2)]
        gap_60 = _rrf_fuse(dense, [], rrf_k=60, top_k=2)
        gap_10 = _rrf_fuse(dense, [], rrf_k=10, top_k=2)
        assert (gap_10[0].score - gap_10[1].score) > (gap_60[0].score - gap_60[1].score)

    def test_returned_chunks_carry_fused_score_not_source_score(self) -> None:
        dense = [_chunk(1, score=0.99)]
        fused = _rrf_fuse(dense, [], rrf_k=60, top_k=1)
        assert fused[0].score != 0.99
        assert abs(fused[0].score - 1.0 / 61) < 1e-9


class TestHybridSearch:
    def test_runs_both_branches_and_fuses_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = VectorStore("unused")
        query_embedding = [0.1, 0.2]
        calls: list[tuple] = []

        def dense_search(chunk_set_id: str, embedding: list[float], k: int) -> list[RetrievedChunk]:
            calls.append(("dense", chunk_set_id, embedding, k))
            return [_chunk(1), _chunk(2)]

        def lexical_search(chunk_set_id: str, query_text: str, k: int) -> list[RetrievedChunk]:
            calls.append(("lexical", chunk_set_id, query_text, k))
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
            ("dense", "test_pipeline", query_embedding, 20),
            ("lexical", "test_pipeline", "what is merge sort?", 20),
        ]
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
        store = VectorStore("unused")
        pool_calls: list[int] = []

        def dense_search(chunk_set_id: str, embedding: list[float], k: int) -> list[RetrievedChunk]:
            pool_calls.append(k)
            return []

        def lexical_search(chunk_set_id: str, query_text: str, k: int) -> list[RetrievedChunk]:
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


_CHUNK_SET_ID = "test_fixed_500_50_real"


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
                chunk_set_id=_CHUNK_SET_ID,
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
        hits = seeded_store.lexical_search(_CHUNK_SET_ID, "collision", k=5)
        assert len(hits) == 1
        assert "collision" in hits[0].text
        assert hits[0].source_id == "A1"
        assert hits[0].score > 0

    def test_multi_term_match_ranks_chunks(self, seeded_store: VectorStore) -> None:
        hits = seeded_store.lexical_search(_CHUNK_SET_ID, "quicksort", k=5)
        chunk_indexes = sorted(h.chunk_index for h in hits)
        assert chunk_indexes == [0, 3]
        assert all(h.score > 0 for h in hits)

    def test_no_match_returns_empty_list(self, seeded_store: VectorStore) -> None:
        hits = seeded_store.lexical_search(_CHUNK_SET_ID, "zebra", k=5)
        assert hits == []

    def test_operator_input_does_not_raise(self, seeded_store: VectorStore) -> None:
        """websearch_to_tsquery must swallow operator-looking input gracefully.

        plainto_tsquery would also not raise here; the point is to lock in
        the parser we picked and prove quoted-phrase + OR syntax reach FTS.
        """
        quoted = seeded_store.lexical_search(_CHUNK_SET_ID, '"hash table"', k=5)
        assert any("hash table" in h.text for h in quoted)
        or_query = seeded_store.lexical_search(_CHUNK_SET_ID, "quicksort OR transaction", k=5)
        or_indexes = {hit.chunk_index for hit in or_query}
        assert 2 in or_indexes
        assert or_indexes & {0, 3}

    def test_chunk_set_id_filter_isolates_rows(self, seeded_store: VectorStore) -> None:
        hits = seeded_store.lexical_search("nonexistent_chunk_set_id", "quicksort", k=5)
        assert hits == []


class TestHybridSearchReal:
    def test_fuses_dense_and_lexical_end_to_end(self, seeded_store: VectorStore) -> None:
        query_embedding = [0.0] * EMBEDDING_DIM
        query_embedding[2] = 1.0

        fused = seeded_store.hybrid_search(
            _CHUNK_SET_ID,
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

    def test_hybrid_filters_by_chunk_set_id(self, seeded_store: VectorStore) -> None:
        fused = seeded_store.hybrid_search(
            "nonexistent_chunk_set_id",
            [0.1] * EMBEDDING_DIM,
            query_text="quicksort",
            k=5,
            rrf_k=60,
        )
        assert fused == []
