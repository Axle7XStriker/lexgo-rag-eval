"""Postgres+pgvector store for the RAG pipelines.

Equal chunker configs share one `chunk_set_id` and one stored copy.
Retrieval filters that id in the query.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import psycopg
from pgvector.psycopg import register_vector
from psycopg.pq import TransactionStatus

from src.db import __file__ as _db_pkg_file

SCHEMA_PATH = Path(_db_pkg_file).parent / "schema.sql"
EMBEDDING_DIM = 1024  # voyage-3-large


@dataclass(frozen=True)
class DocumentRow:
    """A row destined for the `documents` table."""

    source_id: str
    doc_path: str
    title: str | None
    num_pages: int | None
    content_hash: str


@dataclass(frozen=True)
class ChunkRow:
    """A chunk destined for the `chunks` table."""

    chunk_set_id: str
    chunk_index: int
    text: str
    num_tokens: int
    page_start: int
    page_end: int
    content_hash: str
    embedding: list[float]


@dataclass(frozen=True)
class RetrievedChunk:
    """A row returned from `dense_search` — chunk + document context + score."""

    chunk_id: int
    document_id: int
    doc_path: str
    source_id: str
    chunk_set_id: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    score: float


def _row_to_retrieved(row: tuple) -> RetrievedChunk:
    """Adapt a psycopg row into a `RetrievedChunk`.
    Keeps dense + lexical branches in lockstep."""
    return RetrievedChunk(
        chunk_id=row[0],
        document_id=row[1],
        doc_path=row[2],
        source_id=row[3],
        chunk_set_id=row[4],
        chunk_index=row[5],
        text=row[6],
        page_start=row[7],
        page_end=row[8],
        score=row[9],
    )


def _validate_rrf_params(*, rrf_k: int, top_k: int) -> None:
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")


def _rrf_fuse(
    dense: list[RetrievedChunk],
    lexical: list[RetrievedChunk],
    *,
    rrf_k: int,
    top_k: int,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion of two ranked lists → top-`top_k` fused.

    For each chunk (identified by `chunk_id`), sum `1 / (rrf_k + rank)`
    across the two lists it appears in (1-indexed rank). Sort by fused
    score DESC, break ties by chunk_id ASC for determinism.

    The returned `RetrievedChunk.score` is the RRF score (small float,
    upper-bounded by `2 / (rrf_k + 1)`), not either source's original
    score. When a chunk appears in both lists, its non-score fields are
    carried forward from the dense result for deterministic provenance.

    """
    _validate_rrf_params(rrf_k=rrf_k, top_k=top_k)

    scores: dict[int, float] = {}
    for rank, chunk in enumerate(dense, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, chunk in enumerate(lexical, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    by_id: dict[int, RetrievedChunk] = {c.chunk_id: c for c in lexical}
    for chunk in dense:
        by_id[chunk.chunk_id] = chunk

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    fused: list[RetrievedChunk] = []
    for chunk_id, fused_score in ranked[:top_k]:
        base = by_id[chunk_id]
        fused.append(
            RetrievedChunk(
                chunk_id=base.chunk_id,
                document_id=base.document_id,
                doc_path=base.doc_path,
                source_id=base.source_id,
                chunk_set_id=base.chunk_set_id,
                chunk_index=base.chunk_index,
                text=base.text,
                page_start=base.page_start,
                page_end=base.page_end,
                score=fused_score,
            )
        )
    return fused


class VectorStore:
    """Thin sync wrapper around psycopg + pgvector.

    Must be used as a context manager. The connection is opened in
    __enter__; any method call on an unopened store raises RuntimeError.
        with VectorStore(dsn) as store:
            store.ensure_schema()
            ...
    """

    def __init__(self, dsn: str) -> None:
        # Postgres data source name (connection string).
        self._dsn = dsn
        self._conn: psycopg.Connection | None = None

    def __enter__(self) -> Self:
        self.connect()
        return self

    def connect(self) -> None:
        """Open a connection and register pgvector on it.

        No-op when the current connection can run a statement. A missing or
        closed socket is replaced. Deadlock and statement cancel abort the
        transaction and leave the socket open, so that transaction is rolled
        back before the same connection is reused.
        """
        if self._conn is not None and not self._conn.closed:
            if self._conn.info.transaction_status == TransactionStatus.INERROR:
                self._rollback_if_alive()
            if self._conn is not None and not self._conn.closed:
                return
        self._conn = psycopg.connect(self._dsn, autocommit=False, connect_timeout=5)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")  # Before register_vector.
        self._conn.commit()
        register_vector(self._conn)

    def _rollback_if_alive(self) -> None:
        if self._conn is None or self._conn.closed:
            return
        try:
            self._conn.rollback()
        except psycopg.OperationalError:
            return

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None:
            raise RuntimeError("VectorStore not opened; use `with VectorStore(dsn) as store: ...`.")
        return self._conn

    def ensure_schema(self) -> None:
        """Apply schema.sql. Idempotent — every DDL statement is guarded.

        schema.sql is the source of truth for the schema; this method is
        a thin execute-the-file wrapper. Additive changes (new tables,
        new indexes) are picked up on the next call because every DDL is
        guarded with IF NOT EXISTS. Destructive changes (drop column,
        change column type) are not handled here.
        """
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text())
        self.conn.commit()

    def upsert_document(self, doc: DocumentRow) -> int:
        """Insert or refresh a document by `doc_path`; return its id.

        Does NOT commit — caller controls the transaction boundary so
        this can be composed with `upsert_chunks` atomically (see
        `replace_document_chunks`).
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (source_id, doc_path, title, num_pages, content_hash)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (doc_path) DO UPDATE
                    SET source_id    = EXCLUDED.source_id,
                        title        = EXCLUDED.title,
                        num_pages    = EXCLUDED.num_pages,
                        content_hash = EXCLUDED.content_hash
                RETURNING id
                """,
                (doc.source_id, doc.doc_path, doc.title, doc.num_pages, doc.content_hash),
            )
            row = cur.fetchone()
            assert row is not None
            return row[0]

    def upsert_chunks(self, document_id: int, chunks: list[ChunkRow]) -> None:
        """Bulk-insert chunks; on `(document_id, chunk_set_id, chunk_index)` conflict, replace.

        Does NOT commit — caller controls the transaction. Does NOT delete
        stale higher-index chunks left over from a previous ingest with more
        chunks; use `replace_document_chunks` when re-embedding a document.

        Raises `ValueError` up front if any embedding is the wrong dim,
        so a single bad chunk fails fast instead of mid-transaction.
        """
        if not chunks:
            return
        for c in chunks:
            if len(c.embedding) != EMBEDDING_DIM:
                raise ValueError(
                    f"embedding dim {len(c.embedding)} ≠ expected {EMBEDDING_DIM} "
                    f"(document_id={document_id}, chunk_index={c.chunk_index})"
                )
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (document_id, chunk_set_id, chunk_index, text, num_tokens,
                     page_start, page_end, content_hash, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id, chunk_set_id, chunk_index) DO UPDATE
                    SET text         = EXCLUDED.text,
                        num_tokens   = EXCLUDED.num_tokens,
                        page_start   = EXCLUDED.page_start,
                        page_end     = EXCLUDED.page_end,
                        content_hash = EXCLUDED.content_hash,
                        embedding    = EXCLUDED.embedding
                """,
                [
                    (
                        document_id,
                        c.chunk_set_id,
                        c.chunk_index,
                        c.text,
                        c.num_tokens,
                        c.page_start,
                        c.page_end,
                        c.content_hash,
                        c.embedding,
                    )
                    for c in chunks
                ],
            )

    def delete_chunks(self, document_id: int, chunk_set_id: str) -> int:
        """Delete all chunks for `(document_id, chunk_set_id)`; return the row count.

        Does NOT commit. Used by `replace_document_chunks` to purge stale
        chunks (including higher-index orphans a plain UPSERT would miss)
        before re-inserting a fresh chunk set.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE document_id = %s AND chunk_set_id = %s",
                (document_id, chunk_set_id),
            )
            return cur.rowcount

    def replace_document_chunks(
        self,
        doc: DocumentRow,
        chunks: list[ChunkRow],
        *,
        chunk_set_id: str,
    ) -> int:
        """Atomic: upsert `doc`, wipe existing chunks for `(doc, chunk_set_id)`, insert `chunks`.

        Returns the document_id. The document row's content_hash is only
        visible to other readers after all chunks land — a mid-flight failure
        rolls back both writes, so the idempotency short-circuit
        (`skip if content_hash matches`) can trust that a committed doc row
        implies its chunks are present.

        Also purges any orphan chunks with `chunk_index` past the current
        batch's length — the ON CONFLICT UPDATE in `upsert_chunks` alone
        would leave those stranded when a re-ingest produces fewer chunks.

        If the connection was dropped, open a new one and repeat the whole write.
        The first attempt did not commit, so the retry is safe and does not re-embed.
        """
        try:
            return self._write_document_chunks(doc, chunks, chunk_set_id=chunk_set_id)
        except psycopg.OperationalError:
            self.connect()
            return self._write_document_chunks(doc, chunks, chunk_set_id=chunk_set_id)

    def _write_document_chunks(
        self,
        doc: DocumentRow,
        chunks: list[ChunkRow],
        *,
        chunk_set_id: str,
    ) -> int:
        try:
            document_id = self.upsert_document(doc)
            self.delete_chunks(document_id, chunk_set_id)
            self.upsert_chunks(document_id, chunks)
            self.conn.commit()
            return document_id
        except psycopg.OperationalError:
            raise
        except Exception:
            self._rollback_if_alive()
            raise

    # Column list shared by both retrievers so their SELECT bodies stay
    # in lockstep — any new field on `RetrievedChunk` needs updating in
    # one place. `{score_expr}` is the only per-caller substitution.
    _SELECT_COLUMNS = (
        "c.id, c.document_id, d.doc_path, d.source_id, "
        "c.chunk_set_id, c.chunk_index, c.text, c.page_start, c.page_end"
    )

    def dense_search(
        self,
        chunk_set_id: str,
        query_embedding: list[float],
        k: int,
    ) -> list[RetrievedChunk]:
        """Top-k cosine-similarity search over chunks of a given chunk set.

        Returned `score` is cosine similarity in [-1, 1], where higher is
        closer.
        """
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"query embedding dim {len(query_embedding)} ≠ expected {EMBEDDING_DIM}"
            )
        sql = f"""
            SELECT
                {self._SELECT_COLUMNS},
                1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.chunk_set_id = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (query_embedding, chunk_set_id, query_embedding, k))
            rows = cur.fetchall()
        return [_row_to_retrieved(r) for r in rows]

    def lexical_search(
        self,
        chunk_set_id: str,
        query_text: str,
        k: int,
    ) -> list[RetrievedChunk]:
        """Return full-text matches ordered by relevance.

        Scores are meaningful only within this result list. Hybrid search
        combines lexical and dense results by rank rather than raw score.
        """
        sql = f"""
            SELECT
                {self._SELECT_COLUMNS},
                ts_rank_cd(c.text_tsv, query, 32) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id,
                 websearch_to_tsquery('english', %s) AS query
            WHERE c.chunk_set_id = %s
              AND c.text_tsv @@ query
            ORDER BY score DESC
            LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (query_text, chunk_set_id, k))
            rows = cur.fetchall()
        return [_row_to_retrieved(r) for r in rows]

    def hybrid_search(
        self,
        chunk_set_id: str,
        query_embedding: list[float],
        query_text: str,
        k: int,
        rrf_k: int,
    ) -> list[RetrievedChunk]:
        """Fetch top-k from dense + lexical, RRF-fuse, return top-k fused.

        Reciprocal Rank Fusion (Cormack et al. 2009): a chunk's fused score
        is the sum of `1 / (rrf_k + rank)` across the source lists it
        appears in (1-indexed rank). Rank-based, so the two sources'
        wildly-different score scales (cosine in [-1, 1] vs ts_rank_cd
        unbounded) don't need normalization — the algorithm is robust
        by construction.

        Returned `RetrievedChunk.score` is the RRF score (small float,
        roughly bounded by `2 / (rrf_k + 1)`), NOT cosine or ts_rank_cd.
        Downstream renders it verbatim; document this in any UI that
        surfaces the number.
        """
        _validate_rrf_params(rrf_k=rrf_k, top_k=k)
        pool = max(k * 2, 20)  # Wider than k so overlap cannot shrink the fused list.
        dense = self.dense_search(chunk_set_id, query_embedding, k=pool)
        lexical = self.lexical_search(chunk_set_id, query_text, k=pool)
        return _rrf_fuse(dense, lexical, rrf_k=rrf_k, top_k=k)

    def count_chunks(self, chunk_set_id: str | None = None) -> int:
        """Total chunks in the store, optionally filtered by chunk_set_id."""
        with self.conn.cursor() as cur:
            if chunk_set_id is None:
                cur.execute("SELECT COUNT(*) FROM chunks")
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM chunks WHERE chunk_set_id = %s",
                    (chunk_set_id,),
                )
            row = cur.fetchone()
            return row[0] if row else 0
