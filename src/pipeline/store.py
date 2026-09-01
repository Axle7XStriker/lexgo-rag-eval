"""Postgres+pgvector store for the RAG pipelines.

Design notes:
- Sync psycopg 3. Eval loop is a batch job over 100 Q&As, not a serving
  path — an async pool would be complexity without a payoff.
- pgvector's psycopg adapter registered per-connection so `vector` columns
  become Python lists on read and accept lists on write.
- Schema is applied idempotently by `ensure_schema()` reading schema.sql;
  callers can invoke on every ingest run without special-casing "first
  time" vs "subsequent" runs.
- All chunks tagged with `pipeline` (a P1..P4-derived identifier like
  "p1_fixed_500_50"). A single global HNSW index composes with the
  pipeline filter at query time — cheaper than per-pipeline indexes for
  our scale, and P2/P3/P4 A/B evals are a WHERE change, not DDL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import psycopg
from pgvector.psycopg import register_vector

from src.db import __file__ as _db_pkg_file

SCHEMA_PATH = Path(_db_pkg_file).parent / "schema.sql"
EMBEDDING_DIM = 1024  # voyage-3-large


# ── Typed rows ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DocumentRow:
    """A row destined for the `documents` table (pre-insert form)."""

    source_id: str  # A1..A4 / B1..B5
    doc_path: str  # relative to corpus/, matches Citation.doc_path
    title: str | None
    num_pages: int | None
    content_hash: str  # sha256 of extracted text


@dataclass(frozen=True)
class ChunkRow:
    """A chunk destined for the `chunks` table (pre-insert form)."""

    pipeline: str  # e.g. "p1_fixed_500_50"
    chunk_index: int
    text: str
    num_tokens: int
    page_start: int
    page_end: int
    content_hash: str  # sha256 of chunk text
    embedding: list[float]  # length must equal EMBEDDING_DIM


@dataclass(frozen=True)
class RetrievedChunk:
    """A row returned from `dense_search` — chunk + document context + score."""

    chunk_id: int
    document_id: int
    doc_path: str
    source_id: str
    pipeline: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    score: float  # cosine similarity in [-1, 1] (higher is closer)


# ── Store ─────────────────────────────────────────────────────────────


class VectorStore:
    """Thin sync wrapper around psycopg + pgvector.

    Use as a context manager to guarantee the connection closes:

        with VectorStore(dsn) as store:
            store.ensure_schema()
            ...
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: psycopg.Connection | None = None

    def __enter__(self) -> Self:
        self._conn = psycopg.connect(self._dsn, autocommit=False)
        # register_vector needs the `vector` type to exist, but on a fresh
        # DB nothing has installed it yet — ensure_schema() would, but the
        # caller can't reach it until __enter__ returns. Break the cycle by
        # creating the extension here. Idempotent (IF NOT EXISTS) + cheap.
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.commit()
        register_vector(self._conn)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None:
            raise RuntimeError("VectorStore not opened; use `with VectorStore(dsn) as store: ...`.")
        return self._conn

    # ── Schema ────────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        """Apply schema.sql. Idempotent — every DDL statement is guarded."""
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text())
        self.conn.commit()

    # ── Documents ─────────────────────────────────────────────────────

    def upsert_document(self, doc: DocumentRow) -> int:
        """Insert or refresh a document by `doc_path`; return its id.

        On conflict the row's metadata + content_hash are updated. Callers
        that want to skip work when the content is unchanged should
        compare their computed hash against `content_hash` on the existing
        row before calling.
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
            assert row is not None  # RETURNING guarantees a row on INSERT/UPDATE
            self.conn.commit()
            return int(row[0])

    # ── Chunks ────────────────────────────────────────────────────────

    def upsert_chunks(self, document_id: int, chunks: list[ChunkRow]) -> None:
        """Insert or replace chunks for `(document_id, pipeline)`.

        Bulk insert via `executemany`. On
        `(document_id, pipeline, chunk_index)` conflict the row is
        replaced — re-running ingest for the same doc + pipeline is
        cheap for unchanged chunks and a swap for changed ones.

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
                    (document_id, pipeline, chunk_index, text, num_tokens,
                     page_start, page_end, content_hash, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id, pipeline, chunk_index) DO UPDATE
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
                        c.pipeline,
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
        self.conn.commit()

    # ── Retrieval ─────────────────────────────────────────────────────

    def dense_search(
        self,
        pipeline: str,
        query_embedding: list[float],
        k: int,
    ) -> list[RetrievedChunk]:
        """Top-k cosine-similarity search over chunks of a given pipeline.

        Uses pgvector's `<=>` cosine-distance operator (0 = identical
        direction, 1 = orthogonal, 2 = opposite). Returned `score`
        is `1 - distance` = cosine similarity in [-1, 1] (higher is
        closer), which matches the more common downstream convention.
        """
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"query embedding dim {len(query_embedding)} ≠ expected {EMBEDDING_DIM}"
            )
        with self.conn.cursor() as cur:
            # Explicit ::vector cast: on INSERT the destination column type
            # tells psycopg to adapt the Python list as a vector, but as a
            # bare `%s` parameter the type is inferred as double precision[]
            # and pgvector's <=> operator has no such overload.
            cur.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    d.doc_path,
                    d.source_id,
                    c.pipeline,
                    c.chunk_index,
                    c.text,
                    c.page_start,
                    c.page_end,
                    1 - (c.embedding <=> %s::vector) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.pipeline = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, pipeline, query_embedding, k),
            )
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                chunk_id=int(r[0]),
                document_id=int(r[1]),
                doc_path=str(r[2]),
                source_id=str(r[3]),
                pipeline=str(r[4]),
                chunk_index=int(r[5]),
                text=str(r[6]),
                page_start=int(r[7]),
                page_end=int(r[8]),
                score=float(r[9]),
            )
            for r in rows
        ]

    # ── Smoke / observability helpers ─────────────────────────────────

    def count_chunks(self, pipeline: str | None = None) -> int:
        """Total chunks in the store, optionally filtered by pipeline.

        Used by ingest scripts + the PR-B smoke check to prove the schema
        applied and the DB is queryable.
        """
        with self.conn.cursor() as cur:
            if pipeline is None:
                cur.execute("SELECT COUNT(*) FROM chunks")
            else:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE pipeline = %s", (pipeline,))
            row = cur.fetchone()
            return int(row[0]) if row else 0
