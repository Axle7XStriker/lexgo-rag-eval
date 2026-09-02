-- lexgo vector store — documents + chunks + HNSW index.

-- pgvector extension. Guarded with IF NOT EXISTS so re-apply is a no-op.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source PDF. `doc_path` is the canonical identity (matches
-- Citation.doc_path in src.qa_schema); `source_id` denormalizes the
-- doc_path prefix for cheap filter queries.
--
-- Guarded with IF NOT EXISTS so re-apply is a no-op on an existing DB.
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL   PRIMARY KEY,
    source_id     TEXT        NOT NULL,
    doc_path      TEXT        NOT NULL UNIQUE,
    title         TEXT,
    num_pages     INT,
    -- sha256 of extracted text; callers compare against this before
    -- deciding whether to re-chunk the doc.
    content_hash  TEXT        NOT NULL,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS documents_source_id_idx ON documents (source_id);

-- All 4 pipelines (P1..P4) coexist in one table, tagged by `pipeline`.
-- An A/B eval across pipelines is a WHERE filter change, not DDL.
-- Embedding dim is 1024 to match Voyage voyage-3-large.
--
-- Guarded with IF NOT EXISTS + UNIQUE constraint so re-apply is a no-op.
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL    PRIMARY KEY,
    document_id   BIGINT       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    pipeline      TEXT         NOT NULL,
    chunk_index   INT          NOT NULL,
    text          TEXT         NOT NULL,
    num_tokens    INT          NOT NULL,
    page_start    INT          NOT NULL,
    page_end      INT          NOT NULL,
    -- sha256 of `text`; upsert compares against this before overwriting.
    content_hash  TEXT         NOT NULL,
    embedding     vector(1024) NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, pipeline, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_pipeline_idx    ON chunks (pipeline);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- One global HNSW index for cosine similarity. Queries compose the
-- WHERE pipeline = $1 filter with the index scan — cheaper than one
-- HNSW index per pipeline given the modest per-pipeline cardinality
-- (a few thousand chunks in the P1..P4 evals).
--
-- Guarded with IF NOT EXISTS so re-apply is a no-op.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
