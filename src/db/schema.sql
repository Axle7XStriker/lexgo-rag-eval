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

-- Chunks are tagged by `chunk_set_id` — an identifier derived from a
-- pipeline's `ChunkerConfig` only (see `PipelineConfig.chunk_set_id`
-- in src/pipeline/pipeline_config.py). Pipelines whose chunker configs
-- are equal share one physical copy of chunks + embeddings. An A/B eval
-- across pipelines filters on the chunk set each pipeline resolves to —
-- a WHERE filter change, not DDL.
-- Embedding dim is 1024 to match Voyage voyage-3-large.
--
-- Guarded with IF NOT EXISTS + UNIQUE constraint so re-apply is a no-op.
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL    PRIMARY KEY,
    document_id   BIGINT       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_set_id  TEXT         NOT NULL,
    chunk_index   INT          NOT NULL,
    text          TEXT         NOT NULL,
    -- Full-text search column for the P3/P4 hybrid retrievers.
    -- Postgres materializes the tsvector from `text` at insert time, so
    -- ingest can continue using the standard INSERT INTO chunks (...) path.
    -- `english` uses the standard stemmer and stopword dictionary.
    text_tsv      tsvector     GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    num_tokens    INT          NOT NULL,
    page_start    INT          NOT NULL,
    page_end      INT          NOT NULL,
    -- sha256 of `text`; upsert compares against this before overwriting.
    content_hash  TEXT         NOT NULL,
    embedding     vector(1024) NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, chunk_set_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_chunk_set_id_idx ON chunks (chunk_set_id);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx  ON chunks (document_id);

-- One global HNSW index for cosine similarity. Queries compose the
-- WHERE chunk_set_id = $1 filter with the index scan — cheaper than
-- one HNSW index per chunk set given the modest per-set cardinality
-- (a few thousand chunks per set in the P1..P4 evals).
--
-- Guarded with IF NOT EXISTS so re-apply is a no-op.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- GIN is the standard index type for tsvector: fast lookups by lexeme,
-- slower writes than GiST but our write path is a batch ingest, not
-- online serving. Composes with WHERE chunk_set_id = $1 via bitmap-scan,
-- same shape as the HNSW index above.
CREATE INDEX IF NOT EXISTS chunks_text_tsv_gin
    ON chunks USING gin (text_tsv);
