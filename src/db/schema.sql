-- Storage schema for canonical documents and reusable chunk sets.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source PDF. `doc_path` is the canonical identity (matches
-- Citation.doc_path in src.qa_schema); `source_id` denormalizes the
-- doc_path prefix for cheap filter queries.
--
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL   PRIMARY KEY,
    source_id     TEXT        NOT NULL,
    doc_path      TEXT        NOT NULL UNIQUE,
    title         TEXT,
    num_pages     INT,
    content_hash  TEXT        NOT NULL, -- sha256 of extracted text
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS documents_source_id_idx ON documents (source_id);

-- Chunks are tagged by `chunk_set_id` — an identifier derived from a
-- pipeline's `ChunkerConfig` only (see `PipelineConfig.chunk_set_id`
-- in src/pipeline/pipeline_config.py). Pipelines whose chunker configs
-- are equal share one physical copy of chunks + embeddings. An A/B eval
-- across pipelines filters on the chunk set each pipeline resolves to —
-- a WHERE filter change, not DDL.
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL    PRIMARY KEY,
    document_id   BIGINT       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_set_id  TEXT         NOT NULL,
    chunk_index   INT          NOT NULL,
    text          TEXT         NOT NULL,
    text_tsv      tsvector     GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    num_tokens    INT          NOT NULL,
    page_start    INT          NOT NULL,
    page_end      INT          NOT NULL,
    content_hash  TEXT         NOT NULL,
    embedding     vector(1024) NOT NULL, -- voyage-3-large output
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, chunk_set_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_chunk_set_id_idx ON chunks (chunk_set_id);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx  ON chunks (document_id);

 -- One index for every chunk set, written in batch.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Lexical leg, also loaded in batch.
CREATE INDEX IF NOT EXISTS chunks_text_tsv_gin
    ON chunks USING gin (text_tsv);
