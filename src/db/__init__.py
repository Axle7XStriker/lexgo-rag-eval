"""Postgres+pgvector schema for the RAG pipeline.

The schema itself lives in schema.sql — kept as raw SQL (not ORM classes)
because there is exactly one code path that reads it and the SQL is more
scannable than any DSL rendering of it.
"""
