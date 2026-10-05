"""Shared pytest fixtures for the test suite.

Currently just the real-Postgres fixtures (`db_dsn`, `clean_store`). They use
a dedicated test database so cleanup can never truncate application data.
Tests skip gracefully when that database isn't available.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from src.pipeline.store import VectorStore

_DEFAULT_TEST_DSN = "postgresql://lexgo:lexgo@localhost:5432/lexgo_test"


def _dsn_from_env() -> str:
    return os.environ.get("TEST_DATABASE_URL", _DEFAULT_TEST_DSN)


def _assert_test_database(dsn: str) -> None:
    """Refuse any DSN that could point cleanup at application data."""
    database = conninfo_to_dict(dsn).get("dbname", "")
    if not database.endswith("_test"):
        raise ValueError(
            f"TEST_DATABASE_URL must target a database whose name ends in '_test'; got {database!r}"
        )


@pytest.fixture
def db_dsn() -> str:
    """Dedicated test Postgres DSN. Skips if the database isn't reachable."""
    dsn = _dsn_from_env()
    try:
        _assert_test_database(dsn)
    except (ValueError, psycopg.ProgrammingError) as exc:
        pytest.fail(str(exc), pytrace=False)
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            pass
    except Exception as e:
        pytest.skip(f"Postgres not reachable at {dsn}: {e}")
    return dsn


@pytest.fixture
def clean_store(db_dsn: str) -> Iterator[VectorStore]:
    """VectorStore against a fresh schema; wipes `chunks` + `documents` on entry.

    A hard TRUNCATE isolates the test from any pre-existing rows (e.g. from
    a prior `make ingest` run).
      - TRUNCATE removes all rows in one shot (faster than DELETE and doesn't
        write per-row WAL).
      - RESTART IDENTITY resets the SERIAL sequences so `documents.id` starts
        at 1 again — makes test assertions on ids stable across runs.
      - CASCADE follows the chunks→documents foreign key; without it the
        TRUNCATE on `documents` would be refused while chunks reference it.
    """
    with VectorStore(db_dsn) as store:
        store.ensure_schema()
        with store.conn.cursor() as cur:
            cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
        store.conn.commit()
        yield store
