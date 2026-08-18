"""Golden-set QA schema — the contract for 100 hand-authored records.

Every downstream artifact (validator, authoring UI, eval loop, blog table)
reads from this file. The whole eval's credibility rides on these records
being internally consistent, so validation is strict and enforced in one
place.

Storage: `evals/golden/qa.jsonl` — one record per line, hand-authored (LLM
generation prohibited, per CLAUDE.md). Rewrites go through
`save_jsonl_atomic` — never append-mode after the first write — so add/
edit/delete share one code path and a crash can never leave a half-file.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SCHEMA_VERSION = 1


class SourceId(StrEnum):
    A1 = "A1"  # 6.006 lectures 1-12
    A2 = "A2"  # 6.006 recitations 1-12
    A3 = "A3"  # 6.006 problem sets + solutions (PS1-PS4)
    B1 = "B1"  # 6.830 lectures
    B2 = "B2"  # 6.830 papers: storage / indexing / access methods
    B3 = "B3"  # 6.830 papers: query proc / txn / concurrency


SOURCE_LABELS: dict[SourceId, str] = {
    SourceId.A1: "MIT 6.006 — Lecture notes",
    SourceId.A2: "MIT 6.006 — Recitation notes",
    SourceId.A3: "MIT 6.006 — Problem sets + solutions",
    SourceId.B1: "MIT 6.830 — Lecture notes",
    SourceId.B2: "MIT 6.830 — Papers: storage / indexing",
    SourceId.B3: "MIT 6.830 — Papers: query proc / txn / cc",
}


class QAType(StrEnum):
    FACTUAL = "factual"
    CROSS_SOURCE_SYNTHESIS = "cross_source_synthesis"
    SEMANTIC_PARAPHRASE = "semantic_paraphrase"
    OUT_OF_CORPUS = "out_of_corpus"
    ADVERSARIAL = "adversarial"


TARGET_DISTRIBUTION: dict[QAType, int] = {
    QAType.FACTUAL: 40,
    QAType.CROSS_SOURCE_SYNTHESIS: 25,
    QAType.SEMANTIC_PARAPHRASE: 20,
    QAType.OUT_OF_CORPUS: 10,
    QAType.ADVERSARIAL: 5,
}
GOLDEN_TOTAL = sum(TARGET_DISTRIBUTION.values())  # 100

# Single-letter type prefix used in QARecord.id (grep-friendly at a glance).
ID_PREFIX_FOR_TYPE: dict[QAType, str] = {
    QAType.FACTUAL: "f",
    QAType.CROSS_SOURCE_SYNTHESIS: "x",
    QAType.SEMANTIC_PARAPHRASE: "p",
    QAType.OUT_OF_CORPUS: "o",
    QAType.ADVERSARIAL: "a",
}
TYPE_FOR_ID_PREFIX: dict[str, QAType] = {v: k for k, v in ID_PREFIX_FOR_TYPE.items()}


# doc_path is relative to corpus/ and must match the filename pattern used by
# scripts/corpus_manifest.py: <course>/<kind>/<source_id>_<slug>.pdf. The
# source_id is derived from the filename's 2-char prefix (A1 / B3 / ...) — we
# never store it on disk, so there's exactly one source of truth per citation.
DOC_PATH_PATTERN = re.compile(r"^6\.(?:006|830)/[a-z_]+/([AB][123])_[a-z0-9_]+\.pdf$")


class Citation(BaseModel):
    """A single citation: which PDF, where in it, and optionally a supporting quote.

    `source_id` is a derived property (2-char filename prefix of `doc_path`), not
    a stored field — this prevents drift between the fine-grained doc_path and the
    coarse bucket. Persistence only writes {doc_path, page_or_section, quote?}.
    """

    model_config = ConfigDict(extra="forbid")

    doc_path: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Path to the cited PDF, relative to corpus/. E.g. '6.006/lectures/A1_lec03.pdf'."
        ),
    )
    page_or_section: str = Field(min_length=1, max_length=200)
    quote: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _doc_path_format(self) -> Citation:
        if not DOC_PATH_PATTERN.match(self.doc_path):
            raise ValueError(
                f"doc_path {self.doc_path!r} must match "
                f"<course>/<kind>/<source_id>_<slug>.pdf "
                f"(e.g. '6.006/lectures/A1_lec03.pdf')"
            )
        return self

    @cached_property
    def source_id(self) -> SourceId:
        """Auto-derived from the filename prefix of doc_path (e.g. A1_lec03.pdf → A1)."""
        m = DOC_PATH_PATTERN.match(self.doc_path)
        if not m:  # unreachable — the validator would have fired
            raise ValueError(f"cannot derive source_id from {self.doc_path!r}")
        return SourceId(m.group(1))


class QARecord(BaseModel):
    """A single hand-authored golden Q&A record."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    # Type-prefixed slug: f001, x012, p003, o004, a002. Prefix ↔ type is enforced below.
    id: str = Field(pattern=r"^[fxpoa]\d{3}$")
    type: QAType
    question: str = Field(min_length=8, max_length=1000)
    # Char bounds instead of sentence counting — regex sentence splitters are brittle,
    # and "2-4 sentences" is a human judgment made at authoring time. Bounds catch
    # the real failure modes: too-short "answers" and essay-length dumps.
    gold_answer: str = Field(min_length=50, max_length=800)
    gold_citations: list[Citation] = Field(default_factory=list)
    # `sources` is derivable from citations, but stored explicitly so `grep '"B2"'`
    # over qa.jsonl works. Validator enforces exact equality with citation source_ids.
    sources: list[SourceId] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=1000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> QARecord:
        # 1. ID prefix must match type
        expected_prefix = ID_PREFIX_FOR_TYPE[self.type]
        if self.id[0] != expected_prefix:
            raise ValueError(
                f"id {self.id!r} prefix {self.id[0]!r} does not match type "
                f"{self.type.value!r} (expected prefix {expected_prefix!r})"
            )

        cited_sources = {c.source_id for c in self.gold_citations}
        declared_sources = set(self.sources)

        # 2. out_of_corpus: no citations, no sources
        if self.type == QAType.OUT_OF_CORPUS:
            if self.gold_citations:
                raise ValueError("out_of_corpus questions must have no citations")
            if self.sources:
                raise ValueError("out_of_corpus questions must have no sources")
            return self

        # 3. everything else: at least one citation
        if not self.gold_citations:
            raise ValueError(f"type {self.type.value} requires at least one citation")

        # 4. sources must equal citation source_ids exactly
        if declared_sources != cited_sources:
            raise ValueError(
                f"sources {sorted(s.value for s in declared_sources)} != "
                f"citation sources {sorted(s.value for s in cited_sources)}"
            )

        # 5. cross_source_synthesis: >= 2 distinct sources
        if self.type == QAType.CROSS_SOURCE_SYNTHESIS and len(cited_sources) < 2:
            raise ValueError(
                "cross_source_synthesis requires >= 2 distinct source_ids in citations"
            )

        return self


# ── JSONL persistence ─────────────────────────────────────────────────


class LineError(NamedTuple):
    line_no: int  # 1-indexed
    raw: str
    error: str


def load_jsonl(path: Path) -> tuple[list[QARecord], list[LineError]]:
    """Parse qa.jsonl into records and per-line errors.

    Empty lines are skipped silently. Malformed JSON and schema violations
    each yield one LineError with the 1-indexed line number.
    """
    records: list[QARecord] = []
    errors: list[LineError] = []
    if not path.exists():
        return records, errors

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                errors.append(LineError(line_no, raw.rstrip("\n"), f"malformed JSON: {e}"))
                continue
            try:
                records.append(QARecord.model_validate(obj))
            except ValidationError as e:
                errors.append(LineError(line_no, raw.rstrip("\n"), f"schema error: {e}"))
    return records, errors


def save_jsonl_atomic(path: Path, records: list[QARecord], *, backup: bool = True) -> None:
    """Atomically rewrite path with `records`, one JSON object per line.

    Writes to <path>.tmp, fsyncs, then os.replaces — a crash mid-write can
    never leave a half-file. If `backup` and path already exists, copies
    the old file to <path>.bak first (cheap paranoia against a schema bug
    destroying 100 hand-authored records).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_bytes(path.read_bytes())

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json())
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def next_id_for_type(records: list[QARecord], qa_type: QAType) -> str:
    """Return the next available ID for a given type, e.g. 'f042'.

    Reads from the freshly-loaded records list. Numbering per-type independently.
    """
    prefix = ID_PREFIX_FOR_TYPE[qa_type]
    max_n = 0
    for r in records:
        if r.id.startswith(prefix):
            try:
                max_n = max(max_n, int(r.id[1:]))
            except ValueError:
                continue
    return f"{prefix}{max_n + 1:03d}"


def counts_by_type(records: list[QARecord]) -> dict[QAType, int]:
    """Counts of records per type, with 0s for missing types."""
    counter = Counter(r.type for r in records)
    return {t: counter.get(t, 0) for t in QAType}
