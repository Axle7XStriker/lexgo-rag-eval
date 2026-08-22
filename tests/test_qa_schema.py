"""Schema-invariant tests. Small, but every one enforces a rule that would
cost 100 re-authorings if broken — treat as load-bearing.

Grouped into pytest classes by concern:
  - TestCitation: doc_path format, source_id derivation, extra-field rejection
  - TestQARecordInvariants: cross-field rules on QARecord (id prefix, per-type shape)
  - TestQARecordDerivedSources: `sources` computed property
  - TestPersistence: load/save JSONL round-trip, atomic backup
  - TestHelpers: next_id_for_type, distribution + enum coverage
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.qa_schema import (
    GOLDEN_TOTAL,
    TARGET_DISTRIBUTION,
    Citation,
    QARecord,
    QAType,
    SourceId,
    load_jsonl,
    next_id_for_type,
    save_jsonl_atomic,
)

# ── Fixture helpers ───────────────────────────────────────────────────
# Kept as functions (not @pytest.fixture) so tests can mutate the dict
# before validation to exercise negative paths.


def _valid_factual(id_: str = "f001") -> dict:
    return {
        "id": id_,
        "type": QAType.FACTUAL.value,
        "question": "What is the worst-case complexity of merge sort?",
        "gold_answer": (
            "Merge sort has worst-case time complexity O(n log n) because it always "
            "recursively splits the input in half and merges in linear time."
        ),
        "gold_citations": [
            {"doc_path": "6.006/lectures/A1_lec03.pdf", "page_or_section": "slide 12"}
        ],
    }


def _valid_cross_source() -> dict:
    return {
        "id": "x001",
        "type": QAType.CROSS_SOURCE_SYNTHESIS.value,
        "question": (
            "How does the join-cost derivation in the quiz solution align with the "
            "cost model the Selinger lecture introduces?"
        ),
        "gold_answer": (
            "The quiz solution applies the same selectivity-based estimation the "
            "Selinger lecture presents — cardinality of each input drives the join "
            "cost, and plan choice follows from comparing those estimates."
        ),
        "gold_citations": [
            {"doc_path": "6.830/lectures/B1_lec09_selinger.pdf", "page_or_section": "§3"},
            {"doc_path": "6.830/exams/B2_quiz01_sol.pdf", "page_or_section": "Q4"},
        ],
    }


def _valid_out_of_corpus() -> dict:
    return {
        "id": "o001",
        "type": QAType.OUT_OF_CORPUS.value,
        "question": "What is the amortized cost of insertions into a Fibonacci heap?",
        "gold_answer": (
            "This answer is not present in the corpus. Fibonacci heaps are covered "
            "elsewhere in 6.854, not in the 6.006 lecture subset used here."
        ),
        "gold_citations": [],
    }


# ─────────────────────────────────────────────────────────────────────
class TestCitation:
    """Citation validates its doc_path, derives source_id, and forbids extras."""

    def test_doc_path_format_enforced(self) -> None:
        # Wrong course prefix
        with pytest.raises(ValidationError, match="doc_path"):
            Citation.model_validate(
                {"doc_path": "6.999/lectures/A1_lec01.pdf", "page_or_section": "x"}
            )
        # Missing source_id prefix on the filename
        with pytest.raises(ValidationError, match="doc_path"):
            Citation.model_validate(
                {"doc_path": "6.006/lectures/lec01.pdf", "page_or_section": "x"}
            )
        # Bogus source_id prefix on the filename
        with pytest.raises(ValidationError, match="doc_path"):
            Citation.model_validate(
                {"doc_path": "6.006/lectures/Z9_lec01.pdf", "page_or_section": "x"}
            )

    def test_source_id_auto_derived(self) -> None:
        c = Citation(doc_path="6.006/lectures/A1_lec03.pdf", page_or_section="slide 12")
        assert c.source_id == SourceId.A1

    def test_source_id_not_serialized(self) -> None:
        c = Citation(doc_path="6.006/lectures/A1_lec03.pdf", page_or_section="slide 12")
        dumped = c.model_dump()
        # source_id is derived — the disk representation is doc_path only.
        assert "source_id" not in dumped
        assert dumped["doc_path"] == "6.006/lectures/A1_lec03.pdf"

    def test_rejects_explicit_source_id(self) -> None:
        # extra="forbid" catches attempts to override the derived source_id.
        with pytest.raises(ValidationError):
            Citation.model_validate(
                {
                    "doc_path": "6.006/lectures/A1_lec03.pdf",
                    "page_or_section": "x",
                    "source_id": "A1",
                }
            )


# ─────────────────────────────────────────────────────────────────────
class TestQARecordInvariants:
    """QARecord's cross-field rules: id-prefix ↔ type + per-type citation shape."""

    @pytest.mark.parametrize(
        "factory",
        [_valid_factual, _valid_cross_source, _valid_out_of_corpus],
    )
    def test_valid_records_parse(self, factory) -> None:
        QARecord.model_validate(factory())

    def test_id_prefix_must_match_type(self) -> None:
        data = _valid_factual(id_="p001")  # 'p' prefix, but type=factual
        with pytest.raises(ValidationError, match="does not match type"):
            QARecord.model_validate(data)

    def test_out_of_corpus_rejects_citations(self) -> None:
        data = _valid_out_of_corpus()
        data["gold_citations"] = [
            {"doc_path": "6.006/lectures/A1_lec03.pdf", "page_or_section": "x"}
        ]
        with pytest.raises(ValidationError, match="out_of_corpus"):
            QARecord.model_validate(data)

    def test_cross_source_requires_two_distinct_sources(self) -> None:
        data = _valid_cross_source()
        # One citation → only one source_id bucket → violates cross_source_synthesis.
        data["gold_citations"] = [
            {"doc_path": "6.830/lectures/B1_lec04.pdf", "page_or_section": "Lec 4"}
        ]
        with pytest.raises(ValidationError, match="cross_source_synthesis requires"):
            QARecord.model_validate(data)

    def test_non_out_of_corpus_requires_citation(self) -> None:
        data = _valid_factual()
        data["gold_citations"] = []
        with pytest.raises(ValidationError, match="requires at least one citation"):
            QARecord.model_validate(data)

    def test_gold_answer_min_length(self) -> None:
        data = _valid_factual()
        data["gold_answer"] = "too short"  # < 50 chars
        with pytest.raises(ValidationError):
            QARecord.model_validate(data)

    def test_gold_answer_at_boundary_ok(self) -> None:
        data = _valid_factual()
        data["gold_answer"] = "x" * 60
        QARecord.model_validate(data)

    def test_rejects_explicit_sources(self) -> None:
        # `sources` is derived, not a field — extra="forbid" catches it.
        data = _valid_factual()
        data["sources"] = ["A1"]
        with pytest.raises(ValidationError):
            QARecord.model_validate(data)


# ─────────────────────────────────────────────────────────────────────
class TestQARecordDerivedSources:
    """QARecord.sources is a derived @cached_property, not a stored field."""

    def test_single_source(self) -> None:
        r = QARecord.model_validate(_valid_factual())
        assert r.sources == [SourceId.A1]

    def test_multiple_distinct_sources_sorted(self) -> None:
        r = QARecord.model_validate(_valid_cross_source())
        assert r.sources == [SourceId.B1, SourceId.B2]

    def test_out_of_corpus_has_empty_sources(self) -> None:
        r = QARecord.model_validate(_valid_out_of_corpus())
        assert r.sources == []

    def test_sources_not_serialized(self) -> None:
        r = QARecord.model_validate(_valid_factual())
        # `sources` must NOT appear on disk — grep-by-bucket uses `doc_path` instead.
        assert "sources" not in r.model_dump()

    def test_sources_recomputed_on_copy_with_new_citations(self) -> None:
        # Regression guard: cached_property on a Pydantic v2 model must NOT carry
        # a stale cached value across model_copy(update=...). If this ever fails,
        # switch `sources` to @property (recompute-on-access) — cost is negligible.
        r = QARecord.model_validate(_valid_cross_source())
        assert r.sources == [SourceId.B1, SourceId.B2]  # warm the cache
        r2 = r.model_copy(
            update={
                "gold_citations": [
                    Citation(
                        doc_path="6.006/lectures/A1_lec03.pdf",
                        page_or_section="slide 12",
                    )
                ]
            }
        )
        assert r2.sources == [SourceId.A1], (
            f"expected [A1] after model_copy with A1-only citations, got {r2.sources}"
        )


# ─────────────────────────────────────────────────────────────────────
class TestPersistence:
    """JSONL load/save round-trip, error reporting, and atomic backup."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "qa.jsonl"
        r1 = QARecord.model_validate(_valid_factual("f001"))
        r2 = QARecord.model_validate(_valid_factual("f002"))
        save_jsonl_atomic(path, [r1, r2], backup=False)
        loaded, errors = load_jsonl(path)
        assert not errors
        assert [r.id for r in loaded] == ["f001", "f002"]

    def test_load_reports_line_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "qa.jsonl"
        good = json.dumps(_valid_factual("f001"))
        bad_json = "{ not valid json"
        schema_invalid = json.dumps({**_valid_factual("f002"), "type": "not_a_type"})
        path.write_text(f"{good}\n{bad_json}\n{schema_invalid}\n")
        records, errors = load_jsonl(path)
        assert len(records) == 1
        assert len(errors) == 2
        assert errors[0].line_no == 2
        assert errors[1].line_no == 3

    def test_backup_written_on_overwrite(self, tmp_path: Path) -> None:
        path = tmp_path / "qa.jsonl"
        r1 = QARecord.model_validate(_valid_factual("f001"))
        save_jsonl_atomic(path, [r1], backup=False)
        save_jsonl_atomic(path, [r1], backup=True)
        assert path.with_suffix(".jsonl.bak").exists()


# ─────────────────────────────────────────────────────────────────────
class TestHelpers:
    """next_id_for_type + module-level invariants (distribution, enum coverage)."""

    def test_next_id_for_type_gap_ok(self) -> None:
        r1 = QARecord.model_validate(_valid_factual("f001"))
        r2 = QARecord.model_validate(_valid_factual("f003"))  # gap OK — we take max+1
        assert next_id_for_type([r1, r2], QAType.FACTUAL) == "f004"

    def test_next_id_for_empty_type_starts_at_001(self) -> None:
        r1 = QARecord.model_validate(_valid_factual("f001"))
        assert next_id_for_type([r1], QAType.ADVERSARIAL) == "a001"

    def test_target_distribution_sums_to_100(self) -> None:
        assert sum(TARGET_DISTRIBUTION.values()) == GOLDEN_TOTAL == 100

    def test_source_id_covers_all_six(self) -> None:
        assert {s.value for s in SourceId} == {"A1", "A2", "A3", "B1", "B2", "B3"}
