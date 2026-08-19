"""Tests for the CLI validator + its manifest cross-check.

Separate from test_qa_schema — those tests exercise the pydantic model in
isolation; these exercise the eval-run-time invariants (distribution,
duplicate ids, unknown doc_paths).
"""

from __future__ import annotations

from pathlib import Path

from evals.validate_golden import KNOWN_DOC_PATHS, validate_golden
from src.qa_schema import Citation, QARecord, QAType, save_jsonl_atomic

# Pick two real corpus paths from the manifest (any two will do — they're
# guaranteed to exist by construction). Using .pop() on a set is order-nondeterministic;
# sort first for a stable test.
_SORTED_KNOWN = sorted(KNOWN_DOC_PATHS)
_A_PATH = next(p for p in _SORTED_KNOWN if "/A1_" in p or "A1_" in p.split("/")[-1])
_B_PATH = next(p for p in _SORTED_KNOWN if "/B1_" in p or "B1_" in p.split("/")[-1])


def _factual(id_: str, doc_path: str) -> QARecord:
    return QARecord(
        id=id_,
        type=QAType.FACTUAL,
        question="What is the worst-case complexity of merge sort?",
        gold_answer=(
            "Merge sort has worst-case time complexity O(n log n) because it always "
            "recursively splits the input in half and merges in linear time."
        ),
        gold_citations=[Citation(doc_path=doc_path, page_or_section="slide 12")],
    )


def test_validator_clean_report(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    save_jsonl_atomic(path, [_factual("f001", _A_PATH)], backup=False)
    report = validate_golden(path)
    assert report.is_clean
    assert report.total_records == 1
    assert not report.unknown_doc_paths


def test_validator_flags_unknown_doc_path(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    # Well-formatted doc_path that isn't in the manifest.
    fake_but_valid = "6.006/lectures/A1_lec99.pdf"
    assert fake_but_valid not in KNOWN_DOC_PATHS  # sanity
    save_jsonl_atomic(path, [_factual("f001", fake_but_valid)], backup=False)
    report = validate_golden(path)
    assert not report.is_clean
    assert report.unknown_doc_paths == [f"f001 → {fake_but_valid}"]


def test_validator_flags_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    r1 = _factual("f001", _A_PATH)
    r2 = _factual("f001", _B_PATH)
    save_jsonl_atomic(path, [r1, r2], backup=False)
    report = validate_golden(path)
    assert report.duplicate_ids == ["f001"]
    assert not report.is_clean
