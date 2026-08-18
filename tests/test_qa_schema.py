"""Schema-invariant tests. Small, but every one enforces a rule that would
cost 100 re-authorings if broken — treat as load-bearing.
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
        "sources": ["A1"],
    }


def _valid_cross_source() -> dict:
    return {
        "id": "x001",
        "type": QAType.CROSS_SOURCE_SYNTHESIS.value,
        "question": (
            "How does the buffer management approach in Chou-DeWitt compare to what "
            "the lecture recommends?"
        ),
        "gold_answer": (
            "Chou-DeWitt argues query-plan-aware buffer management outperforms LRU on "
            "certain workloads, matching the lecture's point that access-pattern hints "
            "beat generic replacement policies."
        ),
        "gold_citations": [
            {"doc_path": "6.830/lectures/B1_lec04.pdf", "page_or_section": "§2"},
            {"doc_path": "6.830/papers/B2_chou_dewitt_buffer.pdf", "page_or_section": "§3.2"},
        ],
        "sources": ["B1", "B2"],
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
        "sources": [],
    }


# ── Positive: each type parses when constructed correctly ─────────────


@pytest.mark.parametrize(
    "factory",
    [_valid_factual, _valid_cross_source, _valid_out_of_corpus],
)
def test_valid_records_parse(factory) -> None:
    QARecord.model_validate(factory())


# ── Negative: each invariant fires ────────────────────────────────────


def test_id_prefix_must_match_type() -> None:
    data = _valid_factual(id_="p001")  # 'p' prefix, but type=factual
    with pytest.raises(ValidationError, match="does not match type"):
        QARecord.model_validate(data)


def test_out_of_corpus_rejects_citations() -> None:
    data = _valid_out_of_corpus()
    data["gold_citations"] = [{"doc_path": "6.006/lectures/A1_lec03.pdf", "page_or_section": "x"}]
    with pytest.raises(ValidationError, match="out_of_corpus"):
        QARecord.model_validate(data)


def test_out_of_corpus_rejects_sources() -> None:
    data = _valid_out_of_corpus()
    data["sources"] = ["A1"]
    with pytest.raises(ValidationError, match="out_of_corpus"):
        QARecord.model_validate(data)


def test_cross_source_requires_two_distinct_sources() -> None:
    data = _valid_cross_source()
    data["gold_citations"] = [
        {"doc_path": "6.830/lectures/B1_lec04.pdf", "page_or_section": "Lec 4"}
    ]
    data["sources"] = ["B1"]
    with pytest.raises(ValidationError, match="cross_source_synthesis requires"):
        QARecord.model_validate(data)


def test_sources_must_equal_citation_source_ids() -> None:
    data = _valid_factual()
    data["sources"] = ["A2"]  # citations still cite A1
    with pytest.raises(ValidationError, match="citation sources"):
        QARecord.model_validate(data)


def test_non_out_of_corpus_requires_citation() -> None:
    data = _valid_factual()
    data["gold_citations"] = []
    data["sources"] = []
    with pytest.raises(ValidationError, match="requires at least one citation"):
        QARecord.model_validate(data)


def test_gold_answer_min_length() -> None:
    data = _valid_factual()
    data["gold_answer"] = "too short"  # < 50 chars
    with pytest.raises(ValidationError):
        QARecord.model_validate(data)


def test_gold_answer_at_boundary_ok() -> None:
    data = _valid_factual()
    data["gold_answer"] = "x" * 60
    QARecord.model_validate(data)


def test_citation_doc_path_format_enforced() -> None:
    # Wrong course prefix
    with pytest.raises(ValidationError, match="doc_path"):
        Citation.model_validate({"doc_path": "6.999/lectures/A1_lec01.pdf", "page_or_section": "x"})
    # Missing source_id prefix on the filename
    with pytest.raises(ValidationError, match="doc_path"):
        Citation.model_validate({"doc_path": "6.006/lectures/lec01.pdf", "page_or_section": "x"})
    # Bogus source_id prefix on the filename
    with pytest.raises(ValidationError, match="doc_path"):
        Citation.model_validate({"doc_path": "6.006/lectures/Z9_lec01.pdf", "page_or_section": "x"})


def test_citation_source_id_auto_derived() -> None:
    c = Citation(doc_path="6.006/lectures/A1_lec03.pdf", page_or_section="slide 12")
    assert c.source_id == SourceId.A1
    # source_id must NOT serialize — the disk representation is doc_path only.
    dumped = c.model_dump()
    assert "source_id" not in dumped
    assert dumped["doc_path"] == "6.006/lectures/A1_lec03.pdf"


def test_citation_rejects_explicit_source_id() -> None:
    # extra="forbid" on Citation means source_id (which is derived) cannot be passed in.
    with pytest.raises(ValidationError):
        Citation.model_validate(
            {
                "doc_path": "6.006/lectures/A1_lec03.pdf",
                "page_or_section": "x",
                "source_id": "A1",
            }
        )


# ── Persistence + helpers ─────────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    r1 = QARecord.model_validate(_valid_factual("f001"))
    r2 = QARecord.model_validate(_valid_factual("f002"))
    save_jsonl_atomic(path, [r1, r2], backup=False)
    loaded, errors = load_jsonl(path)
    assert not errors
    assert [r.id for r in loaded] == ["f001", "f002"]


def test_load_reports_line_errors(tmp_path: Path) -> None:
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


def test_next_id_for_type() -> None:
    r1 = QARecord.model_validate(_valid_factual("f001"))
    r2 = QARecord.model_validate(_valid_factual("f003"))  # gap OK, we take max+1
    assert next_id_for_type([r1, r2], QAType.FACTUAL) == "f004"
    assert next_id_for_type([r1, r2], QAType.ADVERSARIAL) == "a001"


def test_backup_written_on_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    r1 = QARecord.model_validate(_valid_factual("f001"))
    save_jsonl_atomic(path, [r1], backup=False)
    save_jsonl_atomic(path, [r1], backup=True)
    assert path.with_suffix(".jsonl.bak").exists()


def test_target_distribution_sums_to_100() -> None:
    assert sum(TARGET_DISTRIBUTION.values()) == GOLDEN_TOTAL == 100


def test_source_id_covers_all_six() -> None:
    assert {s.value for s in SourceId} == {"A1", "A2", "A3", "B1", "B2", "B3"}
