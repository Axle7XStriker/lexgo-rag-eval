"""Metric + aggregate tests

Covers:
    - Citation precision
    - Recall@k
    - Out-of-corpus null semantics
    - Skipped records
"""

from __future__ import annotations

import pytest

from evals.metrics import (
    GoldPage,
    PageSpan,
    QAResult,
    aggregate,
    citation_precision_programmatic,
    gold_pages,
    parse_page_spec,
    recall_at_k,
)
from src.qa_schema import QAType


def _span(doc: str, start: int, end: int | None = None) -> PageSpan:
    return PageSpan(doc, start, end if end is not None else start)


def _gold(*pages: tuple[str, str]) -> set[GoldPage]:
    return gold_pages(list(pages))


class TestCitationPrecision:
    def test_full_overlap(self) -> None:
        prec = citation_precision_programmatic(
            [_span("a", 1), _span("b", 2, 4)],
            _gold(("a", "1"), ("b", "3"), ("c", "9")),
        )
        assert prec == 1.0

    def test_right_file_wrong_page_misses(self) -> None:
        prec = citation_precision_programmatic(
            [_span("a", 1), _span("a", 8), _span("b", 2), _span("z", 1)],
            _gold(("a", "1"), ("b", "9")),
        )
        assert prec == pytest.approx(0.25)

    def test_no_overlap(self) -> None:
        prec = citation_precision_programmatic(
            [_span("x", 1), _span("y", 2)],
            _gold(("a", "1"), ("b", "2")),
        )
        assert prec == 0.0

    def test_both_empty_perfect(self) -> None:
        assert citation_precision_programmatic([], set()) == 1.0

    def test_model_empty_gold_present_zero(self) -> None:
        assert citation_precision_programmatic([], _gold(("a", "1"))) == 0.0

    def test_zero_gold_cites_zero_precision(self) -> None:
        assert citation_precision_programmatic([_span("a", 1)], set()) == 0.0

    def test_duplicate_model_cites_counted(self) -> None:
        prec = citation_precision_programmatic(
            [_span("a", 1), _span("a", 1), _span("b", 2)],
            _gold(("a", "1")),
        )
        assert prec == pytest.approx(2 / 3)


class TestRecallAtK:
    def test_span_covers_every_page_in_range(self) -> None:
        r = recall_at_k(
            [_span("a", 1, 3)],
            _gold(("a", "1,3"), ("a", "2")),
        )
        assert r == 1.0

    def test_half_gold_present(self) -> None:
        r = recall_at_k(
            [_span("a", 1), _span("x", 1), _span("y", 1), _span("z", 1), _span("w", 1)],
            _gold(("a", "1"), ("b", "2")),
        )
        assert r == 0.5

    def test_no_gold_present(self) -> None:
        r = recall_at_k(
            [_span("a", 9), _span("b", 9)],
            _gold(("a", "1"), ("b", "2")),
        )
        assert r == 0.0

    def test_out_of_corpus_returns_none(self) -> None:
        assert recall_at_k([_span("a", 1)], set()) is None

    def test_distinct_denominator(self) -> None:
        r = recall_at_k(
            [_span("a", 1), _span("a", 2)],
            _gold(("a", "1"), ("a", "1,2")),
        )
        assert r == 1.0

    def test_window_ignores_spans_past_k(self) -> None:
        r = recall_at_k(
            [
                _span("a", 1),
                _span("a", 2),
                _span("a", 3),
                _span("a", 4),
                _span("a", 5),
                _span("a", 6),
            ],
            _gold(("a", "6")),
            k=5,
        )
        assert r == 0.0

    def test_hyphen_range_and_comma_list(self) -> None:
        r = recall_at_k(
            [_span("book", 53, 54), _span("book", 10)],
            _gold(("book", "53-54"), ("book", "10,11")),
        )
        assert r == pytest.approx(3 / 4)

    def test_unparsed_spec_stays_a_miss(self) -> None:
        r = recall_at_k([_span("a", 12)], _gold(("a", "slide 12"), ("a", "12")))
        assert r == 0.5

    def test_mixed_spec_keeps_unparsed_token(self) -> None:
        r = recall_at_k([_span("a", 53)], _gold(("a", "53, slide 12")))
        assert r == 0.5

    def test_invalid_tokens_stay_unparsed(self) -> None:
        assert parse_page_spec("0") == ([], ["0"])
        assert parse_page_spec("9-2") == ([], ["9-2"])
        assert parse_page_spec("1-100000000") == ([], ["1-100000000"])
        assert parse_page_spec("53-54") == ([53, 54], [])

    def test_empty_retrieval_zero(self) -> None:
        assert recall_at_k([], _gold(("a", "1"))) == 0.0


def _mk_result(
    *,
    qa_id: str = "f001",
    qa_type: QAType = QAType.FACTUAL,
    gold: list[str] | None = None,
    model: list[str] | None = None,
    retrieved: list[str] | None = None,
    prec: float = 1.0,
    recall5: float | None = 1.0,
    judge_correct: bool | None = True,
    judge_cite_valid: bool | None = True,
    judge_rationale: str | None = "ok",
    latency_ms: float = 1000.0,
    gen_in: int = 100,
    gen_out: int = 20,
    gen_cost: float = 0.002,
    judge_in: int = 80,
    judge_out: int = 15,
    judge_cost: float = 0.001,
    error: str | None = None,
) -> QAResult:
    """Convenience constructor with sensible defaults for aggregate tests."""
    if gold is None:
        gold = ["6.006/lectures/A1_lec03.pdf"]
    if model is None:
        model = list(gold)
    if retrieved is None:
        retrieved = list(gold)
    return QAResult(
        qa_id=qa_id,
        qa_type=qa_type,
        question="q",
        gold_answer="ga",
        gold_citation_doc_paths=gold,
        model_answer="ma",
        model_citation_doc_paths=model,
        retrieved_doc_paths_top10=retrieved,
        citation_precision_programmatic=prec,
        recall_at_5=recall5,
        judge_answer_correct=judge_correct,
        judge_citations_semantically_valid=judge_cite_valid,
        judge_rationale=judge_rationale,
        latency_ms=latency_ms,
        generate_input_tokens=gen_in,
        generate_output_tokens=gen_out,
        generate_cost_usd=gen_cost,
        judge_input_tokens=judge_in,
        judge_output_tokens=judge_out,
        judge_cost_usd=judge_cost,
        error=error,
    )


class TestAggregate:
    def test_empty_input(self) -> None:
        m = aggregate([])
        assert m.n_records == 0
        assert m.n_evaluated == 0
        assert m.n_skipped == 0
        assert m.accuracy_overall is None
        assert m.accuracy_by_type == {}
        assert m.citation_precision_programmatic_mean is None
        assert m.mean_recall_at_k is None
        assert m.p50_latency_ms is None
        assert m.p95_latency_ms is None
        assert m.total_generate_judge_cost_usd == 0.0

    def test_all_correct(self) -> None:
        m = aggregate([_mk_result(qa_id=f"f{i:03d}") for i in range(4)])
        assert m.accuracy_overall == 1.0
        assert m.n_evaluated == 4
        assert m.n_skipped == 0

    def test_mixed_correctness_and_type_breakdown(self) -> None:
        results = [
            _mk_result(qa_id="f001", qa_type=QAType.FACTUAL, judge_correct=True),
            _mk_result(qa_id="f002", qa_type=QAType.FACTUAL, judge_correct=False),
            _mk_result(qa_id="p001", qa_type=QAType.SEMANTIC_PARAPHRASE, judge_correct=True),
        ]
        m = aggregate(results)
        assert m.accuracy_overall == pytest.approx(2 / 3)
        assert m.accuracy_by_type[QAType.FACTUAL] == 0.5
        assert m.accuracy_by_type[QAType.SEMANTIC_PARAPHRASE] == 1.0

    def test_skipped_excluded_from_denominators(self) -> None:
        results = [
            _mk_result(qa_id="f001", judge_correct=True),
            _mk_result(qa_id="f002", judge_correct=True),
            _mk_result(
                qa_id="f003",
                judge_correct=None,
                judge_cite_valid=None,
                judge_rationale=None,
                prec=0.0,
                recall5=None,
                error="pipeline_failed: RuntimeError: boom",
            ),
        ]
        m = aggregate(results)
        assert m.n_records == 3
        assert m.n_evaluated == 2
        assert m.n_skipped == 1
        assert m.accuracy_overall == 1.0
        assert QAType.FACTUAL in m.accuracy_by_type

    def test_citation_prec_mean_over_every_evaluated_record(self) -> None:
        results = [
            _mk_result(qa_id="f001", prec=1.0),
            _mk_result(qa_id="f002", prec=0.5),
            _mk_result(qa_id="f003", prec=0.0),
        ]
        m = aggregate(results)
        assert m.citation_precision_programmatic_mean == pytest.approx(0.5)

    def test_out_of_corpus_excluded_from_recall(self) -> None:
        results = [
            _mk_result(qa_id="f001", recall5=1.0),
            _mk_result(qa_id="f002", recall5=0.0),
            _mk_result(
                qa_id="o001",
                qa_type=QAType.OUT_OF_CORPUS,
                gold=[],
                model=[],
                retrieved=["x"],
                prec=1.0,
                recall5=None,
            ),
        ]
        m = aggregate(results)
        assert m.mean_recall_at_k == 0.5

    def test_cost_totals_include_skipped(self) -> None:
        results = [
            _mk_result(qa_id="f001", gen_cost=0.010, judge_cost=0.002),
            _mk_result(
                qa_id="f002",
                gen_cost=0.005,
                judge_cost=0.0,
                judge_correct=None,
                error="judge_failed",
            ),
        ]
        m = aggregate(results)
        assert m.total_generate_cost_usd == pytest.approx(0.015)
        assert m.total_judge_cost_usd == pytest.approx(0.002)
        assert m.total_generate_judge_cost_usd == pytest.approx(0.017)

    def test_p50_p95_small_sample(self) -> None:
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
        results = [_mk_result(qa_id=f"f{i:03d}", latency_ms=lat) for i, lat in enumerate(latencies)]
        m = aggregate(results)
        assert m.p50_latency_ms is not None
        assert m.p95_latency_ms is not None
        assert 200 <= m.p50_latency_ms <= 400
        assert 400 <= m.p95_latency_ms <= 500

    def test_p95_single_sample(self) -> None:
        m = aggregate([_mk_result(latency_ms=1234.5)])
        assert m.p50_latency_ms == 1234.5
        assert m.p95_latency_ms == 1234.5

    def test_judge_citation_rate(self) -> None:
        results = [
            _mk_result(qa_id="f001", judge_cite_valid=True),
            _mk_result(qa_id="f002", judge_cite_valid=False),
            _mk_result(qa_id="f003", judge_cite_valid=True),
        ]
        m = aggregate(results)
        assert m.citation_precision_judge_rate == pytest.approx(2 / 3)

    def test_types_with_no_verdicts_absent_from_breakdown(self) -> None:
        results = [
            _mk_result(qa_id="f001", qa_type=QAType.FACTUAL, judge_correct=True),
            _mk_result(
                qa_id="a001",
                qa_type=QAType.ADVERSARIAL,
                judge_correct=None,
                error="pipeline_failed",
            ),
        ]
        m = aggregate(results)
        assert QAType.FACTUAL in m.accuracy_by_type
        assert QAType.ADVERSARIAL not in m.accuracy_by_type


def test_known_textbook_citations_use_pdf_page_index() -> None:
    from pathlib import Path

    import pymupdf

    clrs = Path("corpus/6.006/textbook/A4_clrs_3ed.pdf")
    dms = Path("corpus/6.830/textbook/B5_dms_3ed.pdf")
    if not clrs.exists() or not dms.exists():
        pytest.skip("textbook PDFs are not in this checkout")

    dms_doc = pymupdf.open(dms)
    wal = dms_doc[52].get_text()
    dms_doc.close()
    assert "Write-Ahead Log" in wal
    assert parse_page_spec("53-54") == ([53, 54], [])

    clrs_doc = pymupdf.open(clrs)
    selection = clrs_doc[49].get_text()
    labeled_page = clrs_doc[1068]
    label = labeled_page.get_label()
    np_text = labeled_page.get_text()
    clrs_doc.close()
    assert "selection sort" in selection.lower()
    assert label == "1048"
    assert "NP-Completeness" in np_text
