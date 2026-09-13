"""Metric + aggregate tests for evals/metrics.py.

Pure-function tests — no I/O, no fixtures beyond hand-crafted lists.
Each test either encodes an invariant the eval loop depends on, or catches
a regression path (out-of-corpus null semantics, skipped records).
"""

from __future__ import annotations

import pytest

from evals.metrics import (
    QAResult,
    aggregate,
    citation_precision_programmatic,
    hit_rate_at_k,
    recall_at_k,
)
from src.qa_schema import QAType

# ── Per-Q&A metric helpers ────────────────────────────────────────────


class TestCitationPrecision:
    def test_full_overlap(self) -> None:
        prec = citation_precision_programmatic(
            model_doc_paths=["a", "b"],
            gold_doc_paths=["a", "b", "c"],
        )
        assert prec == 1.0

    def test_partial_overlap(self) -> None:
        prec = citation_precision_programmatic(
            model_doc_paths=["a", "x", "y", "z"],
            gold_doc_paths=["a", "b"],
        )
        assert prec == pytest.approx(0.25)

    def test_no_overlap(self) -> None:
        prec = citation_precision_programmatic(
            model_doc_paths=["x", "y"],
            gold_doc_paths=["a", "b"],
        )
        assert prec == 0.0

    def test_both_empty_perfect(self) -> None:
        """No gold to cite and model rightly didn't cite → perfect precision."""
        assert citation_precision_programmatic(model_doc_paths=[], gold_doc_paths=[]) == 1.0

    def test_model_empty_gold_present_zero(self) -> None:
        """Gold cites existed but model cited nothing → 0.0 (missed all)."""
        assert citation_precision_programmatic(model_doc_paths=[], gold_doc_paths=["a"]) == 0.0

    def test_zero_gold_cites_zero_precision(self) -> None:
        """Model cites on an out-of-corpus Q&A → 0.0 (all cites are unfounded)."""
        assert citation_precision_programmatic(model_doc_paths=["a"], gold_doc_paths=[]) == 0.0

    def test_duplicate_model_cites_counted(self) -> None:
        """Multiset semantics: the same doc cited twice counts twice on both sides."""
        prec = citation_precision_programmatic(
            model_doc_paths=["a", "a", "b"],
            gold_doc_paths=["a"],
        )
        assert prec == pytest.approx(2 / 3)


class TestHitRateAtK:
    def test_all_top_k_hit(self) -> None:
        """Every retrieved slot is a gold doc → hit rate 1.0."""
        assert (
            hit_rate_at_k(
                retrieved_doc_paths=["a", "b", "c", "d", "e"],
                gold_doc_paths=["a", "b", "c", "d", "e"],
            )
            == 1.0
        )

    def test_partial_hit(self) -> None:
        """2 of the top 5 slots are gold → 0.4."""
        assert hit_rate_at_k(
            retrieved_doc_paths=["a", "x", "y", "b", "z"],
            gold_doc_paths=["a", "b"],
        ) == pytest.approx(0.4)

    def test_no_hit(self) -> None:
        """No gold in top-k → 0.0."""
        assert (
            hit_rate_at_k(
                retrieved_doc_paths=["a", "b", "c"],
                gold_doc_paths=["z"],
            )
            == 0.0
        )

    def test_gold_past_k_ignored(self) -> None:
        """Gold retrieved past k=5 doesn't count — top-k slice only."""
        assert (
            hit_rate_at_k(
                retrieved_doc_paths=["a", "b", "c", "d", "e", "g"],
                gold_doc_paths=["g"],
                k=5,
            )
            == 0.0
        )

    def test_out_of_corpus_returns_none(self) -> None:
        """No gold citations → metric undefined."""
        assert hit_rate_at_k(retrieved_doc_paths=["a"], gold_doc_paths=[]) is None

    def test_empty_retrieval_zero(self) -> None:
        """Pipeline retrieved nothing but gold exists → 0.0 (no denom crash)."""
        assert hit_rate_at_k(retrieved_doc_paths=[], gold_doc_paths=["a"]) == 0.0


class TestRecallAtK:
    def test_all_gold_present(self) -> None:
        r = recall_at_k(
            retrieved_doc_paths=["a", "b", "c", "d", "e"],
            gold_doc_paths=["a", "b"],
        )
        assert r == 1.0

    def test_half_gold_present(self) -> None:
        r = recall_at_k(
            retrieved_doc_paths=["a", "x", "y", "z", "w"],
            gold_doc_paths=["a", "b"],
        )
        assert r == 0.5

    def test_no_gold_present(self) -> None:
        r = recall_at_k(
            retrieved_doc_paths=["x", "y", "z"],
            gold_doc_paths=["a", "b"],
        )
        assert r == 0.0

    def test_out_of_corpus_returns_none(self) -> None:
        assert recall_at_k(retrieved_doc_paths=["a"], gold_doc_paths=[]) is None

    def test_distinct_denominator(self) -> None:
        """Distinct gold count in denominator — duplicates in gold don't inflate it."""
        r = recall_at_k(
            retrieved_doc_paths=["a", "b", "c", "d", "e"],
            gold_doc_paths=["a", "a", "b"],  # 2 distinct
        )
        assert r == 1.0


# ── Aggregate ────────────────────────────────────────────────────────


def _mk_result(
    *,
    qa_id: str = "f001",
    qa_type: QAType = QAType.FACTUAL,
    gold: list[str] | None = None,
    model: list[str] | None = None,
    retrieved: list[str] | None = None,
    prec: float = 1.0,
    hit5: float | None = 1.0,
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
        hit_rate_at_5=hit5,
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
        """Zero records → zero totals, `None` where the metric is undefined."""
        m = aggregate([])
        assert m.n_records == 0
        assert m.n_evaluated == 0
        assert m.n_skipped == 0
        assert m.accuracy_overall is None
        assert m.accuracy_by_type == {}
        assert m.citation_precision_programmatic_mean is None
        assert m.hit_at_k_rate is None
        assert m.mean_recall_at_k is None
        assert m.p50_latency_ms is None
        assert m.p95_latency_ms is None
        assert m.total_cost_usd == 0.0

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
        """A skipped Q&A counts in n_skipped, is excluded from accuracy denom."""
        results = [
            _mk_result(qa_id="f001", judge_correct=True),
            _mk_result(qa_id="f002", judge_correct=True),
            _mk_result(
                qa_id="f003",
                judge_correct=None,
                judge_cite_valid=None,
                judge_rationale=None,
                prec=0.0,
                hit5=None,
                recall5=None,
                error="pipeline_failed: RuntimeError: boom",
            ),
        ]
        m = aggregate(results)
        assert m.n_records == 3
        assert m.n_evaluated == 2
        assert m.n_skipped == 1
        assert m.accuracy_overall == 1.0  # 2/2, skipped record excluded
        assert QAType.FACTUAL in m.accuracy_by_type

    def test_citation_prec_mean_over_every_evaluated_record(self) -> None:
        """Every evaluated record contributes to the mean — the metric is always defined."""
        results = [
            _mk_result(qa_id="f001", prec=1.0),
            _mk_result(qa_id="f002", prec=0.5),
            _mk_result(qa_id="f003", prec=0.0),
        ]
        m = aggregate(results)
        assert m.citation_precision_programmatic_mean == pytest.approx(0.5)

    def test_out_of_corpus_excluded_from_hit_and_recall(self) -> None:
        results = [
            _mk_result(qa_id="f001", hit5=1.0, recall5=1.0),
            _mk_result(qa_id="f002", hit5=0.0, recall5=0.0),
            _mk_result(
                qa_id="o001",
                qa_type=QAType.OUT_OF_CORPUS,
                gold=[],
                model=[],
                retrieved=["x"],
                prec=1.0,
                hit5=None,
                recall5=None,
            ),
        ]
        m = aggregate(results)
        # 2 in-corpus records → mean of 1.0 and 0.0 → 0.5.
        assert m.hit_at_k_rate == 0.5
        assert m.mean_recall_at_k == 0.5

    def test_cost_totals_include_skipped(self) -> None:
        """Skipped records that made partial calls still contribute to cost."""
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
        assert m.total_cost_usd == pytest.approx(0.017)

    def test_p50_p95_small_sample(self) -> None:
        """Percentiles over a small sample use inclusive quantiles + are bounded by inputs."""
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
        results = [_mk_result(qa_id=f"f{i:03d}", latency_ms=lat) for i, lat in enumerate(latencies)]
        m = aggregate(results)
        assert m.p50_latency_ms is not None
        assert m.p95_latency_ms is not None
        # p50 sits at the median; p95 is at or near the top of the sample.
        assert 200 <= m.p50_latency_ms <= 400
        assert 400 <= m.p95_latency_ms <= 500

    def test_p95_single_sample(self) -> None:
        """One-record run: percentiles return that record's latency (no crash)."""
        m = aggregate([_mk_result(latency_ms=1234.5)])
        assert m.p50_latency_ms == 1234.5
        assert m.p95_latency_ms == 1234.5

    def test_judge_citation_rate(self) -> None:
        """Secondary citation metric — rate of judge-approved citations."""
        results = [
            _mk_result(qa_id="f001", judge_cite_valid=True),
            _mk_result(qa_id="f002", judge_cite_valid=False),
            _mk_result(qa_id="f003", judge_cite_valid=True),
        ]
        m = aggregate(results)
        assert m.citation_precision_judge_rate == pytest.approx(2 / 3)

    def test_types_with_no_verdicts_absent_from_breakdown(self) -> None:
        """A QAType present only in a skipped record should not appear in the breakdown."""
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
