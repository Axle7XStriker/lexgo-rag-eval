"""Pure-function metrics for the eval loop.

Two families of helpers:
  1. Per-Q&A metrics — `citation_precision_programmatic`, `recall_at_k`.
     Citation precision is always defined (every combination of empty /
     non-empty model & gold citation sets carries a signal). `recall_at_k`
     returns `None` only for out-of-corpus (no gold docs means no retrieval
     ground truth). The aggregator skips `None`s from denominators.
  2. `QAResult` + `RunMetrics` dataclasses — the frozen record shapes
     `evals/run.py` persists per Q&A and aggregates at the end of a run.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from src.qa_schema import QAType

DEFAULT_K = 5

_PAGE = re.compile(r"^\d+$")
_RANGE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_MAX_RANGE_PAGES = 40


@dataclass(frozen=True)
class PageSpan:
    """Inclusive page range on one document.

    `page_start` and `page_end` are 1-based PDF page indexes, the same numbers
    as `PageText.page_number` and a PDF viewer. They are not the printed folio
    in the page footer. A single page has equal ends.
    """

    doc_path: str
    page_start: int
    page_end: int

    def covers(self, page: int) -> bool:
        return self.page_start <= page <= self.page_end


@dataclass(frozen=True)
class GoldPage:
    """One gold page. `page is None` when `spec` did not parse as page numbers."""

    doc_path: str
    page: int | None = None
    spec: str = ""


def parse_page_spec(spec: str) -> tuple[list[int], list[str]]:
    """Split a page spec into PDF page numbers and unparsed tokens.

    `4`, `4,5`, and `53-54` become page numbers. A comma piece that is not a
    page stays in the second list, including `slide 12` inside `53, slide 12`.
    Page 0, a reversed range, and a range wider than `_MAX_RANGE_PAGES` stay
    unparsed too. The function does not raise on those tokens.
    """
    pages: list[int] = []
    unparsed: list[str] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if _PAGE.fullmatch(token):
            number = int(token)
            if number >= 1:
                pages.append(number)
            else:
                unparsed.append(token)
            continue
        match = _RANGE.fullmatch(token)
        if match is None:
            unparsed.append(token)
            continue
        start, end = int(match.group(1)), int(match.group(2))
        width = end - start + 1
        if start < 1 or start > end or width > _MAX_RANGE_PAGES:
            unparsed.append(token)
            continue
        pages.extend(range(start, end + 1))
    return pages, unparsed


def gold_pages(citations: list[tuple[str, str]]) -> set[GoldPage]:
    """Distinct gold pages from `(doc_path, page_or_section)` pairs.

    Parsed pages and unparsed tokens from the same spec both count. An
    unparsed token stays in the recall denominator and is never covered.
    """
    found: set[GoldPage] = set()
    for doc_path, spec in citations:
        pages, unparsed = parse_page_spec(spec)
        if not pages and not unparsed:
            found.add(GoldPage(doc_path, None, spec))
            continue
        found.update(GoldPage(doc_path, page) for page in pages)
        found.update(GoldPage(doc_path, None, token) for token in unparsed)
    return found


def _span_hits(span: PageSpan, pages: set[GoldPage]) -> bool:
    return any(
        span.doc_path == gold.doc_path and gold.page is not None and span.covers(gold.page)
        for gold in pages
    )


@dataclass(frozen=True)
class QAResult:
    """One evaluated Q&A — persisted as one JSON line in `results.jsonl`.

    Fields group into four blocks, in this order:
      1. Identity (qa_id, qa_type, question).
      2. Gold / model surface (answers, citation doc_paths, retrieved
         doc_paths for the top-10 window).
      3. Per-Q&A metrics (programmatic + judge verdicts).
      4. Accounting (latency + per-provider tokens + per-provider cost).

    `error` is non-null when the Q&A was SKIPPED — generator raised after
    retries, judge raised, or judge produced a malformed JSON. Skipped
    records are excluded from every aggregate denominator except `n_skipped`.
    """

    qa_id: str
    qa_type: QAType
    question: str
    gold_answer: str
    gold_citation_doc_paths: list[str]
    model_answer: str
    model_citation_doc_paths: list[str]
    retrieved_doc_paths_top10: list[str]
    citation_precision_programmatic: float
    recall_at_5: float | None
    judge_answer_correct: bool | None
    judge_citations_semantically_valid: bool | None
    judge_rationale: str | None
    latency_ms: float
    generate_input_tokens: int
    generate_output_tokens: int
    generate_cost_usd: float
    judge_input_tokens: int
    judge_output_tokens: int
    judge_cost_usd: float
    rerank_cost_usd: float = 0.0
    error: str | None = None
    model_citation_spans: tuple[PageSpan, ...] = ()
    retrieved_page_spans_top10: tuple[PageSpan, ...] = ()


@dataclass(frozen=True)
class RunMetrics:
    """Aggregate view of one eval run — the object `summary.md` renders from.

    `accuracy_by_type` covers only QATypes that had at least one evaluated
    (non-skipped) record; missing types are absent, not 0.0 — a report that
    reads 0.0 accuracy on a type with zero samples would be misleading.

    `k` is the retrieval slice depth for the recall aggregate.

    `total_generate_judge_cost_usd` sums Anthropic generate + judge spend
    only. Voyage embed spend is NOT included.
    """

    n_records: int
    n_evaluated: int
    n_skipped: int
    accuracy_overall: float | None
    accuracy_by_type: dict[QAType, float] = field(default_factory=dict)
    citation_precision_programmatic_mean: float | None = None
    citation_precision_judge_rate: float | None = None
    k: int = DEFAULT_K
    mean_recall_at_k: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    total_generate_cost_usd: float = 0.0
    total_judge_cost_usd: float = 0.0
    total_generate_judge_cost_usd: float = 0.0
    total_rerank_cost_usd: float = 0.0
    total_generate_judge_rerank_cost_usd: float = 0.0




def citation_precision_programmatic(
    model_spans: list[PageSpan],
    gold: set[GoldPage],
) -> float:
    """Fraction of cited chunks whose page range covers a gold page on that file.

    Every combination of empty and non-empty model and gold sets has a
    defined value.

    - model=[], gold=[] → 1.0 (nothing to cite, nothing cited)
    - model=[], gold present → 0.0 (model cited nothing)
    - model present, gold=[] → 0.0 (every citation is unfounded)
    - otherwise → cited spans that cover a gold page, divided by cited spans

    Each cited span counts once, including a repeated span. The right file
    on the wrong page is a miss.
    """
    if not model_spans:
        return 1.0 if not gold else 0.0
    hits = sum(1 for span in model_spans if _span_hits(span, gold))
    return hits / len(model_spans)


def recall_at_k(
    retrieved_spans: list[PageSpan],
    gold: set[GoldPage],
    k: int = DEFAULT_K,
) -> float | None:
    """Distinct gold pages covered by the top-k chunk spans, over distinct gold pages.

    Returns `None` when there is no gold page. A span covers a gold page when
    the file matches and the page sits in the span's inclusive range. One
    span can cover several gold pages. A gold spec that is not a page number
    stays in the denominator and is never covered.
    """
    if not gold:
        return None
    window = retrieved_spans[:k]
    covered = sum(
        1
        for page in gold
        if page.page is not None
        and any(span.doc_path == page.doc_path and span.covers(page.page) for span in window)
    )
    return covered / len(gold)


# ── Aggregation ───────────────────────────────────────────────────────


def _mean_or_none(values: list[float]) -> float | None:
    """Arithmetic mean of `values`, or `None` for an empty list."""
    if not values:
        return None
    return sum(values) / len(values)


def _rate_or_none(bools: list[bool]) -> float | None:
    """True-rate over a list of booleans, or `None` when the list is empty."""
    if not bools:
        return None
    return sum(1 for b in bools if b) / len(bools)


def _percentile(values: list[float], pct: float) -> float | None:
    """Approximate percentile (0..100). Returns `None` for empty input."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    cuts = statistics.quantiles(sorted_vals, n=100, method="inclusive")
    idx = max(0, min(len(cuts) - 1, round(pct) - 1))
    return cuts[idx]


def aggregate(results: list[QAResult], *, k: int = DEFAULT_K) -> RunMetrics:
    """Collapse per-Q&A `QAResult`s into a `RunMetrics` summary.

    Skipped records (error is not None) count toward `n_records` and
    `n_skipped` but are excluded from every other denominator — a run with
    two API failures out of fifty should not have its accuracy dragged
    down by two false verdicts that never actually happened.
    """
    n_records = len(results)
    evaluated = [r for r in results if r.error is None]
    n_evaluated = len(evaluated)
    n_skipped = n_records - n_evaluated

    # Accuracy: mean of `judge_answer_correct` treating True/False as 1/0.
    # `None` (judge failed even though the record was 'evaluated') is
    # excluded from the denominator so a malformed judge reply doesn't
    # skew the overall accuracy in either direction.
    verdicts: list[bool] = [
        r.judge_answer_correct for r in evaluated if r.judge_answer_correct is not None
    ]
    accuracy_overall = _rate_or_none(verdicts)

    accuracy_by_type: dict[QAType, float] = {}
    # Group by QAType; only types with >= 1 non-None verdict get a row.
    by_type: dict[QAType, list[bool]] = {}
    for r in evaluated:
        if r.judge_answer_correct is None:
            continue
        by_type.setdefault(r.qa_type, []).append(r.judge_answer_correct)
    for qa_type, bs in by_type.items():
        rate = _rate_or_none(bs)
        if rate is not None:
            accuracy_by_type[qa_type] = rate

    # Citation precision — programmatic mean over every evaluated record.
    prec_values = [r.citation_precision_programmatic for r in evaluated]
    citation_precision_programmatic_mean = _mean_or_none(prec_values)

    # Citation precision — judge-based (secondary). Rate over Q&As where the
    # judge produced a verdict at all.
    cite_judge_bools: list[bool] = [
        r.judge_citations_semantically_valid
        for r in evaluated
        if r.judge_citations_semantically_valid is not None
    ]
    citation_precision_judge_rate = _rate_or_none(cite_judge_bools)

    # Retrieval recall over in-corpus Q&As.
    recall_values = [r.recall_at_5 for r in evaluated if r.recall_at_5 is not None]
    mean_recall_at_k = _mean_or_none(recall_values)

    latencies = [r.latency_ms for r in evaluated]
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)

    # Cost totals include skipped records too — we paid for whatever partial
    # calls did happen, and a cost overrun should show up regardless of
    # whether the Q&A was scored.
    total_gen = sum(r.generate_cost_usd for r in results)
    total_judge = sum(r.judge_cost_usd for r in results)
    total_rerank = sum(r.rerank_cost_usd for r in results)

    return RunMetrics(
        n_records=n_records,
        n_evaluated=n_evaluated,
        n_skipped=n_skipped,
        accuracy_overall=accuracy_overall,
        accuracy_by_type=accuracy_by_type,
        citation_precision_programmatic_mean=citation_precision_programmatic_mean,
        citation_precision_judge_rate=citation_precision_judge_rate,
        k=k,
        mean_recall_at_k=mean_recall_at_k,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        total_generate_cost_usd=total_gen,
        total_judge_cost_usd=total_judge,
        total_generate_judge_cost_usd=total_gen + total_judge,
        total_rerank_cost_usd=total_rerank,
        total_generate_judge_rerank_cost_usd=total_gen + total_judge + total_rerank,
    )


# ── Serialization helpers ─────────────────────────────────────────────


def _span_dict(span: PageSpan) -> dict[str, Any]:
    return {
        "doc_path": span.doc_path,
        "page_start": span.page_start,
        "page_end": span.page_end,
    }


def qaresult_to_dict(r: QAResult) -> dict[str, Any]:
    """Serialize a `QAResult` to a JSON-safe dict. QAType stored as its str value."""
    return {
        "qa_id": r.qa_id,
        "qa_type": r.qa_type.value,
        "question": r.question,
        "gold_answer": r.gold_answer,
        "gold_citation_doc_paths": r.gold_citation_doc_paths,
        "model_answer": r.model_answer,
        "model_citation_doc_paths": r.model_citation_doc_paths,
        "model_citation_spans": [_span_dict(span) for span in r.model_citation_spans],
        "retrieved_doc_paths_top10": r.retrieved_doc_paths_top10,
        "retrieved_page_spans_top10": [_span_dict(span) for span in r.retrieved_page_spans_top10],
        "citation_precision_programmatic": r.citation_precision_programmatic,
        "recall_at_5": r.recall_at_5,
        "judge_answer_correct": r.judge_answer_correct,
        "judge_citations_semantically_valid": r.judge_citations_semantically_valid,
        "judge_rationale": r.judge_rationale,
        "latency_ms": round(r.latency_ms, 2),
        "generate_input_tokens": r.generate_input_tokens,
        "generate_output_tokens": r.generate_output_tokens,
        "generate_cost_usd": round(r.generate_cost_usd, 6),
        "judge_input_tokens": r.judge_input_tokens,
        "judge_output_tokens": r.judge_output_tokens,
        "judge_cost_usd": round(r.judge_cost_usd, 6),
        "rerank_cost_usd": round(r.rerank_cost_usd, 6),
        "error": r.error,
    }


def runmetrics_to_dict(m: RunMetrics) -> dict[str, Any]:
    """Serialize a `RunMetrics` to a JSON-safe dict for the manifest."""
    return {
        "n_records": m.n_records,
        "n_evaluated": m.n_evaluated,
        "n_skipped": m.n_skipped,
        "accuracy_overall": m.accuracy_overall,
        "accuracy_by_type": {t.value: v for t, v in m.accuracy_by_type.items()},
        "citation_precision_programmatic_mean": m.citation_precision_programmatic_mean,
        "citation_precision_judge_rate": m.citation_precision_judge_rate,
        "k": m.k,
        "mean_recall_at_k": m.mean_recall_at_k,
        "p50_latency_ms": (round(m.p50_latency_ms, 2) if m.p50_latency_ms is not None else None),
        "p95_latency_ms": (round(m.p95_latency_ms, 2) if m.p95_latency_ms is not None else None),
        "total_generate_cost_usd": round(m.total_generate_cost_usd, 6),
        "total_judge_cost_usd": round(m.total_judge_cost_usd, 6),
        "total_generate_judge_cost_usd": round(m.total_generate_judge_cost_usd, 6),
        "total_rerank_cost_usd": round(m.total_rerank_cost_usd, 6),
        "total_generate_judge_rerank_cost_usd": round(m.total_generate_judge_rerank_cost_usd, 6),
    }
