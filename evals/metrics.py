"""Pure-function metrics for the eval loop — no I/O, no side effects.

Two families of helpers:
  1. Per-Q&A metrics — `citation_precision_programmatic`, `recall_at_k`.
     Citation precision is always defined (every combination of empty /
     non-empty model & gold citation sets carries a signal). `recall_at_k`
     returns `None` only for out-of-corpus (no gold docs means no retrieval
     ground truth). The aggregator skips `None`s from denominators.
  2. `QAResult` + `RunMetrics` dataclasses — the frozen record shapes
     `evals/run.py` persists per Q&A and aggregates at the end of a run.

Design notes worth remembering:
  - Kept adjacent to the aggregator so the record shape and the code that
    reads it don't drift. If either grows enough to justify a split, move
    the aggregator to `evals/aggregate.py` and keep the dataclasses here.
  - Only `recall_at_k` is exposed as the retrieval headline. An earlier
    `hit_rate_at_k` was multiplicity-weighted over the gold list and either
    collapsed to `recall_at_k` (distinct gold) or inflated it (duplicated
    gold doc_path from multi-page factual Q&As) — misleading either way.
  - Percentiles use `statistics.quantiles(..., method="inclusive")` so
    single-item inputs return that item's value (numpy would raise).
    Small-sample p95 is a rough number by construction — we surface it
    for anomaly detection, not for capacity planning.
  - `total_generate_judge_cost_usd` is exactly what it says: the Anthropic
    generate + judge spend. Voyage embed spend is logged per-call in
    `logs/llm_calls.jsonl` but not aggregated here — the field name is
    honest about that scope so external reports don't misquote it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from src.qa_schema import QAType

# Recall-at-K headline slice. Different from any pipeline's `retriever.top_k` —
# P1..P4 fan out to different top-K values, and reporting a moving-k recall would
# make cross-pipeline comparison meaningless. Baked into the `QAResult.recall_at_5`
# field name because the eval loop only reports one k today; changing
# DEFAULT_K also means renaming that field (and its consumers).
DEFAULT_K = 5


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
    # Always defined — see `citation_precision_programmatic` docstring for
    # how the four (empty/non-empty × model/gold) combinations map to a value.
    citation_precision_programmatic: float
    # `float | None` — `None` iff out-of-corpus (no gold docs, so no
    # retrieval ground truth to score against). Skipped records also carry
    # None here (populated by the eval loop's error path).
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
    error: str | None = None


@dataclass(frozen=True)
class RunMetrics:
    """Aggregate view of one eval run — the object `summary.md` renders from.

    `accuracy_by_type` covers only QATypes that had at least one evaluated
    (non-skipped) record; missing types are absent, not 0.0 — a report that
    reads 0.0 accuracy on a type with zero samples would be misleading.

    `k` is the retrieval slice depth for the recall aggregate (fixed at 5
    today; captured explicitly so future P2..P4 runs that vary it are
    unambiguous in the manifest).

    `total_generate_judge_cost_usd` sums Anthropic generate + judge spend
    only. Voyage embed spend is NOT included — see the module docstring.
    """

    n_records: int  # total records loaded from qa.jsonl
    n_evaluated: int  # records with error is None
    n_skipped: int  # records with error is not None
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


# ── Per-Q&A metrics ───────────────────────────────────────────────────


def citation_precision_programmatic(
    model_doc_paths: list[str],
    gold_doc_paths: list[str],
) -> float:
    """Fraction of the model's cited doc_paths that appear in the gold set.

    Every combination of empty / non-empty (model, gold) has a well-defined
    value — none is dropped from the mean:

    - model=[],  gold=[]  → 1.0  (correct: nothing to cite, nothing cited)
    - model=[],  gold=[x] → 0.0  (model failed to cite when gold existed)
    - model=[x], gold=[]  → 0.0  (every model citation is unfounded)
    - otherwise           → |{p in model: p in gold}| / |model|

    Model citations are compared as a multiset over doc_paths — if the
    model cited the same doc_path twice, both count. That's the honest
    reading of "precision of the model's citations."
    """
    if not model_doc_paths:
        return 1.0 if not gold_doc_paths else 0.0
    gold_set = set(gold_doc_paths)
    hits = sum(1 for p in model_doc_paths if p in gold_set)
    return hits / len(model_doc_paths)


def recall_at_k(
    retrieved_doc_paths: list[str],
    gold_doc_paths: list[str],
    k: int = DEFAULT_K,
) -> float | None:
    """`(# distinct gold doc_paths in top-k) / (# distinct gold doc_paths)`.

    Returns `None` for out-of-corpus (denominator would be zero). Distinctness
    is on both sides — a gold set with the same doc_path listed twice still
    counts as one target, and a retrieved list with duplicates gets one hit.

    When `len(gold) > k`, the metric caps at `k / len(gold)` even if every
    top-k slot is a gold hit — the top-k window physically can't cover more
    distinct docs than it has slots. That's the honest reading, not a bug.
    """
    if not gold_doc_paths:
        return None
    gold_set = set(gold_doc_paths)
    top_k = set(retrieved_doc_paths[:k])
    return len(gold_set & top_k) / len(gold_set)


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
    """Approximate percentile (0..100). Returns `None` for empty input.

    For a single element, returns that element's value — small-sample p95 is
    a rough number by construction, and this keeps `summary.md` populated on
    dev runs with only a handful of records.
    """
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    # statistics.quantiles(n=100) returns 99 cut points → index pct-1.
    # `inclusive` method matches the endpoints of the sample rather than
    # projecting beyond them; it's the more intuitive choice for latency.
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
    # Group by QAType; only types with >= 1 non-None verdict get a row —
    # keeps `summary.md` honest about which buckets actually had samples.
    by_type: dict[QAType, list[bool]] = {}
    for r in evaluated:
        if r.judge_answer_correct is None:
            continue
        by_type.setdefault(r.qa_type, []).append(r.judge_answer_correct)
    for qa_type, bs in by_type.items():
        rate = _rate_or_none(bs)
        if rate is not None:
            accuracy_by_type[qa_type] = rate

    # Citation precision — programmatic mean over every evaluated record
    # (the metric is always defined; every combination carries a signal).
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

    # Retrieval recall over in-corpus Q&As (out-of-corpus rows had recall
    # set to None by the per-Q&A helper above).
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
    )


# ── Serialization helpers ─────────────────────────────────────────────


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
        "retrieved_doc_paths_top10": r.retrieved_doc_paths_top10,
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
    }
