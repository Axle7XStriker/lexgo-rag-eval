"""P1 eval orchestrator — golden Q&As → pipeline → judge → per-run artifacts.

Sequential loop over the golden set. For each record: run the P1 pipeline to
produce an answer, score citations + retrieval programmatically, call the
judge for accuracy + semantic citation validity, and append a `QAResult`
line to `results.jsonl`. On any exception, skip the record with an `error`
string set and continue — a partial run is more valuable than a crashed one.

CLI::

  python -m evals.run [--pipeline p1] [--log-level LEVEL]

Design notes worth remembering:
  - One `VoyageEmbedder`, one `VectorStore`, one `ClaudeGenerator`, one
    `ClaudeJudge` — constructed once in `main()`, reused across every Q&A.
    Same shape as `scripts/ingest.py` opens one embedder + store for the
    document loop.
  - `--pipeline` only accepts `p1` today. P2/P3/P4 will register their own
    retrieval variants against this same harness; the arg is a switch, not
    a plugin registry, until we actually have >1 pipeline.
  - Malformed golden file is FATAL. Same contract as `make validate`:
    refuse to run against a broken qa.jsonl rather than silently evaluating
    the parseable subset.
  - Per-Q&A skips are RECOVERABLE. Any exception raised from the pipeline
    or the judge (post-retry) becomes an `error` field on the QAResult, is
    logged, and the loop continues. Exit code is always 0; the skip count
    lands in `summary.md` for the operator to notice.
  - The per-run `llm_calls.jsonl` snapshot is built by filtering
    `logs/llm_calls.jsonl` for records whose `run_id` matches this run.
    Simpler than a byte-offset diff and robust to concurrent writers
    (we don't have any today, but the property is free).
  - Async fan-out is deferred (~100 Q&As x ~5s x 2 calls ~= 15 min end-to-end).
    Sequential is fine; the seam to parallelize is one `await gather(...)`
    inside the loop body once we actually need it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.metrics import (
    QAResult,
    RunMetrics,
    aggregate,
    citation_precision_programmatic,
    hit_rate_at_k,
    qaresult_to_dict,
    recall_at_k,
    runmetrics_to_dict,
)
from src.config import get_settings
from src.observability import configure_logging, get_logger
from src.pipeline.chunk import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    PIPELINE_TAG,
)
from src.pipeline.embed import VoyageEmbedder
from src.pipeline.generate import ClaudeGenerator
from src.pipeline.judge import PROMPT_VERSION as JUDGE_PROMPT_VERSION
from src.pipeline.judge import ClaudeJudge
from src.pipeline.query import DEFAULT_TOP_K, answer_question
from src.pipeline.query import PROMPT_VERSION as ANSWER_PROMPT_VERSION
from src.pipeline.store import VectorStore
from src.qa_schema import GOLDEN_TOTAL, QARecord, QAType, counts_by_type, load_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA_PATH = REPO_ROOT / "evals" / "golden" / "qa.jsonl"

# Only `p1` for now; adding p2..p4 is a switch here + a retrieval branch
# inside `_run_one`. The registry stays flat until we actually have >1.
SUPPORTED_PIPELINES: tuple[str, ...] = ("p1",)

# Pipeline column tag persisted per QAResult / manifest so future P2..P4 runs
# don't collide in `summary.md` when they're compared side-by-side.
PIPELINE_LABEL_FOR_ARG: dict[str, str] = {"p1": PIPELINE_TAG}


def _git_sha() -> str:
    """Current git commit SHA, or `"unknown"` on any failure.

    Falls back rather than raising because a captured artifact with
    `git_sha: unknown` is still useful for the operator; a crash on
    `git rev-parse` would kill an otherwise successful eval run.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return proc.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _run_one(
    record: QARecord,
    *,
    embedder: VoyageEmbedder,
    store: VectorStore,
    generator: ClaudeGenerator,
    judge: ClaudeJudge,
    pipeline_arg: str,
    run_id: str,
) -> QAResult:
    """Evaluate one record end-to-end. Never raises — catches → error field.

    Returns a `QAResult` with `error` set to a short string when any step
    raises after its own retries have been exhausted. The caller decides
    what to do with that (log + continue is the current policy).
    """
    gold_doc_paths = [c.doc_path for c in record.gold_citations]
    started = time.perf_counter()

    try:
        pipeline_tag = PIPELINE_LABEL_FOR_ARG[pipeline_arg]
        query_result = answer_question(
            query=record.question,
            embedder=embedder,
            store=store,
            generator=generator,
            pipeline_tag=pipeline_tag,
            top_k=DEFAULT_TOP_K,
            run_id=run_id,
        )
    except Exception as e:
        # Everything from `answer_question` (embed, retrieve, generate) is
        # already retried at the provider client level. A raised exception
        # here is a terminal failure for this Q&A — record it and move on.
        elapsed_ms = (time.perf_counter() - started) * 1000
        return QAResult(
            qa_id=record.id,
            qa_type=record.type,
            question=record.question,
            gold_answer=record.gold_answer,
            gold_citation_doc_paths=gold_doc_paths,
            model_answer="",
            model_citation_doc_paths=[],
            retrieved_doc_paths_top10=[],
            citation_precision_programmatic=citation_precision_programmatic([], gold_doc_paths),
            hit_rate_at_5=None,
            recall_at_5=None,
            judge_answer_correct=None,
            judge_citations_semantically_valid=None,
            judge_rationale=None,
            latency_ms=elapsed_ms,
            generate_input_tokens=0,
            generate_output_tokens=0,
            generate_cost_usd=0.0,
            judge_input_tokens=0,
            judge_output_tokens=0,
            judge_cost_usd=0.0,
            error=f"pipeline_failed: {type(e).__name__}: {e}",
        )

    model_doc_paths = [c.doc_path for c in query_result.citations]
    retrieved_doc_paths = [c.doc_path for c in query_result.retrieved_chunks]
    prec = citation_precision_programmatic(model_doc_paths, gold_doc_paths)
    hit5 = hit_rate_at_k(retrieved_doc_paths, gold_doc_paths, k=5)
    recall5 = recall_at_k(retrieved_doc_paths, gold_doc_paths, k=5)

    try:
        verdict = judge.judge(
            question=record.question,
            gold_answer=record.gold_answer,
            gold_citation_doc_paths=gold_doc_paths,
            model_answer=query_result.answer,
            model_citation_doc_paths=model_doc_paths,
            run_id=run_id,
        )
    except Exception as e:
        # Judge failure is recoverable: the pipeline output is preserved,
        # programmatic metrics are still computed, but accuracy is unknown
        # for this Q&A. The record lands with error set so the operator
        # notices; aggregate metrics drop it from the accuracy denominator.
        elapsed_ms = query_result.latency_ms
        return QAResult(
            qa_id=record.id,
            qa_type=record.type,
            question=record.question,
            gold_answer=record.gold_answer,
            gold_citation_doc_paths=gold_doc_paths,
            model_answer=query_result.answer,
            model_citation_doc_paths=model_doc_paths,
            retrieved_doc_paths_top10=retrieved_doc_paths[:10],
            citation_precision_programmatic=prec,
            hit_rate_at_5=hit5,
            recall_at_5=recall5,
            judge_answer_correct=None,
            judge_citations_semantically_valid=None,
            judge_rationale=None,
            latency_ms=elapsed_ms,
            generate_input_tokens=query_result.tokens_input,
            generate_output_tokens=query_result.tokens_output,
            generate_cost_usd=query_result.cost_usd,
            judge_input_tokens=0,
            judge_output_tokens=0,
            judge_cost_usd=0.0,
            error=f"judge_failed: {type(e).__name__}: {e}",
        )

    return QAResult(
        qa_id=record.id,
        qa_type=record.type,
        question=record.question,
        gold_answer=record.gold_answer,
        gold_citation_doc_paths=gold_doc_paths,
        model_answer=query_result.answer,
        model_citation_doc_paths=model_doc_paths,
        retrieved_doc_paths_top10=retrieved_doc_paths[:10],
        citation_precision_programmatic=prec,
        hit_rate_at_5=hit5,
        recall_at_5=recall5,
        judge_answer_correct=verdict.answer_correct,
        judge_citations_semantically_valid=verdict.citations_semantically_valid,
        judge_rationale=verdict.rationale,
        latency_ms=query_result.latency_ms,
        generate_input_tokens=query_result.tokens_input,
        generate_output_tokens=query_result.tokens_output,
        generate_cost_usd=query_result.cost_usd,
        judge_input_tokens=verdict.input_tokens,
        judge_output_tokens=verdict.output_tokens,
        judge_cost_usd=verdict.cost_usd,
        error=None,
    )


def _snapshot_llm_calls(source: Path, dest: Path, run_id: str) -> int:
    """Copy every `logs/llm_calls.jsonl` record whose `run_id` matches → `dest`.

    Returns the number of records written. Runs even when `source` is
    missing (empty snapshot, 0 records) so the manifest link is always
    valid on read.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    if not source.exists():
        dest.write_text("", encoding="utf-8")
        return 0
    with source.open("r", encoding="utf-8") as fin, dest.open("w", encoding="utf-8") as fout:
        for raw in fin:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                # A corrupted line elsewhere in the log is not our problem —
                # the ingest run's snapshot function has the same tolerance.
                continue
            if rec.get("run_id") == run_id:
                fout.write(stripped + "\n")
                written += 1
    return written


def _format_pct(value: float | None) -> str:
    """Format a 0..1 fraction as `NN.N%`, or `n/a` for None."""
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_ms(value: float | None) -> str:
    """Format a latency in ms as `NNNNms`, or `n/a` for None."""
    if value is None:
        return "n/a"
    return f"{value:.0f}ms"


def _per_type_counts(results: list[QAResult]) -> dict[QAType, int]:
    """Per-QAType count of records with a non-`None` judge verdict.

    Used to populate the `n` column of the per-type accuracy table in
    `summary.md`. Skipped records (error is not None) and evaluated-but-
    judge-failed records (judge_answer_correct is None) are excluded, so
    the count matches the denominator of the per-type accuracy rate.
    """
    counts: dict[QAType, int] = {}
    for r in results:
        if r.error is not None or r.judge_answer_correct is None:
            continue
        counts[r.qa_type] = counts.get(r.qa_type, 0) + 1
    return counts


def _write_summary_md(
    path: Path,
    *,
    manifest: dict[str, Any],
    metrics: RunMetrics,
    results: list[QAResult],
) -> None:
    """Render a human-readable markdown summary alongside `manifest.json`.

    Mirrors the manifest's shape but rendered as tables for a human reader.
    Skip list is enumerated so a partial run tells the operator exactly
    which Q&As did not score and why.
    """
    skipped = [r for r in results if r.error is not None]
    per_type = _per_type_counts(results)

    lines: list[str] = []
    lines.append(f"# Eval run — {manifest['run_id']}")
    lines.append("")
    lines.append(f"- pipeline: `{manifest['pipeline']}`")
    lines.append(f"- git_sha: `{manifest['git_sha']}`")
    lines.append(
        f"- prompt versions: answer=`{manifest['prompt_versions']['answer']}`, "
        f"judge=`{manifest['prompt_versions']['judge']}`"
    )
    lines.append(f"- chat model: `{manifest['config']['chat_model']}`")
    lines.append(f"- judge model: `{manifest['config']['judge_model']}`")
    lines.append(f"- embedding model: `{manifest['config']['embedding_model']}`")
    lines.append(f"- top_k: {manifest['config']['top_k']}")
    lines.append(f"- golden path: `{manifest['golden_set']['path']}`")
    lines.append(f"- golden loaded: {manifest['golden_set']['n_records_loaded']} / {GOLDEN_TOTAL}")
    lines.append(f"- wall_clock: {manifest['totals']['wall_clock_seconds']:.1f}s")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| n_evaluated | {metrics.n_evaluated} |")
    lines.append(f"| n_skipped | {metrics.n_skipped} |")
    lines.append(f"| accuracy (overall) | {_format_pct(metrics.accuracy_overall)} |")
    lines.append(
        f"| citation precision (programmatic mean) | "
        f"{_format_pct(metrics.citation_precision_programmatic_mean)} |"
    )
    lines.append(
        f"| citation precision (judge rate) | "
        f"{_format_pct(metrics.citation_precision_judge_rate)} |"
    )
    lines.append(f"| hit-rate@{metrics.k} | {_format_pct(metrics.hit_at_k_rate)} |")
    lines.append(f"| mean recall@{metrics.k} | {_format_pct(metrics.mean_recall_at_k)} |")
    lines.append(f"| p50 latency | {_format_ms(metrics.p50_latency_ms)} |")
    lines.append(f"| p95 latency | {_format_ms(metrics.p95_latency_ms)} |")
    lines.append(f"| cost — generate | ${metrics.total_generate_cost_usd:.4f} |")
    lines.append(f"| cost — judge | ${metrics.total_judge_cost_usd:.4f} |")
    lines.append(f"| cost — total | ${metrics.total_cost_usd:.4f} |")
    lines.append("")
    lines.append("## Accuracy by QA type")
    lines.append("")
    if not metrics.accuracy_by_type:
        lines.append("_no scored records_")
    else:
        lines.append("| type | accuracy | n |")
        lines.append("| --- | --- | --- |")
        for qa_type in QAType:
            if qa_type in metrics.accuracy_by_type:
                lines.append(
                    f"| {qa_type.value} | "
                    f"{_format_pct(metrics.accuracy_by_type[qa_type])} | "
                    f"{per_type.get(qa_type, 0)} |"
                )
    lines.append("")
    lines.append("## Skipped Q&As")
    lines.append("")
    if not skipped:
        lines.append("_none_")
    else:
        for r in skipped:
            lines.append(f"- `{r.qa_id}` ({r.qa_type.value}): {r.error}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        choices=SUPPORTED_PIPELINES,
        default="p1",
        help="Retrieval pipeline to evaluate. Only 'p1' is wired today; "
        "p2..p4 join this switch when they exist.",
    )
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=DEFAULT_QA_PATH,
        help=f"Path to qa.jsonl (default: {DEFAULT_QA_PATH.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        # Default reads settings.log_level (which honors .env) rather than
        # os.environ directly — otherwise a .env-only LOG_LEVEL override is
        # silently ignored for the eval loop but honored everywhere else.
        "--log-level",
        default=settings.log_level,
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger = get_logger("evals.run")

    # Load + strict-validate the golden file up front — refuse to run against
    # a broken qa.jsonl. Same contract as `make validate`.
    records, per_line_errors = load_jsonl(args.qa_path)
    if per_line_errors:
        print(f"\nCannot evaluate: {args.qa_path} has {len(per_line_errors)} parse/schema errors:")
        for e in per_line_errors:
            first_line = e.error.split("\n", 1)[0]
            print(f"  line {e.line_no}: {first_line}")
        print("\nFix the file (or run `make validate`) and re-run `make eval`.\n")
        return 1
    if not records:
        # Zero-record run is not an error — it's normal during golden-set
        # authoring before any Q&As exist. Print + exit cleanly so `make
        # eval` on a fresh clone doesn't look like a broken step.
        print(f"\n{args.qa_path} has zero records — nothing to evaluate. Exiting.\n")
        return 0

    run_started = datetime.now(UTC)
    run_id = f"eval_{run_started.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = settings.evals_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.md"
    snapshot_path = run_dir / "llm_calls.jsonl"

    logger.info(
        "eval_start",
        run_id=run_id,
        pipeline=args.pipeline,
        n_records=len(records),
        qa_path=str(args.qa_path),
    )

    embedder = VoyageEmbedder(
        api_key=settings.voyage_api_key,
        model=settings.embedding_model,
        log_path=settings.llm_call_log,
    )
    generator = ClaudeGenerator(
        api_key=settings.anthropic_api_key,
        model=settings.chat_model,
        log_path=settings.llm_call_log,
    )
    judge = ClaudeJudge(
        api_key=settings.anthropic_api_key,
        model=settings.judge_model,
        log_path=settings.llm_call_log,
    )

    results: list[QAResult] = []
    wall_started = time.perf_counter()
    # Open the results file up front + write per Q&A so a crash mid-run
    # still leaves a partial artifact the operator can inspect.
    with (
        VectorStore(settings.database_url) as store,
        results_path.open("w", encoding="utf-8") as results_fp,
    ):
        for i, record in enumerate(records, start=1):
            qa_result = _run_one(
                record,
                embedder=embedder,
                store=store,
                generator=generator,
                judge=judge,
                pipeline_arg=args.pipeline,
                run_id=run_id,
            )
            results.append(qa_result)
            results_fp.write(json.dumps(qaresult_to_dict(qa_result)) + "\n")
            results_fp.flush()

            marker = (
                "✗"
                if qa_result.error is not None
                else ("✓" if qa_result.judge_answer_correct else "·")
            )
            verdict_str = (
                "skip"
                if qa_result.error is not None
                else f"correct={str(qa_result.judge_answer_correct).lower()}"
            )
            cite_prec_str = (
                f"{qa_result.citation_precision_programmatic:.2f}"
                if qa_result.citation_precision_programmatic is not None
                else "n/a"
            )
            hit_str = (
                f"{qa_result.hit_rate_at_5:.2f}" if qa_result.hit_rate_at_5 is not None else "n/a"
            )
            total_cost = qa_result.generate_cost_usd + qa_result.judge_cost_usd
            print(
                f"  {marker} [{qa_result.qa_id}] {verdict_str}, "
                f"cite_prec={cite_prec_str}, hit@5={hit_str}, "
                f"${total_cost:.4f}, {qa_result.latency_ms / 1000:.1f}s  "
                f"[{i}/{len(records)}]"
            )

    wall_seconds = time.perf_counter() - wall_started
    metrics = aggregate(results)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "pipeline": PIPELINE_LABEL_FOR_ARG[args.pipeline],
        "git_sha": _git_sha(),
        "prompt_versions": {
            "answer": ANSWER_PROMPT_VERSION,
            "judge": JUDGE_PROMPT_VERSION,
        },
        "config": {
            "chat_model": settings.chat_model,
            "judge_model": settings.judge_model,
            "embedding_model": settings.embedding_model,
            "top_k": DEFAULT_TOP_K,
            "chunk_size": DEFAULT_TARGET_TOKENS,
            "chunk_overlap": DEFAULT_OVERLAP_TOKENS,
        },
        "golden_set": {
            "path": str(args.qa_path.relative_to(REPO_ROOT))
            if args.qa_path.is_absolute() and args.qa_path.is_relative_to(REPO_ROOT)
            else str(args.qa_path),
            "n_records_loaded": len(records),
            "counts_by_type": {t.value: n for t, n in counts_by_type(records).items()},
        },
        "totals": {
            "n_evaluated": metrics.n_evaluated,
            "n_skipped": metrics.n_skipped,
            "wall_clock_seconds": round(wall_seconds, 2),
            "cost_usd_generate": round(metrics.total_generate_cost_usd, 6),
            "cost_usd_judge": round(metrics.total_judge_cost_usd, 6),
            "cost_usd_total": round(metrics.total_cost_usd, 6),
        },
        "metrics": runmetrics_to_dict(metrics),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_summary_md(
        summary_path,
        manifest=manifest,
        metrics=metrics,
        results=results,
    )

    snapshot_count = _snapshot_llm_calls(settings.llm_call_log, snapshot_path, run_id=run_id)
    logger.info(
        "eval_done",
        run_id=run_id,
        n_evaluated=metrics.n_evaluated,
        n_skipped=metrics.n_skipped,
        wall_seconds=round(wall_seconds, 2),
        llm_calls_snapshot=snapshot_count,
    )

    print("\n── eval summary ────────────────────────────────────────────")
    print(f"  run_id: {run_id}")
    print(f"  evaluated: {metrics.n_evaluated}    skipped: {metrics.n_skipped}")
    print(
        f"  accuracy: {_format_pct(metrics.accuracy_overall)}    "
        f"cite_prec: {_format_pct(metrics.citation_precision_programmatic_mean)}    "
        f"hit@{metrics.k}: {_format_pct(metrics.hit_at_k_rate)}    "
        f"recall@{metrics.k}: {_format_pct(metrics.mean_recall_at_k)}"
    )
    print(
        f"  latency  p50: {_format_ms(metrics.p50_latency_ms)}    "
        f"p95: {_format_ms(metrics.p95_latency_ms)}"
    )
    print(
        f"  cost     generate: ${metrics.total_generate_cost_usd:.4f}    "
        f"judge: ${metrics.total_judge_cost_usd:.4f}    "
        f"total: ${metrics.total_cost_usd:.4f}"
    )
    artifacts_display = (
        run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir
    )
    print(f"  artifacts: {artifacts_display}")
    print("────────────────────────────────────────────────────────────\n")
    # Always exit 0 — per-Q&A skips are reported in summary.md; the
    # operator decides whether to iterate. A crash before this point still
    # exits non-zero via the usual Python semantics.
    return 0


if __name__ == "__main__":
    sys.exit(main())
