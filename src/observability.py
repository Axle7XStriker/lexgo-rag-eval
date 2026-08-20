"""Logging + LLM-call accounting.

Two responsibilities:
1. Configure structlog for stderr — colorized console rendering when stderr
   is a TTY (interactive CLI + `streamlit run` in a terminal), JSON when
   redirected (CI, log-collection pipelines). Correlation IDs propagated
   via contextvars.
2. Provide `log_llm_call` — appends a single JSON record per LLM call to
   logs/llm_calls.jsonl. Every eval number depends on knowing which calls
   produced it, so per-call provenance is load-bearing infra, not an add-on;
   token + cost accounting comes along for the ride.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog. Idempotent.

    Renderer is chosen by whether stderr is a TTY:
      - TTY  → structlog.dev.ConsoleRenderer (colorized, human-readable).
        This is what you get during interactive CLI runs (`make corpus`,
        `streamlit run`) where JSON-per-line would drown out the
        progress printouts.
      - non-TTY → JSONRenderer (CI, redirection, log-collection pipelines
        that expect one JSON record per line).
    """
    is_tty = sys.stderr.isatty()
    renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=True)
        if is_tty
        else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def log_llm_call(
    log_path: Path,
    *,
    provider: str,
    model: str,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: float,
    run_id: str | None = None,
    prompt_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a single LLM call record to the JSONL log.

    Called at every LLM call site (chat, embed, rerank, judge). Cheap append-only
    write — no batching, no async. If the log dir doesn't exist, creates it.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Canonical fields listed last so they win any key collision with `extra`.
    record: dict[str, Any] = {
        **(extra or {}),
        "ts": datetime.now(UTC).isoformat(),
        "provider": provider,
        "model": model,
        "operation": operation,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "latency_ms": round(latency_ms, 2),
        "run_id": run_id,
        "prompt_version": prompt_version,
    }
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
