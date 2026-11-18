"""Anthropic Claude LLM-as-judge with retry, prompt versioning, and per-call cost logging."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.observability import log_llm_call
from src.pipeline.anthropic_utils import (
    PRICING,
    PROVIDER,
    cost_for,
    is_retryable,
)
from src.pipeline.prompts import load_prompt, render_user_template

DEFAULT_MAX_TOKENS = 1024  # Avoids truncating valid verdict JSON.

PROMPT_ROLE = "judge"
PROMPT_VERSION = "v1"  # Bump when verdict semantics change.

_REQUIRED_USER_PLACEHOLDERS: tuple[str, ...] = (
    "{question}",
    "{gold_answer}",
    "{gold_citations_block}",
    "{model_answer}",
    "{model_citations_block}",
)

_REQUIRED_JSON_KEYS: tuple[str, ...] = (
    "answer_correct",
    "citations_semantically_valid",
    "rationale",
)


class JudgeParseError(ValueError):
    """The judge's reply was not the strict-JSON contract we require.

    Raised for malformed JSON, missing required keys, and wrong value types.
    Caller (the eval loop) catches this and skips the Q&A — a broken judge
    reply must not be silently coerced into a false verdict.
    """


@dataclass(frozen=True)
class JudgeResult:
    """Return shape of `ClaudeJudge.judge` — parsed verdict + accounting.

    The eval loop persists all six fields into `results.jsonl` per Q&A;
    `rationale` shows up in `summary.md` for borderline audits.
    """

    answer_correct: bool
    citations_semantically_valid: bool
    rationale: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _parse_verdict(text: str) -> tuple[bool, bool, str]:
    """Parse the judge's raw reply into (answer_correct, citations_valid, rationale)."""
    stripped = text.strip()
    try:
        obj: Any = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"judge reply is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise JudgeParseError(f"judge reply must be a JSON object, got {type(obj).__name__}")
    missing = [k for k in _REQUIRED_JSON_KEYS if k not in obj]
    if missing:
        raise JudgeParseError(f"judge reply missing required key(s): {', '.join(missing)}")
    if not isinstance(obj["answer_correct"], bool):
        raise JudgeParseError(
            f"judge reply key 'answer_correct' must be a bool, "
            f"got {type(obj['answer_correct']).__name__}"
        )
    if not isinstance(obj["citations_semantically_valid"], bool):
        raise JudgeParseError(
            f"judge reply key 'citations_semantically_valid' must be a bool, "
            f"got {type(obj['citations_semantically_valid']).__name__}"
        )
    if not isinstance(obj["rationale"], str):
        raise JudgeParseError(
            f"judge reply key 'rationale' must be a str, got {type(obj['rationale']).__name__}"
        )
    if not obj["rationale"].strip():
        raise JudgeParseError("judge reply key 'rationale' must not be empty")
    return obj["answer_correct"], obj["citations_semantically_valid"], obj["rationale"]


def _format_citations_block(citation_doc_paths: list[str]) -> str:
    """Render a list of doc_paths as a bulletable block for the prompt.

    Empty list renders as the literal `(none)` — cheaper for the judge to
    reason about than a blank string, and unambiguous for out-of-corpus.
    """
    if not citation_doc_paths:
        return "(none)"
    return "\n".join(f"- {p}" for p in citation_doc_paths)


class ClaudeJudge:
    """Sync Anthropic client wrapper for LLM-as-judge, with retry + cost logging."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        log_path: Path,
        client: Anthropic | None = None,
    ) -> None:
        if model not in PRICING:
            raise ValueError(
                f"unknown Anthropic model {model!r}; add its price to "
                f"ANTHROPIC_PRICING in src/pricing.py before use. "
                f"Known: {sorted(PRICING)}"
            )
        self._model = model
        self._log_path = log_path
        self._client = client or Anthropic(api_key=api_key.get_secret_value())

    @property
    def model(self) -> str:
        return self._model

    def judge(
        self,
        *,
        question: str,
        gold_answer: str,
        gold_citation_doc_paths: list[str],
        model_answer: str,
        model_citation_doc_paths: list[str],
        prompt_version: str = PROMPT_VERSION,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        run_id: str | None = None,
    ) -> JudgeResult:
        """One Anthropic messages.create call → parsed verdict.

        Raises `JudgeParseError` if the reply is not the strict-JSON contract.
        Raises any Anthropic exception the retry policy did not swallow.
        """
        system_body, user_template = load_prompt(
            PROMPT_ROLE,
            prompt_version,
            required_user_placeholders=_REQUIRED_USER_PLACEHOLDERS,
        )
        user_text = render_user_template(
            user_template,
            {
                "question": question,
                "gold_answer": gold_answer,
                "gold_citations_block": _format_citations_block(gold_citation_doc_paths),
                "model_answer": model_answer,
                "model_citations_block": _format_citations_block(model_citation_doc_paths),
            },
        )
        return self._call(
            system=system_body,
            user=user_text,
            prompt_version=prompt_version,
            max_tokens=max_tokens,
            run_id=run_id,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception(is_retryable),
        reraise=True,
    )
    def _call(
        self,
        *,
        system: str,
        user: str,
        prompt_version: str,
        max_tokens: int,
        run_id: str | None,
    ) -> JudgeResult:
        started = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            thinking={"type": "disabled"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        text = "".join(text_parts)

        input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
        cost_usd = cost_for(self._model, input_tokens, output_tokens)

        # Paid even if it fails.
        log_llm_call(
            self._log_path,
            provider=PROVIDER,
            model=self._model,
            operation="judge_answer",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=elapsed_ms,
            run_id=run_id,
            prompt_version=prompt_version,
            extra={
                "max_tokens": max_tokens,
                "thinking": "disabled",
                "stop_reason": getattr(response, "stop_reason", None),
            },
        )
        answer_correct, citations_valid, rationale = _parse_verdict(text)
        return JudgeResult(
            answer_correct=answer_correct,
            citations_semantically_valid=citations_valid,
            rationale=rationale,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )


def make_judge(
    *,
    api_key: SecretStr,
    model: str,
    log_path: Path,
    client: Anthropic | None = None,
) -> ClaudeJudge:
    return ClaudeJudge(api_key=api_key, model=model, log_path=log_path, client=client)
