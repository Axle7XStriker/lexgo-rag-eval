"""Anthropic Claude LLM-as-judge with retry, prompt versioning, and per-call cost logging.

One class, `ClaudeJudge`. One method: `judge(...)` — one API call per Q&A that
returns a `JudgeResult` with both verdicts (answer + citations) plus a short
rationale, parsed from the model's strict-JSON reply.

Design notes worth remembering:
  - Mirrors `ClaudeGenerator` line-for-line: sync `Anthropic` client,
    `SecretStr` key, `tenacity` retries on transient exceptions only,
    fail-fast on unknown model in `__init__`, exactly one `log_llm_call` per
    API call, dependency-injection seam via `client=`. Shared retry / cost /
    provider helpers live in `src.pipeline.anthropic_utils`; shared prompt
    loading + the out-of-corpus sentinel live in `src.pipeline.prompts`.
  - The prompt is a strict-JSON contract. We parse `response.text` with
    `json.loads` and raise `JudgeParseError` on any deviation (malformed
    JSON, missing key, wrong type). The eval loop catches this and skips
    the Q&A rather than silently marking it incorrect — a broken judge
    reply is a signal to iterate on the prompt, not an accuracy datapoint.
  - `temperature=0.0` for the same reproducibility reason as the generator:
    the accuracy delta P1 → P4 must be signal, not judge noise.
"""

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
    DEFAULT_TEMPERATURE,
    PRICING,
    PROVIDER,
    cost_for,
    is_retryable,
)
from src.pipeline.prompts import load_prompt, render_user_template

# Aligned with the generator's 1024. The judge's payload is a 1-3 sentence
# rationale plus a short strict-JSON envelope — well under 1024 in the common
# case — but 512 was tight enough that a verbose rationale on a borderline
# Q&A could stop the reply mid-JSON, trip _parse_verdict, and silently skip
# the record. Extra headroom costs pennies; a silently-skipped Q&A costs
# more (and the operator only sees it as "skip: JSON parse error", not as
# "max_tokens truncation").
DEFAULT_MAX_TOKENS = 1024

# `role` + `version` locate the prompt file at prompts/<role>/<version>.md.
# Bump PROMPT_VERSION on any semantic change to the judge prompt; captured
# eval runs record the version so an accuracy shift is attributable.
PROMPT_ROLE = "judge"
PROMPT_VERSION = "v1"

# Placeholders the judge user template MUST contain — enforced at load time.
_REQUIRED_USER_PLACEHOLDERS: tuple[str, ...] = (
    "{question}",
    "{gold_answer}",
    "{gold_citations_block}",
    "{model_answer}",
    "{model_citations_block}",
)

# JSON keys the judge MUST return. Missing any of them → JudgeParseError.
# `answer_correct` and `citations_semantically_valid` must be bools;
# `rationale` must be a string.
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
    """Parse the judge's raw reply into (answer_correct, citations_valid, rationale).

    The judge prompt forbids Markdown / prose wrappers, but we still handle
    surrounding whitespace — a leading blank line from Claude is not a
    contract violation. Anything else raises `JudgeParseError`.
    """
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
    # Empty / whitespace-only rationale defeats the whole point of the field
    # (it's the audit column in summary.md for borderline verdicts). Reject
    # instead of quietly storing "" and calling the record valid.
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
    """Sync Anthropic client wrapper for LLM-as-judge, with retry + cost logging.

    Not thread-safe (the underlying `Anthropic` client isn't documented as
    such). Fine for the sequential eval loop.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        log_path: Path,
        client: Anthropic | None = None,
    ) -> None:
        # Fail fast on unknown models: a silent $0 per-call cost would poison
        # the blog's cost story with no visible signal.
        if model not in PRICING:
            raise ValueError(
                f"unknown Anthropic model {model!r}; add its price to "
                f"ANTHROPIC_PRICING in src/pricing.py before use. "
                f"Known: {sorted(PRICING)}"
            )
        self._model = model
        self._log_path = log_path
        # `client` injection is the seam tests use — no HTTP mock required.
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
        temperature: float = DEFAULT_TEMPERATURE,
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
            temperature=temperature,
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
        temperature: float,
        run_id: str | None,
    ) -> JudgeResult:
        """One API call. Wrapped in tenacity — do not call directly from tests."""
        started = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        # Concatenate every text block. See ClaudeGenerator for the rationale.
        text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        text = "".join(text_parts)

        input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
        cost_usd = cost_for(self._model, input_tokens, output_tokens)

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
                "temperature": temperature,
                # stop_reason is load-bearing for debugging refusal / max_tokens
                # truncation — cheap to record, expensive to reconstruct later.
                "stop_reason": getattr(response, "stop_reason", None),
            },
        )
        # Parsing happens AFTER the log write. A malformed reply is still a
        # real API call we paid for; recording it keeps cost accounting honest.
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
    """Convenience factory. Kept thin — most callers instantiate directly."""
    return ClaudeJudge(api_key=api_key, model=model, log_path=log_path, client=client)
