"""Anthropic Claude LLM-as-judge with retry, prompt versioning, and per-call cost logging.

One class, `ClaudeJudge`. One method: `judge(...)` — one API call per Q&A that
returns a `JudgeResult` with both verdicts (answer + citations) plus a short
rationale, parsed from the model's strict-JSON reply.

Design notes worth remembering:
  - Mirrors `ClaudeGenerator` line-for-line: sync `Anthropic` client,
    `SecretStr` key, `tenacity` retries on transient exceptions only,
    fail-fast on unknown model in `__init__`, exactly one `log_llm_call` per
    API call, dependency-injection seam via `client=`. The two classes are
    siblings; anything true of one should stay true of the other.
  - The prompt is a strict-JSON contract. We parse `response.text` with
    `json.loads` and raise `JudgeParseError` on any deviation (malformed
    JSON, missing key, wrong type). The eval loop catches this and skips
    the Q&A rather than silently marking it incorrect — a broken judge
    reply is a signal to iterate on the prompt, not an accuracy datapoint.
  - `temperature=0.0` for the same reproducibility reason as the generator:
    the accuracy delta P1 → P4 must be signal, not judge noise.
  - Prompt loading is a small local variant of `_load_prompt` from
    `src/pipeline/query.py`. The judge template has different required
    placeholders (no `{context}`, but `{gold_citations_block}` /
    `{model_citations_block}` / etc.), so factoring one shared loader
    would be lossier than duplicating ~30 lines here. Kept literally
    section-parse-compatible with the answer prompt file format so a
    reader who knows one knows the other.
"""

from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.observability import get_logger, log_llm_call
from src.pricing import ANTHROPIC_PRICING as PRICING
from src.pricing import anthropic_cost

_logger = get_logger("judge")

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
PROVIDER = "anthropic"

# `role` + `version` locate the prompt file at prompts/<role>/<version>.md.
# Bump PROMPT_VERSION on any semantic change to the judge prompt; captured
# eval runs record the version so an accuracy shift is attributable.
PROMPT_ROLE = "judge"
PROMPT_VERSION = "v1"

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# Section markers inside a prompt file. Matches the answer prompt convention
# so both files parse under the same section-based rules.
_SYSTEM_MARKER = "# System"
_USER_TEMPLATE_MARKER = "# User template"

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


def _is_retryable(exc: BaseException) -> bool:
    """True for transient failures; False for deterministic ones.

    Identical policy to `ClaudeGenerator._is_retryable` — retry rate limits,
    connection/timeout errors, and 5xx server errors; fail fast on
    everything else (4xx, auth, unknown model at request time).
    """
    if isinstance(exc, RateLimitError | APIConnectionError | APITimeoutError):
        return True
    if isinstance(exc, APIStatusError):
        # `status_code` is set on typed APIStatusError subclasses; guard with
        # getattr in case a subclass without one slips through.
        code = getattr(exc, "status_code", None)
        return isinstance(code, int) and code >= 500
    return False


def _cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for a call at `model`'s pricing. Missing model → 0.0."""
    cost = anthropic_cost(model, input_tokens, output_tokens)
    if cost is None:
        _logger.warning(
            "anthropic_pricing_missing",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return 0.0
    return cost


@functools.cache
def _load_prompt(role: str, version: str) -> tuple[str, str]:
    """Load a prompt file, return (system_body, user_template).

    Cached per (role, version) — prompt files are immutable once shipped.
    The system body has `{out_of_corpus_sentinel}` pre-substituted at load
    time (imported from `src.pipeline.query` so both the answer prompt and
    the judge prompt share the exact literal); the user template still
    contains its per-request placeholders for the caller to fill.

    Raises:
      FileNotFoundError — no file at prompts/<role>/<version>.md.
      ValueError — missing `# System` / `# User template` sections,
        missing required placeholders in either section.
    """
    # Local import to avoid a package-level cycle with src.pipeline.query,
    # which itself imports nothing from this module today but would create
    # a circular import risk if that ever changes.
    from src.pipeline.query import OUT_OF_CORPUS_SENTINEL

    path = PROMPTS_DIR / role / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    text = path.read_text(encoding="utf-8")

    sys_idx = text.find(_SYSTEM_MARKER)
    user_idx = text.find(_USER_TEMPLATE_MARKER)
    if sys_idx == -1 or user_idx == -1 or user_idx <= sys_idx:
        raise ValueError(
            f"{path}: expected '{_SYSTEM_MARKER}' then '{_USER_TEMPLATE_MARKER}' sections"
        )
    raw_system = text[sys_idx + len(_SYSTEM_MARKER) : user_idx].strip()
    user_template = text[user_idx + len(_USER_TEMPLATE_MARKER) :].strip()

    # Fail fast on template drift.
    #   - The system body MUST reference {out_of_corpus_sentinel}, or the
    #     code constant and the prompt's actual instruction would silently
    #     drift apart.
    #   - The user template MUST contain every per-Q&A placeholder the
    #     caller fills — otherwise a raw literal would be shipped to Claude.
    if "{out_of_corpus_sentinel}" not in raw_system:
        raise ValueError(
            f"{path}: system body must reference '{{out_of_corpus_sentinel}}' "
            f"so OUT_OF_CORPUS_SENTINEL stays the single source of truth."
        )
    required = (
        "{question}",
        "{gold_answer}",
        "{gold_citations_block}",
        "{model_answer}",
        "{model_citations_block}",
    )
    missing = [p for p in required if p not in user_template]
    if missing:
        raise ValueError(
            f"{path}: user template missing required placeholder(s): {', '.join(missing)}"
        )

    system_body = raw_system.format(out_of_corpus_sentinel=OUT_OF_CORPUS_SENTINEL)
    return system_body, user_template


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
                f"unknown Anthropic model {model!r}; add its price to PRICING "
                f"in src/pipeline/generate.py before use. Known: {sorted(PRICING)}"
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
        system_body, user_template = _load_prompt(PROMPT_ROLE, prompt_version)
        user_text = user_template.format(
            question=question,
            gold_answer=gold_answer,
            gold_citations_block=_format_citations_block(gold_citation_doc_paths),
            model_answer=model_answer,
            model_citations_block=_format_citations_block(model_citation_doc_paths),
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
        retry=retry_if_exception(_is_retryable),
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
        cost_usd = _cost_for(self._model, input_tokens, output_tokens)

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
