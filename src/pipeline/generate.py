"""Anthropic Claude generator with retry, prompt versioning, and per-call cost logging.

One class, `ClaudeGenerator`. One method: `generate(system, user, prompt_version)`.

Design notes worth remembering:
  - Mirrors `VoyageEmbedder` line-for-line (see `src/pipeline/embed.py`): sync
    client, `SecretStr` key, `tenacity` retries on transient exceptions only,
    fail-fast on unknown model in `__init__`, exactly one `log_llm_call` per
    API call, dependency-injection seam via `client=` for tests.
  - `PRICING` uses a nested `{input, output}` dict because Claude has separate
    per-1M rates for input and output tokens — Voyage's flat float doesn't fit.
    Anthropic's SDK does NOT return a per-call cost, so cost is computed here.
  - `temperature=0.0` by default: P1 baseline is a scientific eval — outputs
    must be reproducible across runs so the accuracy delta between P1..P4 is
    signal, not noise. Sonnet 4.6 still accepts `temperature` (removed on
    Opus 5 / Sonnet 5 / 4.7+); we pass it explicitly.
  - No `thinking` param: adaptive thinking is a P4 concern at best. The
    baseline generator represents the minimal simple usage.
  - Prompt prefill is NOT supported on Sonnet 4.6 (400 error); we never
    prefill the assistant turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

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

_logger = get_logger("generate")

# USD per 1M tokens, separate input/output rates. Anthropic's SDK does not
# return a per-call cost, so we compute it locally. Update this dict when
# pricing changes — the fail-fast constructor check keeps unknown/mispriced
# models out of the eval numbers.
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    # Add other models here as they get exercised. Keep the nested-dict shape
    # so downstream cost code doesn't branch on model tier.
}

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
PROVIDER = "anthropic"


@dataclass(frozen=True)
class GenerateResult:
    """Return shape of `ClaudeGenerator.generate` — the pipeline reads all four."""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _is_retryable(exc: BaseException) -> bool:
    """True for transient failures; False for deterministic ones.

    RateLimitError, APIConnectionError, APITimeoutError are always transient.
    APIStatusError catches the raw HTTP surface — we retry only 5xx server
    errors, not 4xx client errors (BadRequestError, AuthenticationError,
    NotFoundError, PermissionDeniedError all inherit from APIStatusError but
    correspond to deterministic mistakes we should surface immediately).
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
    """Cost in USD for a call at `model`'s pricing. Missing model → 0.0.

    ClaudeGenerator rejects unknown models at construction, so this fallback
    only trips if PRICING is edited to drop an in-flight model. Warning is a
    belt-and-braces breadcrumb (same pattern as embed._cost_for).
    """
    rates = PRICING.get(model)
    if rates is None:
        _logger.warning(
            "anthropic_pricing_missing",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates[
        "output"
    ]


class ClaudeGenerator:
    """Sync Anthropic client wrapper with retry, cost accounting, and per-call logging.

    Not thread-safe (the underlying `Anthropic` client isn't documented as
    such). Fine for the batch eval loop and single-request Streamlit path.
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

    def generate(
        self,
        *,
        system: str,
        user: str,
        prompt_version: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        run_id: str | None = None,
    ) -> GenerateResult:
        """One Anthropic messages.create call. Wraps _call for tenacity retries."""
        return self._call(
            system=system,
            user=user,
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
    ) -> GenerateResult:
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

        # Concatenate every text block. In practice Sonnet returns a single
        # text block, but joining is safer than indexing content[0] — if a
        # future model tier splits the answer into multiple blocks (or wraps
        # a summarized `thinking` block first), we still capture the answer.
        text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        text = "".join(text_parts)

        input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
        cost_usd = _cost_for(self._model, input_tokens, output_tokens)

        log_llm_call(
            self._log_path,
            provider=PROVIDER,
            model=self._model,
            operation="generate_answer",
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
        return GenerateResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )


def make_generator(
    *,
    api_key: SecretStr,
    model: str,
    log_path: Path,
    client: Anthropic | None = None,
) -> ClaudeGenerator:
    """Convenience factory. Kept thin — most callers instantiate directly."""
    return ClaudeGenerator(api_key=api_key, model=model, log_path=log_path, client=client)
