"""Anthropic Claude generator with retry, prompt versioning, and per-call cost logging.

One class, `ClaudeGenerator`. One method: `generate(system, user, prompt_version)`.

Design notes worth remembering:
  - Mirrors `VoyageEmbedder` line-for-line (see `src/pipeline/embed.py`): sync
    client, `SecretStr` key, `tenacity` retries on transient exceptions only,
    fail-fast on unknown model in `__init__`, exactly one `log_llm_call` per
    API call, dependency-injection seam via `client=` for tests.
  - Shared with `ClaudeJudge`: `is_retryable`, `cost_for`, `PRICING`, and
    `PROVIDER` live in `src.pipeline.anthropic_utils` so a change there
    (new retryable exception, price adjustment) is a one-file edit.
  - Thinking is disabled. Haiku 5.5 turns adaptive thinking on by default
    and those tokens count toward `max_tokens`; leaving it on can consume
    the 1024-token answer budget before any text is returned.
  - Sampling parameters are omitted. Haiku 5.5 returns 400 if `temperature`,
    `top_p`, or `top_k` is sent.
  - Prompt prefill is not used. Haiku 5.5 rejects a trailing assistant turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

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

DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class GenerateResult:
    """Return shape of `ClaudeGenerator.generate` — the pipeline reads all four."""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


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

    def generate(
        self,
        *,
        system: str,
        user: str,
        prompt_version: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        run_id: str | None = None,
    ) -> GenerateResult:
        """One Anthropic messages.create call. Wraps _call for tenacity retries."""
        return self._call(
            system=system,
            user=user,
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
    ) -> GenerateResult:
        """One API call. Wrapped in tenacity — do not call directly from tests."""
        started = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            thinking={"type": "disabled"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        # Concatenate every text block. Joining is safer than indexing
        # content[0]: a split answer, or a thinking block ahead of the text,
        # still leaves us with the reply.
        text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        text = "".join(text_parts)

        input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
        cost_usd = cost_for(self._model, input_tokens, output_tokens)

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
                "thinking": "disabled",
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
