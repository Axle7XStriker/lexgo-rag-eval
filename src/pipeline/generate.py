"""Anthropic Claude generator with retry, prompt versioning, and per-call cost logging."""

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
    """Return shape of `ClaudeGenerator.generate"""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class ClaudeGenerator:
    """Sync Anthropic client wrapper with retry, cost accounting, and per-call logging."""

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

    def generate(
        self,
        *,
        system: str,
        user: str,
        prompt_version: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        run_id: str | None = None,
    ) -> GenerateResult:
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
        """One Anthropic messages.create call. Wraps _call for tenacity retries."""
        started = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            thinking={"type": "disabled"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        text_parts = [  # Keep every text block. A split response is still one answer.
            b.text for b in response.content if getattr(b, "type", None) == "text"
        ]
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
    return ClaudeGenerator(api_key=api_key, model=model, log_path=log_path, client=client)
