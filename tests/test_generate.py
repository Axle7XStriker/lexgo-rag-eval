"""ClaudeGenerator tests. Fully offline via a fake `anthropic.Anthropic` client.

Mirrors tests/test_embed.py: dependency-injection seam via `client=`,
tenacity's `wait` zeroed with monkeypatch for retry tests.

The fake `Anthropic` client + error factories live in `tests/_anthropic_fakes.py`
so `test_judge.py` shares the same wire fakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from anthropic import (
    APIStatusError,
    RateLimitError,
)
from pydantic import SecretStr

from src.pipeline import generate as generate_module
from src.pipeline.anthropic_utils import PRICING, cost_for, is_retryable
from src.pipeline.generate import ClaudeGenerator
from tests._anthropic_fakes import (
    FakeClient,
    FakeMessage,
    FakeMessagesAPI,
    FakeTextBlock,
    FakeUsage,
    auth_error,
    client_error_400,
    connection_error,
    rate_limit,
    read_log,
    server_error,
    timeout_error,
)

# ── Fixtures / helpers ────────────────────────────────────────────────


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "llm_calls.jsonl"


def _make_generator(
    *,
    client: FakeClient,
    log_path: Path,
    model: str = "claude-sonnet-4-6",
) -> ClaudeGenerator:
    return ClaudeGenerator(
        api_key=SecretStr("test-key"),
        model=model,
        log_path=log_path,
        client=client,
    )


# ── Tests ─────────────────────────────────────────────────────────────


class TestHappyPath:
    """A successful generate() returns the text and logs one record."""

    def test_returns_text_and_tokens(self, log_path: Path) -> None:
        """Returns joined answer text + usage counts + cost derived from PRICING."""
        msg = FakeMessage(
            content=[FakeTextBlock(text="the answer is 42 [1]")],
            usage=FakeUsage(input_tokens=120, output_tokens=8),
        )
        client = FakeClient(messages=FakeMessagesAPI(responses=[msg]))
        g = _make_generator(client=client, log_path=log_path)

        result = g.generate(system="sys", user="usr", prompt_version="v1")

        assert result.text == "the answer is 42 [1]"
        assert result.input_tokens == 120
        assert result.output_tokens == 8
        # Cost derived from PRICING via cost_for so a price change here needs
        # no test update — the source of truth is src/pricing.py.
        assert result.cost_usd == pytest.approx(cost_for("claude-sonnet-4-6", 120, 8))

    def test_forwards_request_shape(self, log_path: Path) -> None:
        """generate() forwards model, system, max_tokens, temperature, and messages to the SDK."""
        client = FakeClient()
        g = _make_generator(client=client, log_path=log_path)
        g.generate(
            system="SYS BODY",
            user="USER BODY",
            prompt_version="v1",
            max_tokens=512,
            temperature=0.2,
        )
        assert len(client.messages.calls) == 1
        call = client.messages.calls[0]
        assert call["model"] == "claude-sonnet-4-6"
        assert call["system"] == "SYS BODY"
        assert call["max_tokens"] == 512
        assert call["temperature"] == 0.2
        assert call["messages"] == [{"role": "user", "content": "USER BODY"}]

    def test_multiple_text_blocks_are_joined(self, log_path: Path) -> None:
        """Multiple text blocks are concatenated (defensive against future model tiers)."""
        msg = FakeMessage(
            content=[FakeTextBlock(text="part one "), FakeTextBlock(text="part two")],
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
        client = FakeClient(messages=FakeMessagesAPI(responses=[msg]))
        g = _make_generator(client=client, log_path=log_path)
        result = g.generate(system="s", user="u", prompt_version="v1")
        assert result.text == "part one part two"


class TestLogging:
    """Every successful call writes exactly one llm_calls.jsonl record."""

    def test_log_record_shape(self, log_path: Path) -> None:
        """One successful call writes exactly one log record with all the expected fields."""
        msg = FakeMessage(
            content=[FakeTextBlock(text="hi")],
            usage=FakeUsage(input_tokens=200, output_tokens=50),
            stop_reason="end_turn",
        )
        client = FakeClient(messages=FakeMessagesAPI(responses=[msg]))
        g = _make_generator(client=client, log_path=log_path)
        g.generate(
            system="s",
            user="u",
            prompt_version="v1",
            run_id="run_test_xyz",
            max_tokens=256,
        )

        recs = read_log(log_path)
        assert len(recs) == 1
        r = recs[0]
        assert r["provider"] == "anthropic"
        assert r["model"] == "claude-sonnet-4-6"
        assert r["operation"] == "generate_answer"
        assert r["input_tokens"] == 200
        assert r["output_tokens"] == 50
        assert r["prompt_version"] == "v1"
        assert r["run_id"] == "run_test_xyz"
        assert r["max_tokens"] == 256
        assert r["stop_reason"] == "end_turn"
        assert r["cost_usd"] == pytest.approx(cost_for("claude-sonnet-4-6", 200, 50), abs=1e-6)

    def test_no_log_on_failure(self, log_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exceptions during .create() must NOT write a log line — else cost accounting corrupts."""
        monkeypatch.setattr(
            generate_module.ClaudeGenerator._call.retry,
            "wait",
            lambda *a, **kw: 0,
        )

        def _always_500(_kwargs: dict) -> FakeMessage:
            raise server_error()

        client = FakeClient(messages=FakeMessagesAPI(responses=[_always_500] * 10))
        g = _make_generator(client=client, log_path=log_path)
        with pytest.raises(APIStatusError):
            g.generate(system="s", user="u", prompt_version="v1")
        assert read_log(log_path) == []


class TestRetry:
    """Transient failures retry; deterministic ones do not."""

    @pytest.mark.parametrize(
        ("exc_factory", "label"),
        [
            (rate_limit, "rate_limit"),
            (connection_error, "connection_error"),
            (timeout_error, "timeout"),
            (lambda: server_error(500), "http_500"),
            (lambda: server_error(503), "http_503"),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_retries_on_transient(
        self,
        log_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        exc_factory,
        label: str,
    ) -> None:
        """Each transient failure class triggers a retry; the second attempt succeeds."""
        monkeypatch.setattr(
            generate_module.ClaudeGenerator._call.retry,
            "wait",
            lambda *a, **kw: 0,
        )
        state = {"n": 0}

        def _flaky(_kwargs: dict) -> FakeMessage:
            state["n"] += 1
            if state["n"] < 2:
                raise exc_factory()
            return FakeMessage(
                content=[FakeTextBlock(text="ok")],
                usage=FakeUsage(input_tokens=1, output_tokens=1),
            )

        client = FakeClient(messages=FakeMessagesAPI(responses=[_flaky, _flaky]))
        g = _make_generator(client=client, log_path=log_path)
        result = g.generate(system="s", user="u", prompt_version="v1")
        assert result.text == "ok"
        assert state["n"] == 2

    def test_gives_up_after_max_attempts(
        self, log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A persistent transient failure re-raises after tenacity's attempt budget is exhausted."""
        monkeypatch.setattr(
            generate_module.ClaudeGenerator._call.retry,
            "wait",
            lambda *a, **kw: 0,
        )

        def _always_fails(_kwargs: dict) -> FakeMessage:
            raise rate_limit()

        client = FakeClient(messages=FakeMessagesAPI(responses=[_always_fails] * 10))
        g = _make_generator(client=client, log_path=log_path)
        with pytest.raises(RateLimitError):
            g.generate(system="s", user="u", prompt_version="v1")

    @pytest.mark.parametrize(
        ("exc_factory", "label"),
        [
            (auth_error, "auth"),
            (client_error_400, "bad_request"),
            (lambda: server_error(404), "http_404"),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_fail_fast_on_deterministic(
        self,
        log_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        exc_factory,
        label: str,
    ) -> None:
        """Deterministic errors (auth / 400 / 404) surface after exactly one attempt."""
        # Wait zeroed so if a retry mistakenly happens we still finish fast.
        monkeypatch.setattr(
            generate_module.ClaudeGenerator._call.retry,
            "wait",
            lambda *a, **kw: 0,
        )
        state = {"n": 0}

        def _always(_kwargs: dict) -> FakeMessage:
            state["n"] += 1
            raise exc_factory()

        client = FakeClient(messages=FakeMessagesAPI(responses=[_always] * 10))
        g = _make_generator(client=client, log_path=log_path)
        with pytest.raises(exc_factory().__class__):
            g.generate(system="s", user="u", prompt_version="v1")
        # Deterministic errors must not retry — exactly one attempt.
        assert state["n"] == 1


class TestValidation:
    """Constructor guards on unknown model."""

    def test_unknown_model_rejected_at_init(self, log_path: Path) -> None:
        """Unknown model raises at init — silent $0 cost would poison the eval numbers."""
        with pytest.raises(ValueError, match="unknown Anthropic model"):
            _make_generator(
                client=FakeClient(),
                log_path=log_path,
                model="claude-not-real",
            )


class TestPricing:
    """Cost table stays honest; missing model degrades gracefully."""

    def test_sonnet_present(self) -> None:
        """The currently-used chat model is in PRICING with non-zero input+output rates."""
        assert "claude-sonnet-4-6" in PRICING
        assert PRICING["claude-sonnet-4-6"]["input"] > 0
        assert PRICING["claude-sonnet-4-6"]["output"] > 0

    def test_output_more_expensive_than_input(self) -> None:
        """Sanity: output rate > input rate for every Claude tier (catches an accidental swap)."""
        assert PRICING["claude-sonnet-4-6"]["output"] > PRICING["claude-sonnet-4-6"]["input"]

    def test_unknown_model_zero_cost(self) -> None:
        """Unknown model returns 0.0 cost (and logs a warning), rather than raising."""
        assert cost_for("model-that-does-not-exist", 1_000_000, 1_000_000) == 0.0

    def test_cost_calc(self) -> None:
        """1M input + 1M output = (input_rate + output_rate) USD; expected derived from PRICING."""
        rates = PRICING["claude-sonnet-4-6"]
        expected = rates["input"] + rates["output"]
        assert cost_for("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(expected)


class TestRetryablePredicate:
    """The is_retryable predicate is load-bearing — test it directly too."""

    def test_rate_limit_retryable(self) -> None:
        """Anthropic's RateLimitError is transient — worth retrying."""
        assert is_retryable(rate_limit()) is True

    def test_connection_retryable(self) -> None:
        """A dropped connection is transient — worth retrying."""
        assert is_retryable(connection_error()) is True

    def test_timeout_retryable(self) -> None:
        """APITimeoutError is transient — worth retrying (subclass of APIConnectionError)."""
        assert is_retryable(timeout_error()) is True

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
    def test_5xx_retryable(self, status: int) -> None:
        """5xx server errors are transient — retry across the whole 5xx range."""
        assert is_retryable(server_error(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_4xx_not_retryable(self, status: int) -> None:
        """4xx errors indicate a deterministic client mistake — retrying can't fix them."""
        assert is_retryable(server_error(status)) is False

    def test_random_exception_not_retryable(self) -> None:
        """Non-Anthropic exceptions (a bug in our own code) must not silently retry."""
        assert is_retryable(RuntimeError("nope")) is False
