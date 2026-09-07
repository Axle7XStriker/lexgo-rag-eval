"""ClaudeGenerator tests. Fully offline via a fake `anthropic.Anthropic` client.

Mirrors tests/test_embed.py: dependency-injection seam via `client=`,
tenacity's `wait` zeroed with monkeypatch for retry tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from pydantic import SecretStr

from src.pipeline import generate as generate_module
from src.pipeline.generate import (
    PRICING,
    ClaudeGenerator,
    _cost_for,
    _is_retryable,
)

# ── Anthropic response shape ──────────────────────────────────────────


@dataclass
class _FakeTextBlock:
    """Shape of a `text` block in response.content — only the two fields we read."""

    text: str
    type: str = "text"


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeMessage:
    """Shape of the object client.messages.create returns — only what we read."""

    content: list[_FakeTextBlock]
    usage: _FakeUsage
    stop_reason: str = "end_turn"


# One entry in `_FakeMessagesAPI.responses`: either a canned message or a
# callable invoked with the request kwargs so retry tests can raise then
# succeed. Mirrors the `_ResponseItem` pattern in tests/test_embed.py.
_ResponseItem = _FakeMessage | Callable[[dict], _FakeMessage]


@dataclass
class _FakeMessagesAPI:
    """Duck-types client.messages — only `.create()` because that's all we call."""

    responses: list[_ResponseItem] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def create(self, **kwargs) -> _FakeMessage:
        self.calls.append(kwargs)
        if not self.responses:
            # Sensible default so tests that don't program responses still work.
            return _FakeMessage(
                content=[_FakeTextBlock(text="ok")],
                usage=_FakeUsage(input_tokens=10, output_tokens=5),
            )
        item = self.responses.pop(0)
        if callable(item):
            return item(kwargs)
        return item


@dataclass
class _FakeClient:
    """Duck-types anthropic.Anthropic. Only `.messages` is used."""

    messages: _FakeMessagesAPI = field(default_factory=_FakeMessagesAPI)


# ── Error factories ───────────────────────────────────────────────────

# httpx Request/Response are required by the SDK's typed exceptions. Building
# them once at module level keeps the tests readable.
_HTTP_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _rate_limit(msg: str = "simulated 429") -> RateLimitError:
    resp = httpx.Response(429, request=_HTTP_REQ)
    return RateLimitError(msg, response=resp, body=None)


def _server_error(status: int = 503, msg: str = "simulated 5xx") -> APIStatusError:
    resp = httpx.Response(status, request=_HTTP_REQ)
    return APIStatusError(msg, response=resp, body=None)


def _client_error_400() -> BadRequestError:
    resp = httpx.Response(400, request=_HTTP_REQ)
    return BadRequestError("bad request", response=resp, body=None)


def _auth_error() -> AuthenticationError:
    resp = httpx.Response(401, request=_HTTP_REQ)
    return AuthenticationError("bad key", response=resp, body=None)


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=_HTTP_REQ)


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=_HTTP_REQ)


# ── Fixtures / helpers ────────────────────────────────────────────────


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "llm_calls.jsonl"


def _make_generator(
    *,
    client: _FakeClient,
    log_path: Path,
    model: str = "claude-sonnet-4-6",
) -> ClaudeGenerator:
    return ClaudeGenerator(
        api_key=SecretStr("test-key"),
        model=model,
        log_path=log_path,
        client=client,
    )


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── Tests ─────────────────────────────────────────────────────────────


class TestHappyPath:
    """A successful generate() returns the text and logs one record."""

    def test_returns_text_and_tokens(self, log_path: Path) -> None:
        msg = _FakeMessage(
            content=[_FakeTextBlock(text="the answer is 42 [1]")],
            usage=_FakeUsage(input_tokens=120, output_tokens=8),
        )
        client = _FakeClient(messages=_FakeMessagesAPI(responses=[msg]))
        g = _make_generator(client=client, log_path=log_path)

        result = g.generate(system="sys", user="usr", prompt_version="v1")

        assert result.text == "the answer is 42 [1]"
        assert result.input_tokens == 120
        assert result.output_tokens == 8
        # Cost: 120/1M * 3 + 8/1M * 15
        expected_cost = (120 / 1_000_000) * 3.0 + (8 / 1_000_000) * 15.0
        assert result.cost_usd == pytest.approx(expected_cost)

    def test_forwards_request_shape(self, log_path: Path) -> None:
        client = _FakeClient()
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
        # Sonnet usually returns one block, but if a future tier splits into
        # multiple text blocks we still capture the full answer.
        msg = _FakeMessage(
            content=[_FakeTextBlock(text="part one "), _FakeTextBlock(text="part two")],
            usage=_FakeUsage(input_tokens=1, output_tokens=1),
        )
        client = _FakeClient(messages=_FakeMessagesAPI(responses=[msg]))
        g = _make_generator(client=client, log_path=log_path)
        result = g.generate(system="s", user="u", prompt_version="v1")
        assert result.text == "part one part two"


class TestLogging:
    """Every successful call writes exactly one llm_calls.jsonl record."""

    def test_log_record_shape(self, log_path: Path) -> None:
        msg = _FakeMessage(
            content=[_FakeTextBlock(text="hi")],
            usage=_FakeUsage(input_tokens=200, output_tokens=50),
            stop_reason="end_turn",
        )
        client = _FakeClient(messages=_FakeMessagesAPI(responses=[msg]))
        g = _make_generator(client=client, log_path=log_path)
        g.generate(
            system="s",
            user="u",
            prompt_version="v1",
            run_id="run_test_xyz",
            max_tokens=256,
        )

        recs = _read_log(log_path)
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
        assert r["cost_usd"] == pytest.approx(_cost_for("claude-sonnet-4-6", 200, 50), abs=1e-6)

    def test_no_log_on_failure(self, log_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A raised exception during .create() must not write a log line —
        # tokens_billed accounting would be corrupted.
        monkeypatch.setattr(
            generate_module.ClaudeGenerator._call.retry,
            "wait",
            lambda *a, **kw: 0,
        )

        def _always_500(_kwargs: dict) -> _FakeMessage:
            raise _server_error()

        client = _FakeClient(messages=_FakeMessagesAPI(responses=[_always_500] * 10))
        g = _make_generator(client=client, log_path=log_path)
        with pytest.raises(APIStatusError):
            g.generate(system="s", user="u", prompt_version="v1")
        assert _read_log(log_path) == []


class TestRetry:
    """Transient failures retry; deterministic ones do not."""

    @pytest.mark.parametrize(
        ("exc_factory", "label"),
        [
            (_rate_limit, "rate_limit"),
            (_connection_error, "connection_error"),
            (_timeout_error, "timeout"),
            (lambda: _server_error(500), "http_500"),
            (lambda: _server_error(503), "http_503"),
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
        monkeypatch.setattr(
            generate_module.ClaudeGenerator._call.retry,
            "wait",
            lambda *a, **kw: 0,
        )
        state = {"n": 0}

        def _flaky(_kwargs: dict) -> _FakeMessage:
            state["n"] += 1
            if state["n"] < 2:
                raise exc_factory()
            return _FakeMessage(
                content=[_FakeTextBlock(text="ok")],
                usage=_FakeUsage(input_tokens=1, output_tokens=1),
            )

        client = _FakeClient(messages=_FakeMessagesAPI(responses=[_flaky, _flaky]))
        g = _make_generator(client=client, log_path=log_path)
        result = g.generate(system="s", user="u", prompt_version="v1")
        assert result.text == "ok"
        assert state["n"] == 2

    def test_gives_up_after_max_attempts(
        self, log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            generate_module.ClaudeGenerator._call.retry,
            "wait",
            lambda *a, **kw: 0,
        )

        def _always_fails(_kwargs: dict) -> _FakeMessage:
            raise _rate_limit()

        client = _FakeClient(messages=_FakeMessagesAPI(responses=[_always_fails] * 10))
        g = _make_generator(client=client, log_path=log_path)
        with pytest.raises(RateLimitError):
            g.generate(system="s", user="u", prompt_version="v1")

    @pytest.mark.parametrize(
        ("exc_factory", "label"),
        [
            (_auth_error, "auth"),
            (_client_error_400, "bad_request"),
            (lambda: _server_error(404), "http_404"),
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
        # Wait zeroed so if a retry mistakenly happens we still finish fast.
        monkeypatch.setattr(
            generate_module.ClaudeGenerator._call.retry,
            "wait",
            lambda *a, **kw: 0,
        )
        state = {"n": 0}

        def _always(_kwargs: dict) -> _FakeMessage:
            state["n"] += 1
            raise exc_factory()

        client = _FakeClient(messages=_FakeMessagesAPI(responses=[_always] * 10))
        g = _make_generator(client=client, log_path=log_path)
        with pytest.raises(exc_factory().__class__):
            g.generate(system="s", user="u", prompt_version="v1")
        # Deterministic errors must not retry — exactly one attempt.
        assert state["n"] == 1


class TestValidation:
    """Constructor guards on unknown model."""

    def test_unknown_model_rejected_at_init(self, log_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown Anthropic model"):
            _make_generator(
                client=_FakeClient(),
                log_path=log_path,
                model="claude-not-real",
            )


class TestPricing:
    """Cost table stays honest; missing model degrades gracefully."""

    def test_sonnet_present(self) -> None:
        assert "claude-sonnet-4-6" in PRICING
        assert PRICING["claude-sonnet-4-6"]["input"] > 0
        assert PRICING["claude-sonnet-4-6"]["output"] > 0

    def test_output_more_expensive_than_input(self) -> None:
        # Sanity: Claude output tokens are always priced higher than input.
        # If PRICING is edited the wrong way this catches it immediately.
        assert PRICING["claude-sonnet-4-6"]["output"] > PRICING["claude-sonnet-4-6"]["input"]

    def test_unknown_model_zero_cost(self) -> None:
        assert _cost_for("model-that-does-not-exist", 1_000_000, 1_000_000) == 0.0

    def test_cost_calc(self) -> None:
        # 1M input tokens = $3, 1M output tokens = $15 → $18 total
        assert _cost_for("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(18.0)


class TestRetryablePredicate:
    """The _is_retryable predicate is load-bearing — test it directly too."""

    def test_rate_limit_retryable(self) -> None:
        assert _is_retryable(_rate_limit()) is True

    def test_connection_retryable(self) -> None:
        assert _is_retryable(_connection_error()) is True

    def test_timeout_retryable(self) -> None:
        assert _is_retryable(_timeout_error()) is True

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
    def test_5xx_retryable(self, status: int) -> None:
        assert _is_retryable(_server_error(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_4xx_not_retryable(self, status: int) -> None:
        assert _is_retryable(_server_error(status)) is False

    def test_random_exception_not_retryable(self) -> None:
        assert _is_retryable(RuntimeError("nope")) is False
