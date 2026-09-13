"""Shared fake Anthropic client + error factories used by test_generate and test_judge.

Both ClaudeGenerator and ClaudeJudge wrap the sync `anthropic.Anthropic`
client with the same `.messages.create(**kwargs)` surface, the same
tenacity retry policy, and the same typed-exception classes. The fakes here
are the seam their tests inject via `client=`.

Kept as a plain module (not a conftest fixture) because tests need to
construct these types explicitly per-case — programming canned responses,
raising specific exceptions, inspecting recorded calls — not just receive
a wired-up default.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

# ── Anthropic response shape ──────────────────────────────────────────


@dataclass
class FakeTextBlock:
    """Shape of a `text` block in response.content — only the two fields we read."""

    text: str
    type: str = "text"


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class FakeMessage:
    """Shape of the object client.messages.create returns — only what we read."""

    content: list[FakeTextBlock]
    usage: FakeUsage
    stop_reason: str = "end_turn"


# One entry in `FakeMessagesAPI.responses`: either a canned message or a
# callable invoked with the request kwargs so retry tests can raise then
# succeed. Mirrors the `_ResponseItem` pattern in tests/test_embed.py.
ResponseItem = FakeMessage | Callable[[dict], FakeMessage]


@dataclass
class FakeMessagesAPI:
    """Duck-types client.messages — only `.create()` because that's all we call."""

    responses: list[ResponseItem] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def create(self, **kwargs) -> FakeMessage:
        self.calls.append(kwargs)
        if not self.responses:
            # Sensible default so tests that don't program responses still work.
            return FakeMessage(
                content=[FakeTextBlock(text="ok")],
                usage=FakeUsage(input_tokens=10, output_tokens=5),
            )
        item = self.responses.pop(0)
        if callable(item):
            return item(kwargs)
        return item


@dataclass
class FakeClient:
    """Duck-types anthropic.Anthropic. Only `.messages` is used."""

    messages: FakeMessagesAPI = field(default_factory=FakeMessagesAPI)


# ── Error factories ───────────────────────────────────────────────────

# httpx Request/Response are required by the SDK's typed exceptions. Building
# one once at module level keeps the factory calls readable.
_HTTP_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def rate_limit(msg: str = "simulated 429") -> RateLimitError:
    resp = httpx.Response(429, request=_HTTP_REQ)
    return RateLimitError(msg, response=resp, body=None)


def server_error(status: int = 503, msg: str = "simulated 5xx") -> APIStatusError:
    resp = httpx.Response(status, request=_HTTP_REQ)
    return APIStatusError(msg, response=resp, body=None)


def client_error_400() -> BadRequestError:
    resp = httpx.Response(400, request=_HTTP_REQ)
    return BadRequestError("bad request", response=resp, body=None)


def auth_error() -> AuthenticationError:
    resp = httpx.Response(401, request=_HTTP_REQ)
    return AuthenticationError("bad key", response=resp, body=None)


def connection_error() -> APIConnectionError:
    return APIConnectionError(request=_HTTP_REQ)


def timeout_error() -> APITimeoutError:
    return APITimeoutError(request=_HTTP_REQ)


# ── Log helpers ───────────────────────────────────────────────────────


def read_log(path: Path) -> list[dict]:
    """Read every non-blank line in an `llm_calls.jsonl` file as a dict."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
