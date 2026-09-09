"""ClaudeJudge tests. Fully offline via a fake `anthropic.Anthropic` client.

Mirrors tests/test_generate.py: dependency-injection seam via `client=`,
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

from src.pipeline import judge as judge_module
from src.pipeline.judge import (
    PRICING,
    ClaudeJudge,
    JudgeParseError,
    _cost_for,
    _is_retryable,
)

# ── Anthropic response shape ──────────────────────────────────────────


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeMessage:
    content: list[_FakeTextBlock]
    usage: _FakeUsage
    stop_reason: str = "end_turn"


_ResponseItem = _FakeMessage | Callable[[dict], _FakeMessage]


@dataclass
class _FakeMessagesAPI:
    responses: list[_ResponseItem] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def create(self, **kwargs) -> _FakeMessage:
        self.calls.append(kwargs)
        if not self.responses:
            # Sensible default so tests that don't program responses still work.
            return _FakeMessage(
                content=[
                    _FakeTextBlock(
                        text=json.dumps(
                            {
                                "answer_correct": True,
                                "citations_semantically_valid": True,
                                "rationale": "default fake verdict",
                            }
                        )
                    )
                ],
                usage=_FakeUsage(input_tokens=100, output_tokens=20),
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


@pytest.fixture(autouse=True)
def _judge_prompt_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the judge's prompt loader at a minimal fixture prompt in tmp_path.

    Keeps tests decoupled from the shipped `prompts/judge/v1.md` wording so
    a prompt-content edit doesn't break unit tests here.
    """
    prompt_path = tmp_path / "prompts" / "judge" / "v1.md"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(
        """---
name: test-judge
version: 1
---

# System

TEST JUDGE. Sentinel: '{out_of_corpus_sentinel}'.

# User template

Q: {question}
GA: {gold_answer}
GC: {gold_citations_block}
MA: {model_answer}
MC: {model_citations_block}
""",
        encoding="utf-8",
    )
    judge_module._load_prompt.cache_clear()
    monkeypatch.setattr(judge_module, "PROMPTS_DIR", tmp_path / "prompts")
    yield
    judge_module._load_prompt.cache_clear()


def _make_judge(
    *,
    client: _FakeClient,
    log_path: Path,
    model: str = "claude-sonnet-4-6",
) -> ClaudeJudge:
    return ClaudeJudge(
        api_key=SecretStr("test-key"),
        model=model,
        log_path=log_path,
        client=client,
    )


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _canned_verdict(
    *,
    answer_correct: bool = True,
    citations_valid: bool = True,
    rationale: str = "looks fine",
    input_tokens: int = 200,
    output_tokens: int = 30,
) -> _FakeMessage:
    return _FakeMessage(
        content=[
            _FakeTextBlock(
                text=json.dumps(
                    {
                        "answer_correct": answer_correct,
                        "citations_semantically_valid": citations_valid,
                        "rationale": rationale,
                    }
                )
            )
        ],
        usage=_FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _judge_kwargs(**overrides) -> dict:
    """Default kwargs for judge.judge() — tests only override what they care about."""
    base = {
        "question": "what is merge sort?",
        "gold_answer": "merge sort is O(n log n)",
        "gold_citation_doc_paths": ["6.006/lectures/A1_lec03.pdf"],
        "model_answer": "it's O(n log n)",
        "model_citation_doc_paths": ["6.006/lectures/A1_lec03.pdf"],
    }
    base.update(overrides)
    return base


# ── Tests ─────────────────────────────────────────────────────────────


class TestHappyPath:
    """A successful judge() parses the JSON reply into a JudgeResult."""

    def test_returns_parsed_verdict(self, log_path: Path) -> None:
        client = _FakeClient(
            messages=_FakeMessagesAPI(
                responses=[
                    _canned_verdict(
                        answer_correct=True,
                        citations_valid=False,
                        rationale="cited unrelated docs",
                        input_tokens=180,
                        output_tokens=25,
                    )
                ]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        r = j.judge(**_judge_kwargs())
        assert r.answer_correct is True
        assert r.citations_semantically_valid is False
        assert r.rationale == "cited unrelated docs"
        assert r.input_tokens == 180
        assert r.output_tokens == 25
        assert r.cost_usd == pytest.approx(_cost_for("claude-sonnet-4-6", 180, 25))

    def test_handles_surrounding_whitespace(self, log_path: Path) -> None:
        """Leading/trailing whitespace in the reply is stripped before json.loads."""
        payload = (
            "\n\n  "
            + json.dumps(
                {
                    "answer_correct": False,
                    "citations_semantically_valid": True,
                    "rationale": "answer omits a claim",
                }
            )
            + "  \n"
        )
        client = _FakeClient(
            messages=_FakeMessagesAPI(
                responses=[
                    _FakeMessage(
                        content=[_FakeTextBlock(text=payload)],
                        usage=_FakeUsage(input_tokens=10, output_tokens=5),
                    )
                ]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        r = j.judge(**_judge_kwargs())
        assert r.answer_correct is False
        assert r.citations_semantically_valid is True

    def test_forwards_request_shape(self, log_path: Path) -> None:
        """judge() sends the substituted prompt as system + user to the SDK."""
        client = _FakeClient()  # default fake verdict
        j = _make_judge(client=client, log_path=log_path)
        j.judge(
            **_judge_kwargs(
                question="Q?",
                model_answer="MA",
                gold_answer="GA",
                gold_citation_doc_paths=[],
                model_citation_doc_paths=[],
            )
        )
        assert len(client.messages.calls) == 1
        call = client.messages.calls[0]
        assert call["model"] == "claude-sonnet-4-6"
        # Sentinel substituted from the code constant, not a raw literal.
        from src.pipeline.query import OUT_OF_CORPUS_SENTINEL

        assert OUT_OF_CORPUS_SENTINEL in call["system"]
        assert "{out_of_corpus_sentinel}" not in call["system"]
        # User body contains substituted placeholders.
        assert "Q: Q?" in call["messages"][0]["content"]
        assert "GA: GA" in call["messages"][0]["content"]
        assert "MA: MA" in call["messages"][0]["content"]
        # Empty citation lists render as `(none)` — unambiguous for the model.
        assert "GC: (none)" in call["messages"][0]["content"]
        assert "MC: (none)" in call["messages"][0]["content"]


class TestParseErrors:
    """Non-conforming judge replies raise JudgeParseError (not silent coerce)."""

    def test_malformed_json(self, log_path: Path) -> None:
        client = _FakeClient(
            messages=_FakeMessagesAPI(
                responses=[
                    _FakeMessage(
                        content=[_FakeTextBlock(text="not valid json {")],
                        usage=_FakeUsage(input_tokens=1, output_tokens=1),
                    )
                ]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        with pytest.raises(JudgeParseError, match="not valid JSON"):
            j.judge(**_judge_kwargs())

    def test_missing_required_key(self, log_path: Path) -> None:
        """A reply missing `rationale` raises JudgeParseError with the field named."""
        bad = json.dumps({"answer_correct": True, "citations_semantically_valid": True})
        client = _FakeClient(
            messages=_FakeMessagesAPI(
                responses=[
                    _FakeMessage(
                        content=[_FakeTextBlock(text=bad)],
                        usage=_FakeUsage(input_tokens=1, output_tokens=1),
                    )
                ]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        with pytest.raises(JudgeParseError, match="rationale"):
            j.judge(**_judge_kwargs())

    def test_wrong_type_answer_correct(self, log_path: Path) -> None:
        """`answer_correct: "yes"` (string, not bool) raises JudgeParseError."""
        bad = json.dumps(
            {
                "answer_correct": "yes",
                "citations_semantically_valid": True,
                "rationale": "hi",
            }
        )
        client = _FakeClient(
            messages=_FakeMessagesAPI(
                responses=[
                    _FakeMessage(
                        content=[_FakeTextBlock(text=bad)],
                        usage=_FakeUsage(input_tokens=1, output_tokens=1),
                    )
                ]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        with pytest.raises(JudgeParseError, match="answer_correct"):
            j.judge(**_judge_kwargs())

    def test_non_object_reply(self, log_path: Path) -> None:
        """A JSON array reply is well-formed JSON but not the required object shape."""
        client = _FakeClient(
            messages=_FakeMessagesAPI(
                responses=[
                    _FakeMessage(
                        content=[_FakeTextBlock(text=json.dumps([1, 2, 3]))],
                        usage=_FakeUsage(input_tokens=1, output_tokens=1),
                    )
                ]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        with pytest.raises(JudgeParseError, match="JSON object"):
            j.judge(**_judge_kwargs())

    def test_parse_error_still_logs(self, log_path: Path) -> None:
        """A malformed reply is still a paid API call — cost accounting stays honest."""
        client = _FakeClient(
            messages=_FakeMessagesAPI(
                responses=[
                    _FakeMessage(
                        content=[_FakeTextBlock(text="not json")],
                        usage=_FakeUsage(input_tokens=42, output_tokens=7),
                    )
                ]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        with pytest.raises(JudgeParseError):
            j.judge(**_judge_kwargs())
        recs = _read_log(log_path)
        assert len(recs) == 1
        assert recs[0]["operation"] == "judge_answer"
        assert recs[0]["input_tokens"] == 42


class TestLogging:
    """Every successful call writes exactly one llm_calls.jsonl record."""

    def test_log_record_shape(self, log_path: Path) -> None:
        client = _FakeClient(
            messages=_FakeMessagesAPI(
                responses=[_canned_verdict(input_tokens=222, output_tokens=44)]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        j.judge(**_judge_kwargs(), run_id="run_test_xyz")
        recs = _read_log(log_path)
        assert len(recs) == 1
        r = recs[0]
        assert r["provider"] == "anthropic"
        assert r["model"] == "claude-sonnet-4-6"
        assert r["operation"] == "judge_answer"
        assert r["input_tokens"] == 222
        assert r["output_tokens"] == 44
        assert r["prompt_version"] == "v1"
        assert r["run_id"] == "run_test_xyz"
        assert r["stop_reason"] == "end_turn"
        assert r["cost_usd"] == pytest.approx(_cost_for("claude-sonnet-4-6", 222, 44), abs=1e-6)

    def test_exactly_one_log_per_call(self, log_path: Path) -> None:
        """Even after multiple judge() calls, count == calls (no double logging)."""
        client = _FakeClient(
            messages=_FakeMessagesAPI(
                responses=[_canned_verdict(), _canned_verdict(), _canned_verdict()]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        for _ in range(3):
            j.judge(**_judge_kwargs())
        assert len(_read_log(log_path)) == 3


class TestRetry:
    """Transient failures retry; deterministic ones do not."""

    def test_retries_on_500(self, log_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            judge_module.ClaudeJudge._call.retry,
            "wait",
            lambda *a, **kw: 0,
        )
        state = {"n": 0}

        def _flaky(_kwargs: dict) -> _FakeMessage:
            state["n"] += 1
            if state["n"] < 2:
                raise _server_error(500)
            return _canned_verdict()

        client = _FakeClient(messages=_FakeMessagesAPI(responses=[_flaky, _flaky]))
        j = _make_judge(client=client, log_path=log_path)
        r = j.judge(**_judge_kwargs())
        assert r.answer_correct is True
        assert state["n"] == 2

    def test_fail_fast_on_400(self, log_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            judge_module.ClaudeJudge._call.retry,
            "wait",
            lambda *a, **kw: 0,
        )
        state = {"n": 0}

        def _always(_kwargs: dict) -> _FakeMessage:
            state["n"] += 1
            raise _client_error_400()

        client = _FakeClient(messages=_FakeMessagesAPI(responses=[_always] * 10))
        j = _make_judge(client=client, log_path=log_path)
        with pytest.raises(BadRequestError):
            j.judge(**_judge_kwargs())
        # Deterministic errors must not retry — exactly one attempt.
        assert state["n"] == 1


class TestValidation:
    def test_unknown_model_rejected_at_init(self, log_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown Anthropic model"):
            _make_judge(client=_FakeClient(), log_path=log_path, model="claude-not-real")


class TestPricing:
    """Cost math mirrors ClaudeGenerator — same rate table, same helper."""

    def test_cost_calc(self) -> None:
        rates = PRICING["claude-sonnet-4-6"]
        expected = rates["input"] + rates["output"]
        assert _cost_for("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(expected)

    def test_unknown_model_zero_cost(self) -> None:
        assert _cost_for("model-that-does-not-exist", 1_000_000, 1_000_000) == 0.0


class TestRetryablePredicate:
    """Same policy as ClaudeGenerator._is_retryable — smoke-check the shared shape."""

    def test_rate_limit_retryable(self) -> None:
        assert _is_retryable(_rate_limit()) is True

    def test_connection_retryable(self) -> None:
        assert _is_retryable(_connection_error()) is True

    def test_timeout_retryable(self) -> None:
        assert _is_retryable(_timeout_error()) is True

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_5xx_retryable(self, status: int) -> None:
        assert _is_retryable(_server_error(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_4xx_not_retryable(self, status: int) -> None:
        assert _is_retryable(_server_error(status)) is False

    def test_random_exception_not_retryable(self) -> None:
        assert _is_retryable(RuntimeError("nope")) is False


class TestAuthErrorFailFast:
    """Direct check: auth error surfaces from `judge` after exactly one attempt."""

    def test_auth_error(self, log_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            judge_module.ClaudeJudge._call.retry,
            "wait",
            lambda *a, **kw: 0,
        )
        state = {"n": 0}

        def _always(_kwargs: dict) -> _FakeMessage:
            state["n"] += 1
            raise _auth_error()

        client = _FakeClient(messages=_FakeMessagesAPI(responses=[_always] * 5))
        j = _make_judge(client=client, log_path=log_path)
        with pytest.raises(AuthenticationError):
            j.judge(**_judge_kwargs())
        assert state["n"] == 1
