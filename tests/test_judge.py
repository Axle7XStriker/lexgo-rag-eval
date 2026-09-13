"""ClaudeJudge tests. Fully offline via a fake `anthropic.Anthropic` client.

Mirrors tests/test_generate.py: dependency-injection seam via `client=`,
tenacity's `wait` zeroed with monkeypatch for retry tests. The Anthropic
client fake + error factories are imported from `tests/_anthropic_fakes`.

Scope: judge-specific behavior only — the strict-JSON contract, judge
prompt loading, and the `operation="judge_answer"` log record. Retry,
pricing, and the `is_retryable` policy live in test_generate.py; both
classes route through the shared helpers in `src.pipeline.anthropic_utils`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from src.pipeline import prompts as prompts_module
from src.pipeline.anthropic_utils import cost_for
from src.pipeline.judge import ClaudeJudge, JudgeParseError
from src.pipeline.prompts import OUT_OF_CORPUS_SENTINEL
from tests._anthropic_fakes import (
    FakeClient,
    FakeMessage,
    FakeMessagesAPI,
    FakeTextBlock,
    FakeUsage,
    read_log,
)

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
    prompts_module.load_prompt.cache_clear()
    monkeypatch.setattr(prompts_module, "PROMPTS_DIR", tmp_path / "prompts")
    yield
    prompts_module.load_prompt.cache_clear()


def _make_judge(
    *,
    client: FakeClient,
    log_path: Path,
    model: str = "claude-sonnet-4-6",
) -> ClaudeJudge:
    return ClaudeJudge(
        api_key=SecretStr("test-key"),
        model=model,
        log_path=log_path,
        client=client,
    )


def _canned_verdict(
    *,
    answer_correct: bool = True,
    citations_valid: bool = True,
    rationale: str = "looks fine",
    input_tokens: int = 200,
    output_tokens: int = 30,
) -> FakeMessage:
    return FakeMessage(
        content=[
            FakeTextBlock(
                text=json.dumps(
                    {
                        "answer_correct": answer_correct,
                        "citations_semantically_valid": citations_valid,
                        "rationale": rationale,
                    }
                )
            )
        ],
        usage=FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens),
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
        client = FakeClient(
            messages=FakeMessagesAPI(
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
        assert r.cost_usd == pytest.approx(cost_for("claude-sonnet-4-6", 180, 25))

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
        client = FakeClient(
            messages=FakeMessagesAPI(
                responses=[
                    FakeMessage(
                        content=[FakeTextBlock(text=payload)],
                        usage=FakeUsage(input_tokens=10, output_tokens=5),
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
        client = FakeClient(messages=FakeMessagesAPI(responses=[_canned_verdict()]))
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
        client = FakeClient(
            messages=FakeMessagesAPI(
                responses=[
                    FakeMessage(
                        # Bare token — obviously not JSON, and doesn't share
                        # substrings with the JudgeParseError message.
                        content=[FakeTextBlock(text="???")],
                        usage=FakeUsage(input_tokens=1, output_tokens=1),
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
        client = FakeClient(
            messages=FakeMessagesAPI(
                responses=[
                    FakeMessage(
                        content=[FakeTextBlock(text=bad)],
                        usage=FakeUsage(input_tokens=1, output_tokens=1),
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
        client = FakeClient(
            messages=FakeMessagesAPI(
                responses=[
                    FakeMessage(
                        content=[FakeTextBlock(text=bad)],
                        usage=FakeUsage(input_tokens=1, output_tokens=1),
                    )
                ]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        with pytest.raises(JudgeParseError, match="answer_correct"):
            j.judge(**_judge_kwargs())

    def test_non_object_reply(self, log_path: Path) -> None:
        """A JSON array reply is well-formed JSON but not the required object shape."""
        client = FakeClient(
            messages=FakeMessagesAPI(
                responses=[
                    FakeMessage(
                        content=[FakeTextBlock(text=json.dumps([1, 2, 3]))],
                        usage=FakeUsage(input_tokens=1, output_tokens=1),
                    )
                ]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        with pytest.raises(JudgeParseError, match="JSON object"):
            j.judge(**_judge_kwargs())

    def test_parse_error_still_logs(self, log_path: Path) -> None:
        """A malformed reply is still a paid API call — cost accounting stays honest."""
        client = FakeClient(
            messages=FakeMessagesAPI(
                responses=[
                    FakeMessage(
                        content=[FakeTextBlock(text="???")],
                        usage=FakeUsage(input_tokens=42, output_tokens=7),
                    )
                ]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        with pytest.raises(JudgeParseError):
            j.judge(**_judge_kwargs())
        recs = read_log(log_path)
        assert len(recs) == 1
        assert recs[0]["operation"] == "judge_answer"
        assert recs[0]["input_tokens"] == 42


class TestLogging:
    """Every successful call writes exactly one llm_calls.jsonl record."""

    def test_log_record_shape(self, log_path: Path) -> None:
        client = FakeClient(
            messages=FakeMessagesAPI(
                responses=[_canned_verdict(input_tokens=222, output_tokens=44)]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        j.judge(**_judge_kwargs(), run_id="run_test_xyz")
        recs = read_log(log_path)
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
        assert r["cost_usd"] == pytest.approx(cost_for("claude-sonnet-4-6", 222, 44), abs=1e-6)

    def test_exactly_one_log_per_call(self, log_path: Path) -> None:
        """Even after multiple judge() calls, count == calls (no double logging)."""
        client = FakeClient(
            messages=FakeMessagesAPI(
                responses=[_canned_verdict(), _canned_verdict(), _canned_verdict()]
            )
        )
        j = _make_judge(client=client, log_path=log_path)
        for _ in range(3):
            j.judge(**_judge_kwargs())
        assert len(read_log(log_path)) == 3


class TestValidation:
    def test_unknown_model_rejected_at_init(self, log_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown Anthropic model"):
            _make_judge(client=FakeClient(), log_path=log_path, model="claude-not-real")
