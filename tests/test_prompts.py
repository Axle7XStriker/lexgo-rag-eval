"""Prompt loader tests — SafeDict semantics + line-anchored section markers.

These lock in two invariants the query + judge pipelines depend on:

  1. A literal `{...}` in a prompt body that isn't one of the loader's
     placeholders passes through as-is instead of raising KeyError. Without
     this, a prompt author who writes a JSON example like
     `{"answer_correct": true}` inside the system/user body would kill every
     run at first call.
  2. `# System` / `# User template` are matched at line starts only, not as
     substrings — a body comment that mentions `'# User template'` in text
     must not shift the parsed boundary and silently truncate the section.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import prompts as prompts_module
from src.pipeline.prompts import (
    OUT_OF_CORPUS_SENTINEL,
    load_prompt,
    render_user_template,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """The loader is `@functools.cache`d — reset between tests so each test
    reads its own on-disk fixture instead of a cached prior payload."""
    load_prompt.cache_clear()
    yield
    load_prompt.cache_clear()


def _write_prompt(dir: Path, role: str, version: str, body: str) -> Path:
    """Helper: write a prompt file at prompts/<role>/<version>.md under `dir`."""
    p = dir / role / f"{version}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


class TestSafePlaceholderPassthrough:
    """Unknown `{...}` tokens in the body must NOT crash the loader / renderer."""

    def test_literal_braces_in_system_body_pass_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A JSON-schema example inside the system body must not raise KeyError.

        Under plain `str.format`, `{answer_correct}` in the body would be
        looked up in the kwargs and blow up with `KeyError: 'answer_correct'`
        on every call — killing the run. With SafeDict, the literal survives
        and is delivered to Claude as-is.
        """
        monkeypatch.setattr(prompts_module, "PROMPTS_DIR", tmp_path)
        _write_prompt(
            tmp_path,
            "answer",
            "v1",
            "# System\n\n"
            "Sentinel: '{out_of_corpus_sentinel}'.\n\n"
            'Reply format: {"answer_correct": true, "note": "explain"}.\n\n'
            "# User template\n\n"
            "Q: {question}\nCTX: {context}\n",
        )
        system_body, _user_template = load_prompt(
            "answer",
            "v1",
            required_user_placeholders=("{question}", "{context}"),
        )
        assert OUT_OF_CORPUS_SENTINEL in system_body
        assert '{"answer_correct": true, "note": "explain"}' in system_body

    def test_literal_braces_in_user_template_pass_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """render_user_template must not blow up on literal `{key}` it didn't fill."""
        monkeypatch.setattr(prompts_module, "PROMPTS_DIR", tmp_path)
        _write_prompt(
            tmp_path,
            "answer",
            "v1",
            "# System\n\n"
            "Sentinel: '{out_of_corpus_sentinel}'.\n\n"
            "# User template\n\n"
            "Q: {question}\n"
            "CTX: {context}\n"
            'Format your reply as {"quote": "..."}\n',
        )
        _, user_template = load_prompt(
            "answer",
            "v1",
            required_user_placeholders=("{question}", "{context}"),
        )
        rendered = render_user_template(
            user_template, {"question": "what?", "context": "ctx-body"}
        )
        assert "Q: what?" in rendered
        assert "CTX: ctx-body" in rendered
        # Unfilled JSON example survives verbatim.
        assert '{"quote": "..."}' in rendered


class TestLineAnchoredMarkers:
    """The section markers are line-anchored, not substring-anchored."""

    def test_marker_word_inside_body_does_not_shift_boundary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`'# User template'` appearing inside a system body comment must NOT
        be picked up by the section finder — otherwise raw_system would
        truncate at the comment and the actual system body would be
        discarded."""
        monkeypatch.setattr(prompts_module, "PROMPTS_DIR", tmp_path)
        _write_prompt(
            tmp_path,
            "answer",
            "v1",
            "# System\n\n"
            "Sentinel: '{out_of_corpus_sentinel}'.\n"
            "Note: don't confuse this with the '# User template' section below.\n"
            "This sentence must survive parsing.\n\n"
            "# User template\n\n"
            "Q: {question}\nCTX: {context}\n",
        )
        system_body, user_template = load_prompt(
            "answer",
            "v1",
            required_user_placeholders=("{question}", "{context}"),
        )
        # If the finder had matched the in-body substring, this line would
        # have been chopped out of system_body.
        assert "This sentence must survive parsing." in system_body
        assert "Q: {question}" in user_template

    def test_missing_line_anchored_marker_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A marker embedded mid-line (no line-start) is not accepted."""
        monkeypatch.setattr(prompts_module, "PROMPTS_DIR", tmp_path)
        _write_prompt(
            tmp_path,
            "answer",
            "v1",
            "prefix # System\n\n"
            "Sentinel: '{out_of_corpus_sentinel}'.\n\n"
            "# User template\n\n"
            "Q: {question}\nCTX: {context}\n",
        )
        with pytest.raises(ValueError, match="expected '# System'"):
            load_prompt(
                "answer",
                "v1",
                required_user_placeholders=("{question}", "{context}"),
            )
