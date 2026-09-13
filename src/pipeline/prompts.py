"""Shared prompt-loading helpers and constants for the P1 pipeline.

Both `src.pipeline.query` (answer prompt) and `src.pipeline.judge` (judge
prompt) parse a section-based prompt file at `prompts/<role>/<version>.md`,
substitute `{out_of_corpus_sentinel}` at load time, and validate that every
per-request placeholder the caller will fill is actually present. Keeping
one loader + one sentinel here means the two callers can't drift on either
the parse rules or the sentinel literal.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

# Repo-relative prompts directory. `parents[2]` from `src/pipeline/prompts.py`
# lands on the repo root — the same anchor query.py / judge.py used before.
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# Section markers inside a prompt file. Anchored to line starts (re.MULTILINE)
# rather than substring `find` — otherwise a body comment like
# `contrast with the '# User template' section` would silently shift the
# parsed boundary and truncate the system body.
_SYSTEM_MARKER_RE = re.compile(r"^# System[ \t]*$", re.MULTILINE)
_USER_TEMPLATE_MARKER_RE = re.compile(r"^# User template[ \t]*$", re.MULTILINE)

# Exact sentinel string Claude must return when the context doesn't cover
# the question. Threaded INTO both the answer and judge prompt templates as
# `{out_of_corpus_sentinel}` so this constant is the single source of truth —
# the prompt files cannot drift to a different literal string.
OUT_OF_CORPUS_SENTINEL = "This isn't covered in the provided course materials."


def _substitute(template: str, substitutions: dict[str, str]) -> str:
    """Literal `{key}` → value substitution over an explicit allowlist.

    Deliberately does NOT go through `str.format` / `str.format_map`. The
    format mini-language treats `:` as the field/spec separator, so a
    prompt body that contains a JSON example like
    `{"answer_correct": true}` blows up with `ValueError: Invalid format
    specifier` even under a passthrough dict. Plain per-key `str.replace`
    is the only substitution semantic that leaves arbitrary `{…}` bodies
    (JSON snippets, code fragments, hints like `avoid {backticks}`) alone
    while still filling our intended placeholders.

    Consequence: only exact `{key}` tokens are substituted. `{ key }` (with
    spaces), `{{key}}` (doubled), and any nested/computed field-name syntax
    is left as-is. That's the intended contract.
    """
    for key, value in substitutions.items():
        template = template.replace("{" + key + "}", value)
    return template


@functools.cache
def load_prompt(
    role: str,
    version: str,
    *,
    required_user_placeholders: tuple[str, ...],
) -> tuple[str, str]:
    """Load a prompt file, return (system_body, user_template).

    Cached per (role, version, required_user_placeholders) — prompt files are
    immutable once shipped. The system body has `{out_of_corpus_sentinel}`
    pre-substituted at load time; the user template still contains its
    per-request placeholders for the caller to fill.

    `required_user_placeholders` is the tuple of `{...}` tokens the caller
    guarantees to substitute — every one must be present in the user
    template or loading fails fast (would otherwise ship a raw literal to
    Claude).

    Substitution is per-key `str.replace` over an explicit allowlist (see
    `_substitute`), so any `{...}` in the body that isn't a listed placeholder
    (JSON schema examples, code, literal `{backticks}`) is delivered as-is.

    Raises:
      FileNotFoundError — no file at prompts/<role>/<version>.md.
      ValueError — missing `# System` / `# User template` line-anchored
        section headers, or missing required placeholders in either section.
    """
    path = PROMPTS_DIR / role / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    text = path.read_text(encoding="utf-8")

    sys_match = _SYSTEM_MARKER_RE.search(text)
    user_match = _USER_TEMPLATE_MARKER_RE.search(text)
    if sys_match is None or user_match is None or user_match.start() <= sys_match.start():
        raise ValueError(
            f"{path}: expected '# System' then '# User template' section headers "
            f"(each on its own line)"
        )
    raw_system = text[sys_match.end() : user_match.start()].strip()
    user_template = text[user_match.end() :].strip()

    # Fail fast on template drift.
    #   - The system body MUST reference {out_of_corpus_sentinel}, or the
    #     code constant and the prompt's actual instruction would silently
    #     drift apart.
    #   - The user template MUST contain every per-request placeholder the
    #     caller fills — otherwise a raw literal would be shipped to Claude.
    if "{out_of_corpus_sentinel}" not in raw_system:
        raise ValueError(
            f"{path}: system body must reference '{{out_of_corpus_sentinel}}' "
            f"so OUT_OF_CORPUS_SENTINEL stays the single source of truth."
        )
    missing = [p for p in required_user_placeholders if p not in user_template]
    if missing:
        raise ValueError(
            f"{path}: user template missing required placeholder(s): {', '.join(missing)}"
        )

    system_body = _substitute(raw_system, {"out_of_corpus_sentinel": OUT_OF_CORPUS_SENTINEL})
    return system_body, user_template


def render_user_template(user_template: str, substitutions: dict[str, str]) -> str:
    """Substitute `substitutions` into `user_template` with the safe allowlist.

    Same `{key}` syntax as before, but implemented via per-key `str.replace`
    so unknown `{...}` tokens in the template (JSON snippets, code) pass
    through untouched and can't raise KeyError / ValueError at fill time.
    Use this at every call site that fills a prompt's user template.
    """
    return _substitute(user_template, substitutions)
