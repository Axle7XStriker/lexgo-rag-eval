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
from pathlib import Path

# Repo-relative prompts directory. `parents[2]` from `src/pipeline/prompts.py`
# lands on the repo root — the same anchor query.py / judge.py used before.
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# Section markers inside a prompt file. Markdown headings so the file also
# renders well in a browser / editor preview.
_SYSTEM_MARKER = "# System"
_USER_TEMPLATE_MARKER = "# User template"

# Exact sentinel string Claude must return when the context doesn't cover
# the question. Threaded INTO both the answer and judge prompt templates as
# `{out_of_corpus_sentinel}` so this constant is the single source of truth —
# the prompt files cannot drift to a different literal string.
OUT_OF_CORPUS_SENTINEL = "This isn't covered in the provided course materials."


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

    Raises:
      FileNotFoundError — no file at prompts/<role>/<version>.md.
      ValueError — missing `# System` / `# User template` sections, missing
        required placeholders in either section.
    """
    path = PROMPTS_DIR / role / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    text = path.read_text(encoding="utf-8")

    sys_idx = text.find(_SYSTEM_MARKER)
    user_idx = text.find(_USER_TEMPLATE_MARKER)
    if sys_idx == -1 or user_idx == -1 or user_idx <= sys_idx:
        raise ValueError(
            f"{path}: expected '{_SYSTEM_MARKER}' then '{_USER_TEMPLATE_MARKER}' sections"
        )
    raw_system = text[sys_idx + len(_SYSTEM_MARKER) : user_idx].strip()
    user_template = text[user_idx + len(_USER_TEMPLATE_MARKER) :].strip()

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

    system_body = raw_system.format(out_of_corpus_sentinel=OUT_OF_CORPUS_SENTINEL)
    return system_body, user_template
