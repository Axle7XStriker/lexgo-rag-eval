"""Author page — hand-curate the 100-Q&A golden set.

LLM generation of Q&As is prohibited (see CLAUDE.md) — this UI is the tool
the human uses to enter every record. Structure:

  1. Progress dashboard — live counts vs the 40/25/20/10/5 target distribution.
  2. Errors expander — surfaces any invalid records in qa.jsonl with line numbers.
  3. "Add" tab — form to enter a new Q&A. Auto-generates the ID, auto-derives
     `sources` from citations, validates via pydantic, atomic-writes on save.
  4. "Browse / edit / delete" tab — filterable table; click a row to edit
     in-place or delete with two-step confirm.

Every mutation goes through the same read → mutate → atomic-rewrite → rerun
loop. Never append-mode: keeping one write path means fewer surprises when
we're 80 records deep with two weeks of authoring on the line.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import ValidationError

from evals.validate_golden import validate_golden
from scripts.corpus_manifest import MANIFEST
from src.qa_schema import (
    GOLDEN_TOTAL,
    TARGET_DISTRIBUTION,
    Citation,
    QARecord,
    QAType,
    SourceId,
    load_jsonl,
    next_id_for_type,
    save_jsonl_atomic,
)
from src.ui_helpers import load_settings_or_stop, render_page_header, render_sidebar

REPO_ROOT = Path(__file__).resolve().parents[1]
QA_PATH = REPO_ROOT / "evals" / "golden" / "qa.jsonl"

# Sorted list of every corpus doc_path — feeds the citation dropdown.
# Sorted so the dropdown groups by source_id (A1 lectures → A2 recitations → ... → B3).
CORPUS_DOC_PATHS: list[str] = sorted(entry.dest_path for entry in MANIFEST)

st.set_page_config(page_title="lexgo — author", page_icon="✍️", layout="wide")

settings = load_settings_or_stop()
render_sidebar(settings)
render_page_header(
    "Author golden Q&As",
    "Hand-authored, LLM-generation prohibited. Every entry lands in "
    "`evals/golden/qa.jsonl` and is the load-bearing artifact of the whole eval.",
)

QA_PATH.parent.mkdir(parents=True, exist_ok=True)
if not QA_PATH.exists():
    QA_PATH.touch()

# ── Load state ─────────────────────────────────────────────────────
records, load_errors = load_jsonl(QA_PATH)
report = validate_golden(QA_PATH)


# ── Progress dashboard ─────────────────────────────────────────────
def render_progress_dashboard() -> None:
    """Render 6 tiles (5 QAType + 1 total) with st.metric + st.progress.

    Each tile shows `<count> / <target>`, a signed delta (e.g. `+2`, `-3`,
    or `on target`), and a proportional progress bar clamped to [0, 1].
    Reads from the outer `report` closure so callers don't need to pass it.
    """
    cols = st.columns(len(TARGET_DISTRIBUTION) + 1)
    for i, (qa_type, target) in enumerate(TARGET_DISTRIBUTION.items()):
        count = report.counts.get(qa_type, 0)
        delta = count - target
        with cols[i]:
            st.metric(
                label=qa_type.value.replace("_", " "),
                value=f"{count} / {target}",
                delta=f"{delta:+d}" if count != target else "on target",
                delta_color=("normal" if count == target else ("off" if delta < 0 else "inverse")),
            )
            st.progress(min(count / target, 1.0) if target else 0.0)
    with cols[-1]:
        st.metric(
            label="**total**",
            value=f"{report.total_records} / {GOLDEN_TOTAL}",
            delta=f"{report.total_records - GOLDEN_TOTAL:+d}"
            if report.total_records != GOLDEN_TOTAL
            else "complete",
        )
        st.progress(min(report.total_records / GOLDEN_TOTAL, 1.0))


render_progress_dashboard()

if report.errors or report.duplicate_ids:
    with st.expander(
        f"⚠ {len(report.errors)} invalid records"
        f"{f' + {len(report.duplicate_ids)} duplicate ids' if report.duplicate_ids else ''}",
        expanded=True,
    ):
        for e in report.errors:
            st.code(f"line {e.line_no}: {e.error.splitlines()[0]}\n  {e.raw[:200]}")
        for id_ in report.duplicate_ids:
            st.warning(f"duplicate id: {id_}")

st.divider()


# ── Helpers ────────────────────────────────────────────────────────
def _empty_citation_row() -> dict[str, Any]:
    """Seed row for the citation editor. Uses the first manifest path as the
    default doc_path selection so the dropdown always renders a valid value."""
    return {"doc_path": CORPUS_DOC_PATHS[0], "page_or_section": "", "quote": ""}


def _citation_editor(key: str, initial: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """st.data_editor for citations. Returns current rows; empty rows dropped downstream.

    `source_id` is auto-derived from `doc_path` on save (via Citation.source_id) —
    users pick the specific PDF, source_id is inferred from its filename prefix.
    """
    edited = st.data_editor(
        initial,
        key=key,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "doc_path": st.column_config.SelectboxColumn(
                "PDF",
                options=CORPUS_DOC_PATHS,
                required=True,
                width="large",
                help=(
                    "Specific corpus PDF cited. "
                    "Source bucket (A1..B3) is inferred from filename prefix."
                ),
            ),
            "page_or_section": st.column_config.TextColumn(
                "Page / section",
                required=True,
                width="medium",
                help="e.g. 'slide 12', '§4.2, p.87', 'Problem 3'",
            ),
            "quote": st.column_config.TextColumn(
                "Quote (optional)",
                width="large",
                help="Optional short excerpt supporting the answer.",
            ),
        },
    )
    return edited if isinstance(edited, list) else list(edited)


def _build_record(
    *,
    qa_type: QAType,
    id_: str,
    question: str,
    gold_answer: str,
    citation_rows: list[dict[str, Any]],
    notes: str | None,
    created_at: datetime | None = None,
) -> QARecord:
    # Drop rows that are missing either required field.
    kept = [
        r
        for r in citation_rows
        if (str(r.get("doc_path") or "").strip()) and (str(r.get("page_or_section") or "").strip())
    ]
    citations = [
        Citation(
            doc_path=str(row["doc_path"]).strip(),
            page_or_section=str(row["page_or_section"]).strip(),
            quote=(str(row["quote"]).strip() or None) if row.get("quote") is not None else None,
        )
        for row in kept
    ]
    # `sources` on QARecord is a derived @cached_property computed from citations.
    now = datetime.now(UTC)
    return QARecord(
        id=id_,
        type=qa_type,
        question=question.strip(),
        gold_answer=gold_answer.strip(),
        gold_citations=citations,
        notes=(notes.strip() or None) if notes else None,
        created_at=created_at or now,
        updated_at=now,
    )


def _save_and_rerun(new_records: list[QARecord], msg: str) -> None:
    """Atomic-rewrite qa.jsonl and force a Streamlit rerun.

    The canonical write path used by add / edit / delete — keeping one code
    path means every mutation gets the same crash-safety guarantees and the
    same dashboard-refresh behavior.
    """
    save_jsonl_atomic(QA_PATH, new_records)
    st.success(msg)
    st.rerun()


# ── Tabs ───────────────────────────────────────────────────────────
tab_add, tab_browse = st.tabs(["➕ Add", "📋 Browse / edit / delete"])


# ─────────────────────────── ADD TAB ───────────────────────────────
with tab_add:
    st.markdown("#### New Q&A")
    st.caption(
        "Type + citations are edited outside the form so the citation editor "
        "reacts to row add/remove immediately. On save, form fields clear; "
        "type + citations persist across saves (usually you're batching entries "
        "against the same source)."
    )

    add_type_str = st.selectbox(
        "Type",
        options=[t.value for t in QAType],
        key="add_type",
        help="See CLAUDE.md for what each type measures.",
    )
    add_type = QAType(add_type_str)

    is_ooc = add_type == QAType.OUT_OF_CORPUS
    if is_ooc:
        st.info(
            "Out-of-corpus questions must have **no citations** and **no sources** "
            "— the correct answer is that the corpus doesn't cover it."
        )
        citation_rows: list[dict[str, Any]] = []
    else:
        st.markdown("**Citations** — one row per source cited. Empty rows are ignored.")
        initial_rows: list[dict[str, Any]] = st.session_state.get("add_citations_initial") or [
            _empty_citation_row()
        ]
        citation_rows = _citation_editor("add_citations_editor", initial_rows)

    # Preview derived sources (auto-computed from doc_path filename prefix).
    if citation_rows:
        preview_sources = sorted(
            {
                str(r["doc_path"]).rsplit("/", 1)[-1][:2]
                for r in citation_rows
                if r.get("doc_path") and str(r.get("page_or_section") or "").strip()
            }
        )
        if preview_sources:
            st.caption(
                f"**Sources (auto-derived):** {', '.join(preview_sources)} "
                f"({len(preview_sources)} distinct)"
            )

    with st.form("add_qa_form", clear_on_submit=True):
        question = st.text_area(
            "Question",
            key="add_question",
            height=80,
            placeholder="e.g. What is the worst-case time complexity of merge sort?",
        )
        gold_answer = st.text_area(
            "Gold answer (2-4 sentences)",
            key="add_gold_answer",
            height=140,
            placeholder="Direct, grounded answer. 50-800 chars.",
        )
        notes = st.text_area(
            "Notes (optional)",
            key="add_notes",
            height=60,
            placeholder="Anything the judge or you-later should know.",
        )
        submitted = st.form_submit_button("Save Q&A", type="primary")

    if submitted:
        # ID: freshly computed from currently loaded records for this type.
        new_id = next_id_for_type(records, add_type)
        try:
            new_record = _build_record(
                qa_type=add_type,
                id_=new_id,
                question=question,
                gold_answer=gold_answer,
                citation_rows=citation_rows,
                notes=notes,
            )
        except (ValidationError, ValueError) as e:
            st.error("Validation failed — record not saved.")
            st.code(str(e))
        else:
            new_records = [*records, new_record]
            _save_and_rerun(new_records, f"Saved {new_record.id}")


# ─────────────────────────── BROWSE TAB ─────────────────────────────
with tab_browse:
    if not records:
        st.info("No records yet. Add your first Q&A on the ➕ Add tab.")
    else:
        st.markdown(f"**{len(records)} records** in `{QA_PATH.relative_to(REPO_ROOT)}`.")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            type_filter = st.multiselect(
                "Filter: type",
                options=[t.value for t in QAType],
                default=[],
                help="Empty = all types.",
            )
        with col_f2:
            source_filter = st.multiselect(
                "Filter: source",
                options=[s.value for s in SourceId],
                default=[],
                help="Empty = all sources. Matches records that cite ANY of the selected sources.",
            )

        def _matches(r: QARecord) -> bool:
            if type_filter and r.type.value not in type_filter:
                return False
            if source_filter:
                record_sources = {s.value for s in r.sources}
                if not (record_sources & set(source_filter)):
                    return False
            return True

        filtered = [r for r in records if _matches(r)]
        st.caption(f"Showing {len(filtered)} / {len(records)}")

        table_rows = [
            {
                "id": r.id,
                "type": r.type.value,
                "sources": ", ".join(s.value for s in r.sources) or "—",
                "question": r.question if len(r.question) <= 90 else r.question[:87] + "…",
                "citations": len(r.gold_citations),
                "updated_at": r.updated_at.strftime("%Y-%m-%d %H:%M"),
            }
            for r in filtered
        ]

        selection_state = st.dataframe(
            table_rows,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="browse_table",
        )

        # st.dataframe returns a state object with .selection when on_select is set.
        selected_rows: list[int] = (
            getattr(selection_state, "selection", {}).get("rows", [])
            if hasattr(selection_state, "selection")
            else []
        )

        st.divider()

        if not selected_rows:
            st.info("Select a row above to edit or delete.")
        else:
            selected_record = filtered[selected_rows[0]]
            _render_edit_panel_kwargs = dict(
                record=selected_record,
                all_records=records,
            )
            st.markdown(f"#### Edit `{selected_record.id}`")

            edit_type_str = st.selectbox(
                "Type",
                options=[t.value for t in QAType],
                index=[t.value for t in QAType].index(selected_record.type.value),
                key=f"edit_type_{selected_record.id}",
                help="Changing type also changes the id prefix — new id will be auto-generated.",
            )
            edit_type = QAType(edit_type_str)
            is_ooc_edit = edit_type == QAType.OUT_OF_CORPUS

            if is_ooc_edit:
                st.info("Out-of-corpus: citations + sources will be dropped on save.")
                edit_citation_rows: list[dict[str, Any]] = []
            else:
                initial_edit_rows = [
                    {
                        "doc_path": c.doc_path,
                        "page_or_section": c.page_or_section,
                        "quote": c.quote or "",
                    }
                    for c in selected_record.gold_citations
                ] or [_empty_citation_row()]
                edit_citation_rows = _citation_editor(
                    f"edit_citations_{selected_record.id}",
                    initial_edit_rows,
                )

            edit_question = st.text_area(
                "Question",
                value=selected_record.question,
                key=f"edit_q_{selected_record.id}",
                height=80,
            )
            edit_gold_answer = st.text_area(
                "Gold answer",
                value=selected_record.gold_answer,
                key=f"edit_a_{selected_record.id}",
                height=140,
            )
            edit_notes = st.text_area(
                "Notes",
                value=selected_record.notes or "",
                key=f"edit_n_{selected_record.id}",
                height=60,
            )

            col_save, col_delete, col_spacer = st.columns([1, 1, 4])
            with col_save:
                save_clicked = st.button(
                    "💾 Save changes", type="primary", key=f"save_{selected_record.id}"
                )
            with col_delete:
                delete_clicked = st.button(
                    "🗑️ Delete", type="secondary", key=f"delete_{selected_record.id}"
                )

            pending_delete = st.session_state.get("pending_delete_id")

            if save_clicked:
                # If type unchanged, keep the id. If changed, mint a new one for the new type.
                keep_id = (
                    selected_record.id
                    if edit_type == selected_record.type
                    else next_id_for_type(
                        [r for r in records if r.id != selected_record.id],
                        edit_type,
                    )
                )
                try:
                    updated = _build_record(
                        qa_type=edit_type,
                        id_=keep_id,
                        question=edit_question,
                        gold_answer=edit_gold_answer,
                        citation_rows=edit_citation_rows,
                        notes=edit_notes,
                        created_at=selected_record.created_at,
                    )
                except (ValidationError, ValueError) as e:
                    st.error("Validation failed — nothing saved.")
                    st.code(str(e))
                else:
                    new_records = [updated if r.id == selected_record.id else r for r in records]
                    _save_and_rerun(new_records, f"Updated {updated.id}")

            if delete_clicked:
                st.session_state["pending_delete_id"] = selected_record.id
                st.rerun()

            if pending_delete == selected_record.id:
                st.warning(f"Really delete **{selected_record.id}**? This cannot be undone.")
                col_yes, col_no, _ = st.columns([1, 1, 4])
                with col_yes:
                    if st.button(
                        "Yes, delete", type="primary", key=f"confirm_del_{selected_record.id}"
                    ):
                        new_records = [r for r in records if r.id != selected_record.id]
                        st.session_state.pop("pending_delete_id", None)
                        _save_and_rerun(new_records, f"Deleted {selected_record.id}")
                with col_no:
                    if st.button("Cancel", key=f"cancel_del_{selected_record.id}"):
                        st.session_state.pop("pending_delete_id", None)
                        st.rerun()
