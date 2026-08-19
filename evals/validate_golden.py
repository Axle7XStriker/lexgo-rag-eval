"""CLI lint for the hand-authored golden Q&A set.

Two entry points share the same validation logic:
  - `validate_golden(path) -> ValidationReport` — importable, used by the
    Streamlit authoring UI's progress dashboard.
  - `main() -> int` — CLI wrapper (`make validate` / `python -m evals.validate_golden`).

Exit codes (CLI only):
  0 — nothing broken. This includes:
      * the file is empty or has fewer than 100 records (authoring in progress);
      * exactly 100 records AND the distribution matches TARGET_DISTRIBUTION.
  1 — something needs fixing. This fires when ANY of:
      * a line failed to parse as JSON or violated the QARecord schema;
      * two records share the same id;
      * a citation's doc_path is not in scripts/corpus_manifest.py
        (retrieval will not index it, so the eval would be broken);
      * total_records == 100 AND the per-type distribution ≠ 40/25/20/10/5
        (the split IS the eval design — silent drift would falsify results).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from scripts.corpus_manifest import MANIFEST
from src.qa_schema import (
    GOLDEN_TOTAL,
    TARGET_DISTRIBUTION,
    LineError,
    QAType,
    counts_by_type,
    load_jsonl,
)

# The single source of truth for "which PDFs are in the corpus." Any citation
# whose doc_path is not here means either (a) a typo, or (b) someone forgot to
# add the PDF to the manifest. Both are eval-breaking — treat as errors.
KNOWN_DOC_PATHS: set[str] = {entry.dest_path for entry in MANIFEST}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA_PATH = REPO_ROOT / "evals" / "golden" / "qa.jsonl"


@dataclass
class ValidationReport:
    """Aggregate view of a qa.jsonl scan — one object drives both the CLI
    output and the Streamlit dashboard.

    Fields:
      total_records: successfully-parsed records (does not count errors).
      errors: per-line parse or schema errors, with 1-indexed line numbers.
      counts: parsed-record count per QAType, with zeros for missing types.
      duplicate_ids: any id appearing on more than one record, sorted.
      unknown_doc_paths: "<record_id> → <doc_path>" strings for every
        citation whose doc_path is not in scripts/corpus_manifest.py.
    """

    total_records: int
    errors: list[LineError] = field(default_factory=list)
    counts: dict[QAType, int] = field(default_factory=dict)
    duplicate_ids: list[str] = field(default_factory=list)
    unknown_doc_paths: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True iff no errors, no duplicate ids, and no unknown doc_paths."""
        return not self.errors and not self.duplicate_ids and not self.unknown_doc_paths

    @property
    def is_complete(self) -> bool:
        """True iff total_records == GOLDEN_TOTAL (100)."""
        return self.total_records == GOLDEN_TOTAL

    @property
    def distribution_ok(self) -> bool:
        """True iff per-type counts exactly match TARGET_DISTRIBUTION (40/25/20/10/5)."""
        return self.counts == TARGET_DISTRIBUTION


def validate_golden(path: Path) -> ValidationReport:
    """Load qa.jsonl and compute the ValidationReport summary in one pass.

    Missing file is treated the same as an empty file: total_records=0 with
    no errors. Both the CLI and the Streamlit dashboard call this — the
    returned dataclass is the shared surface.
    """
    records, errors = load_jsonl(path)
    id_counter = Counter(r.id for r in records)
    duplicate_ids = sorted(id_ for id_, n in id_counter.items() if n > 1)
    unknown_doc_paths: list[str] = []
    for r in records:
        for c in r.gold_citations:
            if c.doc_path not in KNOWN_DOC_PATHS:
                unknown_doc_paths.append(f"{r.id} → {c.doc_path}")
    return ValidationReport(
        total_records=len(records),
        errors=errors,
        counts=counts_by_type(records),
        duplicate_ids=duplicate_ids,
        unknown_doc_paths=unknown_doc_paths,
    )


def _bar(count: int, target: int, width: int = 20) -> str:
    """Return an ASCII progress bar of the form `[========------------]`.

    `filled` cells reflect count/target (clamped to width). A zero target
    returns an all-empty bar rather than crashing.
    """
    if target <= 0:
        return "[" + "-" * width + "]"
    filled = min(width, round(count / target * width))
    return "[" + "=" * filled + "-" * (width - filled) + "]"


def _print_report(report: ValidationReport, path: Path) -> None:
    """Print a human-readable report of the scan to stdout.

    Sample output on an in-progress golden set:

        lexgo golden set: evals/golden/qa.jsonl
          loaded: 17 / 100 records    errors: 0    duplicate ids: 0

          progress:
            factual                      8 / 40   [====----------------]
            cross_source_synthesis       4 / 25   [===-----------------]
            semantic_paraphrase          3 / 20   [===-----------------]
            out_of_corpus                1 / 10   [==------------------]
            adversarial                  1 / 5    [====----------------]

    When errors, duplicate ids, or unknown doc_paths are present, they are
    listed above the progress table. When the set is complete + distribution
    correct, a `✓ 100 records, distribution matches target` line is appended.
    """
    print(f"\nlexgo golden set: {path}")
    print(
        f"  loaded: {report.total_records} / {GOLDEN_TOTAL} records"
        f"    errors: {len(report.errors)}    duplicate ids: {len(report.duplicate_ids)}"
    )

    if report.errors:
        print("\n  errors:")
        for e in report.errors:
            first_line = e.error.split("\n", 1)[0]
            print(f"    line {e.line_no}: {first_line}")
    if report.duplicate_ids:
        print("\n  duplicate ids:")
        for id_ in report.duplicate_ids:
            print(f"    - {id_}")
    if report.unknown_doc_paths:
        print("\n  unknown doc_paths (not in scripts/corpus_manifest.py):")
        for entry in report.unknown_doc_paths:
            print(f"    - {entry}")

    print("\n  progress:")
    for qa_type, target in TARGET_DISTRIBUTION.items():
        count = report.counts.get(qa_type, 0)
        marker = "  " if count == target else ("!!" if count > target else "  ")
        print(f"    {qa_type.value:<26} {count:>3d} / {target:<3d}  {_bar(count, target)} {marker}")
    print()

    if report.is_complete and not report.distribution_ok:
        print("  ⚠ 100 records but distribution ≠ target (40/25/20/10/5)")
    elif report.is_complete and report.distribution_ok:
        print("  ✓ 100 records, distribution matches target")


def _typed_report_exit_code(report: ValidationReport) -> int:
    """Map a ValidationReport to the CLI exit code — see module docstring."""
    if not report.is_clean:
        return 1
    if report.is_complete and not report.distribution_ok:
        return 1
    return 0


def main() -> int:
    """CLI entry point. Parses --path, runs `validate_golden`, prints the
    report, and returns the exit code documented in the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_QA_PATH,
        help=f"Path to qa.jsonl (default: {DEFAULT_QA_PATH.relative_to(REPO_ROOT)}).",
    )
    args = parser.parse_args()

    report = validate_golden(args.path)
    _print_report(report, args.path)
    return _typed_report_exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
