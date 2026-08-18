"""CLI lint for the hand-authored golden Q&A set.

Two entry points:
  - `validate_golden(path) -> ValidationReport` — importable, used by the
    Streamlit authoring UI's progress dashboard.
  - `main() -> int` — CLI wrapper (`make validate` / `python -m evals.validate_golden`),
    exits 0/1 for CI.

Exit code contract:
  - Parse errors OR duplicate IDs → exit 1 (something is broken).
  - Under 100 records with a clean set → exit 0 (in progress).
  - Exactly 100 records with the WRONG distribution → exit 1
    (the 40/25/20/10/5 split IS the eval design; drift silently fails it).
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
    QARecord,
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
    total_records: int
    errors: list[LineError] = field(default_factory=list)
    counts: dict[QAType, int] = field(default_factory=dict)
    duplicate_ids: list[str] = field(default_factory=list)
    # doc_path values cited by records that are not in the corpus manifest.
    # Formatted as "id → doc_path" so the operator can grep-jump to the record.
    unknown_doc_paths: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.errors and not self.duplicate_ids and not self.unknown_doc_paths

    @property
    def is_complete(self) -> bool:
        return self.total_records == GOLDEN_TOTAL

    @property
    def distribution_ok(self) -> bool:
        return self.counts == TARGET_DISTRIBUTION


def validate_golden(path: Path) -> ValidationReport:
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
    if target <= 0:
        return "[" + "-" * width + "]"
    filled = min(width, round(count / target * width))
    return "[" + "=" * filled + "-" * (width - filled) + "]"


def _print_report(report: ValidationReport, path: Path) -> None:
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
    if not report.is_clean:
        return 1
    if report.is_complete and not report.distribution_ok:
        return 1
    return 0


def main() -> int:
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


# Re-exports so external callers don't need to reach into src.qa_schema.
__all__ = [
    "QARecord",
    "ValidationReport",
    "main",
    "validate_golden",
]
