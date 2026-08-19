"""Pinned corpus manifest — the 6 source docs behind the RAG eval.

Two source courses, three source_ids per course. Filenames are prefixed with
their source_id so `ls corpus/` and `grep A2 evals/golden/qa.jsonl` are both
self-documenting. Zero-padded lecture numbers preserve alphanumeric sort.

Kinds:
  - "ocw_resource_page": URL points to an OCW resource landing page; the
    fetcher parses its HTML for the actual (hashed) PDF asset URL.
  - "direct_pdf": URL points directly at the PDF bytes (third-party mirrors).

`urls` is a tuple of fallback URLs; the fetcher tries them in order and
takes the first that returns a valid PDF (magic bytes `%PDF`).

`optional=True` means the fetcher logs the miss and exits 0 anyway — used
for papers that lack a reliably-open host (R*-Tree in particular).
"""

from dataclasses import dataclass
from typing import Literal

from src.qa_schema import SourceId

KindLiteral = Literal["ocw_resource_page", "direct_pdf"]

OCW_006_BASE = "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources"
OCW_830_BASE = "https://ocw.mit.edu/courses/6-830-database-systems-fall-2010/resources"


@dataclass(frozen=True)
class ManifestEntry:
    """One corpus PDF as tracked by the fetcher.

    Fields:
      source_id: which of the 6 buckets this PDF belongs to (SourceId enum,
        shared with src/qa_schema.py — single source of truth).
      description: human-readable label for logs / error messages.
      kind: "ocw_resource_page" means the URL is an OCW landing page whose
        HTML must be parsed to find the actual PDF href; "direct_pdf" means
        the URL points at the PDF bytes directly.
      urls: fallback list — the fetcher tries them in order and takes the
        first that returns a valid PDF (magic bytes %PDF).
      dest_path: where to store it under corpus/, relative. Filenames start
        with the source_id (e.g. A1_lec03.pdf) so `ls` groups by bucket.
      optional: if True, a full-list failure is logged but does not fail
        the run (used for papers with no reliably-open host).
    """

    source_id: SourceId
    description: str
    kind: KindLiteral
    urls: tuple[str, ...]
    dest_path: str  # relative to corpus/
    optional: bool = False


def _ocw_006_lec(n: int) -> ManifestEntry:
    return ManifestEntry(
        source_id=SourceId.A1,
        description=f"6.006 F11 lecture {n:02d} notes",
        kind="ocw_resource_page",
        urls=(f"{OCW_006_BASE}/mit6_006f11_lec{n:02d}/",),
        dest_path=f"6.006/lectures/A1_lec{n:02d}.pdf",
    )


def _ocw_006_rec(n: int, *, optional: bool = False) -> ManifestEntry:
    return ManifestEntry(
        source_id=SourceId.A2,
        description=f"6.006 F11 recitation {n:02d} notes",
        kind="ocw_resource_page",
        urls=(f"{OCW_006_BASE}/mit6_006f11_rec{n:02d}/",),
        dest_path=f"6.006/recitations/A2_rec{n:02d}.pdf",
        optional=optional,
    )


def _ocw_006_ps(n: int, sol: bool) -> ManifestEntry:
    suffix = "_sol" if sol else ""
    label = "solutions" if sol else "problems"
    return ManifestEntry(
        source_id=SourceId.A3,
        description=f"6.006 F11 problem set {n} — {label}",
        kind="ocw_resource_page",
        urls=(f"{OCW_006_BASE}/mit6_006f11_ps{n}{suffix}/",),
        dest_path=f"6.006/problem_sets/A3_ps{n}{suffix}.pdf",
    )


def _ocw_830_lec(slug: str, dest_stem: str, *, optional: bool = False) -> ManifestEntry:
    return ManifestEntry(
        source_id=SourceId.B1,
        description=f"6.830 F10 lecture — {slug}",
        kind="ocw_resource_page",
        urls=(f"{OCW_830_BASE}/{slug}/",),
        dest_path=f"6.830/lectures/B1_{dest_stem}.pdf",
        optional=optional,
    )


# ── A1: 6.006 lectures 1-12 (models → sorting → trees → hashing → numerics)
_A1: list[ManifestEntry] = [_ocw_006_lec(n) for n in range(1, 13)]

# ── A2: 6.006 recitations 1-12. rec03/rec04 currently return 503 from OCW
# (Fastly cache errors — may resolve later; marked optional so they don't
# block the fetcher). Verified against the recitations page: the other 10
# resource pages exist and render correctly.
_OPTIONAL_A2 = {3, 4}
_A2: list[ManifestEntry] = [_ocw_006_rec(n, optional=n in _OPTIONAL_A2) for n in range(1, 13)]

# ── A3: 6.006 PS1-PS4 with solutions (8 PDFs)
_A3: list[ManifestEntry] = [_ocw_006_ps(n, sol) for n in range(1, 5) for sol in (False, True)]

# ── B1: 6.830 lectures. Enumerated against the OCW lecture-notes index —
# lec08 is genuinely skipped in the published materials (jumps 07/07b → 09/09_selinger).
# Sub-lecture PDFs (07b, 09_selinger, 20b) are published and verified.
_B1_LECTURES: list[tuple[str, str]] = [
    (f"mit6_830f10_lec{n:02d}", f"lec{n:02d}") for n in range(1, 21) if n != 8
] + [
    ("mit6_830f10_lec07b", "lec07b"),
    ("mit6_830f10_lec09_selinger", "lec09_selinger"),
    ("mit6_830f10_lec20b", "lec20b"),
]
_B1: list[ManifestEntry] = [_ocw_830_lec(slug, stem) for slug, stem in _B1_LECTURES]

# MIT CSAIL 6.830 paper cache — verified to host every paper we need.
# Kept as the primary URL for every paper below; third-party mirrors are fallbacks.
_MIT_6830_CACHE = "https://people.csail.mit.edu/tdanford/6830papers"

# ── B2: 3 papers on storage / indexing / access methods
_B2: list[ManifestEntry] = [
    ManifestEntry(
        source_id=SourceId.B2,
        description="Chou & DeWitt — Evaluation of Buffer Management Strategies (VLDB 1985)",
        kind="direct_pdf",
        urls=(
            f"{_MIT_6830_CACHE}/chou-dewitt-eval-buffer-management.pdf",
            "https://www.vldb.org/conf/1985/P127.PDF",
        ),
        dest_path="6.830/papers/B2_chou_dewitt_buffer.pdf",
    ),
    ManifestEntry(
        source_id=SourceId.B2,
        description="Beckmann et al. — The R*-Tree (SIGMOD 1990)",
        kind="direct_pdf",
        urls=(
            f"{_MIT_6830_CACHE}/beckmann-r-star-tree.pdf",
            "https://infolab.usc.edu/csci599/Fall2001/paper/rstar-tree.pdf",
        ),
        dest_path="6.830/papers/B2_rstar_tree.pdf",
    ),
    ManifestEntry(
        source_id=SourceId.B2,
        description="Stonebraker et al. — C-Store: A Column-oriented DBMS (VLDB 2005)",
        kind="direct_pdf",
        urls=(f"{_MIT_6830_CACHE}/stonebraker-cstore.pdf",),
        dest_path="6.830/papers/B2_cstore.pdf",
    ),
]

# ── B3: 4 papers on query proc / transactions / concurrency
_B3: list[ManifestEntry] = [
    ManifestEntry(
        source_id=SourceId.B3,
        description="Selinger et al. — Access Path Selection in a Relational DBMS (SIGMOD 1979)",
        kind="direct_pdf",
        urls=(
            f"{_MIT_6830_CACHE}/selinger-access-path-selection.pdf",
            "https://people.eecs.berkeley.edu/~brewer/cs262/3-selinger79.pdf",
        ),
        dest_path="6.830/papers/B3_selinger.pdf",
    ),
    ManifestEntry(
        source_id=SourceId.B3,
        description="Franklin — Concurrency Control and Recovery (1997)",
        kind="direct_pdf",
        urls=(
            f"{_MIT_6830_CACHE}/franklin-concurrency-control.pdf",
            "https://courses.cs.washington.edu/courses/cse544/11wi/papers/franklin97.pdf",
        ),
        dest_path="6.830/papers/B3_franklin.pdf",
    ),
    ManifestEntry(
        source_id=SourceId.B3,
        description="Kung & Robinson — On Optimistic Methods for Concurrency Control (TODS 1981)",
        kind="direct_pdf",
        urls=(f"{_MIT_6830_CACHE}/kung-optimistic-methods.pdf",),
        dest_path="6.830/papers/B3_kung_robinson.pdf",
    ),
    ManifestEntry(
        source_id=SourceId.B3,
        description="Gray et al. — Granularity of Locks and Degrees of Consistency (1976)",
        kind="direct_pdf",
        urls=(f"{_MIT_6830_CACHE}/gray-lock-granularity.pdf",),
        dest_path="6.830/papers/B3_gray_granularity.pdf",
    ),
]

MANIFEST: list[ManifestEntry] = _A1 + _A2 + _A3 + _B1 + _B2 + _B3
