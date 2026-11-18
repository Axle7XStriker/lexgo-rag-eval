"""Idempotent corpus downloader.

Download the corpus into corpus/ under its pinned dest_path without replacing
valid local PDFs. Writes are atomic, responses must contain PDF bytes, and
optional manifest entries do not fail the run.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from scripts.corpus_manifest import MANIFEST, ManifestEntry
from src.observability import configure_logging, get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "corpus"
USER_AGENT = "lexgo-rag-eval/0.1 (+https://github.com/Axle7XStriker/lexgo-rag-eval)"
POLITENESS_SLEEP_S = 1.0
REQUEST_TIMEOUT_S = 30
PDF_MAGIC = b"%PDF"
MAX_RESPONSE_BYTES = 200 * 1024 * 1024  # Bounds unexpectedly large mirror responses.
ALLOWED_SCHEMES = frozenset({"http", "https"})  # Excludes local-file sources.
_RETRIABLE_STATUS = frozenset({429, *range(500, 600)})

_PDF_HREF_RE = re.compile(
    r"""
    href \s* = \s*
    (?:
        " ( [^"]+ \.pdf (?:\?[^"]*)? ) "
      | ' ( [^']+ \.pdf (?:\?[^']*)? ) '
      | ( [^\s'"<>]+ \.pdf (?:\?\S*)? )
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


class FetchError(RuntimeError):
    """A single URL attempt failed for a deterministic (non-retriable) reason.

    Wraps failure modes that will not resolve by retrying, so `fetch_entry` can
    fall back to the next URL in the entry's `urls` tuple immediately:
      - 4xx HTTP status (URL is gone or wrong),
      - no `<a href="*.pdf">` on an OCW resource page (HTML shape changed),
      - response bytes don't begin with the %PDF magic (server returned an
        HTML error page with 200 OK, or a redirect to a different asset),
      - URL scheme is not http/https.

    See `TransientHTTPError` for the retriable counterpart.
    """


class TransientHTTPError(RuntimeError):
    """HTTP 429 or 5xx — worth retrying with backoff.

    Kept separate from FetchError so tenacity's retry predicate can target
    only the transient cases and avoid burning ~15s on deterministic 4xx.
    """


@dataclass
class FetchResult:
    """Outcome of one manifest entry — one of ok / skipped_present / failed / missing_optional."""

    entry: ManifestEntry
    status: str  # "ok" | "skipped_present" | "failed" | "missing_optional"
    url_used: str | None = None
    error: str | None = None


def _headers() -> dict[str, str]:
    """HTTP headers sent with every request."""
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,text/html;q=0.9,application/xhtml+xml;q=0.9,*/*;q=0.5",
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TransientHTTPError, TimeoutError)),
    reraise=True,
)
def _http_get(url: str) -> bytes:
    """Fetch the body of `url` as bytes.

    Raises:
      FetchError — for deterministic failures (bad scheme, 4xx). Not retried.
      TransientHTTPError — 429 or 5xx. Retried up to 3x with 1-10s backoff.
      TimeoutError — socket timeout. Retried the same way.
    """
    if urlparse(url).scheme not in ALLOWED_SCHEMES:
        raise FetchError(f"unsupported URL scheme in {url!r}; only http/https are allowed")
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            data = resp.read(MAX_RESPONSE_BYTES + 1)  # One extra byte flags oversize.
    except urllib.error.HTTPError as e:
        if e.code in _RETRIABLE_STATUS:
            raise TransientHTTPError(f"HTTP {e.code} for {url}") from e
        raise FetchError(f"HTTP {e.code} for {url}") from e
    if len(data) > MAX_RESPONSE_BYTES:
        raise FetchError(
            f"response from {url} exceeds {MAX_RESPONSE_BYTES} bytes; refusing to buffer"
        )
    return data


def _extract_pdf_href(html: str, base_url: str, prefer_containing: str | None = None) -> str:
    """Return a .pdf href from `html` as an absolute URL, else raise.

    Handles double-quoted, single-quoted, and unquoted `href` attributes, and
    tolerates a `?query` suffix. If `prefer_containing` is given, prefers the
    first href whose path contains that string (case-insensitive).
    """
    hrefs = [g for triple in _PDF_HREF_RE.findall(html) for g in triple if g]
    if not hrefs:
        raise FetchError(f"no .pdf href found on {base_url}")
    if prefer_containing:
        needle = prefer_containing.lower()
        for h in hrefs:
            if needle in h.lower():
                return urljoin(base_url, h)
    return urljoin(base_url, hrefs[0])


def _slug_from_resource_url(url: str) -> str | None:
    """Extract the OCW resource slug from a URL like `.../resources/mit6_006f11_lec03/`."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    return parts[-1] if parts else None


def _download_one(entry: ManifestEntry, url: str) -> bytes:
    """Fetch one candidate `url` for `entry` and return its PDF bytes."""
    if entry.kind == "ocw_resource_page":
        page_bytes = _http_get(url)
        page_html = page_bytes.decode("utf-8", errors="replace")
        pdf_url = _extract_pdf_href(
            page_html,
            base_url=url,
            prefer_containing=_slug_from_resource_url(url),
        )
        pdf_bytes = _http_get(pdf_url)
    else:
        pdf_bytes = _http_get(url)
    if not pdf_bytes.startswith(PDF_MAGIC):
        raise FetchError(f"response from {url} is not a PDF (first bytes: {pdf_bytes[:8]!r})")
    return pdf_bytes


def _atomic_write(dest: Path, data: bytes) -> None:
    """Write `data` to `dest` via a .part sidecar"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def fetch_entry(entry: ManifestEntry, *, force: bool = False, logger=None) -> FetchResult:
    """Fetch a single manifest entry, trying its URLs in order until one works.

    Idempotency: returns `skipped_present` without any HTTP work when the dest
    already exists non-empty (unless force=True). Each URL is downloaded via
    `_download_one`, which fails through to the next candidate on any FetchError.
    If all URLs fail, the result status is `failed` (or `missing_optional` when
    entry.optional is True, so the outer run can still exit 0).
    """
    dest = CORPUS_ROOT / entry.dest_path
    if not force and dest.exists() and dest.stat().st_size > 0:
        return FetchResult(entry, status="skipped_present")
    last_error: str | None = None
    for url in entry.urls:
        try:
            pdf_bytes = _download_one(entry, url)
        except (
            FetchError,
            TransientHTTPError,
            urllib.error.URLError,
            TimeoutError,
            RetryError,
        ) as e:
            last_error = f"{type(e).__name__}: {e}"
            if logger:
                logger.warning(
                    "fetch_url_failed",
                    source_id=entry.source_id,
                    dest_path=entry.dest_path,
                    url=url,
                    error=last_error,
                )
            continue
        _atomic_write(dest, pdf_bytes)
        return FetchResult(entry, status="ok", url_used=url)
    status = "missing_optional" if entry.optional else "failed"
    return FetchResult(entry, status=status, error=last_error)


def _print_summary(results: list[FetchResult]) -> None:
    """Print an ok/skipped/failed tally to stdout, then list any failures.

    Sample output::

        ── fetch_corpus summary ─────────────────────────────────────
          ok:               57    skipped:    2    failed:     0    missing (optional): 2

          Optional missing (source manually if needed):
            - A2/6.006/recitations/A2_rec03.pdf
            - A2/6.006/recitations/A2_rec04.pdf
        ─────────────────────────────────────────────────────────────
    """
    buckets: dict[str, list[FetchResult]] = {
        "ok": [],
        "skipped_present": [],
        "failed": [],
        "missing_optional": [],
    }
    for r in results:
        buckets[r.status].append(r)

    print("\n── fetch_corpus summary ─────────────────────────────────────")
    print(
        f"  ok:               {len(buckets['ok']):3d}"
        f"    skipped:  {len(buckets['skipped_present']):3d}"
        f"    failed:   {len(buckets['failed']):3d}"
        f"    missing (optional): {len(buckets['missing_optional']):3d}"
    )
    if buckets["failed"]:
        print("\n  REQUIRED failures (fetcher exits 1):")
        for r in buckets["failed"]:
            print(f"    - {r.entry.source_id}/{r.entry.dest_path} :: {r.error}")
    if buckets["missing_optional"]:
        print("\n  Optional missing (source manually if needed):")
        for r in buckets["missing_optional"]:
            print(f"    - {r.entry.source_id}/{r.entry.dest_path}")
    print("─────────────────────────────────────────────────────────────\n")


def main() -> int:
    """CLI entry: iterate MANIFEST, fetch each entry, print progress + summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned URL + dest for each entry; download nothing.",
    )
    parser.add_argument(
        "--only",
        choices=["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5"],
        help="Restrict to entries with this source_id.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the dest file already exists.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Log level (default: INFO or $LOG_LEVEL).",
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger = get_logger("fetch_corpus")

    entries = [e for e in MANIFEST if not args.only or e.source_id == args.only]
    logger.info(
        "fetch_corpus_start", n_entries=len(entries), dry_run=args.dry_run, force=args.force
    )

    if args.dry_run:
        for e in entries:
            print(f"  {e.source_id}  {e.dest_path}")
            for u in e.urls:
                print(f"      -> {u}{' (optional)' if e.optional else ''}")
        return 0

    results: list[FetchResult] = []
    for i, entry in enumerate(entries):
        result = fetch_entry(entry, force=args.force, logger=logger)
        results.append(result)
        marker = {
            "ok": "✓",
            "skipped_present": "·",
            "failed": "✗",
            "missing_optional": "?",
        }[result.status]
        print(f"  {marker} [{entry.source_id}] {entry.dest_path}  ({result.status})")
        if result.status == "ok" and i < len(entries) - 1:
            time.sleep(POLITENESS_SLEEP_S)

    _print_summary(results)
    any_required_failed = any(r.status == "failed" for r in results)
    return 1 if any_required_failed else 0


if __name__ == "__main__":
    sys.exit(main())
