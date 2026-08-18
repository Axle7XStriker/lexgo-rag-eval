"""Idempotent corpus downloader.

Reads scripts/corpus_manifest.py, downloads each entry into corpus/ under
its pinned dest_path, skips anything already present + non-empty. Two URL
kinds: `ocw_resource_page` (parse HTML for the PDF asset) and `direct_pdf`
(fetch bytes directly).

Design notes worth remembering:
  - Stdlib-only HTTP/HTML — no BeautifulSoup, no httpx. `tenacity` (already
    a project dep) wraps the raw GET with 3-attempt exponential backoff.
  - Atomic write via .part → os.replace so a crash mid-download can never
    leave a half-file that passes the idempotency check.
  - PDF magic-byte sniff — OCW returns HTML error pages with a 200 OK for
    some malformed URLs; checking `%PDF` catches those.
  - 1.0s inter-request politeness sleep. OCW is small non-profit infra.
  - `optional=True` entries never fail the run — they log MISSING and the
    human sources the PDF manually.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from re import IGNORECASE, findall
from urllib.parse import urljoin

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
USER_AGENT = "lexgo-rag-eval/0.1 (portfolio project; contact via github.com/Axle7XStriker)"
POLITENESS_SLEEP_S = 1.0
REQUEST_TIMEOUT_S = 30
PDF_MAGIC = b"%PDF"


class FetchError(RuntimeError):
    """A single URL failed to produce a valid PDF."""


@dataclass
class FetchResult:
    entry: ManifestEntry
    status: str  # "ok" | "skipped_present" | "failed" | "missing_optional"
    url_used: str | None = None
    error: str | None = None


def _headers() -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Accept": ("application/pdf,text/html;q=0.9,application/xhtml+xml;q=0.9,*/*;q=0.5"),
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((urllib.error.URLError, TimeoutError, FetchError)),
    reraise=True,
)
def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        status = getattr(resp, "status", 200)
        if status >= 400:
            raise FetchError(f"HTTP {status} for {url}")
        return resp.read()


def _extract_pdf_href(html: str, base_url: str) -> str:
    """Find the first .pdf href on an OCW resource page and return an absolute URL."""
    hrefs = findall(r'href="([^"]+\.pdf(?:\?[^"]*)?)"', html, IGNORECASE)
    if not hrefs:
        raise FetchError(f"no .pdf href found on {base_url}")
    return urljoin(base_url, hrefs[0])


def _download_one(entry: ManifestEntry, url: str) -> bytes:
    if entry.kind == "ocw_resource_page":
        page_bytes = _http_get(url)
        page_html = page_bytes.decode("utf-8", errors="replace")
        pdf_url = _extract_pdf_href(page_html, base_url=url)
        pdf_bytes = _http_get(pdf_url)
    else:
        pdf_bytes = _http_get(url)
    if not pdf_bytes.startswith(PDF_MAGIC):
        raise FetchError(f"response from {url} is not a PDF (first bytes: {pdf_bytes[:8]!r})")
    return pdf_bytes


def _atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dest)


def fetch_entry(entry: ManifestEntry, *, force: bool = False, log=None) -> FetchResult:
    dest = CORPUS_ROOT / entry.dest_path
    if not force and dest.exists() and dest.stat().st_size > 0:
        return FetchResult(entry, status="skipped_present")
    last_error: str | None = None
    for url in entry.urls:
        try:
            pdf_bytes = _download_one(entry, url)
        except (FetchError, urllib.error.URLError, TimeoutError, RetryError) as e:
            last_error = f"{type(e).__name__}: {e}"
            if log:
                log.warning(
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned URL + dest for each entry; download nothing.",
    )
    parser.add_argument(
        "--only",
        choices=["A1", "A2", "A3", "B1", "B2", "B3"],
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
    log = get_logger("fetch_corpus")

    entries = [e for e in MANIFEST if not args.only or e.source_id == args.only]
    log.info("fetch_corpus_start", n_entries=len(entries), dry_run=args.dry_run, force=args.force)

    if args.dry_run:
        for e in entries:
            print(f"  {e.source_id}  {e.dest_path}")
            for u in e.urls:
                print(f"      -> {u}{' (optional)' if e.optional else ''}")
        return 0

    results: list[FetchResult] = []
    for i, entry in enumerate(entries):
        result = fetch_entry(entry, force=args.force, log=log)
        results.append(result)
        marker = {
            "ok": "✓",
            "skipped_present": "·",
            "failed": "✗",
            "missing_optional": "?",
        }[result.status]
        print(f"  {marker} [{entry.source_id}] {entry.dest_path}  ({result.status})")
        # Sleep between downloads that actually hit the network.
        if result.status == "ok" and i < len(entries) - 1:
            time.sleep(POLITENESS_SLEEP_S)

    _print_summary(results)
    any_required_failed = any(r.status == "failed" for r in results)
    return 1 if any_required_failed else 0


if __name__ == "__main__":
    sys.exit(main())
