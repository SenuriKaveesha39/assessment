"""Runtime discovery and download of the Conditions of Carriage document.

Per the Part A success criteria, the document must be located and fetched by
the running solution -- never downloaded by hand and dropped into the repo.
A real (headless) browser is used to load the entry page, because the link
location is only guaranteed by rendered page content, not by a URL we're
allowed to hardcode.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

ENTRY_URL = "https://sandpit-air-website.replit.app/travel-information/"
LINK_TEXT_PATTERN = re.compile(r"conditions of carriage", re.IGNORECASE)


class DocumentNotFoundError(RuntimeError):
    """Raised when the entry page no longer links to a Conditions of Carriage document."""


@dataclass
class AcquiredDocument:
    path: Path
    source_url: str
    link_text: str
    retrieved_at: str
    sha256: str
    size_bytes: int


def discover_document_url(entry_url: str = ENTRY_URL) -> tuple[str, str]:
    """Render the entry page and locate the Conditions of Carriage link.

    Returns (absolute_document_url, link_text).
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(entry_url, wait_until="networkidle")

            for link in page.locator("a").all():
                text = (link.text_content() or "").strip()
                href = link.get_attribute("href")
                if href and LINK_TEXT_PATTERN.search(text):
                    absolute_url = page.evaluate(
                        "(href) => new URL(href, document.baseURI).href", href
                    )
                    return absolute_url, text

            raise DocumentNotFoundError(
                f"No 'Conditions of Carriage' link found on {entry_url}"
            )
        finally:
            browser.close()


def download_document(document_url: str, dest_dir: Path) -> AcquiredDocument:
    """Fetch the document bytes at runtime via Playwright's own request API."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = document_url.rsplit("/", 1)[-1] or "conditions-of-carriage.docx"
    dest_path = dest_dir / filename

    with sync_playwright() as p:
        request_context = p.request.new_context()
        try:
            response = request_context.get(document_url)
            if not response.ok:
                raise DocumentNotFoundError(
                    f"Download failed: HTTP {response.status} from {document_url}"
                )
            body = response.body()
        finally:
            request_context.dispose()

    dest_path.write_bytes(body)

    return AcquiredDocument(
        path=dest_path,
        source_url=document_url,
        link_text="",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
    )


def acquire(entry_url: str = ENTRY_URL, dest_dir: Path = Path("data/raw")) -> AcquiredDocument:
    """End-to-end acquisition: discover the link, then download the document."""
    document_url, link_text = discover_document_url(entry_url)
    acquired = download_document(document_url, dest_dir)
    acquired.link_text = link_text
    return acquired
