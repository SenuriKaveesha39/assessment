"""End-to-end Part A pipeline: acquire -> parse -> chunk -> index.

Run via `python scripts/run_ingestion.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .acquisition import ENTRY_URL, acquire
from .indexer import DEFAULT_INDEX_DIR, build_chunks, index_chunks
from .parser import parse_document

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_PATH = PROCESSED_DIR / "sections.json"


def run(entry_url: str = ENTRY_URL) -> dict:
    logger.info("Discovering and downloading Conditions of Carriage from %s", entry_url)
    acquired = acquire(entry_url=entry_url, dest_dir=RAW_DIR)
    logger.info(
        "Downloaded %s (%d bytes, sha256=%s) from %s",
        acquired.path,
        acquired.size_bytes,
        acquired.sha256[:12],
        acquired.source_url,
    )

    parsed = parse_document(
        acquired.path,
        source_url=acquired.source_url,
        retrieved_at=acquired.retrieved_at,
        sha256=acquired.sha256,
        size_bytes=acquired.size_bytes,
    )
    logger.info("Parsed %d sections from %s", len(parsed.sections), parsed.metadata.title)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.write_text(parsed.model_dump_json(indent=2))
    logger.info("Wrote structured sections to %s", PROCESSED_PATH)

    chunks = build_chunks(parsed)
    count = index_chunks(chunks, index_dir=DEFAULT_INDEX_DIR)
    logger.info("Indexed %d chunks into %s", count, DEFAULT_INDEX_DIR)

    return {
        "document": acquired.path.name,
        "source_url": acquired.source_url,
        "version": parsed.metadata.version,
        "effective_date": parsed.metadata.effective_date,
        "sections": len(parsed.sections),
        "chunks": count,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run()
    print(json.dumps(summary, indent=2))
