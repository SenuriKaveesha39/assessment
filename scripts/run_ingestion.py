#!/usr/bin/env python
"""CLI entry point for Part A: acquire the Conditions of Carriage document at
runtime, parse it into semantic sections, and build the retrieval index.

Usage:
    python scripts/run_ingestion.py [--entry-url URL]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingestion.acquisition import ENTRY_URL  # noqa: E402
from ingestion.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-url", default=ENTRY_URL, help="Travel information entry page")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run(entry_url=args.entry_url)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
