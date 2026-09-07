#!/usr/bin/env python
"""Part B CLI: run the agent against a passenger query and print its verified answer.

Usage:
    python scripts/answer_query.py [--query "..."] [--index-dir data/index]

Requires ANTHROPIC_API_KEY to be set (e.g. in a .env file at the repo root).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agent.runner import answer_query  # noqa: E402
from agent.verification import verify_answer  # noqa: E402
from ingestion.indexer import DEFAULT_INDEX_DIR  # noqa: E402
from ingestion.pipeline import PROCESSED_PATH  # noqa: E402

DEFAULT_QUERY = (
    "A passenger is travelling on an International Business fare to South Africa, "
    "with a ticket issued on 1 January 2027. What is their checked baggage allowance "
    "and what is their carry-on baggage allowance?"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not Path(args.index_dir).exists():
        raise SystemExit(
            f"No index found at {args.index_dir}. Run scripts/run_ingestion.py first."
        )

    result = answer_query(args.query, index_dir=Path(args.index_dir))
    trace = result.pop("_trace")

    if result["type"] == "chat":
        print(result["message"])
        print(f"\n({len(trace)} tool call(s) made; see log above for detail)", file=sys.stderr)
        return

    verification = verify_answer(result, PROCESSED_PATH)
    print(json.dumps({"answer": result, "verification": verification}, indent=2))
    print(f"\n({len(trace)} tool call(s) made; see log above for detail)", file=sys.stderr)

    if not verification["all_grounded"]:
        print("\nWARNING: one or more cited facts were not found verbatim in the cited section.", file=sys.stderr)


if __name__ == "__main__":
    main()
