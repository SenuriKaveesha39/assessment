#!/usr/bin/env python
"""Part C CLI: answer the passenger query (Part B), then populate and save the
confirmation letter from that verified answer.

Usage:
    python scripts/generate_letter.py [--query "..."] [--template PATH] [--output PATH]

Requires ANTHROPIC_API_KEY to be set (e.g. in a .env file at the repo root),
and an index already built by scripts/run_ingestion.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agent.runner import answer_query, create_agent  # noqa: E402
from agent.verification import verify_answer  # noqa: E402
from ingestion.indexer import DEFAULT_INDEX_DIR  # noqa: E402
from ingestion.pipeline import PROCESSED_PATH  # noqa: E402
from letters.template import build_merge_fields, populate_letter  # noqa: E402

DEFAULT_QUERY = (
    "A passenger is travelling on an International Business fare to South Africa, "
    "with a ticket issued on 1 January 2027. What is their checked baggage allowance "
    "and what is their carry-on baggage allowance?"
)
DEFAULT_TEMPLATE = Path("template/Sandpit Air Baggage Allowance Confirmation.docx")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output", default=None, help="Defaults to output/<timestamp>-baggage-confirmation.docx")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    if not Path(args.index_dir).exists():
        raise SystemExit(f"No index found at {args.index_dir}. Run scripts/run_ingestion.py first.")
    if not Path(args.template).exists():
        raise SystemExit(f"Template not found at {args.template}.")

    logger.info("Answering passenger query via agent...")
    # One-shot CLI: still goes through the checkpointer + thread_id path,
    # just with a fresh, never-seen thread_id since there's only one message.
    graph = create_agent(index_dir=Path(args.index_dir))
    answer = answer_query(graph, args.query, thread_id=uuid.uuid4().hex)
    answer.pop("_trace", None)

    if answer["type"] == "chat":
        raise SystemExit(
            f"Agent treated this as a non-entitlement message rather than a fare/baggage "
            f"question, so there is nothing to put in a letter. It replied: {answer['message']}"
        )

    verification = verify_answer(answer, PROCESSED_PATH)
    if not verification["all_grounded"]:
        ungrounded = [k for k, v in verification["checks"].items() if not v["grounded"]]
        raise SystemExit(
            f"Refusing to generate a letter: {ungrounded} were not verified against the "
            f"cited source sections. Escalating to a human agent rather than sending an "
            f"unverified confirmation. Answer was: {answer}"
        )
    logger.info("All facts grounded in cited sections: %s", verification["checks"])

    fields = build_merge_fields(answer, PROCESSED_PATH)
    logger.info("Merge fields: %s", fields)

    output_path = (
        Path(args.output)
        if args.output
        else Path("output") / f"{datetime.now():%Y%m%d-%H%M%S}-baggage-confirmation.docx"
    )
    populate_letter(Path(args.template), fields, output_path)
    logger.info("Letter written to %s", output_path)
    print(str(output_path))


if __name__ == "__main__":
    main()
