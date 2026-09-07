#!/usr/bin/env python
"""Interactive CLI chatbot for Sandpit Air passenger queries.

Ask baggage/fare/entitlement questions in a loop; each answer is produced by
the agent (Part B) via the retrieval tool and independently grounding-checked.
Type /letter at any point to generate the confirmation letter (Part C) from
the most recent answer.

Usage:
    python scripts/chat.py                  # interactive loop
    python scripts/chat.py --query "..."    # answer one question and exit

Requires ANTHROPIC_API_KEY (e.g. in a .env file) and an index already built
by scripts/run_ingestion.py.
"""

from __future__ import annotations

import argparse
import logging
import select
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agent.runner import AgentError, answer_query  # noqa: E402
from agent.verification import verify_answer  # noqa: E402
from ingestion.indexer import DEFAULT_INDEX_DIR  # noqa: E402
from ingestion.pipeline import PROCESSED_PATH  # noqa: E402
from letters.template import build_merge_fields, populate_letter  # noqa: E402

DEFAULT_TEMPLATE = Path("template/Sandpit Air Baggage Allowance Confirmation.docx")

HELP_TEXT = """\
Commands:
  <question>   Ask a passenger query, e.g. "What is the checked baggage
                allowance for a Business fare to Japan issued today?"
  /letter       Generate the confirmation letter from the most recent answer
  /help         Show this message
  /quit, /exit  Leave the chat
"""


def _print_answer(answer: dict, verification: dict) -> None:
    print(f"\n{answer['summary']}\n")
    print(f"  Destination region:  {answer['destination_region']}  (Section {answer['destination_region_sections']})")
    print(f"  Fare type:           {answer['fare_type']}")
    print(f"  Ticket issue date:   {answer['ticket_issue_date']}  -> {answer['fare_era']} fare conditions (Section {answer['fare_era_sections']})")
    print(f"  Checked baggage:     {answer['checked_baggage_allowance']}  (Section {answer['checked_baggage_sections']})")
    print(f"  Carry-on baggage:    {answer['carry_on_allowance']}  (Section {answer['carry_on_sections']})")
    if verification["all_grounded"]:
        print("  [verified: every cited fact was found in its cited section]\n")
    else:
        ungrounded = [k for k, v in verification["checks"].items() if not v["grounded"]]
        print(
            f"  [NOT VERIFIED: {ungrounded} could not be confirmed against their cited "
            f"sections -- do not rely on this answer. Escalating to a human agent is the "
            f"safe next step; use /letter only once a verified answer is shown.]\n"
        )


def _generate_letter(answer: dict, verification: dict, template_path: Path) -> None:
    if not verification["all_grounded"]:
        ungrounded = [k for k, v in verification["checks"].items() if not v["grounded"]]
        print(
            f"Refusing to generate a letter: {ungrounded} could not be verified against "
            f"their cited sections. This has been held back for a human agent to review "
            f"rather than sent to the passenger -- ask again or rephrase, or escalate."
        )
        return
    if not template_path.exists():
        print(f"Template not found at {template_path}.")
        return

    fields = build_merge_fields(answer, PROCESSED_PATH)
    output_path = Path("output") / f"{datetime.now():%Y%m%d-%H%M%S}-baggage-confirmation.docx"
    populate_letter(template_path, fields, output_path)
    print(f"Letter written to {output_path}\n")


def _handle_query(query: str, *, index_dir: Path) -> tuple[dict, dict] | None:
    try:
        result = answer_query(query, index_dir=index_dir)
    except AgentError as e:
        print(f"Agent error: {e}")
        return None
    result.pop("_trace", None)

    if result["type"] == "chat":
        print(f"\n{result['message']}\n")
        return None

    verification = verify_answer(result, PROCESSED_PATH)
    _print_answer(result, verification)
    return result, verification


def _read_query(prompt: str = "You: ") -> str:
    """Read one line, then greedily absorb any lines already sitting in stdin.

    input() stops at the first newline, so pasting a query that has hard
    line breaks (e.g. copied from a document wrapped at 80 columns) would
    otherwise silently split it into several incomplete queries -- one per
    remaining line, each fed to the agent on its own with no way to tell
    that this happened. A real interactive Enter-press never has further
    lines already buffered, so treating "more is available right now" as
    "this was one paste" is a safe way to reassemble it into one query.
    """
    parts = [input(prompt)]
    while select.select([sys.stdin], [], [], 0)[0]:
        extra = sys.stdin.readline()
        if not extra:
            break
        parts.append(extra.rstrip("\n"))
    return " ".join(p.strip() for p in parts if p.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", default=None, help="Answer one question and exit instead of looping.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise SystemExit(f"No index found at {index_dir}. Run scripts/run_ingestion.py first.")

    if args.query:
        _handle_query(args.query, index_dir=index_dir)
        return

    print("Sandpit Air passenger assistant. Type /help for commands, /quit to leave.\n")
    last_result: tuple[dict, dict] | None = None

    while True:
        try:
            line = _read_query().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print(HELP_TEXT)
            continue
        if line == "/letter":
            if last_result is None:
                print("Ask a question first, then /letter to generate the confirmation.")
                continue
            _generate_letter(*last_result, template_path=Path(args.template))
            continue

        result = _handle_query(line, index_dir=index_dir)
        if result is not None:
            last_result = result


if __name__ == "__main__":
    main()
