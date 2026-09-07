#!/usr/bin/env python
"""Interactive CLI chatbot for Sandpit Air passenger queries.

Ask baggage/fare/entitlement questions in a loop; each answer is produced by
the agent (Part B) via the retrieval tool and independently grounding-checked.
The conversation has real memory (LangGraph checkpointer, keyed by a
per-session thread_id) -- follow-up questions can refer back to earlier ones.
Every fully-verified entitlement answer immediately generates its
confirmation letter (Part C) -- no separate step needed; the output path is
printed right there. /letter re-generates a fresh copy of the last one on
demand, /clear starts a new conversation, /quit leaves.

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

from agent.runner import AgentError  # noqa: E402
from agent.session import ChatSession  # noqa: E402
from agent.verification import verify_answer  # noqa: E402
from ingestion.indexer import DEFAULT_INDEX_DIR  # noqa: E402
from ingestion.pipeline import PROCESSED_PATH  # noqa: E402
from letters.template import build_merge_fields, populate_letter  # noqa: E402

DEFAULT_TEMPLATE = Path("template/Sandpit Air Baggage Allowance Confirmation.docx")

HELP_TEXT = """\
Commands (the leading "/" is optional -- "exit" works the same as "/exit"):
  <question>   Ask a passenger query, e.g. "What is the checked baggage
                allowance for a Business fare to Japan issued today?"
                A verified entitlement answer auto-generates its
                confirmation letter -- the output path is printed with it.
  /letter       Re-generate a fresh copy of the letter for the most recent answer
  /clear        Start a fresh conversation (forgets everything asked so far)
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


def _handle_query(query: str, *, session: ChatSession, template_path: Path) -> tuple[dict, dict] | None:
    try:
        result = session.ask(query)
    except AgentError as e:
        print(f"Agent error: {e}")
        return None
    result.pop("_trace", None)

    if result["type"] == "chat":
        print(f"\n{result['message']}\n")
        return None

    verification = verify_answer(result, PROCESSED_PATH)
    _print_answer(result, verification)

    # Every verified entitlement answer gets its confirmation letter
    # immediately -- no separate /letter step. /letter still works too, e.g.
    # to regenerate a fresh copy on demand.
    if verification["all_grounded"]:
        _generate_letter(result, verification, template_path)

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

    session = ChatSession(index_dir=index_dir)

    if args.query:
        _handle_query(args.query, session=session, template_path=Path(args.template))
        session.exit()
        return

    print("Sandpit Air passenger assistant. Type /help for commands, /quit to leave.\n")

    while True:
        try:
            line = _read_query().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        # Accept commands with or without the leading "/" -- a passenger
        # (or an interactive user) typing bare "exit" expects that to leave,
        # not get forwarded to the LLM as a question. Exact-word match only,
        # so this never fires on a real query that merely mentions "clear".
        command = line.lower().lstrip("/")
        if command in ("quit", "exit"):
            break
        if command == "help":
            print(HELP_TEXT)
            continue
        if command == "clear":
            session.clear()
            print("Conversation cleared -- starting fresh.\n")
            continue
        if command == "letter":
            if session.last_result is None:
                print("Ask a question first, then /letter to generate the confirmation.")
                continue
            print("Regenerating a fresh copy of the confirmation letter...")
            _generate_letter(*session.last_result, template_path=Path(args.template))
            continue

        result = _handle_query(line, session=session, template_path=Path(args.template))
        if result is not None:
            session.last_result = result

    session.exit()
    print("Goodbye.")


if __name__ == "__main__":
    main()
