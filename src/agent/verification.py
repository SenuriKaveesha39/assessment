"""Post-hoc grounding check on the agent's final answer.

The `submit_answer` schema already forces the agent to cite a section number
per fact. This module is the independent check that the citation is real:
for each fact, does the *cited section's own text* (from the parsed
sections.json, not the agent's say-so) actually contain the value claimed?
This is deliberately outside the agent's control -- it re-derives the check
from ground truth rather than trusting the model's self-report, and it is
applied to every fact (destination region, fare era, checked baggage, carry-
on), not just the first, per the assessment's success criteria.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_SECTION_NUM_RE = re.compile(r"\d+")
_WEIGHT_RE = re.compile(r"\d+(?:\.\d+)?\s*kg", re.IGNORECASE)


def _load_parsed(processed_path: Path) -> dict:
    return json.loads(processed_path.read_text())


def _sections_by_number(parsed: dict) -> dict[str, dict]:
    return {s["section_number"]: s for s in parsed["sections"] if s["section_number"]}


def _section_text(section: dict) -> str:
    parts = [section["title"], *section["paragraphs"], *section["bullets"]]
    for table in section["tables"]:
        for row in table["rows"]:
            parts.append(", ".join(f"{col}: {val}" for col, val in row["cells"].items()))
    return " ".join(parts).lower()


def _cited_numbers(raw: str) -> list[str]:
    return _SECTION_NUM_RE.findall(raw or "")


def _primary_needle(value: str) -> str:
    # An allowance like "46 kg" may be phrased by the agent without the
    # document's surrounding words (e.g. "Business:" / ", carried across two
    # pieces"). The weight figure itself is the load-bearing fact, so prefer
    # matching just that over demanding an exact-string match.
    match = _WEIGHT_RE.search(value)
    return match.group(0) if match else value


def verify_answer(answer: dict, processed_path: Path) -> dict:
    parsed = _load_parsed(processed_path)
    sections = _sections_by_number(parsed)
    cutoff_date = (parsed["metadata"].get("effective_date") or "").lower()

    def check(needle: str, cited_field: str) -> dict:
        numbers = _cited_numbers(answer.get(cited_field, ""))
        needle_lower = needle.lower().strip()
        grounded_sections = [
            n for n in numbers if n in sections and needle_lower and needle_lower in _section_text(sections[n])
        ]
        return {
            "cited_sections": numbers,
            "grounded_sections": grounded_sections,
            "grounded": bool(grounded_sections),
        }

    checks = {
        "destination_region": check(answer["destination_region"], "destination_region_sections"),
        # The document text never literally says "current" or "legacy" -- what
        # it states is the cutoff date. Verifying against the document's own
        # effective_date (read from metadata, not hardcoded) keeps this check
        # correct even if a future revision moves the cutoff.
        "fare_era": check(cutoff_date, "fare_era_sections"),
        "checked_baggage_allowance": check(
            _primary_needle(answer["checked_baggage_allowance"]), "checked_baggage_sections"
        ),
        "carry_on_allowance": check(_primary_needle(answer["carry_on_allowance"]), "carry_on_sections"),
    }

    return {
        "checks": checks,
        "all_grounded": all(c["grounded"] for c in checks.values()),
    }
