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
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_MARKER_RE = re.compile(r"\(Sections?\s+[\d,\s]+\)", re.IGNORECASE)
# A sentence is treated as "factual" (and therefore required to carry a
# citation) if it mentions a number or one of the domain terms every
# entitlement answer revolves around. This is deliberately broad -- a false
# positive here just means a harmless sentence gets asked for a citation it
# didn't need, whereas a false negative would let an uncited claim slip past
# the whole point of this check.
_FACTUAL_HINT_RE = re.compile(
    r"\d|\bkg\b|allowance|baggage|fare|region|current|legacy|carry-on|checked|entitlement",
    re.IGNORECASE,
)


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


def _check_summary_citation_coverage(summary: str) -> dict:
    """Flag any sentence in the prose summary that states a fact with no citation.

    The per-field checks above confirm that destination_region_sections etc.
    are individually grounded, but the passenger-facing text is `summary`,
    not those fields -- a model could cite Section 6 correctly in the
    structured field while still writing an uncited aside in the prose (e.g.
    slipping in an extra claim while explaining the reasoning). This re-reads
    the summary itself and checks every sentence that looks factual for an
    inline "(Section N)" marker, independent of what the structured fields
    say.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(summary or "") if s.strip()]
    uncited = [s for s in sentences if _FACTUAL_HINT_RE.search(s) and not _CITATION_MARKER_RE.search(s)]
    return {
        "sentences_checked": len(sentences),
        "uncited_sentences": uncited,
        "grounded": not uncited,
    }


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
        "summary_citations": _check_summary_citation_coverage(answer.get("summary", "")),
    }

    return {
        "checks": checks,
        "all_grounded": all(c["grounded"] for c in checks.values()),
    }
