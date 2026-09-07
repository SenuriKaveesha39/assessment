"""The `submit_answer` and `respond_to_passenger` tools, as LangChain `@tool`s.

Modelling the final answer as a tool call (rather than parsing free text)
does two things at once: it forces every field the passenger correspondence
needs (Part C's merge fields) to be produced together, and it forces a
distinct citation list *per fact* rather than one citation for the whole
answer -- which is what lets the cross-check in `verification.py` confirm
that every reported fact, not just the first, is grounded in a retrieved
section.

`@tool(parse_docstring=True)` derives each tool's name from the function
name, its description from the docstring's summary line, and each
parameter's JSON Schema (type, enum, description, required-ness from
whether it has a default) from the type hints and the docstring's Args
section -- LangChain validates at import time that the two stay in sync, so
there's no separate schema dict to keep in step with the signature by hand.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool


@tool(parse_docstring=True)
def respond_to_passenger(message: str) -> dict:
    """Reply directly to the passenger when their message is NOT a fare/baggage/entitlement question that needs the Conditions of Carriage -- e.g. a greeting, small talk, an out-of-scope request, or a clarifying question you need to ask them (e.g. their destination or ticket issue date) before you can look anything up. Never state a fare condition, baggage allowance, or other document fact through this tool -- any such fact must go through search_conditions_of_carriage and submit_answer instead.

    Args:
        message: The reply to show the passenger.
    """
    return {"message": message}


@tool(parse_docstring=True)
def submit_answer(
    destination_region: str,
    destination_region_sections: str,
    fare_type: str,
    ticket_issue_date: str,
    fare_era: Literal["current", "legacy"],
    fare_era_sections: str,
    checked_baggage_allowance: str,
    checked_baggage_sections: str,
    carry_on_allowance: str,
    carry_on_sections: str,
    summary: str,
) -> dict:
    """Submit the final, fully verified answer to the passenger's query. Call this exactly once, and only after every field below is backed by text you actually retrieved with search_conditions_of_carriage -- not general knowledge.

    Args:
        destination_region: The Conditions of Carriage region the passenger's destination resolves to, e.g. 'Africa'.
        destination_region_sections: The section number(s) (e.g. '6' or '5,6') whose retrieved text supports this value.
        fare_type: The fare family/class from the query, normalized to the document's terms, e.g. 'International Business'.
        ticket_issue_date: The ticket issue date exactly as given in the passenger's query, e.g. '1 January 2027'.
        fare_era: Whether the ticket issue date falls under current or legacy fare conditions.
        fare_era_sections: Section(s) stating the cutoff date used to decide current vs. legacy.
        checked_baggage_allowance: The checked baggage allowance that applies, e.g. '46 kg'.
        checked_baggage_sections: The section number(s) whose retrieved text supports this value.
        carry_on_allowance: The carry-on baggage allowance that applies, e.g. '14 kg, carried across two pieces'.
        carry_on_sections: The section number(s) whose retrieved text supports this value.
        summary: Full prose answer to the passenger's question, with an inline '(Section N)' citation immediately after every factual claim.
    """
    return {
        "destination_region": destination_region,
        "destination_region_sections": destination_region_sections,
        "fare_type": fare_type,
        "ticket_issue_date": ticket_issue_date,
        "fare_era": fare_era,
        "fare_era_sections": fare_era_sections,
        "checked_baggage_allowance": checked_baggage_allowance,
        "checked_baggage_sections": checked_baggage_sections,
        "carry_on_allowance": carry_on_allowance,
        "carry_on_sections": carry_on_sections,
        "summary": summary,
    }
