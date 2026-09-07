"""The `submit_answer` tool: the agent's structured, per-fact-cited final output.

Modelling the final answer as a tool call (rather than parsing free text)
does two things at once: it forces every field the passenger correspondence
needs (Part C's merge fields) to be produced together, and it forces a
distinct citation list *per fact* rather than one citation for the whole
answer -- which is what lets the cross-check in `verification.py` confirm
that every reported fact, not just the first, is grounded in a retrieved
section.
"""

from __future__ import annotations

SUBMIT_ANSWER_TOOL_NAME = "submit_answer"
CHAT_TOOL_NAME = "respond_to_passenger"

CHAT_TOOL_SCHEMA = {
    "name": CHAT_TOOL_NAME,
    "description": (
        "Reply directly to the passenger when their message is NOT a fare/baggage/"
        "entitlement question that needs the Conditions of Carriage -- e.g. a greeting, "
        "small talk, an out-of-scope request, or a clarifying question you need to ask "
        "them (e.g. their destination or ticket issue date) before you can look anything "
        "up. Never state a fare condition, baggage allowance, or other document fact "
        "through this tool -- any such fact must go through search_conditions_of_carriage "
        "and submit_answer instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The reply to show the passenger."},
        },
        "required": ["message"],
    },
}

_CITED_STRING = {
    "type": "string",
    "description": "The section number(s) (e.g. '6' or '5,6') whose retrieved text supports this value.",
}

SUBMIT_ANSWER_TOOL_SCHEMA = {
    "name": SUBMIT_ANSWER_TOOL_NAME,
    "description": (
        "Submit the final, fully verified answer to the passenger's query. "
        "Call this exactly once, and only after every field below is backed by text you "
        "actually retrieved with search_conditions_of_carriage -- not general knowledge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "destination_region": {
                "type": "string",
                "description": "The Conditions of Carriage region the passenger's destination resolves to, e.g. 'Africa'.",
            },
            "destination_region_sections": _CITED_STRING,
            "fare_type": {
                "type": "string",
                "description": "The fare family/class from the query, normalized to the document's terms, e.g. 'International Business'.",
            },
            "ticket_issue_date": {
                "type": "string",
                "description": "The ticket issue date exactly as given in the passenger's query, e.g. '1 January 2027'.",
            },
            "fare_era": {
                "type": "string",
                "enum": ["current", "legacy"],
                "description": "Whether the ticket issue date falls under current or legacy fare conditions.",
            },
            "fare_era_sections": {
                **_CITED_STRING,
                "description": "Section(s) stating the cutoff date used to decide current vs. legacy.",
            },
            "checked_baggage_allowance": {
                "type": "string",
                "description": "The checked baggage allowance that applies, e.g. '46 kg'.",
            },
            "checked_baggage_sections": _CITED_STRING,
            "carry_on_allowance": {
                "type": "string",
                "description": "The carry-on baggage allowance that applies, e.g. '14 kg, carried across two pieces'.",
            },
            "carry_on_sections": _CITED_STRING,
            "summary": {
                "type": "string",
                "description": (
                    "Full prose answer to the passenger's question, with an inline "
                    "'(Section N)' citation immediately after every factual claim."
                ),
            },
        },
        "required": [
            "destination_region",
            "destination_region_sections",
            "fare_type",
            "ticket_issue_date",
            "fare_era",
            "fare_era_sections",
            "checked_baggage_allowance",
            "checked_baggage_sections",
            "carry_on_allowance",
            "carry_on_sections",
            "summary",
        ],
    },
}
