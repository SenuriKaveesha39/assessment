"""The retrieval tool exposed to the LLM agent, plus its Anthropic tool schema.

The tool is a thin wrapper over `ingestion.indexer.query_index`. It is the
*only* way the agent can learn what the Conditions of Carriage says -- no
document text is ever placed in the system prompt, so any fact in the
agent's final answer had to come from a tool call the transcript can show.
"""

from __future__ import annotations

from pathlib import Path

from ingestion.indexer import DEFAULT_INDEX_DIR, query_index

SEARCH_TOOL_NAME = "search_conditions_of_carriage"

# Squared-L2 distance on all-MiniLM-L6-v2's normalized embeddings (see
# indexer.py's normalization note). Empirically calibrated against this
# index: genuinely relevant queries ("checked baggage allowance Business
# international Africa") scored 0.54-0.86, while queries about things the
# Conditions of Carriage doesn't cover at all ("weather in Sydney", "reset
# my email password", "pet policy") scored 1.49-2.04. 1.2 sits cleanly in
# the gap between those two clusters. This is a signal to the agent that a
# search came back weak, not a hard cutoff enforced here -- the agent (or a
# human reviewing its trace) decides whether to refine the query or escalate.
LOW_CONFIDENCE_DISTANCE = 1.2

SEARCH_TOOL_SCHEMA = {
    "name": SEARCH_TOOL_NAME,
    "description": (
        "Search the indexed Sandpit Air Conditions of Carriage for relevant passages. "
        "Returns the matching sections' text along with their section number, title, "
        "and applicability metadata (service_scope, fare_era). This is the only source "
        "of truth for fare conditions, baggage allowances and travel entitlements -- "
        "never answer from general airline knowledge. "
        "Use service_scope/fare_era filters once you know them (e.g. after resolving "
        "a destination country to a region, or a ticket issue date to a fare era) to "
        "avoid retrieving a section that looks similar but doesn't actually apply."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query, e.g. 'checked baggage allowance Business international'.",
            },
            "service_scope": {
                "type": "string",
                "enum": ["domestic", "international"],
                "description": "Restrict to sections that apply to this scope. Omit if not yet known or not relevant.",
            },
            "fare_era": {
                "type": "string",
                "enum": ["current", "legacy"],
                "description": "Restrict to sections that apply to this fare era. Omit if not yet known or not relevant.",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of passages to return (default 4).",
            },
        },
        "required": ["query"],
    },
}


def run_search_tool(tool_input: dict, index_dir: Path = DEFAULT_INDEX_DIR) -> dict:
    result = query_index(
        tool_input["query"],
        n_results=tool_input.get("n_results", 4),
        index_dir=index_dir,
        service_scope=tool_input.get("service_scope"),
        fare_era=tool_input.get("fare_era"),
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    if not documents:
        return {
            "results": [],
            "low_confidence": True,
            "note": (
                "No sections matched. Try a broader query or remove a filter. If repeated "
                "searches stay empty or low-confidence, tell the passenger via "
                "respond_to_passenger that this needs a human agent -- do not guess."
            ),
        }

    low_confidence = distances[0] > LOW_CONFIDENCE_DISTANCE
    response = {
        "results": [
            {
                "section_number": meta["section_number"],
                "section_title": meta["section_title"],
                "service_scope": meta["service_scope"] or "domestic,international",
                "fare_era": meta["fare_era"],
                "text": doc,
                "relevance_distance": round(dist, 4),
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ],
        "low_confidence": low_confidence,
    }
    if low_confidence:
        response["note"] = (
            "None of these results are strongly relevant (best relevance_distance "
            f"{distances[0]:.2f} > {LOW_CONFIDENCE_DISTANCE} threshold). Try refining the "
            "query or filters first; if repeated searches stay low-confidence, tell the "
            "passenger via respond_to_passenger that this needs a human agent rather than "
            "reporting an uncertain figure."
        )
    return response
