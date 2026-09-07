"""The retrieval tool exposed to the LLM agent, as a LangChain `@tool`.

The tool is a thin wrapper over `ingestion.indexer.query_index` (Chroma
similarity search followed by cross-encoder reranking). It is the *only* way
the agent can learn what the Conditions of Carriage says -- no document
text is ever placed in the system prompt, so any fact in the agent's final
answer had to come from a tool call the transcript can show.

`index_dir` varies per `answer_query()` call (different CLI invocations can
point at different index directories), but a `@tool`-decorated function's
schema is fixed at decoration time and its parameters are all LLM-visible --
so rather than adding a non-LLM-controlled `index_dir` parameter to the
schema, `make_search_tool` builds a fresh tool per call, closing over
`index_dir` in the function body instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from langchain_core.tools import tool

from ingestion.indexer import DEFAULT_INDEX_DIR, TOP_K, query_index

# The cross-encoder reranker's own relevance score (0-1, higher is better --
# opposite direction from the bi-encoder's embedding distance). Calibrated
# against this index: genuinely relevant queries ("checked baggage allowance
# Business international Africa") scored 0.90+, while queries about things
# the Conditions of Carriage doesn't cover at all ("weather in Sydney",
# "reset my email password", "pet policy") scored 0.0. 0.5 sits cleanly in
# that gap. This is a signal to the agent that a search came back weak, not a
# hard cutoff enforced here -- the agent (or the applicability_check graph
# node, or a human reviewing the trace) decides what to do about it.
LOW_CONFIDENCE_SCORE = 0.5


def make_search_tool(index_dir: Path = DEFAULT_INDEX_DIR):
    @tool(parse_docstring=True)
    def search_conditions_of_carriage(
        query: str,
        service_scope: Optional[Literal["domestic", "international"]] = None,
        fare_era: Optional[Literal["current", "legacy"]] = None,
        n_results: int = TOP_K,
    ) -> dict:
        """Search the indexed Sandpit Air Conditions of Carriage for relevant passages. Returns the matching sections' text along with their section number, title, and applicability metadata (service_scope, fare_era). This is the only source of truth for fare conditions, baggage allowances and travel entitlements -- never answer from general airline knowledge. Use service_scope/fare_era filters once you know them (e.g. after resolving a destination country to a region, or a ticket issue date to a fare era) to avoid retrieving a section that looks similar but doesn't actually apply.

        Args:
            query: Natural-language search query, e.g. 'checked baggage allowance Business international'.
            service_scope: Restrict to sections that apply to this scope. Omit if not yet known or not relevant.
            fare_era: Restrict to sections that apply to this fare era. Omit if not yet known or not relevant.
            n_results: Number of passages to return (default and max is set by the retriever).
        """
        result = query_index(
            query,
            n_results=min(n_results, TOP_K),
            index_dir=index_dir,
            service_scope=service_scope,
            fare_era=fare_era,
        )

        documents = result["documents"]
        metadatas = result["metadatas"]
        scores = result["scores"]

        if not documents:
            return {
                "results": [],
                "low_confidence": True,
                "note": (
                    "No sections matched. Try a broader query or remove a filter. If "
                    "repeated searches stay empty or low-confidence, tell the passenger "
                    "via respond_to_passenger that this needs a human agent -- do not guess."
                ),
            }

        low_confidence = scores[0] < LOW_CONFIDENCE_SCORE
        response = {
            "results": [
                {
                    "section_number": meta["section_number"],
                    "section_title": meta["section_title"],
                    "service_scope": meta["service_scope"] or "domestic,international",
                    "fare_era": meta["fare_era"],
                    "text": doc,
                    "relevance_score": round(score, 4),
                }
                for doc, meta, score in zip(documents, metadatas, scores)
            ],
            "low_confidence": low_confidence,
        }
        if low_confidence:
            response["note"] = (
                f"None of these results are strongly relevant (best relevance_score "
                f"{scores[0]:.2f} < {LOW_CONFIDENCE_SCORE} threshold). Try refining the "
                "query or filters first; if repeated searches stay low-confidence, tell "
                "the passenger via respond_to_passenger that this needs a human agent "
                "rather than reporting an uncertain figure."
            )
        return response

    return search_conditions_of_carriage
