"""Entry point that compiles and invokes the LangGraph agent for one passenger message.

The actual reasoning loop -- agent / tools / applicability_check / finalize --
lives in graph.py. This module just wires up the initial state, bounds how
many graph steps a single message may take, and reconstructs a flat
tool-call trace from the resulting message history (for citation auditing
and the "(N tool calls made)" line the CLIs print).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from .graph import build_graph
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("SANDPIT_AIR_AGENT_MODEL", "claude-haiku-4-5-20251001")
MAX_TURNS = 10


class AgentError(RuntimeError):
    pass


def _reconstruct_trace(messages: list) -> list[dict]:
    tool_calls_by_id = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                tool_calls_by_id[tc["id"]] = {"tool": tc["name"], "input": tc["args"]}
    return [
        {**tool_calls_by_id.get(msg.tool_call_id, {}), "output": msg.content}
        for msg in messages
        if isinstance(msg, ToolMessage)
    ]


def answer_query(query: str, *, index_dir: Path, model: str = DEFAULT_MODEL) -> dict:
    """Run the agent graph for one passenger message.

    Returns a tagged dict: {"type": "answer", ...submit_answer fields...} for
    an entitlement question, or {"type": "chat", "message": ...} for anything
    else (greetings, small talk, a clarifying question back to the passenger).
    Either way, `_trace` carries every search tool call made (for citation
    auditing).
    """
    graph = build_graph(index_dir=index_dir, model=model)
    initial_state = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)],
        "resolved_scope": None,
        "resolved_era": None,
        "pending_tool_results": [],
        "final_result": None,
    }

    try:
        # Each logical turn is at most two graph steps (agent -> tools ->
        # applicability_check, or agent -> remind); triple MAX_TURNS for margin.
        final_state = graph.invoke(initial_state, config={"recursion_limit": MAX_TURNS * 3})
    except GraphRecursionError as e:
        raise AgentError(f"Agent did not submit an answer within {MAX_TURNS} turns") from e

    result = final_state.get("final_result")
    if result is None:
        raise AgentError("Agent graph ended without submitting an answer")

    result["_trace"] = _reconstruct_trace(final_state["messages"])
    return result
