"""Entry point that builds and drives the LangGraph agent for passenger messages.

The actual reasoning loop -- agent / tools / applicability_check / finalize --
lives in graph.py. This module wires up an in-memory checkpointer so state
(message history, resolved_scope, resolved_era) persists per thread_id across
multiple calls to answer_query(): build the agent once with create_agent(),
then reuse it for every message in a conversation, varying only thread_id.
A fresh, never-seen thread_id starts a conversation with a clean slate --
that's what /clear uses to reset a chat session without rebuilding the LLM
client, tools, or graph.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from .graph import build_graph
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("SANDPIT_AIR_AGENT_MODEL", "claude-haiku-4-5-20251001")
MAX_TURNS = 10


class AgentError(RuntimeError):
    pass


def create_agent(index_dir: Path, model: str = DEFAULT_MODEL):
    """Build the compiled LangGraph agent once, with an in-memory checkpointer.

    The checkpointer only lives for this process -- a new run starts with no
    history, which matches a CLI chat session's lifetime. Swapping in a
    persistent checkpointer (SqliteSaver, PostgresSaver, ...) to survive
    restarts is a one-line change here; nothing else in the agent needs to
    know about it.
    """
    return build_graph(index_dir=index_dir, model=model, checkpointer=InMemorySaver())


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


def answer_query(graph, query: str, *, thread_id: str) -> dict:
    """Run one passenger message through a graph built by create_agent().

    `thread_id` selects which conversation this message belongs to: reusing
    the same id across calls carries the message history and resolved
    scope/era forward (real multi-turn memory via the graph's checkpointer);
    a thread_id the checkpointer has never seen starts fresh.

    Returns a tagged dict: {"type": "answer", ...submit_answer fields...} for
    an entitlement question, or {"type": "chat", "message": ...} for anything
    else (greetings, small talk, a clarifying question back to the passenger).
    Either way, `_trace` carries every search tool call made this session (for
    citation auditing).
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_TURNS * 3}

    # Only seed the system prompt on a thread's first turn -- add_messages
    # appends rather than replaces, so resending it every turn would pile up
    # duplicate SystemMessages in state that persists across calls.
    is_new_thread = not graph.get_state(config).values.get("messages")
    new_messages = [HumanMessage(content=query)]
    if is_new_thread:
        new_messages = [SystemMessage(content=SYSTEM_PROMPT), *new_messages]

    try:
        # recursion_limit bounds this single invoke() call's steps, not the
        # thread's cumulative history -- each passenger message gets its own
        # full turn budget regardless of how many prior turns happened.
        final_state = graph.invoke({"messages": new_messages}, config=config)
    except GraphRecursionError as e:
        raise AgentError(f"Agent did not submit an answer within {MAX_TURNS} turns") from e

    result = final_state.get("final_result")
    if result is None:
        raise AgentError("Agent graph ended without submitting an answer")

    result["_trace"] = _reconstruct_trace(final_state["messages"])
    return result
