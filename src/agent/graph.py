"""LangGraph orchestration of the retrieval agent.

Graph shape::

    START -> agent -[no tool call]-> remind -> agent
                    -[search call(s)]-> tools -> applicability_check -> agent
                    -[submit_answer / respond_to_passenger]-> finalize -> END

`tools` and `applicability_check` are two distinct nodes with a real
division of labour, not one step wearing two names: `tools` only executes
search_conditions_of_carriage (Chroma similarity search + cross-encoder
rerank) and stashes the raw results; `applicability_check` is the one that
flags every passage against whatever service_scope/fare_era the agent has
already resolved (inferred from the filters it chose to pass into the
search tool on this or an earlier call) and is what actually turns those
raw results into the ToolMessages the LLM sees next. A passage whose own
service_scope/fare_era contradicts what's already been established gets an
explicit "MISMATCH" annotation injected before the LLM ever reads it again,
rather than relying on the model to notice a metadata field buried in the
passage text. This is the graph's code-enforced answer to "cross-check
retrieved context against the constraints in the query" -- applied once,
uniformly, to every passage, rather than hoped for from prompting alone.

State management: `build_graph` takes an optional `checkpointer` and, when
one is supplied, compiles a graph that persists `AgentState` per thread_id
(see runner.py's create_agent/answer_query). Reusing the same compiled graph
and thread_id across multiple passenger messages gives real multi-turn
memory -- the message history AND resolved_scope/resolved_era carry over --
without this module needing to know anything about sessions itself.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .answer_schema import respond_to_passenger, submit_answer
from .tools import make_search_tool

logger = logging.getLogger(__name__)

_REMIND_MESSAGE = (
    "You must respond with a tool call: search_conditions_of_carriage if you still "
    "need information, submit_answer if an entitlement question is fully answered, "
    "or respond_to_passenger otherwise."
)


class PendingResult(TypedDict):
    tool_call_id: str
    result: dict
    service_scope_filter: Optional[str]
    fare_era_filter: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    resolved_scope: Optional[str]
    resolved_era: Optional[str]
    pending_tool_results: list[PendingResult]
    final_result: Optional[dict]


def _annotate_applicability(result: dict, resolved_scope: Optional[str], resolved_era: Optional[str]) -> dict:
    for item in result.get("results", []):
        scope_field = item.get("service_scope", "")
        scope_ok = (
            resolved_scope is None
            or scope_field == "domestic,international"
            or resolved_scope in scope_field.split(",")
        )
        era_field = item.get("fare_era", "unspecified")
        era_ok = resolved_era is None or era_field == "unspecified" or era_field == resolved_era

        if scope_ok and era_ok:
            item["applicability_check"] = "OK"
            continue

        mismatches = []
        if not scope_ok:
            mismatches.append(
                f"this passage's service_scope is {scope_field!r} but the resolved scope is {resolved_scope!r}"
            )
        if not era_ok:
            mismatches.append(
                f"this passage's fare_era is {era_field!r} but the resolved era is {resolved_era!r}"
            )
        item["applicability_check"] = "MISMATCH: " + "; ".join(mismatches) + " -- do not rely on this passage."
    return result


def build_graph(index_dir: Path, model: str, checkpointer=None):
    search_tool = make_search_tool(index_dir=index_dir)
    llm = ChatAnthropic(model=model, temperature=0, max_tokens=2048)
    llm_with_tools = llm.bind_tools([search_tool, submit_answer, respond_to_passenger])

    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def route_from_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return "remind"
        names = {tc["name"] for tc in last.tool_calls}
        if submit_answer.name in names or respond_to_passenger.name in names:
            return "finalize"
        return "tools"

    def remind_node(state: AgentState) -> dict:
        logger.warning("Model responded without a tool call, re-prompting")
        return {"messages": [HumanMessage(content=_REMIND_MESSAGE)]}

    def tools_node(state: AgentState) -> dict:
        """Execute every search call in the latest AI message. Retrieval only --
        no applicability reasoning happens here, that's the next node's job."""
        last = state["messages"][-1]
        pending: list[PendingResult] = []
        for tc in last.tool_calls:
            if tc["name"] != search_tool.name:
                continue
            logger.info("search(%s)", tc["args"])
            result = search_tool.invoke(tc["args"])
            pending.append(
                {
                    "tool_call_id": tc["id"],
                    "result": result,
                    "service_scope_filter": tc["args"].get("service_scope"),
                    "fare_era_filter": tc["args"].get("fare_era"),
                }
            )
        return {"pending_tool_results": pending}

    def applicability_check_node(state: AgentState) -> dict:
        """Code-driven cross-check: annotate each pending result against whatever
        scope/era the agent has resolved so far, then turn it into the ToolMessage
        the LLM actually sees. Nothing here is LLM-decided."""
        scope = state.get("resolved_scope")
        era = state.get("resolved_era")
        tool_messages = []
        for entry in state["pending_tool_results"]:
            scope = entry["service_scope_filter"] or scope
            era = entry["fare_era_filter"] or era
            annotated = _annotate_applicability(entry["result"], scope, era)
            tool_messages.append(ToolMessage(content=json.dumps(annotated), tool_call_id=entry["tool_call_id"]))
        return {
            "messages": tool_messages,
            "resolved_scope": scope,
            "resolved_era": era,
            "pending_tool_results": [],
        }

    def finalize_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        for tc in last.tool_calls:
            if tc["name"] == submit_answer.name:
                logger.info("submit_answer")
                return {"final_result": {"type": "answer", **tc["args"]}}
            if tc["name"] == respond_to_passenger.name:
                logger.info("respond_to_passenger")
                return {"final_result": {"type": "chat", "message": tc["args"]["message"]}}
        raise RuntimeError("finalize_node reached without a terminal tool call")

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("remind", remind_node)
    graph.add_node("tools", tools_node)
    graph.add_node("applicability_check", applicability_check_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_from_agent, {"remind": "remind", "tools": "tools", "finalize": "finalize"}
    )
    graph.add_edge("remind", "agent")
    graph.add_edge("tools", "applicability_check")
    graph.add_edge("applicability_check", "agent")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
