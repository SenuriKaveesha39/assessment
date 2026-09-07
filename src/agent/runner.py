"""The agentic tool-use loop: reason, call tools, evaluate, repeat until submit_answer.

Uses the Anthropic Messages API directly (not a framework) so the reasoning
loop -- and the point at which we force the model back on track if it tries
to answer in plain text instead of calling a tool -- is fully visible.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import anthropic

from .answer_schema import (
    CHAT_TOOL_NAME,
    CHAT_TOOL_SCHEMA,
    SUBMIT_ANSWER_TOOL_NAME,
    SUBMIT_ANSWER_TOOL_SCHEMA,
)
from .prompts import SYSTEM_PROMPT
from .tools import SEARCH_TOOL_NAME, SEARCH_TOOL_SCHEMA, run_search_tool

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("SANDPIT_AIR_AGENT_MODEL", "claude-haiku-4-5-20251001")
MAX_TURNS = 10


class AgentError(RuntimeError):
    pass


def answer_query(query: str, *, index_dir: Path, model: str = DEFAULT_MODEL) -> dict:
    """Run the agent loop for one passenger message.

    Returns a tagged dict: {"type": "answer", ...submit_answer fields...} for
    an entitlement question, or {"type": "chat", "message": ...} for anything
    else (greetings, small talk, a clarifying question back to the passenger).
    Either way, `_trace` carries every tool call made (for citation auditing).
    """
    client = anthropic.Anthropic()
    tools = [SEARCH_TOOL_SCHEMA, SUBMIT_ANSWER_TOOL_SCHEMA, CHAT_TOOL_SCHEMA]
    messages: list[dict] = [{"role": "user", "content": query}]
    trace: list[dict] = []

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
            # This SDK's typed signature dropped `temperature` for the newer
            # model family, but the API still accepts it as a pass-through.
            extra_body={"temperature": 0},
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if block.type == "tool_use"]

        if not tool_uses:
            # The system prompt requires a tool call every turn; nudge back
            # on track rather than accepting an un-cited free-text answer.
            logger.warning("Turn %d: model responded without a tool call, re-prompting", turn)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You must respond with a tool call: search_conditions_of_carriage "
                        "if you still need information, submit_answer if an entitlement "
                        "question is fully answered, or respond_to_passenger otherwise."
                    ),
                }
            )
            continue

        tool_results = []
        for tool_use in tool_uses:
            if tool_use.name == SEARCH_TOOL_NAME:
                logger.info("Turn %d: search(%s)", turn, tool_use.input)
                result = run_search_tool(tool_use.input, index_dir=index_dir)
                trace.append({"tool": tool_use.name, "input": tool_use.input, "output": result})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(result),
                    }
                )
            elif tool_use.name == SUBMIT_ANSWER_TOOL_NAME:
                logger.info("Turn %d: submit_answer", turn)
                return {"type": "answer", **tool_use.input, "_trace": trace}
            elif tool_use.name == CHAT_TOOL_NAME:
                logger.info("Turn %d: respond_to_passenger", turn)
                return {"type": "chat", "message": tool_use.input["message"], "_trace": trace}
            else:
                raise AgentError(f"Unknown tool requested: {tool_use.name}")

        messages.append({"role": "user", "content": tool_results})

    raise AgentError(f"Agent did not submit an answer within {MAX_TURNS} turns")
