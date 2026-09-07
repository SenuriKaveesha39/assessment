"""Session state for the interactive chat CLI.

Wraps the compiled LangGraph agent (built once) with the bits specific to
one running conversation: which thread_id its checkpointed state lives
under, and the most recent verified answer (so /letter has something to act
on). Kept separate from chat.py so the REPL's I/O loop doesn't get tangled
up with session lifecycle -- /clear and /exit are just methods here.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from .runner import answer_query, create_agent


class ChatSession:
    def __init__(self, index_dir: Path, model: str | None = None):
        kwargs = {"model": model} if model else {}
        self.graph = create_agent(index_dir=index_dir, **kwargs)
        self.thread_id = uuid.uuid4().hex
        self.last_result: tuple[dict, dict] | None = None

    def ask(self, query: str) -> dict:
        """Answer one passenger message, remembered under this session's thread_id."""
        return answer_query(self.graph, query, thread_id=self.thread_id)

    def clear(self) -> None:
        """Start a fresh conversation.

        A new, never-seen thread_id makes the graph's checkpointer treat the
        next message as turn one -- message history and resolved
        scope/era are gone -- without rebuilding the LLM client, tools, or
        graph. Also drops the answer /letter would act on, since it belonged
        to the conversation being cleared.
        """
        self.thread_id = uuid.uuid4().hex
        self.last_result = None

    def exit(self) -> None:
        """Session teardown.

        The in-memory checkpointer needs no explicit cleanup -- it's
        garbage-collected with the process. This is the one place to add
        cleanup (e.g. closing a persistent checkpointer's DB connection) if
        create_agent ever switches to one.
        """
        pass
