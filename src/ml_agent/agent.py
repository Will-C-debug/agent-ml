"""Assemble the ML engineer agent from LangChain building blocks.

- ``langchain.agents.create_agent`` provides the tool-calling loop (a LangGraph graph).
- Middleware adds planning (``write_todos``), human approval for ``run_python``, error recovery
  for tools, a per-turn model-call limit and, for Claude, prompt caching.
- A LangGraph SQLite checkpointer persists every conversation thread in the workspace.
"""

from __future__ import annotations

import sqlite3

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from ml_agent.config import Settings, build_chat_model
from ml_agent.prompts import build_system_prompt
from ml_agent.tools import build_tools
from ml_agent.workspace import Workspace

# Exceptions a tool raises because of bad input or data; the agent sees the message and adapts.
RECOVERABLE_ERRORS = (
    ValueError,
    KeyError,
    TypeError,
    FileNotFoundError,
    IsADirectoryError,
    PermissionError,
    RuntimeError,
    ImportError,
    MemoryError,
)


def tool_error_message(exc: Exception, request: ToolCallRequest) -> str | None:
    if isinstance(exc, RECOVERABLE_ERRORS):
        detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        return f"Error in {request.tool_call['name']} ({type(exc).__name__}): {detail}"
    return None


def open_checkpointer(ws: Workspace) -> SqliteSaver:
    """A SQLite checkpointer stored in the workspace, so conversations can be resumed."""
    conn = sqlite3.connect(ws.checkpoint_db, check_same_thread=False)
    return SqliteSaver(conn)


def build_agent(
    settings: Settings,
    *,
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build the compiled LangGraph agent and return it with its workspace."""
    ws = Workspace(settings.workspace)
    middleware: list[AgentMiddleware] = [
        TodoListMiddleware(),
        ModelCallLimitMiddleware(run_limit=settings.max_model_calls, exit_behavior="end"),
    ]
    if not settings.auto_approve:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "run_python": InterruptOnConfig(
                        allowed_decisions=["approve", "edit", "reject"],
                        description="The agent wants to run this Python code on your machine.",
                    )
                }
            )
        )
    middleware.append(ToolErrorMiddleware(tool_error_message))
    if model is None and settings.provider == "anthropic":
        from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

        middleware.append(AnthropicPromptCachingMiddleware(ttl="1h"))

    agent = create_agent(
        model or build_chat_model(settings),
        tools=build_tools(ws, settings),
        system_prompt=build_system_prompt(ws, settings),
        middleware=middleware,
        checkpointer=checkpointer,
        name="ml_engineer",
    )
    return agent, ws
