"""
Deep Agent graph.

State: messages (append-only), todos (last-write wins), vfs_keys (last-write wins).
Checkpointer: AsyncPostgresSaver — full conversation history survives server restarts,
              supports async streaming via astream_events.

build_agent() is async and must be called from an async context (e.g. FastAPI lifespan).
For sync/CLI use, see the README for the sync PostgresSaver alternative.
"""

import os
from typing import Annotated, Any

import psycopg
from psycopg.rows import dict_row
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agent.config import MODEL_GENERAL, make_llm
from agent.db import DATABASE_URL, todos_get, vfs_list
from agent.skills import analyze_data, search_knowledge, write_code
from agent.tools import (
    bash_execute,
    list_files,
    read_file,
    request_options,
    write_file,
    write_todos,
)

TOOLS = [
    write_file,
    read_file,
    list_files,
    bash_execute,
    write_todos,
    request_options,
    analyze_data,
    write_code,
    search_knowledge,
]

_SYSTEM_PROMPT = """You are a Deep Agent — a careful, multi-step AI assistant.

When you receive a task:
1. Call write_todos with your full plan before starting any work.
2. Use write_file to store large artifacts (code, reports, data). Never put file *content* in your response — only paths.
3. Use bash_execute to run scripts. Files you wrote with write_file are available in the working directory.
4. Use the skill tools (analyze_data, write_code, search_knowledge) for specialised LLM work.
5. Call request_options when you need a human decision before proceeding.
6. When done, summarize what was accomplished and list the VFS paths of any artifacts.

Keep your reasoning concise. Prefer tools over inline content."""


class DeepAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    todos:    list[str]
    vfs_keys: list[str]


def _agent_node(state: DeepAgentState, config: RunnableConfig) -> dict[str, Any]:
    llm = make_llm(MODEL_GENERAL).bind_tools(TOOLS)
    messages = [SystemMessage(_SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages, config=config)

    tid = config.get("configurable", {}).get("thread_id", "default")
    return {
        "messages": [response],
        "todos":    todos_get(tid),
        "vfs_keys": vfs_list(tid),
    }


def _should_continue(state: DeepAgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


async def build_agent():
    """Build the compiled graph with an async PostgreSQL checkpointer.

    Opens a persistent async psycopg3 connection for the lifetime of the
    process. Call this once from FastAPI lifespan or an equivalent startup hook.
    """
    conn = await psycopg.AsyncConnection.connect(
        DATABASE_URL,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()  # idempotent — creates langgraph_checkpoint_* tables

    graph = StateGraph(DeepAgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
