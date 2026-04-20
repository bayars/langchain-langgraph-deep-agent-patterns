"""
Deep Agent — LangGraph StateGraph with:
  - Virtual File System (VFS) scoped per thread
  - Planning via write_todos tool
  - Skills (analyze_data, write_code, search_knowledge) each using their own LLM
  - Human-in-the-Loop via LangGraph interrupt()
"""

import json
from collections import defaultdict
from typing import Annotated, Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from typing_extensions import TypedDict

from .config import make_llm, MODEL_GENERAL
from .skills import analyze_data, search_knowledge, write_code

# Thread-scoped side-channel stores (keyed by thread_id)
_vfs_store: dict[str, dict[str, str]] = defaultdict(dict)
_todos_store: dict[str, list[str]] = defaultdict(list)


# ── State ────────────────────────────────────────────────────────────────────

class DeepAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    todos: list[str]
    completed_todos: list[str]
    vfs_keys: list[str]  # mirrors _vfs_store keys for state visibility


# ── VFS / Meta Tools ─────────────────────────────────────────────────────────

@tool
def write_file(path: str, content: str, config: RunnableConfig) -> str:
    """Write content to a file in the agent's virtual file system."""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    _vfs_store[thread_id][path] = content
    return f"File written: {path!r} ({len(content)} chars)"


@tool
def read_file(path: str, config: RunnableConfig) -> str:
    """Read a file from the agent's virtual file system."""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    vfs = _vfs_store[thread_id]
    if path not in vfs:
        return f"File not found: {path!r}. Available: {list(vfs.keys())}"
    return f"=== {path} ===\n{vfs[path]}"


@tool
def list_files(config: RunnableConfig) -> str:
    """List all files in the agent's virtual file system."""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    files = list(_vfs_store[thread_id].keys())
    return "VFS files: " + json.dumps(files) if files else "VFS is empty."


@tool
def write_todos(todos: list[str], config: RunnableConfig) -> str:
    """
    Set the agent's todo/plan list. Call this first to establish your plan
    before executing tasks.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    _todos_store[thread_id] = todos
    lines = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(todos))
    return f"Plan set ({len(todos)} steps):\n{lines}"


@tool
def request_options(question: str, options: list[str], config: RunnableConfig) -> str:
    """
    Pause execution and ask the human to choose from a list of options.
    Use this when you need human guidance before proceeding.
    """
    # LangGraph interrupt — pauses the graph and surfaces data to the GUI
    choice = interrupt({"question": question, "options": options})
    return f"Human selected: {choice!r}"


# ── Graph ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Deep Agent with the following capabilities:

1. PLANNING: Always start complex tasks by calling write_todos() with a step-by-step plan.
2. VIRTUAL FILE SYSTEM: Use write_file/read_file/list_files to store and retrieve artifacts.
   Save analysis results, generated code, and reports as files.
3. SKILLS: Delegate specialized work to skill tools:
   - analyze_data(data) — deep analysis using a reasoning model
   - write_code(task, language) — write code using a specialized coding model
   - search_knowledge(query) — search the internal knowledge base
4. HUMAN-IN-THE-LOOP: Use request_options(question, options) when you need a human decision.

Workflow for complex tasks:
  1. Call write_todos() with your plan
  2. Execute each step, using skills and VFS tools
  3. Save final artifacts to VFS
  4. Summarize what was accomplished
"""

_ALL_TOOLS = [
    write_todos,
    write_file,
    read_file,
    list_files,
    analyze_data,
    write_code,
    search_knowledge,
    request_options,
]


def _agent_node(state: DeepAgentState, config: RunnableConfig) -> dict[str, Any]:
    llm = make_llm(MODEL_GENERAL).bind_tools(_ALL_TOOLS)
    thread_id = config.get("configurable", {}).get("thread_id", "default")

    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm.invoke(messages, config=config)

    # Sync vfs_keys so state reflects current VFS
    vfs_keys = list(_vfs_store[thread_id].keys())
    todos = _todos_store.get(thread_id, state.get("todos", []))

    return {
        "messages": [response],
        "todos": todos,
        "vfs_keys": vfs_keys,
    }


def _should_continue(state: DeepAgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def create_deep_agent():
    graph = StateGraph(DeepAgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(_ALL_TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    # interrupt() inside request_options tool handles HITL — no interrupt_before needed
    return graph.compile(checkpointer=memory)


# Expose side-channel accessors for servers
def get_vfs(thread_id: str) -> dict[str, str]:
    return dict(_vfs_store.get(thread_id, {}))


def get_todos(thread_id: str) -> list[str]:
    return list(_todos_store.get(thread_id, []))


DEEP_AGENT = create_deep_agent()
