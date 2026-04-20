"""
Simple LangGraph ReAct agent — demonstrates basic tool-use loop.
Uses a module-level dict as an in-memory key-value data store.
"""

import json
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from .config import make_llm, MODEL_GENERAL

_data_store: dict[str, str] = {}


@tool
def write_data(key: str, value: str) -> str:
    """Write a key-value pair to the shared data store."""
    _data_store[key] = value
    return f"Stored: {key!r} = {value!r}"


@tool
def read_data(key: str) -> str:
    """Read a value from the data store by key."""
    if key in _data_store:
        return f"{key!r} = {_data_store[key]!r}"
    return f"Key {key!r} not found in data store."


@tool
def list_data() -> str:
    """List all keys currently in the data store."""
    if not _data_store:
        return "Data store is empty."
    return "Keys: " + json.dumps(list(_data_store.keys()))


def get_store_snapshot() -> dict:
    return dict(_data_store)


def create_simple_agent():
    llm = make_llm(MODEL_GENERAL)
    memory = MemorySaver()
    tools = [write_data, read_data, list_data]
    return create_react_agent(llm, tools, checkpointer=memory)


SIMPLE_AGENT = create_simple_agent()
