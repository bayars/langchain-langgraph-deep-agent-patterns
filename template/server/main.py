"""
Aegra-compatible REST API — LangGraph Platform spec.

All state is persisted to PostgreSQL. No module-level dicts.

SSE event names (identical to LangGraph Platform):
  metadata        — run_id, assistant_id
  updates         — per-node state delta
  messages/partial — streaming token from LLM
  tool_start      — tool invocation beginning
  tool_end        — tool invocation result
  interrupt       — HITL pause (question + options)
  values          — full state snapshot at end of run
  end             — run complete
  error           — exception message
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agent.db import (
    DATABASE_URL,
    init_schema,
    run_create,
    run_update_status,
    runs_list,
    thread_create,
    thread_get,
    thread_get_by_session,
    thread_update_status,
    todos_get,
    vfs_get_all,
)
from agent.graph import TOOLS, build_agent

ASSISTANTS = {
    "deep": {
        "id":          "deep",
        "name":        "Deep Agent",
        "description": "LangGraph Deep Agent — VFS + skills + bash + HITL",
        "tools":       [t.name for t in TOOLS],
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema(DATABASE_URL)
    app.state.agent = await build_agent()
    yield


app = FastAPI(title="Deep Agent — Aegra API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────────────

class CreateThreadRequest(BaseModel):
    metadata: dict = {}


class CreateRunRequest(BaseModel):
    assistant_id: str
    input: dict
    config: dict = {}


class ResumeRunRequest(BaseModel):
    values: str | dict


class ChatStreamRequest(BaseModel):
    message: str
    thread_id: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data)}


async def _stream_run(agent, thread_id: str, run_id: str, agent_input, config: dict):
    """
    Async generator that drives astream_events and yields SSE-formatted dicts.
    Handles both normal completion and GraphInterrupt (HITL).
    """
    run_update_status(run_id, "running")
    thread_update_status(thread_id, "running")

    langgraph_config = {
        "configurable": {"thread_id": thread_id, **(config.get("configurable", {}))},
    }

    yield _sse("metadata", {"run_id": run_id, "assistant_id": "deep"})

    try:
        async for event in agent.astream_events(agent_input, config=langgraph_config, version="v2"):
            name = event.get("event", "")
            data = event.get("data", {})

            if name == "on_chain_start" and event.get("name") in ("agent", "tools"):
                yield _sse("updates", {"node": event["name"], "run_id": run_id})

            elif name == "on_chat_model_stream":
                chunk = data.get("chunk", {})
                content = getattr(chunk, "content", "") or ""
                if content:
                    yield _sse("messages/partial", {"content": content, "type": "AIMessageChunk"})

            elif name == "on_tool_start":
                yield _sse("tool_start", {
                    "tool":  event.get("name", ""),
                    "input": data.get("input", {}),
                })

            elif name == "on_tool_end":
                output = data.get("output", "")
                if hasattr(output, "content"):
                    output = output.content
                yield _sse("tool_end", {
                    "tool":   event.get("name", ""),
                    "output": str(output)[:500],
                })

    except Exception as exc:
        import traceback
        exc_type = type(exc).__name__
        print(f"[stream error] {exc_type}: {exc}\n{traceback.format_exc()}", flush=True)

        if "GraphInterrupt" in exc_type or "__interrupt__" in str(exc):
            payload = _extract_interrupt(exc)
            run_update_status(run_id, "interrupted")
            thread_update_status(thread_id, "interrupted")
            yield _sse("interrupt", payload)
            return

        run_update_status(run_id, "error")
        thread_update_status(thread_id, "idle")
        yield _sse("error", {"message": f"{exc_type}: {exc or traceback.format_exc().splitlines()[-1]}"})
        return

    # Emit final state snapshot (must use async API with AsyncPostgresSaver)
    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    if state and state.values:
        sv = dict(state.values)
        sv["vfs"]    = vfs_get_all(thread_id)
        sv["todos"]  = todos_get(thread_id)
        sv["messages"] = [
            {"role": getattr(m, "type", "unknown"), "content": getattr(m, "content", "")}
            for m in sv.get("messages", [])
        ]
        yield _sse("values", sv)

    run_update_status(run_id, "done")
    thread_update_status(thread_id, "idle")
    yield _sse("end", {})


def _extract_interrupt(exc: Exception) -> dict:
    try:
        payload = exc.args[0] if exc.args else {}
        if isinstance(payload, (list, tuple)) and payload:
            val = payload[0].value if hasattr(payload[0], "value") else payload[0]
        else:
            val = payload
        if isinstance(val, dict):
            return {"question": val.get("question", "Input required"), "options": val.get("options", [])}
        return {"question": str(val), "options": []}
    except Exception:
        return {"question": "Agent requires input", "options": []}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/assistants")
def list_assistants():
    return list(ASSISTANTS.values())


@app.post("/threads")
def create_thread(req: CreateThreadRequest):
    thread_id = str(uuid.uuid4())
    return thread_create(thread_id, req.metadata)


@app.get("/threads/{thread_id}")
def get_thread(thread_id: str):
    t = thread_get(thread_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")
    return t


@app.get("/threads/{thread_id}/state")
async def get_thread_state(request: Request, thread_id: str):
    if not thread_get(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    agent = request.app.state.agent
    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    values = dict(state.values) if state and state.values else {}
    values["vfs"]    = vfs_get_all(thread_id)
    values["todos"]  = todos_get(thread_id)
    values["messages"] = [
        {"role": getattr(m, "type", "unknown"), "content": getattr(m, "content", "")}
        for m in values.get("messages", [])
    ]
    return {"thread_id": thread_id, "values": values}


@app.post("/threads/{thread_id}/runs/stream")
def stream_run(request: Request, thread_id: str, req: CreateRunRequest):
    if req.assistant_id not in ASSISTANTS:
        raise HTTPException(status_code=400, detail=f"Unknown assistant: {req.assistant_id!r}")
    if not thread_get(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")

    run_id = str(uuid.uuid4())
    run_create(run_id, thread_id, req.assistant_id)
    agent = request.app.state.agent

    async def generator():
        async for chunk in _stream_run(agent, thread_id, run_id, req.input, req.config):
            yield chunk

    return EventSourceResponse(generator())


@app.post("/threads/{thread_id}/runs/{run_id}/resume")
def resume_run(request: Request, thread_id: str, run_id: str, req: ResumeRunRequest):
    if not thread_get(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")

    new_run_id = str(uuid.uuid4())
    run_create(new_run_id, thread_id, "deep")
    agent = request.app.state.agent

    async def generator():
        async for chunk in _stream_run(agent, thread_id, new_run_id, Command(resume=req.values), {}):
            yield chunk

    return EventSourceResponse(generator())


@app.get("/sessions/{session_id}/state")
async def get_session_state(request: Request, session_id: str):
    """
    Return VFS + todos for a Langflow session.
    Looks up the Aegra thread tagged with metadata.lf_session = session_id.
    """
    thread = thread_get_by_session(session_id)
    if not thread:
        return {"session_id": session_id, "vfs": {}, "todos": []}
    tid = thread["thread_id"]
    return {
        "session_id": session_id,
        "thread_id":  tid,
        "vfs":        vfs_get_all(tid),
        "todos":      todos_get(tid),
    }


@app.get("/threads/{thread_id}/runs")
def list_runs(thread_id: str):
    if not thread_get(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    return runs_list(thread_id)


@app.post("/chat/stream")
def chat_stream(request: Request, body: ChatStreamRequest):
    """
    Simplified SSE endpoint (FastAPI mode in the GUI).

    Accepts {message, thread_id?} — creates a thread internally if none
    is provided. Translates Aegra SSE events into the simpler
    {type: token|tool_start|tool_end|todos_update|vfs_update|interrupt|error|done}
    schema so callers don't need to manage threads or understand the full
    LangGraph Platform event spec.
    """
    tid = body.thread_id or str(uuid.uuid4())
    if not thread_get(tid):
        thread_create(tid, {})

    run_id = str(uuid.uuid4())
    run_create(run_id, tid, "deep")
    agent = request.app.state.agent

    async def generator():
        agent_input = {"messages": [{"role": "user", "content": body.message}]}
        async for chunk in _stream_run(agent, tid, run_id, agent_input, {}):
            event_name = chunk.get("event", "")
            try:
                data = json.loads(chunk.get("data", "{}"))
            except (json.JSONDecodeError, TypeError):
                continue

            simplified = None

            if event_name == "messages/partial":
                content = data.get("content", "")
                if content:
                    simplified = {"type": "token", "content": content}

            elif event_name == "tool_start":
                simplified = {
                    "type": "tool_start",
                    "tool":  data.get("tool", ""),
                    "input": data.get("input", {}),
                }

            elif event_name == "tool_end":
                simplified = {
                    "type":   "tool_end",
                    "tool":   data.get("tool", ""),
                    "output": data.get("output", ""),
                }

            elif event_name == "values":
                if data.get("todos"):
                    yield {"data": json.dumps({"type": "todos_update", "todos": data["todos"]})}
                if data.get("vfs"):
                    yield {"data": json.dumps({"type": "vfs_update", "vfs": data["vfs"]})}
                continue

            elif event_name == "interrupt":
                simplified = {
                    "type":     "interrupt",
                    "question": data.get("question", "Input required"),
                    "options":  data.get("options", []),
                }

            elif event_name == "error":
                simplified = {"type": "error", "message": data.get("message", "Unknown error")}

            elif event_name == "end":
                simplified = {"type": "done"}

            if simplified:
                yield {"data": json.dumps(simplified)}

    return EventSourceResponse(generator())
