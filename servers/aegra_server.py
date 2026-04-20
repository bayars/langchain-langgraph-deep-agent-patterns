"""
Aegra-style Server — LangGraph Platform REST API spec (port 8001).

Aegra is the open-source (Apache 2.0) alternative to the official LangGraph
Platform. It implements the same REST contract so any LangGraph SDK client
(JS or Python) can connect to it without modification.

Key differences vs the FastAPI server:
  - Typed HTTP resources: /threads, /runs (vs ad-hoc /api/simple/stream)
  - Standard SSE event names: metadata / updates / values / interrupt / error / end
  - Thread lifecycle management (create → run → inspect state)
  - Frontend can use the official @langchain/langgraph-sdk instead of raw fetch
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langgraph.types import Command
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agents.deep_agent import DEEP_AGENT, get_todos, get_vfs
from agents.simple_agent import SIMPLE_AGENT

app = FastAPI(title="Aegra-style LangGraph Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
PRESENTATION_DIR = Path(__file__).parent.parent / "presentation"

# ── In-memory stores (Aegra uses Postgres; we use dicts for the demo) ─────────

_threads: dict[str, dict] = {}
_runs: dict[str, dict] = {}   # run_id → run metadata

ASSISTANTS = {
    "simple": {"id": "simple", "name": "Simple ReAct Agent", "graph_id": "simple"},
    "deep":   {"id": "deep",   "name": "Deep Agent (VFS + Skills + HITL)", "graph_id": "deep"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_agent(assistant_id: str):
    if assistant_id == "simple":
        return SIMPLE_AGENT
    if assistant_id == "deep":
        return DEEP_AGENT
    raise HTTPException(status_code=404, detail=f"Unknown assistant: {assistant_id!r}")


# ── Request Models ────────────────────────────────────────────────────────────

class CreateThreadRequest(BaseModel):
    metadata: dict = Field(default_factory=dict)


class CreateRunRequest(BaseModel):
    assistant_id: str = "deep"
    input: dict = Field(default_factory=dict)  # {"messages": [{"role": "user", "content": "..."}]}
    config: dict = Field(default_factory=dict)
    stream_mode: str = "values"  # "values" | "updates" | "messages"


class ResumeRunRequest(BaseModel):
    values: Any  # the human's choice


# ── Thread Endpoints ──────────────────────────────────────────────────────────

@app.get("/assistants")
async def list_assistants():
    """List available agent assistants."""
    return list(ASSISTANTS.values())


@app.get("/api/test/stream")
async def test_stream():
    """Instant mock SSE — Aegra event schema. No Ollama needed."""
    import asyncio

    async def generator():
        yield {"event": "metadata", "data": json.dumps({"run_id": "test-run"})}
        await asyncio.sleep(0.2)
        for chunk in ["Hello ", "from ", "Aegra! "]:
            yield {"event": "messages/partial", "data": json.dumps({"content": chunk})}
            await asyncio.sleep(0.3)
        yield {"event": "tool_start", "data": json.dumps({"tool": "write_data", "input": {"key": "demo"}})}
        await asyncio.sleep(0.4)
        yield {"event": "tool_end", "data": json.dumps({"tool": "write_data", "output": "Stored!"})}
        await asyncio.sleep(0.3)
        for chunk in ["Platform ", "API ", "is ", "working ✓"]:
            yield {"event": "messages/partial", "data": json.dumps({"content": chunk})}
            await asyncio.sleep(0.2)
        yield {"event": "end", "data": json.dumps({})}

    return EventSourceResponse(generator())


@app.post("/threads", status_code=201)
async def create_thread(req: CreateThreadRequest = CreateThreadRequest()):
    thread_id = str(uuid.uuid4())
    thread = {
        "thread_id": thread_id,
        "created_at": _now(),
        "updated_at": _now(),
        "metadata": req.metadata,
        "status": "idle",
    }
    _threads[thread_id] = thread
    return thread


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    if thread_id not in _threads:
        raise HTTPException(404, f"Thread {thread_id!r} not found")
    return _threads[thread_id]


@app.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str):
    """Return the latest state snapshot for the thread."""
    if thread_id not in _threads:
        raise HTTPException(404, f"Thread {thread_id!r} not found")

    thread = _threads[thread_id]
    assistant_id = thread.get("assistant_id", "deep")
    agent = _get_agent(assistant_id)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = agent.get_state(config)
        values = state.values if state else {}
        # Enrich with side-channel VFS/todos
        vfs = get_vfs(thread_id)
        todos = get_todos(thread_id)
        return {
            "thread_id": thread_id,
            "values": {**values, "vfs": vfs, "todos": todos},
            "next": list(state.next) if state else [],
            "metadata": state.metadata if state else {},
        }
    except Exception as exc:
        return {"thread_id": thread_id, "values": {}, "error": str(exc)}


# ── Run / Stream Endpoints ────────────────────────────────────────────────────

@app.post("/threads/{thread_id}/runs/stream")
async def create_run_stream(thread_id: str, req: CreateRunRequest):
    """
    Create a run and stream its events using the LangGraph Platform event schema.
    This is the endpoint the LangGraph JS/Python SDK calls.
    """
    if thread_id not in _threads:
        raise HTTPException(404, f"Thread {thread_id!r} not found")

    run_id = str(uuid.uuid4())
    agent = _get_agent(req.assistant_id)

    _threads[thread_id]["assistant_id"] = req.assistant_id
    _threads[thread_id]["status"] = "running"
    _threads[thread_id]["updated_at"] = _now()
    _runs[run_id] = {
        "run_id": run_id,
        "thread_id": thread_id,
        "assistant_id": req.assistant_id,
        "status": "running",
        "created_at": _now(),
    }

    config = {"configurable": {"thread_id": thread_id, **req.config}}
    inputs = req.input

    async def generate():
        # ── event: metadata ──────────────────────────────────────
        yield {
            "event": "metadata",
            "data": json.dumps({"run_id": run_id, "assistant_id": req.assistant_id}),
        }

        try:
            async for event in agent.astream_events(inputs, config=config, version="v2"):
                name = event.get("event", "")
                data = event.get("data", {})

                # ── event: updates (per-node state deltas) ────────
                if name == "on_chain_end" and event.get("name") in ("agent", "tools"):
                    node_name = event.get("name")
                    output = data.get("output", {})
                    vfs = get_vfs(thread_id)
                    todos = get_todos(thread_id)
                    yield {
                        "event": "updates",
                        "data": json.dumps({
                            "node": node_name,
                            "state": {
                                "messages_count": len(output.get("messages", [])),
                                "todos": todos,
                                "vfs_keys": list(vfs.keys()),
                            },
                        }),
                    }

                # ── event: messages/partial (token streaming) ─────
                elif name == "on_chat_model_stream":
                    chunk = data.get("chunk", {})
                    content = getattr(chunk, "content", "") or ""
                    if content:
                        yield {
                            "event": "messages/partial",
                            "data": json.dumps({"content": content, "type": "AIMessageChunk"}),
                        }

                # ── event: tool invocations ───────────────────────
                elif name == "on_tool_start":
                    yield {
                        "event": "tool_start",
                        "data": json.dumps({
                            "tool": event.get("name"),
                            "input": data.get("input", {}),
                        }),
                    }
                elif name == "on_tool_end":
                    output = data.get("output", "")
                    if hasattr(output, "content"):
                        output = output.content
                    yield {
                        "event": "tool_end",
                        "data": json.dumps({
                            "tool": event.get("name"),
                            "output": str(output)[:500],
                        }),
                    }

        except Exception as exc:
            if "GraphInterrupt" in type(exc).__name__:
                try:
                    payload = exc.args[0] if exc.args else {}
                    if isinstance(payload, (list, tuple)) and len(payload) > 0:
                        interrupt_val = payload[0].value if hasattr(payload[0], "value") else payload[0]
                    else:
                        interrupt_val = payload

                    _threads[thread_id]["status"] = "interrupted"
                    _runs[run_id]["status"] = "interrupted"
                    _runs[run_id]["interrupt"] = interrupt_val

                    # ── event: interrupt ─────────────────────────
                    yield {
                        "event": "interrupt",
                        "data": json.dumps({"run_id": run_id, "value": interrupt_val}),
                    }
                except Exception:
                    yield {
                        "event": "interrupt",
                        "data": json.dumps({"run_id": run_id, "value": {"question": "Agent needs input", "options": []}}),
                    }
            else:
                # ── event: error ─────────────────────────────────
                _threads[thread_id]["status"] = "error"
                _runs[run_id]["status"] = "error"
                yield {
                    "event": "error",
                    "data": json.dumps({"run_id": run_id, "message": str(exc)}),
                }
            return

        # ── event: values (final full state) ─────────────────────
        vfs = get_vfs(thread_id)
        todos = get_todos(thread_id)
        _threads[thread_id]["status"] = "idle"
        _runs[run_id]["status"] = "success"

        yield {
            "event": "values",
            "data": json.dumps({
                "thread_id": thread_id,
                "vfs": vfs,
                "todos": todos,
            }),
        }

        # ── event: end ────────────────────────────────────────────
        yield {"event": "end", "data": json.dumps({})}

    return EventSourceResponse(generate())


@app.post("/threads/{thread_id}/runs/{run_id}/resume")
async def resume_run(thread_id: str, run_id: str, req: ResumeRunRequest):
    """Resume an interrupted run. Maps to LangGraph Command(resume=...)."""
    if thread_id not in _threads:
        raise HTTPException(404, f"Thread {thread_id!r} not found")
    if run_id not in _runs:
        raise HTTPException(404, f"Run {run_id!r} not found")

    thread = _threads[thread_id]
    assistant_id = thread.get("assistant_id", "deep")
    agent = _get_agent(assistant_id)
    config = {"configurable": {"thread_id": thread_id}}

    new_run_id = str(uuid.uuid4())
    _runs[new_run_id] = {
        "run_id": new_run_id,
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "status": "running",
        "created_at": _now(),
        "resumed_from": run_id,
    }

    async def generate():
        yield {
            "event": "metadata",
            "data": json.dumps({"run_id": new_run_id, "resumed_from": run_id}),
        }

        try:
            async for event in agent.astream_events(
                Command(resume=req.values), config=config, version="v2"
            ):
                name = event.get("event", "")
                data = event.get("data", {})

                if name == "on_chat_model_stream":
                    chunk = data.get("chunk", {})
                    content = getattr(chunk, "content", "") or ""
                    if content:
                        yield {
                            "event": "messages/partial",
                            "data": json.dumps({"content": content, "type": "AIMessageChunk"}),
                        }
                elif name == "on_tool_end":
                    output = data.get("output", "")
                    yield {
                        "event": "tool_end",
                        "data": json.dumps({"tool": event.get("name"), "output": str(output)[:300]}),
                    }
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}
            return

        vfs = get_vfs(thread_id)
        todos = get_todos(thread_id)
        _threads[thread_id]["status"] = "idle"
        _runs[new_run_id]["status"] = "success"

        yield {
            "event": "values",
            "data": json.dumps({"thread_id": thread_id, "vfs": vfs, "todos": todos}),
        }
        yield {"event": "end", "data": json.dumps({})}

    return EventSourceResponse(generate())


@app.get("/threads/{thread_id}/runs")
async def list_runs(thread_id: str):
    runs = [r for r in _runs.values() if r["thread_id"] == thread_id]
    return sorted(runs, key=lambda r: r["created_at"])


# ── Static ─────────────────────────────────────────────────────────────────────

@app.get("/presentation")
async def presentation():
    return FileResponse(PRESENTATION_DIR / "index.html")


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("servers.aegra_server:app", host="0.0.0.0", port=8001, reload=True)
