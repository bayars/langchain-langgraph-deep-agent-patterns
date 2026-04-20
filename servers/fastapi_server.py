"""
FastAPI Server — RAW SSE approach (port 8000).

Key characteristic: you manually translate astream_events into a custom
hand-rolled JSON event schema. The frontend must know and parse this schema.
No standard contract — tight coupling between server and client.
"""

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agents.deep_agent import DEEP_AGENT, get_todos, get_vfs
from agents.simple_agent import SIMPLE_AGENT, get_store_snapshot

app = FastAPI(title="DeepAgent Demo — FastAPI (Raw SSE)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
PRESENTATION_DIR = Path(__file__).parent.parent / "presentation"


# ── Request Models ────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    message: str
    thread_id: str = ""


class ResumeRequest(BaseModel):
    choice: str


# ── SSE Helpers ───────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Serialize one SSE data line."""
    return json.dumps(payload)


async def _stream_agent(agent, message: str, thread_id: str):
    """
    RAW SSE generator — manually maps astream_events to a custom event schema.
    This is the FastAPI "manual" approach: explicit if/elif on event_name.
    """
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [{"role": "user", "content": message}]}

    try:
        async for event in agent.astream_events(inputs, config=config, version="v2"):
            name = event.get("event", "")
            data = event.get("data", {})
            tags = event.get("tags", [])

            if name == "on_chat_model_stream":
                chunk = data.get("chunk", {})
                content = getattr(chunk, "content", "") or ""
                if content:
                    yield _sse({"type": "token", "content": content})

            elif name == "on_tool_start":
                yield _sse({
                    "type": "tool_start",
                    "tool": event.get("name", ""),
                    "input": data.get("input", {}),
                })

            elif name == "on_tool_end":
                output = data.get("output", "")
                if hasattr(output, "content"):
                    output = output.content
                yield _sse({
                    "type": "tool_end",
                    "tool": event.get("name", ""),
                    "output": str(output)[:500],
                })

                # Sync side-panel state after each tool
                vfs = get_vfs(thread_id)
                todos = get_todos(thread_id)
                if vfs:
                    yield _sse({"type": "vfs_update", "files": vfs})
                if todos:
                    yield _sse({"type": "todos_update", "todos": todos})

    except Exception as exc:
        error_str = str(exc)
        # LangGraph surfaces HITL as a GraphInterrupt exception
        if "GraphInterrupt" in type(exc).__name__ or "__interrupt__" in error_str:
            import re
            # Extract interrupt payload if possible
            try:
                payload = exc.args[0] if exc.args else {}
                if isinstance(payload, (list, tuple)) and len(payload) > 0:
                    interrupt_val = payload[0].value if hasattr(payload[0], "value") else payload[0]
                else:
                    interrupt_val = payload
                yield _sse({
                    "type": "interrupt",
                    "question": interrupt_val.get("question", "Please choose:") if isinstance(interrupt_val, dict) else str(interrupt_val),
                    "options": interrupt_val.get("options", []) if isinstance(interrupt_val, dict) else [],
                })
            except Exception:
                yield _sse({"type": "interrupt", "question": "Agent needs input", "options": []})
            return

        yield _sse({"type": "error", "message": error_str})
        return

    yield _sse({"type": "done"})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/test/stream")
async def test_stream():
    """Instant mock SSE — no Ollama needed. Use to verify browser SSE pipeline."""
    import asyncio

    async def generator():
        for chunk in ["Hello ", "from ", "the ", "agent! "]:
            yield {"data": json.dumps({"type": "token", "content": chunk})}
            await asyncio.sleep(0.3)
        yield {"data": json.dumps({"type": "tool_start", "tool": "write_data", "input": {"key": "demo", "value": "works"}})}
        await asyncio.sleep(0.4)
        yield {"data": json.dumps({"type": "tool_end", "tool": "write_data", "output": "Stored: 'demo' = 'works'"})}
        await asyncio.sleep(0.3)
        for chunk in ["SSE ", "pipeline ", "is ", "working ✓"]:
            yield {"data": json.dumps({"type": "token", "content": chunk})}
            await asyncio.sleep(0.2)
        yield {"data": json.dumps({"type": "done"})}

    return EventSourceResponse(generator())


@app.post("/api/simple/stream")
async def stream_simple(req: RunRequest):
    thread_id = req.thread_id or str(uuid.uuid4())

    async def generator():
        async for data in _stream_agent(SIMPLE_AGENT, req.message, thread_id):
            yield {"data": data}

    return EventSourceResponse(generator())


@app.post("/api/deep/stream")
async def stream_deep(req: RunRequest):
    thread_id = req.thread_id or str(uuid.uuid4())

    async def generator():
        async for data in _stream_agent(DEEP_AGENT, req.message, thread_id):
            yield {"data": data}

    return EventSourceResponse(generator())


@app.post("/api/resume/{thread_id}")
async def resume_thread(thread_id: str, req: ResumeRequest):
    """Resume an interrupted deep agent thread with a human choice."""
    config = {"configurable": {"thread_id": thread_id}}

    events = []
    try:
        async for event in DEEP_AGENT.astream_events(
            Command(resume=req.choice), config=config, version="v2"
        ):
            name = event.get("event", "")
            data = event.get("data", {})
            if name == "on_chat_model_stream":
                chunk = data.get("chunk", {})
                content = getattr(chunk, "content", "") or ""
                if content:
                    events.append({"type": "token", "content": content})
            elif name == "on_tool_end":
                output = data.get("output", "")
                events.append({"type": "tool_end", "tool": event.get("name"), "output": str(output)[:300]})
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    vfs = get_vfs(thread_id)
    todos = get_todos(thread_id)
    return {
        "status": "resumed",
        "thread_id": thread_id,
        "events": events,
        "vfs": vfs,
        "todos": todos,
    }


@app.get("/api/threads/{thread_id}/vfs")
async def get_thread_vfs(thread_id: str):
    return {"thread_id": thread_id, "files": get_vfs(thread_id)}


@app.get("/api/threads/{thread_id}/todos")
async def get_thread_todos(thread_id: str):
    return {"thread_id": thread_id, "todos": get_todos(thread_id)}


@app.get("/api/store")
async def get_simple_store():
    return {"store": get_store_snapshot()}


# ── Static Files ──────────────────────────────────────────────────────────────

@app.get("/test")
async def test_page():
    return FileResponse(FRONTEND_DIR / "test.html")


@app.get("/presentation")
async def presentation():
    return FileResponse(PRESENTATION_DIR / "index.html")


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("servers.fastapi_server:app", host="0.0.0.0", port=8000, reload=True)
