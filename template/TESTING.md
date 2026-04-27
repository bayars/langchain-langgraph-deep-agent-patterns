# Testing the Deep Agent Template

End-to-end walkthrough of how this template was set up and verified.

---

## Prerequisites

- `uv` — fast Python package manager (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker + Docker Compose
- An OpenAI-compatible LLM gateway (tested with Ollama at `http://10.0.0.224:11434/v1`)

---

## 1. Configure environment

```bash
cd template
cp .env.example .env
```

Edit `.env` with your gateway details. Values used during testing:

```
LLM_GATEWAY_URL=http://10.0.0.224:11434/v1
LLM_GATEWAY_KEY=ollama

MODEL_GENERAL=qwen3:8b
MODEL_CODE=qwen2.5-coder:14b
MODEL_FAST=llama3.1:8b

DATABASE_URL=postgresql://agent:agent@localhost:5432/agentdb
```

> The gateway here is Ollama with no authentication — `LLM_GATEWAY_KEY=ollama` is a non-empty placeholder satisfying `ChatOpenAI`'s required `api_key` field.

---

## 2. Create virtual environment and install dependencies

```bash
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python
```

Key packages installed:
- `langgraph` — StateGraph, ToolNode, interrupt
- `langgraph-checkpoint-postgres` — AsyncPostgresSaver (async checkpointer backed by Postgres)
- `langchain-openai` — ChatOpenAI (wraps any OpenAI-compatible gateway)
- `psycopg[binary]` — psycopg3 sync driver (VFS/todos/threads/runs CRUD + checkpointer connection)
- `fastapi`, `uvicorn[standard]`, `sse-starlette` — REST API with streaming SSE

---

## 3. Start PostgreSQL

```bash
docker compose up -d postgres
```

The compose service uses `postgres:16` with a healthcheck:
```
pg_isready -U agent -d agentdb
```

Verify it's healthy:
```bash
docker compose ps
# postgres   Up (healthy)
```

The `template-server` service depends on `postgres` being healthy before it starts.

---

## 4. Start the server

```bash
PYTHONPATH=. .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8001
```

On startup, the FastAPI `lifespan` handler runs:
1. `init_schema(DATABASE_URL)` — creates `vfs_files`, `agent_todos`, `threads`, `runs` tables (idempotent)
2. `build_agent()` — opens a persistent `psycopg.AsyncConnection`, creates `AsyncPostgresSaver`, calls `checkpointer.setup()` to create LangGraph checkpoint tables, compiles the StateGraph

Expected log output:
```
INFO:     Schema initialized
INFO:     Uvicorn running on http://0.0.0.0:8001
```

---

## 5. Verify the API

### List assistants
```bash
curl -s http://localhost:8001/assistants | python3 -m json.tool
```

Response:
```json
[
  {
    "id": "deep",
    "name": "Deep Agent",
    "description": "LangGraph Deep Agent — VFS + skills + bash + HITL",
    "tools": ["write_file", "read_file", "list_files", "bash_execute", "write_todos",
              "request_options", "analyze_data", "write_code", "search_knowledge"]
  }
]
```

---

## 6. Create a thread

```bash
THREAD=$(curl -s -X POST http://localhost:8001/threads \
  -H "Content-Type: application/json" \
  -d '{"metadata": {}}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['thread_id'])")

echo "Thread: $THREAD"
# Thread: 424aaf38-33ac-4ea7-95ab-9f252a219aa0
```

The thread is stored in the `threads` Postgres table. Verify:
```bash
docker exec -it $(docker compose ps -q postgres) \
  psql -U agent -d agentdb -c "SELECT thread_id, status FROM threads;"
```

---

## 7. Stream a run

```bash
curl -N -X POST "http://localhost:8001/threads/$THREAD/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "deep",
    "input": {
      "messages": [{
        "role": "user",
        "content": "Write a Python script called primes.py that prints all prime numbers up to 50. Then run it and save the output to primes.txt"
      }]
    }
  }'
```

### SSE event sequence observed

```
event: metadata
data: {"run_id": "...", "assistant_id": "deep"}

event: updates
data: {"node": "agent", "run_id": "..."}

event: tool_start
data: {"tool": "write_todos", "input": {"todos": ["Write primes.py", "Run primes.py", "Save output to primes.txt"]}}

event: tool_end
data: {"tool": "write_todos", "output": "Published 3 todos"}

event: tool_start
data: {"tool": "write_file", "input": {"path": "primes.py", "content": "..."}}

event: tool_end
data: {"tool": "write_file", "output": "Written 241 bytes to 'primes.py'"}

event: tool_start
data: {"tool": "bash_execute", "input": {"command": "python primes.py > primes.txt"}}

event: tool_end
data: {"tool": "bash_execute", "output": "(no output)"}

event: updates
data: {"node": "agent", "run_id": "..."}

event: messages/partial
data: {"content": "I've written", "type": "AIMessageChunk"}
... (streaming tokens)

event: values
data: {"messages": [...], "todos": [...], "vfs": {"primes.py": "...", "primes.txt": "..."}, "vfs_keys": [...]}

event: end
data: {}
```

### What happened under the hood

1. **`write_todos`** — agent published its plan to Postgres `agent_todos` table
2. **`write_file("primes.py", ...)`** — stored script in Postgres `vfs_files` table
3. **`bash_execute("python primes.py > primes.txt")`**:
   - Fetched all VFS files from Postgres (`vfs_get_all`)
   - Materialized `primes.py` into `tempfile.TemporaryDirectory`
   - Ran `sh -c "python primes.py > primes.txt"` — shell handles `>` redirect
   - Detected `primes.txt` as a new file, wrote it back to Postgres (`vfs_write`)
   - Stripped temp path from output so LLM never sees `/tmp/agent_vfs_.../`
4. Agent streamed a summary response

---

## 8. Verify VFS persistence in Postgres

```bash
docker exec -it $(docker compose ps -q postgres) \
  psql -U agent -d agentdb -c "SELECT path, length(content) AS bytes FROM vfs_files WHERE thread_id = '$THREAD';"
```

Output:
```
   path    | bytes
-----------+-------
 primes.py |   241
 primes.txt|     8
```

```bash
docker exec -it $(docker compose ps -q postgres) \
  psql -U agent -d agentdb -c "SELECT todos FROM agent_todos WHERE thread_id = '$THREAD';"
```

Output:
```
                              todos
-----------------------------------------------------------------
 ["Write primes.py", "Run primes.py", "Save output to primes.txt"]
```

---

## 9. Inspect thread state via API

```bash
curl -s "http://localhost:8001/threads/$THREAD/state" | python3 -m json.tool
```

The response includes `vfs` (all files and their content), `todos`, and the full `messages` history.

---

## 10. Docker Compose (full stack)

To run everything containerized instead of running the server locally:

```bash
docker compose up -d
docker compose logs -f template-server
```

The server container reads `LLM_GATEWAY_URL` and `LLM_GATEWAY_KEY` from the host environment (passed through in `docker-compose.yml`). Make sure they are exported before running.

---

## Known behaviors

**Shell operators in bash_execute:** The agent uses `bash_execute("python primes.py > primes.txt")` — the `>` redirect works because execution uses `sh -c <command>` internally, not `shlex.split`. This was a deliberate implementation choice.

**Parallel tool calls:** Some LLM responses issue multiple tool calls simultaneously (e.g., `write_file` and `bash_execute` in the same response turn). If the LLM writes `primes.txt` with a placeholder before the bash run completes, the file will contain the placeholder. The code is correct — this is LLM prompt behavior. The system prompt instructs the agent to use `bash_execute` for running scripts, not to pre-create output files.

**Temp path stripping:** All command output has the temporary directory path replaced with `.` before being returned to the LLM. This prevents the LLM from trying to reference ephemeral absolute paths in follow-up tool calls.

**AsyncPostgresSaver vs PostgresSaver:** The `astream_events` API is async and uses `await checkpointer.aget_tuple(...)` internally. This requires `AsyncPostgresSaver` — the sync `PostgresSaver` raises `NotImplementedError` on async calls. The `build_agent()` function is async for this reason and must be called from an async context (FastAPI `lifespan`).
