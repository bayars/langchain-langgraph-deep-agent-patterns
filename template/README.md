# Deep Agent Template

A production-ready LangGraph Deep Agent template with:

- **PostgreSQL VFS** — files persist across sessions, never enter the context window
- **bash_execute** — run code inside a temp dir populated from VFS; changes sync back automatically
- **PostgreSQL checkpointer** — full conversation history survives restarts
- **Aegra REST API** — LangGraph Platform-compatible streaming API (port 8001)
- **Deep Agents CLI** — works out of the box with `deepagents`

---

## Prerequisites

- Python 3.11+
- Docker + Docker Compose
- LLM gateway credentials (OpenAI-compatible endpoint)

---

## Quick start

```bash
# 1. Configure
cp .env.example .env
# Edit .env: fill in LLM_GATEWAY_URL, LLM_GATEWAY_KEY, model aliases

# 2. Start
docker compose up -d

# 3. Verify
docker compose logs template-server
# Should show: "Schema initialized" then "Uvicorn running on http://0.0.0.0:8001"
```

---

## Mode A — Deep Agents CLI

```bash
pip install deepagents-cli

# Point the CLI at your LLM gateway (same key as in .env)
export OPENAI_API_BASE="$LLM_GATEWAY_URL"
export OPENAI_API_KEY="$LLM_GATEWAY_KEY"

# Interactive TUI — AGENTS.md is used for context
deepagents --model openai:general

# Non-interactive / scripted
deepagents -m "Write a Python script to list prime numbers up to 100, run it, save output to primes.txt" -y

# ACP server mode (like opencode serve) — attach from another terminal
deepagents --acp
deepagents --attach <server-url>
```

> The CLI uses its own built-in bash, read_file, write_file tools alongside `AGENTS.md` context.
> For the CLI to use *your* VFS (PostgreSQL), connect it to the Aegra server via MCP — see `.mcp.json.example`.

---

## Mode B — Aegra REST API

```bash
# Create a thread
THREAD=$(curl -s -X POST http://localhost:8001/threads \
  -H "Content-Type: application/json" \
  -d '{"metadata": {}}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['thread_id'])")

echo "Thread: $THREAD"

# Stream a run (tokens arrive live)
curl -N -X POST "http://localhost:8001/threads/$THREAD/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "deep",
    "input": {
      "messages": [{"role": "user", "content": "Write hello.py that prints Hello World, then run it"}]
    }
  }'

# Inspect VFS after run
curl "http://localhost:8001/threads/$THREAD/state" | python3 -m json.tool

# Resume a HITL interrupt
curl -X POST "http://localhost:8001/threads/$THREAD/runs/REPLACE_RUN_ID/resume" \
  -H "Content-Type: application/json" \
  -d '{"values": "Full report"}'
```

---

## SSE event reference

| Event | Payload |
|-------|---------|
| `metadata` | `{run_id, assistant_id}` |
| `updates` | `{node, run_id}` |
| `messages/partial` | `{content, type}` — streaming token |
| `tool_start` | `{tool, input}` |
| `tool_end` | `{tool, output}` |
| `interrupt` | `{question, options}` — HITL pause |
| `values` | full state snapshot `{messages, todos, vfs, vfs_keys}` |
| `end` | `{}` — run complete |
| `error` | `{message}` |

---

## How VFS + bash_execute works

```
1. write_file("script.py", <code>)
      → INSERT INTO vfs_files (thread_id, "script.py", <code>)

2. bash_execute("python script.py")
      → SELECT * FROM vfs_files WHERE thread_id = ?   ← fetch all files
      → write each file to /tmp/agent_vfs_<id>/       ← temp dir
      → subprocess.run(["python", "script.py"], cwd=tempdir)
      → for each new/changed file in tempdir → INSERT/UPDATE vfs_files

3. read_file("output.txt")
      → SELECT content FROM vfs_files WHERE path = "output.txt"
```

Files never enter the LLM context window — only paths do. The agent can
work on arbitrarily large codebases without context overflow.

---

## Adapting this template

| What to change | Where |
|----------------|-------|
| LLM gateway URL / key | `.env` |
| Model aliases | `.env` (`MODEL_GENERAL`, `MODEL_CODE`) |
| `search_knowledge` implementation | `agent/skills.py` — replace with Glean API call |
| System prompt | `agent/graph.py` — `_SYSTEM_PROMPT` |
| Add a new tool | `agent/tools.py` + add to `TOOLS` in `agent/graph.py` |
| Scale VFS storage | `agent/db.py` — swap psycopg3 for asyncpg or S3 |
| Add auth | `server/main.py` — add FastAPI dependency for API key header |

---

## Project structure

```
template/
├── agent/
│   ├── config.py    LLM gateway — make_llm(), model constants
│   ├── db.py        PostgreSQL — VFS, todos, threads, runs
│   ├── tools.py     LangGraph tools — VFS ops + bash_execute + HITL
│   ├── skills.py    Skill tools — analyze_data, write_code, search_knowledge
│   └── graph.py     StateGraph + PostgresSaver checkpointer + AGENT singleton
├── server/
│   └── main.py      Aegra REST API (FastAPI + SSE)
├── AGENTS.md        Deep Agents CLI context
├── .env.example     Required env vars
├── docker-compose.yml  postgres:16 + template-server
└── Dockerfile
```
