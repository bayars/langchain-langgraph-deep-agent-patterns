"""
PostgreSQL persistence layer.

Four tables:
  vfs_files   — per-thread virtual file system (path → content)
  agent_todos — per-thread todo list (JSONB array)
  threads     — Aegra thread registry
  runs        — Aegra run registry

All functions are synchronous and accept an explicit conn_str so they are
easy to test in isolation (pass a test-database URL, no monkey-patching needed).

The LangGraph checkpointer (PostgresSaver) manages its own tables separately.
"""

import json
import os
from datetime import datetime, timezone

import psycopg

DATABASE_URL: str = os.environ["DATABASE_URL"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vfs_files (
    thread_id  TEXT        NOT NULL,
    path       TEXT        NOT NULL,
    content    TEXT        NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (thread_id, path)
);

CREATE TABLE IF NOT EXISTS agent_todos (
    thread_id  TEXT        NOT NULL PRIMARY KEY,
    todos      JSONB       NOT NULL DEFAULT '[]',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS threads (
    thread_id  TEXT        NOT NULL PRIMARY KEY,
    metadata   JSONB       NOT NULL DEFAULT '{}',
    status     TEXT        NOT NULL DEFAULT 'idle',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT        NOT NULL PRIMARY KEY,
    thread_id    TEXT        NOT NULL REFERENCES threads(thread_id),
    assistant_id TEXT        NOT NULL,
    status       TEXT        NOT NULL DEFAULT 'pending',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def init_schema(conn_str: str = DATABASE_URL) -> None:
    """Create all tables if they do not exist. Safe to call on every startup."""
    with psycopg.connect(conn_str) as conn:
        conn.execute(_SCHEMA)
        conn.commit()


# ── VFS ───────────────────────────────────────────────────────────────────────

def vfs_write(thread_id: str, path: str, content: str, conn_str: str = DATABASE_URL) -> None:
    with psycopg.connect(conn_str) as conn:
        conn.execute(
            """
            INSERT INTO vfs_files (thread_id, path, content)
            VALUES (%s, %s, %s)
            ON CONFLICT (thread_id, path)
            DO UPDATE SET content = EXCLUDED.content, updated_at = NOW()
            """,
            (thread_id, path, content),
        )
        conn.commit()


def vfs_read(thread_id: str, path: str, conn_str: str = DATABASE_URL) -> str | None:
    with psycopg.connect(conn_str) as conn:
        row = conn.execute(
            "SELECT content FROM vfs_files WHERE thread_id = %s AND path = %s",
            (thread_id, path),
        ).fetchone()
    return row[0] if row else None


def vfs_list(thread_id: str, conn_str: str = DATABASE_URL) -> list[str]:
    with psycopg.connect(conn_str) as conn:
        rows = conn.execute(
            "SELECT path FROM vfs_files WHERE thread_id = %s ORDER BY path",
            (thread_id,),
        ).fetchall()
    return [r[0] for r in rows]


def vfs_get_all(thread_id: str, conn_str: str = DATABASE_URL) -> dict[str, str]:
    with psycopg.connect(conn_str) as conn:
        rows = conn.execute(
            "SELECT path, content FROM vfs_files WHERE thread_id = %s",
            (thread_id,),
        ).fetchall()
    return {r[0]: r[1] for r in rows}


# ── Todos ─────────────────────────────────────────────────────────────────────

def todos_write(thread_id: str, todos: list[str], conn_str: str = DATABASE_URL) -> None:
    with psycopg.connect(conn_str) as conn:
        conn.execute(
            """
            INSERT INTO agent_todos (thread_id, todos)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (thread_id)
            DO UPDATE SET todos = EXCLUDED.todos, updated_at = NOW()
            """,
            (thread_id, json.dumps(todos)),
        )
        conn.commit()


def todos_get(thread_id: str, conn_str: str = DATABASE_URL) -> list[str]:
    with psycopg.connect(conn_str) as conn:
        row = conn.execute(
            "SELECT todos FROM agent_todos WHERE thread_id = %s",
            (thread_id,),
        ).fetchone()
    return row[0] if row else []


# ── Threads ───────────────────────────────────────────────────────────────────

def thread_create(
    thread_id: str,
    metadata: dict | None = None,
    conn_str: str = DATABASE_URL,
) -> dict:
    meta = json.dumps(metadata or {})
    with psycopg.connect(conn_str) as conn:
        row = conn.execute(
            """
            INSERT INTO threads (thread_id, metadata)
            VALUES (%s, %s::jsonb)
            RETURNING thread_id, metadata, status, created_at, updated_at
            """,
            (thread_id, meta),
        ).fetchone()
        conn.commit()
    return _thread_row(row)


def thread_get(thread_id: str, conn_str: str = DATABASE_URL) -> dict | None:
    with psycopg.connect(conn_str) as conn:
        row = conn.execute(
            "SELECT thread_id, metadata, status, created_at, updated_at FROM threads WHERE thread_id = %s",
            (thread_id,),
        ).fetchone()
    return _thread_row(row) if row else None


def thread_get_by_session(session_id: str, conn_str: str = DATABASE_URL) -> dict | None:
    """Find the most-recent thread tagged with metadata.lf_session = session_id."""
    with psycopg.connect(conn_str) as conn:
        row = conn.execute(
            """
            SELECT thread_id, metadata, status, created_at, updated_at
            FROM threads
            WHERE metadata->>'lf_session' = %s
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    return _thread_row(row) if row else None


def thread_update_status(thread_id: str, status: str, conn_str: str = DATABASE_URL) -> None:
    with psycopg.connect(conn_str) as conn:
        conn.execute(
            "UPDATE threads SET status = %s, updated_at = NOW() WHERE thread_id = %s",
            (status, thread_id),
        )
        conn.commit()


def _thread_row(row: tuple) -> dict:
    return {
        "thread_id":  row[0],
        "metadata":   row[1],
        "status":     row[2],
        "created_at": row[3].isoformat(),
        "updated_at": row[4].isoformat(),
    }


# ── Runs ──────────────────────────────────────────────────────────────────────

def run_create(
    run_id: str,
    thread_id: str,
    assistant_id: str,
    conn_str: str = DATABASE_URL,
) -> dict:
    with psycopg.connect(conn_str) as conn:
        row = conn.execute(
            """
            INSERT INTO runs (run_id, thread_id, assistant_id)
            VALUES (%s, %s, %s)
            RETURNING run_id, thread_id, assistant_id, status, created_at, updated_at
            """,
            (run_id, thread_id, assistant_id),
        ).fetchone()
        conn.commit()
    return _run_row(row)


def run_update_status(run_id: str, status: str, conn_str: str = DATABASE_URL) -> None:
    with psycopg.connect(conn_str) as conn:
        conn.execute(
            "UPDATE runs SET status = %s, updated_at = NOW() WHERE run_id = %s",
            (status, run_id),
        )
        conn.commit()


def runs_list(thread_id: str, conn_str: str = DATABASE_URL) -> list[dict]:
    with psycopg.connect(conn_str) as conn:
        rows = conn.execute(
            "SELECT run_id, thread_id, assistant_id, status, created_at, updated_at FROM runs WHERE thread_id = %s ORDER BY created_at DESC",
            (thread_id,),
        ).fetchall()
    return [_run_row(r) for r in rows]


def _run_row(row: tuple) -> dict:
    return {
        "run_id":       row[0],
        "thread_id":    row[1],
        "assistant_id": row[2],
        "status":       row[3],
        "created_at":   row[4].isoformat(),
        "updated_at":   row[5].isoformat(),
    }
