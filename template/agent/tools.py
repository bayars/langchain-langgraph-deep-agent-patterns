"""
LangGraph tools for the Deep Agent.

VFS tools persist to PostgreSQL — no in-memory state.
bash_execute materializes VFS files to a temporary directory, runs the command,
then syncs any new or modified files back to the database.
"""

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import interrupt

from agent.db import (
    DATABASE_URL,
    todos_write,
    vfs_get_all,
    vfs_list,
    vfs_read,
    vfs_write,
)


def _tid(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("thread_id", "default")


# ── File system ───────────────────────────────────────────────────────────────

@tool
def write_file(path: str, content: str, config: RunnableConfig) -> str:
    """Write content to a virtual file stored in the database."""
    vfs_write(_tid(config), path, content)
    return f"Written {len(content)} bytes to '{path}'"


@tool
def read_file(path: str, config: RunnableConfig) -> str:
    """Read a virtual file from the database."""
    content = vfs_read(_tid(config), path)
    if content is None:
        return f"File not found: '{path}'"
    return content


@tool
def list_files(config: RunnableConfig) -> str:
    """List all files in the virtual file system for this thread."""
    paths = vfs_list(_tid(config))
    return "\n".join(paths) if paths else "No files yet."


# ── Bash execution ────────────────────────────────────────────────────────────

@tool
def bash_execute(command: str, config: RunnableConfig) -> str:
    """
    Execute a shell command in a temporary working directory pre-populated
    with all VFS files for this thread.

    After execution, any files that were created or modified inside the working
    directory are automatically written back to the VFS database.

    Example workflow:
      1. write_file("analysis.py", <code>)
      2. bash_execute("python analysis.py")   ← runs the file, captures output
      3. read_file("output.txt")              ← read any file the script produced
    """
    tid = _tid(config)
    vfs_files = vfs_get_all(tid)

    with tempfile.TemporaryDirectory(prefix=f"agent_vfs_{tid[:8]}_") as workdir:
        workdir_path = Path(workdir)

        # Materialize VFS files into the temp directory
        for rel_path, content in vfs_files.items():
            dest = workdir_path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Run via sh -c so shell operators (>, |, &&, ;) work naturally.
        try:
            result = subprocess.run(
                ["sh", "-c", command],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            return "Error: sh not found — cannot execute command"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 30 seconds"

        # Strip the temp path from output so the LLM never sees ephemeral paths
        output = (result.stdout + result.stderr).replace(workdir, ".").strip()

        # Sync modified or new files back to VFS
        for file_path in workdir_path.rglob("*"):
            if not file_path.is_file():
                continue
            rel = str(file_path.relative_to(workdir_path))
            new_content = file_path.read_text(encoding="utf-8", errors="replace")
            if new_content != vfs_files.get(rel):
                vfs_write(tid, rel, new_content)

    return output[:4000] if output else "(no output)"


# ── Planning ──────────────────────────────────────────────────────────────────

@tool
def write_todos(todos: list[str], config: RunnableConfig) -> str:
    """Publish a task plan. Todos are visible to reviewers in real time."""
    todos_write(_tid(config), todos)
    return f"Published {len(todos)} todos"


# ── Human-in-the-loop ─────────────────────────────────────────────────────────

@tool
def request_options(question: str, options: list[str]) -> str:
    """
    Pause execution and ask the human to choose an option.
    The agent resumes automatically after the human responds.
    """
    choice = interrupt({"question": question, "options": options})
    return f"Human chose: {choice!r}"
