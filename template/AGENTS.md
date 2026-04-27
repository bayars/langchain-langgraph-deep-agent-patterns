# Deep Agent Template

A LangGraph Deep Agent that demonstrates persistent VFS-backed file management,
multi-model skill delegation, bash execution, and human-in-the-loop interrupts.

## What this agent does

Given a task, this agent will:
1. Publish a plan (`write_todos`) before starting work
2. Store all file artifacts in a PostgreSQL-backed VFS — never in the context window
3. Delegate to specialized LLMs for analysis and code writing
4. Run code with `bash_execute` — VFS files are materialized to a temp dir for execution
5. Pause for human approval when needed (`request_options`)
6. Resume exactly where it left off after a human responds

## Available tools

| Tool | What it does |
|------|-------------|
| `write_file(path, content)` | Save a file to the VFS (persisted to PostgreSQL) |
| `read_file(path)` | Read a file from the VFS |
| `list_files()` | List all files in the current thread's VFS |
| `bash_execute(command)` | Run a shell command; VFS files are available in the working directory |
| `write_todos(todos)` | Publish a task plan visible to reviewers |
| `request_options(question, options)` | Pause and ask the human to choose |
| `analyze_data(data)` | Analyze data using the general LLM |
| `write_code(task, language)` | Write code using the code-specialized LLM |
| `search_knowledge(query)` | Search and synthesize an answer |

## Important usage notes

- All files persist across sessions via PostgreSQL — use `write_file` to store work
- Use `bash_execute` to run scripts you wrote with `write_file`; output is captured and returned
- Files written by a script during `bash_execute` are automatically saved back to VFS
- VFS paths are relative (e.g. `analysis.txt`, `src/main.py`)
- State (messages, todos, VFS) survives server restarts thanks to PostgreSQL checkpointing

## Example prompts

```
Write a Python script called analysis.py that generates a list of prime numbers
up to 100, then run it and save the output to primes.txt.
```

```
Analyze this sales data: Q1=120k, Q2=145k, Q3=98k, Q4=210k.
Save findings to report.txt and write a matplotlib chart script.
Ask me before running the chart.
```
