# Rook Persistent Agent (experimental)

This is an isolated background-agent layer for Rook. It does **not** replace Rook's randomly initialized model or its weights. The loop calls Rook through a command-line adapter you configure, stores each session in SQLite, and periodically asks Rook to produce one useful question or investigation seed from its own current context.

## Important distinction

A scheduler can run inference repeatedly; that is not evidence of consciousness or continuous subjective experience. This project provides persistence, timed execution, and a journal. It does not silently grant shell, browser, GitHub, or deployment permissions to model-generated text.

## Requirements

- Python 3.10+
- A working inference command for Rook's own model
- No third-party Python packages required

## Configure the model adapter

Set `ROOK_COMMAND` to a command that accepts a prompt on standard input and writes the model's response to standard output. Example for a project-specific adapter:

```bash
export ROOK_COMMAND="python rook_adapter.py"
export ROOK_INTERVAL_SECONDS=1800
export ROOK_DB_PATH=./data/rook_memory.sqlite3
python rook-companion/agent.py
```

`rook_adapter.py` is deliberately not invented here: it must be written to match the actual model/tokenizer/inference API in this repository. The agent will not fall back to a hosted model if the adapter is missing.

The adapter should load Rook's own checkpoint and return plain text. Keep model output treated as untrusted data. This agent stores it, but does not execute instructions contained in it.

## One-shot test

```bash
ROOK_COMMAND="python rook_adapter.py" python rook-companion/agent.py --once
```

## Journal

SQLite stores timestamps, prompts, responses, and a short status. Back up the database if you want to preserve its journal. Do not commit secrets or the database to Git.

## Current limitations

- No deployment has been performed by this repository change.
- No inference adapter is included until the actual Rook model interface is verified.
- No autonomous web browsing, code execution, messaging, or model-weight training is enabled.
- For truly persistent execution, run it on a machine/service that remains available. A scheduled job is not the same as a process that runs continuously.
