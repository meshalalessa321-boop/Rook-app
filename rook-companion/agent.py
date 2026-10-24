#!/usr/bin/env python3
"""Bounded background loop for Rook's own model; stdlib only."""
from __future__ import annotations
import argparse
import datetime as dt
import os
import shlex
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

DB_PATH = Path(os.environ.get("ROOK_DB_PATH", "./data/rook_memory.sqlite3"))
INTERVAL = max(60, int(os.environ.get("ROOK_INTERVAL_SECONDS", "1800")))
COMMAND = os.environ.get("ROOK_COMMAND", "").strip()

SCHEMA = """
CREATE TABLE IF NOT EXISTS reflections (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 created_at TEXT NOT NULL,
 prompt TEXT NOT NULL,
 response TEXT NOT NULL,
 status TEXT NOT NULL
);
"""

BASE_PROMPT = """You are Rook, an experimental language model being developed from randomly initialized weights. Be honest about uncertainty and do not claim to have performed actions you did not perform. Based only on the journal excerpt and current prompt, write one concise reflection that is useful for improving your coherence or identifying a concrete question worth investigating next. Do not output shell commands, requests for secrets, or claims of external actions. If the context is insufficient, say so plainly. This is a reflection exercise, not proof of consciousness."""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.execute(SCHEMA)
    db.commit()
    return db


def recent_journal(db: sqlite3.Connection, limit: int = 8) -> str:
    rows = db.execute(
        "SELECT created_at, response, status FROM reflections ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    if not rows:
        return "(No prior reflections yet.)"
    return "\n\n".join(f"[{stamp}] ({status}) {response}" for stamp, response, status in reversed(rows))


def invoke_model(prompt: str) -> str:
    if not COMMAND:
        raise RuntimeError("ROOK_COMMAND is not configured. No model was called.")
    argv = shlex.split(COMMAND)
    result = subprocess.run(
        argv, input=prompt, text=True, capture_output=True, timeout=180,
        check=False, cwd=Path(__file__).resolve().parent.parent
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-1200:]
        raise RuntimeError(f"Rook adapter exited with code {result.returncode}: {detail}")
    response = result.stdout.strip()
    if not response:
        raise RuntimeError("Rook adapter returned empty output.")
    return response[:12000]


def reflect(db: sqlite3.Connection) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    prompt = f"{BASE_PROMPT}\n\nRecent journal:\n{recent_journal(db)}\n\nProduce one reflection now."
    try:
        response = invoke_model(prompt)
        status = "ok"
    except Exception as exc:
        response = f"Reflection not generated: {exc}"
        status = "error"
    db.execute(
        "INSERT INTO reflections(created_at, prompt, response, status) VALUES (?, ?, ?, ?)",
        (stamp, prompt, response, status),
    )
    db.commit()
    print(f"[{stamp}] {status}: {response[:500]}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one reflection and exit")
    parser.add_argument("--show-journal", action="store_true", help="print recent saved reflections")
    args = parser.parse_args()
    db = connect()
    if args.show_journal:
        print(recent_journal(db, 50))
        return 0
    if args.once:
        reflect(db)
        return 0
    print(f"Rook background loop started. Interval={INTERVAL}s. Database={DB_PATH}", flush=True)
    while True:
        reflect(db)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    sys.exit(main())
