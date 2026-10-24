#!/usr/bin/env python3
"""Scheduled engineering partner for Rook. Produces evidence-based work reports."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(".")
REPORT = Path("rook-engineer-report.md")
MODEL = os.environ.get("ROOK_ENGINEER_MODEL", "gpt-6.1-sol")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MAX_FILES = 45
MAX_FILE_CHARS = 9000
MAX_TOTAL_CHARS = 80000
SKIP_PARTS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "data",
    "checkpoints", "weights", "dist", "build", ".next"
}
ALLOWED_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".json", ".txt", ".sh"}


def collect_source() -> str:
    parts: list[str] = []
    total = 0
    files = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in ALLOWED_SUFFIXES
        and not any(part in SKIP_PARTS for part in p.parts)
        and p.name not in {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
        and not p.name.startswith(".env")
    )
    for path in files[:MAX_FILES]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_CHARS]
        except OSError:
            continue
        block = f"\n\n===== FILE: {path.as_posix()} =====\n{content}"
        if total + len(block) > MAX_TOTAL_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts) or "(No eligible source files found.)"


def ask_engineer(source: str) -> str:
    system = """You are the autonomous engineering partner for Rook, a small language model being trained from random initialization. Your job is to make concrete, evidence-based progress toward coherent language generation while preserving Rook's own model weights and training-from-scratch requirement. Do not recommend swapping in a pretrained model as Rook's brain. You may recommend using a hosted model as a separate engineering assistant only. Inspect the supplied repository excerpts. Distinguish facts from hypotheses; cite exact file paths and relevant symbols. Prioritize one next engineering task that can be performed safely and tested. Do not claim to have executed code, inspected files not shown, deployed anything, or changed the repository. Return a concise Markdown report with: status, strongest evidence, biggest blocker, one highest-value next task, concrete implementation steps, and tests that would prove success. If repository excerpts are insufficient, say exactly what needs to be inspected next."""
    payload = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Review the repository source below and produce the next engineering work report.\n\n" + source}
        ],
        "max_output_tokens": 3500
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"OpenAI API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach OpenAI API: {exc}") from exc

    texts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))
    result = "\n".join(texts).strip()
    if not result:
        raise RuntimeError("The API returned no output_text. Check model access and response schema.")
    return result


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    if not API_KEY:
        print("OPENAI_API_KEY is not configured. Add it as a repository Actions secret; no API call was made.", file=sys.stderr)
        return 2
    source = collect_source()
    report = f"# Rook Engineering Report\n\nGenerated: {now}\nModel: {MODEL}\n\n"
    report += ask_engineer(source)
    REPORT.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
