#!/usr/bin/env python3
"""Prepare a documented, open-licensed text corpus from WikiText-2.

Uses the original Wikipedia-derived WikiText-2 training split only. Articles
are kept as separate documents so rook_brain.py can split documents before
training. This is a bootstrap corpus, not a complete conversational corpus.
"""
import json
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
TEXTS = CORPUS / "texts"
MANIFEST = CORPUS / "manifest.jsonl"
DATASET_ID = "Salesforce/wikitext"
CONFIG = "wikitext-2-v1"
DATASET_URL = "https://huggingface.co/datasets/Salesforce/wikitext"
LICENSE = "CC BY-SA 3.0"
HEADING = re.compile(r"^= ([^=].*?) =$")


def main():
    print(f"Loading {DATASET_ID} / {CONFIG} train split...", flush=True)
    print("Source license: CC BY-SA 3.0 (Wikipedia-derived text); confirm the dataset card before redistributing source text.", flush=True)
    dataset = load_dataset(DATASET_ID, CONFIG, split="train")
    TEXTS.mkdir(parents=True, exist_ok=True)
    for old in TEXTS.glob("*.txt"):
        old.unlink()

    documents = []
    title = "opening"
    lines = []

    def flush():
        nonlocal title, lines
        body = "\n".join(lines).strip()
        if len(body.encode("utf-8")) >= 300:
            documents.append((title, body))
        lines = []

    for row in dataset:
        line = str(row.get("text", "")).rstrip()
        match = HEADING.match(line)
        if match:
            flush()
            title = match.group(1).strip()
        elif line.strip():
            lines.append(line.strip())
        elif lines and lines[-1] != "":
            lines.append("")
    flush()

    manifest_rows = []
    for i, (doc_title, body) in enumerate(documents):
        relpath = f"texts/article-{i:06d}.txt"
        (CORPUS / relpath).write_text(body, encoding="utf-8")
        manifest_rows.append({
            "path": relpath,
            "source": f"WikiText-2 / Wikipedia-derived article: {doc_title}",
            "license": LICENSE,
            "url": DATASET_URL,
            "language": "en",
        })

    if len(manifest_rows) < 3:
        raise SystemExit(f"Expected at least 3 article documents, got {len(manifest_rows)}.")
    MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "dataset": DATASET_ID,
        "config": CONFIG,
        "source_split": "train",
        "documents": len(manifest_rows),
        "manifest": str(MANIFEST),
        "note": "Bootstrap text corpus only; not a complete conversational corpus.",
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
