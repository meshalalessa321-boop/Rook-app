#!/usr/bin/env python3
"""Add TinyStories to the documented corpus for full-sequence pretraining.

TinyStories is simple, clean English written for very small language models, which
is why it is a good fit for a ~15M-parameter model. Stories are packed into shards
(many stories per .txt file) so the corpus stays a few hundred files instead of
hundreds of thousands. rook_brain.py splits whole shards into train/validation/test,
so no story is shared across splits.

Run order: prepare_wikitext.py first (it rewrites the manifest), then this script,
then prepare_conversations.py. Existing rows from other sources are preserved.

Adding data changes the corpus fingerprint, so start the next training run with
`python training/rook_brain.py --fresh ...` as rook_brain.py instructs.

Usage:
    python training/prepare_tinystories.py                  # 50,000 stories
    python training/prepare_tinystories.py --max-stories 300000
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
OUT = CORPUS / "stories"
MANIFEST = CORPUS / "manifest.jsonl"
DATASET_ID = "roneneldan/TinyStories"
DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
# Confirm this against the dataset card before redistributing the text.
LICENSE = "CDLA-Sharing-1.0"
SOURCE = f"{DATASET_ID} story shard"
SHARD_STORIES = 100


def shard_stories(stories, shard_size=SHARD_STORIES):
    """Group stories into shard texts, separated by blank lines. Deterministic."""
    shards, current = [], []
    for story in stories:
        text = str(story or "").strip()
        if len(text.encode("utf-8")) < 50:
            continue
        current.append(text)
        if len(current) == shard_size:
            shards.append("\n\n".join(current))
            current = []
    if len(current) >= 3:
        shards.append("\n\n".join(current))
    return shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stories", type=int, default=50000)
    args = parser.parse_args()

    print(f"Loading the first {args.max_stories} stories of {DATASET_ID} train split...", flush=True)
    print(f"Source license: {LICENSE}. Confirm on the dataset card before redistributing.", flush=True)
    # Regular (non-streaming) load: streaming left background threads running and
    # crashed the interpreter on exit (exit code 134) in GitHub Actions.
    dataset = load_dataset(DATASET_ID, split=f"train[:{args.max_stories}]")

    shards = shard_stories(row.get("text", "") for row in dataset)
    if len(shards) < 10:
        raise SystemExit(f"Expected at least 10 story shards, got {len(shards)}. Check the dataset schema.")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.txt"):
        old.unlink()

    existing = []
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != SOURCE:
                existing.append(row)

    rows, total_bytes = [], 0
    for index, text in enumerate(shards):
        relpath = f"stories/tinystories-{index:05d}.txt"
        (CORPUS / relpath).write_text(text, encoding="utf-8")
        total_bytes += len(text.encode("utf-8"))
        rows.append({"path": relpath, "source": SOURCE, "license": LICENSE,
                     "url": DATASET_URL, "language": "en"})

    MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in existing + rows),
        encoding="utf-8",
    )
    print(json.dumps({"dataset": DATASET_ID, "license": LICENSE, "shards": len(rows),
                      "stories": sum(shard.count("\n\n") + 1 for shard in shards),
                      "bytes": total_bytes, "objective": "full-sequence next-token prediction",
                      "pretrained_weights": False}), flush=True)


if __name__ == "__main__":
    main()
