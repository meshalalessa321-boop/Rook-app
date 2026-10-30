#!/usr/bin/env python3
"""Add everyday human-to-human dialogue (DailyDialog) to the documented corpus.

Why: nearly all of Rook's text is stories, so a greeting gets a story back. Real
two-person conversation teaches what comes after "hello" without casting him in
any role: both speakers are people, and nobody is the helper or the one helped.

Rules this script follows:
  * Speakers are labelled "Person A" / "Person B" in alternation. No assistant,
    user, system or identity text is added, and no instruction is given.
  * Both speakers' lines are ordinary next-token targets (full-sequence training).
  * Dialogues are packed into shards so rook_brain.py splits whole shards into
    train/validation/test; no dialogue is shared between splits.
  * The manifest row carries a `weight` (how many times the training split repeats
    these shards). Validation and test are never repeated.

License: DailyDialog is CC BY-NC-SA 4.0 (non-commercial, share-alike). That is fine
for a personal project; check it before using Rook commercially or redistributing.

Run after prepare_wikitext.py (which rewrites the manifest) and prepare_tinystories.py.
"""
import argparse
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
OUT = CORPUS / "dialogue"
MANIFEST = CORPUS / "manifest.jsonl"
DATASET_ID = "li2017dailydialog/daily_dialog"
DATASET_URL = "https://huggingface.co/datasets/li2017dailydialog/daily_dialog"
ORIGINAL_ZIP = "http://yanran.li/files/ijcnlp_dailydialog.zip"
LICENSE = "CC BY-NC-SA 4.0"
SOURCE = f"{DATASET_ID} dialogue shard"
SHARD_DIALOGUES = 20
WEIGHT = 3
LABELS = ("Person A", "Person B")

_APOS = re.compile(r"(\w)\s+['’‘]\s*(\w)")
_APOS_TAIL = re.compile(r"\s+['’‘]\s*(s|m|re|ve|ll|d|t)\b")
_SPACE_BEFORE = re.compile(r"\s+([.,!?;:%])")
_OPEN_PAREN = re.compile(r"([(\[])\s+")
_CLOSE_PAREN = re.compile(r"\s+([)\]])")


def clean_utterance(text):
    """Undo DailyDialog's tokenized spacing: "don ' t go ?" -> "don't go?"."""
    text = " ".join(str(text or "").split())
    text = _APOS_TAIL.sub(r"'\1", text)
    text = _APOS.sub(r"\1'\2", text)
    text = _SPACE_BEFORE.sub(r"\1", text)
    text = _OPEN_PAREN.sub(r"\1", text)
    text = _CLOSE_PAREN.sub(r"\1", text)
    return text.replace("’", "'").replace("‘", "'").strip()


def format_dialogue(utterances):
    """Alternate Person A / Person B. Returns None for dialogues too short to be a conversation."""
    lines = []
    for utterance in utterances:
        text = clean_utterance(utterance)
        if text:
            lines.append(f"{LABELS[len(lines) % 2]}: {text}")
    return "\n".join(lines) if len(lines) >= 2 else None


def shard_dialogues(dialogues, shard_size=SHARD_DIALOGUES):
    """Pack formatted dialogues into shard texts separated by blank lines. Deterministic."""
    shards, current = [], []
    for dialogue in dialogues:
        if not dialogue:
            continue
        current.append(dialogue)
        if len(current) == shard_size:
            shards.append("\n\n".join(current))
            current = []
    if len(current) >= 3:
        shards.append("\n\n".join(current))
    return shards


def _from_hub(limit):
    from datasets import load_dataset
    attempts = [
        lambda split: load_dataset(DATASET_ID, split=split),
        lambda split: load_dataset("daily_dialog", split=split),
        lambda split: load_dataset(DATASET_ID, split=split, revision="refs/convert/parquet"),
    ]
    last = None
    for attempt in attempts:
        try:
            dialogues = []
            for split in ("train", "validation", "test"):
                for row in attempt(split):
                    dialogues.append(list(row["dialog"]))
                    if limit and len(dialogues) >= limit:
                        return dialogues
            return dialogues
        except Exception as exc:  # try the next way of loading it
            last = exc
            print(f"Hub load attempt failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    raise RuntimeError(f"Could not load DailyDialog from the Hub: {last}")


def _from_original_zip(limit):
    print(f"Downloading the original DailyDialog zip from {ORIGINAL_ZIP} ...", flush=True)
    with urllib.request.urlopen(ORIGINAL_ZIP, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    names = [n for n in archive.namelist() if n.endswith("dialogues_text.txt")]
    if not names:
        raise RuntimeError("dialogues_text.txt not found in the original archive")
    dialogues = []
    for line in archive.read(names[0]).decode("utf-8").splitlines():
        parts = [p for p in line.split("__eou__") if p.strip()]
        if parts:
            dialogues.append(parts)
        if limit and len(dialogues) >= limit:
            break
    return dialogues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-dialogues", type=int, default=0, help="0 = all (about 13,000)")
    parser.add_argument("--min-dialogues", type=int, default=2000, help="Fail if fewer usable dialogues are found")
    args = parser.parse_args()

    print(f"Loading DailyDialog. License: {LICENSE} (non-commercial, share-alike).", flush=True)
    try:
        raw = _from_hub(args.max_dialogues)
    except Exception as exc:
        print(f"{exc}\nTrying the original download instead.", flush=True)
        raw = _from_original_zip(args.max_dialogues)
    dialogues = [format_dialogue(u) for u in raw]
    dialogues = [d for d in dialogues if d]
    if len(dialogues) < args.min_dialogues:
        raise SystemExit(f"Expected at least {args.min_dialogues} dialogues, got {len(dialogues)}. Check the dataset schema.")
    shards = shard_dialogues(dialogues)
    if len(shards) < 5:
        raise SystemExit(f"Expected at least 5 dialogue shards, got {len(shards)}.")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.txt"):
        old.unlink()
    existing = []
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("source") != SOURCE:
                    existing.append(row)
    rows, total_bytes = [], 0
    for index, text in enumerate(shards):
        relpath = f"dialogue/dailydialog-{index:05d}.txt"
        (CORPUS / relpath).write_text(text, encoding="utf-8")
        total_bytes += len(text.encode("utf-8"))
        rows.append({"path": relpath, "source": SOURCE, "license": LICENSE, "url": DATASET_URL,
                     "language": "en", "weight": WEIGHT})
    MANIFEST.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in existing + rows), encoding="utf-8")
    print(json.dumps({"dataset": DATASET_ID, "license": LICENSE, "dialogues": len(dialogues), "shards": len(rows),
                      "bytes": total_bytes, "training_weight": WEIGHT, "speaker_labels": list(LABELS),
                      "objective": "full-sequence next-token prediction", "pretrained_weights": False}), flush=True)
    print("Sample:\n" + shards[0][:400], flush=True)


if __name__ == "__main__":
    main()
