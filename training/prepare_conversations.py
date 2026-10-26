#!/usr/bin/env python3
"""Add a small, licensed English conversation mixture for full-sequence pretraining.

Both user and assistant messages remain ordinary next-token targets. This does
not perform answer-only fine-tuning and does not prescribe Rook's identity.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
OUT = CORPUS / "conversations"
MANIFEST = CORPUS / "manifest.jsonl"
DATASET_ID = "OpenAssistant/oasst1"
DATASET_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1"
LICENSE = "Apache-2.0"
MAX_DOCUMENTS = 2000
MAX_DOCUMENT_BYTES = 4000


def build_transcripts(rows, limit=MAX_DOCUMENTS):
    """Return deterministic, one-branch-per-root English conversation documents."""
    nodes = {}
    for row in rows:
        if row.get("deleted") or str(row.get("lang", "")).lower() != "en":
            continue
        role = row.get("role")
        text = str(row.get("text") or "").strip()
        node_id = str(row.get("message_id") or "").strip()
        if not node_id or not text or role not in ("prompter", "assistant"):
            continue
        nodes[node_id] = {
            "id": node_id,
            "parent": str(row.get("parent_id") or "").strip() or None,
            "role": "User" if role == "prompter" else "Assistant",
            "text": text,
        }

    children = {}
    for node_id, node in nodes.items():
        parent = node["parent"]
        if parent in nodes:
            children.setdefault(parent, []).append(node_id)
    for child_ids in children.values():
        child_ids.sort()

    @lru_cache(maxsize=None)
    def deepest_path(node_id):
        child_ids = children.get(node_id, [])
        if not child_ids:
            return (node_id,)
        paths = [deepest_path(child_id) for child_id in child_ids]
        return (node_id,) + max(paths, key=lambda path: (len(path), path))

    roots = [
        node_id for node_id, node in nodes.items()
        if node["role"] == "User" and node["parent"] not in nodes
    ]
    roots.sort(key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())

    documents = []
    for root in roots:
        path = deepest_path(root)
        if len(path) < 2:
            continue
        lines = []
        byte_count = 0
        for node_id in path:
            node = nodes[node_id]
            line = f'{node["role"]}: {node["text"]}'
            size = len(line.encode("utf-8")) + (1 if lines else 0)
            if byte_count + size > MAX_DOCUMENT_BYTES:
                break
            lines.append(line)
            byte_count += size
        if len(lines) < 2 or byte_count < 300:
            continue
        documents.append({
            "root_id": root,
            "text": "\n".join(lines),
        })
        if len(documents) >= limit:
            break
    return documents


def main():
    print(f"Loading {DATASET_ID} train split...", flush=True)
    print("Source license: Apache-2.0. Conversation turns are trained as full-sequence text.", flush=True)
    dataset = load_dataset(DATASET_ID, split="train")
    documents = build_transcripts(dataset)
    if len(documents) < 100:
        raise SystemExit(
            f"Expected at least 100 English conversation documents; found {len(documents)}. "
            "Check the dataset schema/language field before training."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.txt"):
        old.unlink()

    existing = []
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("source", "")).startswith(f"{DATASET_ID} conversation"):
                existing.append(row)

    rows = []
    total_bytes = 0
    for index, document in enumerate(documents):
        relpath = f"conversations/oasst-{index:05d}.txt"
        text = document["text"]
        (CORPUS / relpath).write_text(text, encoding="utf-8")
        total_bytes += len(text.encode("utf-8"))
        rows.append({
            "path": relpath,
            "source": f"{DATASET_ID} conversation transcript",
            "license": LICENSE,
            "url": DATASET_URL,
            "language": "en",
        })

    MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in existing + rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "dataset": DATASET_ID,
        "license": LICENSE,
        "language": "en",
        "documents": len(rows),
        "bytes": total_bytes,
        "objective": "full-sequence next-token prediction",
        "speaker_role_loss_mask": False,
        "pretrained_weights": False,
    }), flush=True)


if __name__ == "__main__":
    main()
