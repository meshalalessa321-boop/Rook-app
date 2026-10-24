#!/usr/bin/env python3
"""Train a small, from-scratch byte-pair tokenizer for Rook.

This learns merges from Rook's documented local training corpus. It does not
load pretrained weights or call an external inference API. Special token IDs
are reserved after the learned vocabulary.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
MANIFEST = CORPUS / "manifest.jsonl"
DEFAULT_OUT = ROOT / "artifacts" / "rook-brain" / "tokenizer.json"


def read_corpus():
    if not MANIFEST.exists():
        raise SystemExit(f"Missing corpus manifest: {MANIFEST}")
    texts = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        path = (CORPUS / row["path"]).resolve()
        if CORPUS.resolve() not in path.parents or not path.is_file():
            raise SystemExit(f"Unsafe or missing corpus path: {row.get('path')}")
        texts.append(path.read_text(encoding="utf-8").strip())
    if len(texts) < 3:
        raise SystemExit("Need at least three corpus documents to train a tokenizer.")
    return texts


def train_bpe(texts, target_vocab=2048, min_pair_frequency=2):
    # Start from bytes, so every UTF-8 string remains representable.
    sequences = [list(text.encode("utf-8")) for text in texts]
    merges = []
    vocab_size = 256
    while vocab_size < target_vocab:
        pairs = Counter()
        for seq in sequences:
            pairs.update(zip(seq, seq[1:]))
        if not pairs:
            break
        pair, count = pairs.most_common(1)[0]
        if count < min_pair_frequency:
            break
        new_id = 256 + len(merges)
        merged = []
        for seq in sequences:
            out = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            merged.append(out)
        sequences = merged
        merges.append([int(pair[0]), int(pair[1])])
        vocab_size += 1
    # IDs: bytes 0..255, learned merges 256.., then PAD/BOS/EOS.
    return {
        "format": "rook-byte-pair-v1",
        "base": "utf-8 bytes",
        "byte_vocab_size": 256,
        "merges": merges,
        "vocab_size": 256 + len(merges) + 3,
        "pad_id": 256 + len(merges),
        "bos_id": 257 + len(merges),
        "eos_id": 258 + len(merges),
        "target_vocab_size": target_vocab,
        "min_pair_frequency": min_pair_frequency,
        "training_documents": len(texts),
        "pretrained_tokenizer": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--min-pair-frequency", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.vocab_size < 259:
        parser.error("--vocab-size must be at least 259")
    tokenizer = train_bpe(read_corpus(), args.vocab_size, args.min_pair_frequency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tokenizer, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "rook_tokenizer_trained", "output": str(args.output),
                      "vocab_size": tokenizer["vocab_size"],
                      "learned_merges": len(tokenizer["merges"]),
                      "documents": tokenizer["training_documents"],
                      "pretrained_tokenizer": False}), flush=True)


if __name__ == "__main__":
    main()
