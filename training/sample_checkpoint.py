#!/usr/bin/env python3
"""Print a few sample replies from a Rook checkpoint, for comparing training runs.

Prompts use neutral "Person A:" / "Person B:" labels only. Nothing is added to the
text: no identity, no instructions, no role. Generation stops at the next speaker label.
"""
import argparse
import json
import re
from pathlib import Path

import torch

from rook_tokenizer import ByteBPE, byte_tokenizer
from train_rook_next import Rook

PROMPTS = [
    "Hello!",
    "How are you today?",
    "What did you do this weekend?",
    "Do you like music?",
    "I'm a little tired.",
]
TURN = re.compile(r"\n?Person [AB]:")


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    tokenizer = ByteBPE.from_json(checkpoint["tokenizer"]) if checkpoint.get("tokenizer") else byte_tokenizer()
    model = Rook(cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"], vocab=tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, tokenizer, int(cfg["ctx"])


@torch.no_grad()
def reply(model, tokenizer, ctx, line, temperature=0.7, top_k=32, max_tokens=60, seed=0):
    gen = torch.Generator().manual_seed(seed)
    prompt = [tokenizer.bos] + tokenizer.encode(f"Person A: {line}\nPerson B:")
    out = []
    for _ in range(max_tokens):
        seq = (prompt + out)[-ctx:]
        tokens = torch.full((1, ctx), tokenizer.pad, dtype=torch.long)
        tokens[0, :len(seq)] = torch.tensor(seq)
        scores = model(tokens)[0, len(seq) - 1].clone()
        scores[tokenizer.pad] = scores[tokenizer.bos] = -1e9
        if temperature <= 0:
            nxt = int(scores.argmax())
        else:
            top = torch.topk(scores, min(top_k, scores.numel()))
            probs = torch.softmax(top.values / temperature, dim=-1)
            nxt = int(top.indices[torch.multinomial(probs, 1, generator=gen)])
        if nxt == tokenizer.eos:
            break
        out.append(nxt)
        if TURN.search(tokenizer.decode(out)):
            break
    text = TURN.split(tokenizer.decode(out))[0]
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default="")
    parser.add_argument("--max-tokens", type=int, default=60)
    args = parser.parse_args()
    model, tokenizer, ctx = load(Path(args.checkpoint))
    samples = [{"prompt": p, "reply": reply(model, tokenizer, ctx, p, max_tokens=args.max_tokens, seed=i)}
               for i, p in enumerate(PROMPTS)]
    for s in samples:
        print(f"Person A: {s['prompt']}\nPerson B: {s['reply']}\n")
    if args.out:
        Path(args.out).write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
