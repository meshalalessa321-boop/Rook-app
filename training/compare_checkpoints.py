#!/usr/bin/env python3
"""Score a checkpoint: loss on the same held-out text, plus repeated chat tests.

Read-only. Used to compare two training settings; it never trains or saves a model.
Reply generation mirrors rook-talk.html (temperature 0.72, top 32, last 4 messages).
"""
import argparse, json, random, re
from pathlib import Path

import torch

from rook_brain import load_documents, split_docs, examples, evaluate
from rook_tokenizer import ByteBPE
from train_rook_next import Rook

TURN = re.compile(r"\n\s*Person [AB]:")
OPENERS = ["Hi", "How are you today?", "What did you do this weekend?", "Do you like music?", "I'm a little tired."]


@torch.no_grad()
def reply(model, tok, ctx, history, current):
    prompt = "".join(f"Person {s}: {t}\n" for s, t in history[-4:]) + f"Person A: {current}\nPerson B:"
    base, out = [tok.bos] + tok.encode(prompt), []
    limit = 256 + tok.n
    allowed = torch.zeros(model.vocab, dtype=torch.bool)
    for i in range(limit):
        allowed[i] = all(b == 10 or 32 <= b < 127 for b in tok.decode([i]).encode("utf-8"))
    allowed[tok.eos] = True
    for step in range(90):
        seq = (base + out)[-ctx:]
        x = torch.full((1, ctx), tok.pad, dtype=torch.long)
        x[0, :len(seq)] = torch.tensor(seq)
        row = model(x)[0, len(seq) - 1].clone()
        row[~allowed] = float("-inf")
        if step < 3:
            row[tok.eos] = float("-inf")
        top, ids = torch.topk(row, 32)
        nxt = int(ids[torch.multinomial(torch.softmax(top / 0.72, 0), 1)])
        if nxt == tok.eos:
            break
        out.append(nxt)
        if TURN.search(tok.decode(out)):
            break
    return TURN.split(tok.decode(out))[0].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--trials", type=int, default=20)
    a = p.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg, tok = ck["config"], ByteBPE.from_json(ck["tokenizer"])
    model = Rook(cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"], vocab=cfg["vocab"])
    model.load_state_dict(ck["model"])
    model.eval()
    docs, _ = load_documents()
    _, valid_docs, test_docs = split_docs(docs, 1701)
    result = {"label": a.label, "step": int(ck["step"])}
    for name, part in (("validation", valid_docs), ("test", test_docs)):
        result[name] = evaluate(model, examples(part, cfg["ctx"], tok), "cpu", limit=1500, pad=tok.pad)

    hits, names = 0, []
    for trial in range(a.trials):
        torch.manual_seed(trial)
        history = []
        for message in ("Hi", "My name is Meshal", "What's my name?"):
            answer = reply(model, tok, cfg["ctx"], history, message)
            history += [("A", message), ("B", answer)]
        hits += bool(re.search(r"\bmeshal\b", history[-1][1], re.I))
        names.append(history)
    result["name_test"] = {"answered_meshal": hits, "trials": a.trials, "chats": names}
    result["openers"] = {}
    for text in OPENERS:
        answers = []
        for trial in range(5):
            torch.manual_seed(1000 + trial)
            answers.append(reply(model, tok, cfg["ctx"], [], text))
        result["openers"][text] = answers
    Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("COMPARE_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
