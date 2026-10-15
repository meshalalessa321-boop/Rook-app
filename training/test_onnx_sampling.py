import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL = Path("models/rook-step-105500.onnx")
assert MODEL.is_file() and MODEL.stat().st_size > 50_000_000
session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
ctx, pad, bos, eos, vocab = 256, 256, 257, 258, 259
prompt = [bos] + list("User: What is your name?\nAssistant:".encode("utf-8"))
prompt = prompt[-ctx:]
tokens = np.full((1, ctx), pad, dtype=np.int64)
tokens[0, -len(prompt):] = prompt
rng = np.random.default_rng(2026)
generated = []
for step in range(48):
    logits = session.run(["logits"], {"tokens": tokens})[0][0, -1].astype(np.float64)
    logits[pad] = -1e9
    logits[bos] = -1e9
    if step < 18:
        logits[eos] = -1e9
    top = np.argpartition(logits, -48)[-48:]
    vals = logits[top]
    probs = np.exp((vals - vals.max()) / 0.9)
    probs /= probs.sum()
    nxt = int(rng.choice(top, p=probs))
    if nxt == eos:
        break
    if not 0 <= nxt <= 255:
        break
    generated.append(nxt)
    seq = (prompt + generated)[-ctx:]
    tokens.fill(pad)
    tokens[0, -len(seq):] = seq
text = bytes(generated).decode("utf-8", errors="replace")
text = "".join(ch for ch in text if ch in "\\n\\t" or ord(ch) >= 32).strip()
print("PROMPT: What is your name?")
print("GENERATED TOKENS:", len(generated))
print("MODEL REPLY SAMPLE:", repr(text))
assert len(generated) >= 18, "Rook stopped before the minimum generation length"
assert text, "Rook generated no readable text"
