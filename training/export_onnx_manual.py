#!/usr/bin/env python3
"""Export Rook using an explicit attention implementation supported by ONNX."""
import hashlib, json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import torch.nn.functional as F
from train_rook_next import PAD, Rook

REPO = Path(__file__).resolve().parents[1]
checkpoint = torch.load(REPO / "artifacts/rook-next/best.pt", map_location="cpu", weights_only=False)
cfg = checkpoint["config"]
assert cfg["vocab"] == 259 and cfg["ctx"] == 256
base = Rook(cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"])
base.load_state_dict(checkpoint["model"])
base.eval()

class ExportRook(nn.Module):
    def __init__(self, model, heads):
        super().__init__()
        self.model = model
        self.heads = heads
        self.head_dim = model.t.embedding_dim // heads

    def forward(self, tokens):
        x = tokens[:, -self.model.c:]
        positions = torch.arange(x.size(1), device=x.device)
        x = self.model.t(x) + self.model.p(positions)[None]
        for block in self.model.b:
            z = block.n1(x)
            qkv = F.linear(z, block.a.in_proj_weight, block.a.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            bsz, seq, width = q.shape
            q = q.reshape(bsz, seq, self.heads, self.head_dim).transpose(1, 2)
            k = k.reshape(bsz, seq, self.heads, self.head_dim).transpose(1, 2)
            v = v.reshape(bsz, seq, self.heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            causal = torch.triu(torch.ones((seq, seq), dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(causal[None, None], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            attended = torch.matmul(probs, v).transpose(1, 2).contiguous().reshape(bsz, seq, width)
            attended = F.linear(attended, block.a.out_proj.weight, block.a.out_proj.bias)
            x = x + attended
            x = x + block.f(block.n2(x))
        return self.model.o(self.model.n(x))

wrapped = ExportRook(base, cfg["heads"]).eval()
for p in wrapped.parameters():
    p.requires_grad_(False)
dummy = torch.full((1, cfg["ctx"]), PAD, dtype=torch.long)
with torch.no_grad():
    expected = base(dummy)
    actual = wrapped(dummy)
    max_diff = float((expected - actual).abs().max())
    assert torch.allclose(expected, actual, atol=2e-4, rtol=2e-4), f"manual attention mismatch: {max_diff}"
out_dir = REPO / "artifacts/rook-onnx"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "model.onnx"
with torch.no_grad():
    torch.onnx.export(
        wrapped, (dummy,), str(out),
        input_names=["tokens"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
onnx.checker.check_model(str(out))
session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
probe = np.full((1, cfg["ctx"]), PAD, dtype=np.int64)
logits = session.run(["logits"], {"tokens": probe})[0]
assert logits.shape == (1, cfg["ctx"], cfg["vocab"]), logits.shape
assert np.isfinite(logits).all()
prompt = [257] + list("User: Hi.\nAssistant:".encode("utf-8"))
prompt = prompt[-cfg["ctx"]:]
tokens = np.full((1, cfg["ctx"]), PAD, dtype=np.int64)
tokens[0, -len(prompt):] = prompt
generated = []
for _ in range(32):
    scores = session.run(["logits"], {"tokens": tokens})[0][0, -1].copy()
    scores[PAD] = -1e9
    scores[257] = -1e9
    nxt = int(np.argmax(scores))
    if nxt == 258:
        break
    if not 0 <= nxt <= 255:
        break
    generated.append(nxt)
    seq = (prompt + generated)[-cfg["ctx"]:]
    tokens.fill(PAD)
    tokens[0, -len(seq):] = seq
sample = bytes(generated).decode("utf-8", errors="replace")
manifest = {
    "format": "ONNX", "source_checkpoint": "best.pt",
    "training_step": int(checkpoint["step"]),
    "validation_loss": float(checkpoint["val_loss"]),
    "config": cfg,
    "parameters": int(sum(p.numel() for p in base.parameters())),
    "pretrained_weights": False, "external_inference_api": False,
    "input": {"name": "tokens", "dtype": "int64", "shape": [1, cfg["ctx"]]},
    "output": {"name": "logits", "dtype": "float32", "shape": [1, cfg["ctx"], cfg["vocab"]]},
    "model_bytes": out.stat().st_size,
    "model_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "attention_equivalence_max_abs_diff": max_diff,
    "smoke_test": "passed",
    "greedy_sample_for_prompt_User_Hi": sample,
}
(out_dir / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
