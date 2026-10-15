#!/usr/bin/env python3
"""Export Rook's verified trained checkpoint to ONNX and test real inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from train_rook_next import PAD, Rook

REPO = Path(__file__).resolve().parents[1]
checkpoint_path = REPO / "artifacts/rook-next/best.pt"
out_dir = REPO / "artifacts/rook-onnx"
out_dir.mkdir(parents=True, exist_ok=True)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
cfg = checkpoint["config"]
assert cfg["vocab"] == 259 and cfg["ctx"] == 256
assert cfg.get("position_scheme") == "prefix-right-pad-v2", "Refusing to export a checkpoint trained with incompatible token positions."
model = Rook(cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"])
model.load_state_dict(checkpoint["model"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

dummy = torch.full((1, cfg["ctx"]), PAD, dtype=torch.long)
out = out_dir / "model.onnx"
with torch.no_grad():
    torch.onnx.export(
        model, (dummy,), str(out),
        input_names=["tokens"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
onnx.checker.check_model(str(out))
session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
probe = np.full((1, cfg["ctx"]), PAD, dtype=np.int64)
logits = session.run(["logits"], {"tokens": probe})[0]
assert logits.shape == (1, cfg["ctx"], cfg["vocab"]), logits.shape
assert np.isfinite(logits).all()

# Test a real prompt and greedy token generation, not only tensor loading.
prompt = [257] + list("User: Hi.\nAssistant:".encode("utf-8"))
prompt = prompt[-cfg["ctx"]:]
tokens = np.full((1, cfg["ctx"]), PAD, dtype=np.int64)
tokens[0, :len(prompt)] = prompt
generated = []
for _ in range(24):
    seq = (prompt + generated)[-cfg["ctx"]:]
    tokens.fill(PAD)
    tokens[0, :len(seq)] = seq
    scores = session.run(["logits"], {"tokens": tokens})[0][0, len(seq) - 1]
    scores[PAD] = -1e9
    scores[257] = -1e9
    nxt = int(np.argmax(scores))
    if nxt == 258:
        break
    if 0 <= nxt <= 255:
        generated.append(nxt)
    else:
        break
sample = bytes(generated).decode("utf-8", errors="replace")
manifest = {
    "format": "ONNX",
    "source_checkpoint": "best.pt",
    "training_step": int(checkpoint["step"]),
    "validation_loss": float(checkpoint["val_loss"]),
    "config": cfg,
    "parameters": int(sum(p.numel() for p in model.parameters())),
    "pretrained_weights": False,
    "external_inference_api": False,
    "input": {"name": "tokens", "dtype": "int64", "shape": [1, cfg["ctx"]]},
    "output": {"name": "logits", "dtype": "float32", "shape": [1, cfg["ctx"], cfg["vocab"]]},
    "model_bytes": out.stat().st_size,
    "model_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "position_scheme": "prefix-right-pad-v2",
    "smoke_test": "passed",
    "greedy_sample_for_prompt_User_Hi": sample,
}
(out_dir / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
