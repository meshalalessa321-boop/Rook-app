#!/usr/bin/env python3
"""Export the verified from-scratch Rook checkpoint to ONNX for local browser inference."""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from train_rook_next import PAD, Rook

ROOT = Path("artifacts/rook-onnx")
ROOT.mkdir(parents=True, exist_ok=True)
checkpoint_path = Path("artifacts/rook-next/best.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
cfg = checkpoint["config"]
assert cfg["vocab"] == 259
assert cfg["ctx"] == 256
assert int(checkpoint["step"]) >= 1

model = Rook(cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"])
model.load_state_dict(checkpoint["model"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

dummy = torch.full((1, cfg["ctx"]), PAD, dtype=torch.long)
out = ROOT / "model.onnx"
with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        input_names=["tokens"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

onnx.checker.check_model(str(out))
session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
test = np.full((1, cfg["ctx"]), PAD, dtype=np.int64)
result = session.run(["logits"], {"tokens": test})[0]
assert result.shape == (1, cfg["ctx"], cfg["vocab"]), result.shape
assert np.isfinite(result).all()
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
    "smoke_test": "passed",
}
(ROOT / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
