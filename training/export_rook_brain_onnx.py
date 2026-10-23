#!/usr/bin/env python3
"""Export the current random-initialized Rook Brain checkpoint for browser inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from train_rook_next import PAD, BOS, EOS, V, Rook

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "training" / "artifacts" / "rook-brain" / "best.pt"
MODEL_OUT = REPO / "models" / "rook-brain.onnx"
MANIFEST_OUT = REPO / "rook-status" / "brain.json"
CTX = 256


def main():
    if not CHECKPOINT.is_file():
        raise SystemExit(f"Missing trained Rook Brain checkpoint: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    if cfg.get("vocab") != V or cfg.get("ctx") != CTX:
        raise SystemExit(f"Unexpected Rook Brain config: {cfg}")
    if cfg.get("position_scheme") != "prefix-right-pad-v2":
        raise SystemExit("Refusing to export incompatible token-position scheme.")
    if checkpoint.get("objective") != "full-sequence next-token prediction":
        raise SystemExit("Refusing to export a checkpoint without verified full-sequence objective metadata.")
    if not isinstance(checkpoint.get("model"), dict):
        raise SystemExit("Checkpoint has no model state_dict.")

    model = Rook(cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.full((1, CTX), PAD, dtype=torch.long)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.onnx.export(
        model, (dummy,), str(MODEL_OUT),
        input_names=["tokens"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    onnx.checker.check_model(str(MODEL_OUT))

    session = ort.InferenceSession(str(MODEL_OUT), providers=["CPUExecutionProvider"])
    probe = np.full((1, CTX), PAD, dtype=np.int64)
    logits = session.run(["logits"], {"tokens": probe})[0]
    expected_shape = (1, CTX, V)
    if logits.shape != expected_shape or not np.isfinite(logits).all():
        raise RuntimeError(f"ONNX smoke test failed: shape={logits.shape}")

    # Exercise real autoregressive inference using the same prefix/right-padding
    # position scheme as training. This test validates execution, not quality.
    prompt = [BOS] + list("User: Hi.\nAssistant:".encode("utf-8"))
    generated = []
    for _ in range(24):
        seq = (prompt + generated)[-CTX:]
        tokens = np.full((1, CTX), PAD, dtype=np.int64)
        tokens[0, :len(seq)] = seq
        scores = session.run(["logits"], {"tokens": tokens})[0][0, len(seq) - 1]
        scores[PAD] = -1e9
        scores[BOS] = -1e9
        nxt = int(np.argmax(scores))
        if nxt == EOS:
            break
        if not 0 <= nxt <= 255:
            raise RuntimeError(f"Unexpected generated token id: {nxt}")
        generated.append(nxt)
    sample = bytes(generated).decode("utf-8", errors="replace")
    digest = hashlib.sha256(MODEL_OUT.read_bytes()).hexdigest()
    manifest = {
        "status": "verified",
        "model": "./models/rook-brain.onnx",
        "format": "ONNX",
        "source_checkpoint": "training/artifacts/rook-brain/best.pt",
        "training_step": int(checkpoint["step"]),
        "validation_loss": float(checkpoint["val_loss"]),
        "objective": checkpoint["objective"],
        "config": cfg,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "pretrained_weights": False,
        "external_inference_api": False,
        "position_scheme": "prefix-right-pad-v2",
        "input": {"name": "tokens", "dtype": "int64", "shape": [1, CTX]},
        "output": {"name": "logits", "dtype": "float32", "shape": [1, CTX, V]},
        "model_bytes": MODEL_OUT.stat().st_size,
        "model_sha256": digest,
        "tensor_smoke_test": "passed",
        "autoregressive_smoke_test": "passed",
        "sample_for_test_prompt": sample,
        "quality_note": "Successful export and inference do not imply coherent conversation; this is an early-stage language-modeling checkpoint.",
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
