#!/usr/bin/env python3
"""Export the current random-initialized Rook Brain checkpoint for browser inference.

Writes the ONNX model, a brain.json description, and (for BPE models) tokenizer.json.
The generation self-test uses neutral speaker labels, never an assistant/user role.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from train_rook_next import Rook
from rook_tokenizer import ByteBPE, byte_tokenizer

REPO = Path(__file__).resolve().parents[1]
BASE_CHECKPOINT = REPO / "training" / "artifacts" / "rook-brain" / "best.pt"
MODEL_OUT = REPO / "models" / "rook-brain.onnx"
MANIFEST_OUT = REPO / "rook-status" / "brain.json"
TOKENIZER_OUT = REPO / "rook-status" / "tokenizer.json"
TEST_PROMPT = "Person A: Hi.\nPerson B:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(BASE_CHECKPOINT))
    parser.add_argument("--model-out", default=str(MODEL_OUT))
    parser.add_argument("--manifest-out", default=str(MANIFEST_OUT))
    parser.add_argument("--tokenizer-out", default=str(TOKENIZER_OUT))
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    model_out, manifest_out, tokenizer_out = Path(args.model_out), Path(args.manifest_out), Path(args.tokenizer_out)

    # Export only the full-sequence checkpoint. No answer-only fine-tuned variant exists.
    if not checkpoint_path.is_file():
        raise SystemExit(f"Missing Rook checkpoint: {checkpoint_path}. Run the training pipeline first.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    tokenizer = ByteBPE.from_json(checkpoint["tokenizer"]) if checkpoint.get("tokenizer") else byte_tokenizer()
    vocab, ctx = tokenizer.vocab_size, int(cfg.get("ctx", 0))
    if cfg.get("vocab") != vocab or ctx <= 0:
        raise SystemExit(f"Unexpected Rook Brain config: {cfg} (tokenizer vocab {vocab})")
    if tokenizer.n and cfg.get("tokenizer_sha256") != tokenizer.digest():
        raise SystemExit("Checkpoint tokenizer does not match its recorded checksum.")
    if cfg.get("position_scheme") != "prefix-right-pad-v2":
        raise SystemExit("Refusing to export incompatible token-position scheme.")
    if checkpoint.get("objective") != "full-sequence next-token prediction":
        raise SystemExit("Refusing to export a checkpoint without verified full-sequence objective metadata.")
    if not isinstance(checkpoint.get("model"), dict):
        raise SystemExit("Checkpoint has no model state_dict.")

    model = Rook(cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"], vocab=vocab)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.full((1, ctx), tokenizer.pad, dtype=torch.long)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.onnx.export(
        model, (dummy,), str(model_out),
        input_names=["tokens"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    onnx.checker.check_model(str(model_out))

    session = ort.InferenceSession(str(model_out), providers=["CPUExecutionProvider"])
    probe = np.full((1, ctx), tokenizer.pad, dtype=np.int64)
    logits = session.run(["logits"], {"tokens": probe})[0]
    expected_shape = (1, ctx, vocab)
    if logits.shape != expected_shape or not np.isfinite(logits).all():
        raise RuntimeError(f"ONNX smoke test failed: shape={logits.shape}")

    # The ONNX graph must agree with the PyTorch model it was exported from.
    sample_ids = np.array([[tokenizer.bos] + tokenizer.encode(TEST_PROMPT)], dtype=np.int64)
    padded = np.full((1, ctx), tokenizer.pad, dtype=np.int64)
    padded[0, :sample_ids.shape[1]] = sample_ids[0]
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(padded)).numpy()
    onnx_logits = session.run(["logits"], {"tokens": padded})[0]
    last = sample_ids.shape[1] - 1
    if not np.allclose(torch_logits[0, last], onnx_logits[0, last], atol=1e-3):
        raise RuntimeError("ONNX output differs from the PyTorch model.")

    # Exercise real autoregressive inference using the same prefix/right-padding
    # position scheme as training. This test validates execution, not quality.
    prompt = [tokenizer.bos] + tokenizer.encode(TEST_PROMPT)
    generated = []
    for _ in range(24):
        seq = (prompt + generated)[-ctx:]
        tokens = np.full((1, ctx), tokenizer.pad, dtype=np.int64)
        tokens[0, :len(seq)] = seq
        scores = session.run(["logits"], {"tokens": tokens})[0][0, len(seq) - 1]
        scores[tokenizer.pad] = -1e9
        scores[tokenizer.bos] = -1e9
        nxt = int(np.argmax(scores))
        if nxt == tokenizer.eos:
            break
        if not 0 <= nxt < 256 + tokenizer.n:
            raise RuntimeError(f"Unexpected generated token id: {nxt}")
        generated.append(nxt)
    sample = tokenizer.decode(generated)
    digest = hashlib.sha256(model_out.read_bytes()).hexdigest()

    tokenizer_info = None
    if tokenizer.n:
        tokenizer_out.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_out.write_text(json.dumps(tokenizer.to_json(), separators=(",", ":")), encoding="utf-8")
        tokenizer_info = {"format": "rook-bytebpe-v1", "file": "tokenizer.json", "vocab_size": vocab,
                          "merges": tokenizer.n, "sha256": tokenizer.digest(),
                          "learned_from": "Rook's own training split; no pretrained tokenizer"}
    manifest = {
        "status": "verified",
        "model": "./models/rook-brain.onnx",
        "format": "ONNX",
        "source_checkpoint": str(checkpoint_path.relative_to(REPO)) if checkpoint_path.is_relative_to(REPO) else checkpoint_path.name,
        "stage": checkpoint.get("stage", "full_sequence_pretraining"),
        "finetune_steps": int(checkpoint.get("finetune_steps", 0)),
        "training_step": int(checkpoint["step"]),
        "validation_loss": float(checkpoint["val_loss"]),
        "objective": checkpoint["objective"],
        "config": cfg,
        "tokenizer": tokenizer_info,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "pretrained_weights": False,
        "external_inference_api": False,
        "position_scheme": "prefix-right-pad-v2",
        "input": {"name": "tokens", "dtype": "int64", "shape": [1, ctx]},
        "output": {"name": "logits", "dtype": "float32", "shape": [1, ctx, vocab]},
        "model_bytes": model_out.stat().st_size,
        "model_sha256": digest,
        "tensor_smoke_test": "passed",
        "onnx_matches_pytorch": "passed",
        "autoregressive_smoke_test": "passed",
        "sample_for_test_prompt": sample,
        "quality_note": "Runtime checks verify execution only. Review the generated sample and held-out test loss before treating this as conversationally capable.",
    }
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
