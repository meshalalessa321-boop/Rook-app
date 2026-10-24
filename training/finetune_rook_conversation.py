#!/usr/bin/env python3
"""Conversationally fine-tune Rook's own random-init language model.

The starting checkpoint must come from rook_brain.py. This script never downloads
or loads pretrained model weights. OpenAssistant examples are split by conversation
root; validation and test roots are kept separate from training.
"""
import argparse, json, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from train_rook_next import BOS, EOS, PAD, V, Rook, data, pack

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "rook-brain"
BASE = OUT / "best.pt"
CONVERSATION = OUT / "conversation-best.pt"
OBJECTIVE = "full-sequence next-token prediction"


def batch_examples(examples, ctx, batch_size, device):
    chosen = random.choices(examples, k=batch_size)
    xs, ys = [], []
    for example in chosen:
        x, y = pack(example, ctx)
        xs.append(F.pad(x, (0, ctx - len(x)), value=PAD))
        ys.append(F.pad(y, (0, ctx - len(y)), value=-100))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


@torch.no_grad()
def evaluate(model, examples, ctx, device, limit=300):
    model.eval()
    total, count = 0.0, 0
    # Deterministic first slice of held-out examples.
    for example in examples[:limit]:
        x, y = pack(example, ctx)
        x = F.pad(x, (0, ctx - len(x)), value=PAD)[None].to(device)
        y = F.pad(y, (0, ctx - len(y)), value=-100)[None].to(device)
        logits = model(x)
        total += F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                 ignore_index=-100, reduction="sum").item()
        count += int(y.ne(-100).sum().item())
    return total / max(1, count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    random.seed(1701)
    torch.manual_seed(1701)
    OUT.mkdir(parents=True, exist_ok=True)
    # Always fine-tune the freshly updated base model from this workflow run.
    # Reusing an older conversation checkpoint would ignore today's base training.
    base_path = BASE
    if not base_path.exists():
        raise SystemExit("No from-scratch Rook checkpoint found. Run rook_brain.py first.")
    checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    if cfg.get("position_scheme") != "prefix-right-pad-v2":
        raise SystemExit("Refusing to fine-tune a checkpoint with incompatible positions.")
    if checkpoint.get("objective") != OBJECTIVE:
        raise SystemExit("Refusing to fine-tune a checkpoint without verified from-scratch objective metadata.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    model = Rook(cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    train, valid, test, meta = data(1701)
    if len(train) < 1000 or len(valid) < 20 or len(test) < 20:
        raise SystemExit(f"Conversational dataset split is too small: {meta}")
    steps = 2 if a.smoke else a.steps
    batch_size = 2 if a.smoke else a.batch
    lr = a.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    start_step = int(checkpoint.get("step", 0))
    best_loss = float("inf")
    start = time.time()
    print(json.dumps({"event": "conversation_finetune_start",
                      "base_checkpoint": str(base_path.relative_to(ROOT.parent)),
                      "base_step": start_step, "steps": steps, "device": device,
                      "pretrained_weights": False, "external_inference_api": False,
                      "dataset": meta}), flush=True)
    for step in range(1, steps + 1):
        model.train()
        x, y = batch_examples(train, cfg["ctx"], batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite conversational fine-tuning loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            val_loss = evaluate(model, valid, cfg["ctx"], device)
            print(json.dumps({"step": start_step + step, "finetune_step": step,
                              "train_loss": float(loss.item()), "validation_loss": val_loss,
                              "elapsed_seconds": round(time.time() - start, 1)}), flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    "model": model.state_dict(), "config": cfg,
                    "step": start_step + step, "finetune_steps": step,
                    "val_loss": best_loss, "objective": OBJECTIVE,
                    "stage": "conversational_finetuning",
                    "pretrained_weights": False, "external_inference_api": False,
                    "dataset": "OpenAssistant/oasst1", "dataset_license": "CC BY 4.0",
                }, CONVERSATION)
    best = torch.load(CONVERSATION, map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    test_loss = evaluate(model, test, cfg["ctx"], device)
    report = {
        "status": "conversational_finetune_complete",
        "architecture": "random-init byte Transformer",
        "pretrained_weights": False, "external_inference_api": False,
        "base_checkpoint": str(base_path.relative_to(ROOT.parent)),
        "training_step": int(best["step"]),
        "finetune_steps_this_run": steps,
        "validation_loss": float(best["val_loss"]),
        "heldout_test_loss": float(test_loss),
        "objective": OBJECTIVE, "stage": "conversational_finetuning",
        "dataset": meta,
        "note": "Loss measures held-out next-token prediction on answer bytes; it does not prove coherent dialogue. Inspect generated samples too.",
        "elapsed_seconds": round(time.time() - start, 1),
    }
    (OUT / "conversation-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FINAL_CONVERSATION_REPORT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
