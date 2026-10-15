#!/usr/bin/env python3
"""Resumable, self-checkpointing Rook training worker.

Uses only a randomly initialized model and local training code. No pretrained
weights or external inference API are used. Intended for repeated cloud-worker
runs with a persistent checkpoint directory.
"""
import argparse
import json
import os
import random
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from train_rook_next import PAD, Rook, data, ev, pack, V

STOP = False


def request_stop(_signum, _frame):
    global STOP
    STOP = True


def atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--max-runtime-seconds", type=int, default=20400)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until the time limit")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    random.seed(1701)
    torch.manual_seed(1701)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1701)
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = Path("artifacts/rook-next")
    out.mkdir(parents=True, exist_ok=True)
    resume_path = out / "resume.pt"
    best_path = out / "best.pt"
    report_path = out / "report.json"

    train, valid, test, meta = data(1701)
    atomic_json_save(meta, out / "dataset_manifest.json")
    if len(train) <= 1000 or len(valid) <= 20 or len(test) <= 20:
        raise RuntimeError(f"Dataset/split sanity check failed: {meta}")

    config = {"d": args.d, "layers": args.layers, "heads": args.heads,
              "ctx": args.ctx, "vocab": V}
    model = Rook(args.d, args.layers, args.heads, args.ctx).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    step = 0
    best_val = float("inf")
    history = []
    resumed_from_checkpoint = resume_path.exists()
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("config") != config:
            raise RuntimeError(f"Checkpoint config mismatch: saved={state.get('config')} requested={config}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        best_val = float(state.get("best_val_loss", float("inf")))
        history = state.get("history", [])
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if device == "cuda" and state.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng_state"]])
        print(json.dumps({"event": "resume", "step": step, "best_val_loss": best_val}), flush=True)
    else:
        print(json.dumps({"event": "fresh_random_initialization", "step": 0,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "pretrained_weights": False, "external_inference_api": False}), flush=True)

    def save_state():
        atomic_torch_save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": step,
            "best_val_loss": best_val,
            "history": history[-500:],
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            "saved_at_unix": time.time(),
        }, resume_path)

    run_start_step = step
    started = time.monotonic()
    last_loss = None
    while not STOP:
        if time.monotonic() - started >= args.max_runtime_seconds:
            break
        if args.max_steps and (step - run_start_step) >= args.max_steps:
            break

        model.train()
        xs, ys = [], []
        for _ in range(args.batch):
            x, y = pack(random.choice(train), args.ctx)
            xs.append(F.pad(x, (args.ctx - len(x), 0), value=PAD))
            ys.append(F.pad(y, (args.ctx - len(y), 0), value=-100))
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda" and torch.cuda.is_bf16_supported()):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        if not torch.isfinite(loss):
            save_state()
            raise RuntimeError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        last_loss = float(loss.item())

        if step % args.eval_every == 0 or step == 1:
            val_loss = ev(model, valid, device, args.ctx, limit=100)
            record = {"step": step, "train_loss": last_loss, "val_loss": val_loss,
                      "elapsed_seconds": round(time.monotonic() - started, 1)}
            history.append(record)
            print(json.dumps(record), flush=True)
            if val_loss < best_val:
                best_val = val_loss
                atomic_torch_save({"model": model.state_dict(), "config": config,
                                   "step": step, "val_loss": val_loss,
                                   "pretrained_weights": False}, best_path)
        if step % args.save_every == 0:
            save_state()

    save_state()
    elapsed = round(time.monotonic() - started, 1)
    report = {
        "status": "checkpointed",
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "architecture": "random-init byte Transformer",
        "parameters": sum(p.numel() for p in model.parameters()),
        "pretrained_weights": False,
        "external_inference_api": False,
        "config": config,
        "dataset": meta,
        "step": step,
        "best_validation_loss": None if best_val == float("inf") else best_val,
        "last_train_loss": last_loss,
        "history": history[-100:],
        "elapsed_seconds_this_run": elapsed,
        "device": device,
        "checkpoint": str(resume_path),
        "best_checkpoint": str(best_path) if best_path.exists() else None,
    }
    if best_path.exists():
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        report["best_step"] = best["step"]
        report["heldout_test_loss_sample"] = ev(model, test, device, args.ctx, limit=100)
    atomic_json_save(report, report_path)
    print("ROOK_WORKER_REPORT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
