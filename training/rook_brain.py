#!/usr/bin/env python3
"""Rook Brain: random-init byte language-model pretraining on documented local text.

This is deliberately NOT instruction tuning. Every non-padding token is a
next-token target; no speaker role or answer span is privileged.
"""
import argparse, hashlib, json, random, time, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from train_rook_next import PAD, BOS, EOS, V, Rook

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
MANIFEST = CORPUS / "manifest.jsonl"
OUT = ROOT / "artifacts" / "rook-brain"

def load_documents():
    if not MANIFEST.exists():
        raise SystemExit(
            f"Missing {MANIFEST}. Add one JSON object per line with path, source, "
            "license, url, and language. Only include text you have the right to use."
        )
    docs, seen, audit = [], set(), []
    for line_no, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Bad manifest JSON on line {line_no}: {exc}")
        required = ("path", "source", "license", "url", "language")
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            raise SystemExit(f"Manifest line {line_no} missing: {', '.join(missing)}")
        path = (CORPUS / row["path"]).resolve()
        if CORPUS.resolve() not in path.parents:
            raise SystemExit(f"Refusing path outside corpus folder: {row['path']}")
        if not path.is_file() or path.suffix.lower() != ".txt":
            raise SystemExit(f"Expected an existing .txt file inside corpus/: {row['path']}")
        text = path.read_text(encoding="utf-8", errors="strict").strip()
        if len(text.encode("utf-8")) < 300:
            audit.append({"path": row["path"], "status": "skipped_short", "bytes": len(text.encode("utf-8"))})
            continue
        digest = hashlib.sha256(" ".join(text.split()).casefold().encode("utf-8")).hexdigest()
        if digest in seen:
            audit.append({"path": row["path"], "status": "skipped_duplicate", "sha256": digest})
            continue
        seen.add(digest)
        docs.append({"path": row["path"], "text": text, "source": row["source"],
                     "license": row["license"], "url": row["url"], "language": row["language"],
                     "sha256": digest})
        audit.append({"path": row["path"], "status": "included", "bytes": len(text.encode("utf-8")),
                      "license": row["license"], "language": row["language"], "sha256": digest})
    if len(docs) < 3:
        raise SystemExit(f"Need at least 3 distinct, non-duplicate text documents; found {len(docs)}.")
    return docs, audit

def split_docs(docs, seed):
    # Split whole documents, not overlapping windows from the same document.
    rng = random.Random(seed)
    order = list(range(len(docs)))
    rng.shuffle(order)
    n = len(order)
    n_test = max(1, round(n * .05))
    n_val = max(1, round(n * .05))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    test_ids = set(order[:n_test])
    val_ids = set(order[n_test:n_test+n_val])
    train_ids = set(order[n_test+n_val:])
    return ([docs[i] for i in sorted(train_ids)],
            [docs[i] for i in sorted(val_ids)],
            [docs[i] for i in sorted(test_ids)])

def examples(docs, ctx):
    result = []
    for doc in docs:
        ids = [BOS] + list(doc["text"].encode("utf-8")) + [EOS]
        # Adjacent windows retain every byte as a next-token target, including
        # the first token after each window boundary.
        for start in range(0, max(1, len(ids)-1), ctx):
            segment = ids[start:start+ctx+1]
            if len(segment) < 2:
                continue
            x, y = segment[:-1], segment[1:]
            if len(x) < ctx:
                x += [PAD] * (ctx-len(x))
                y += [PAD] * (ctx-len(y))
            result.append((x, y))
    return result

@torch.no_grad()
def evaluate(model, batches, device, limit=500):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in batches[:limit]:
        xt = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
        yt = torch.tensor(y, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(xt)
        loss = F.cross_entropy(logits.reshape(-1, V), yt.reshape(-1), ignore_index=PAD, reduction="sum")
        total_loss += float(loss.item())
        total_tokens += int((yt != PAD).sum().item())
    return {"loss": total_loss / max(1, total_tokens),
            "perplexity": float(torch.exp(torch.tensor(total_loss / max(1, total_tokens))).item()),
            "tokens": total_tokens, "windows": min(len(batches), limit)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Run two tiny steps to validate the pipeline.")
    p.add_argument("--fresh", action="store_true", help="Ignore saved Rook Brain state and start from random weights.")
    a = p.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    docs, audit = load_documents()
    manifest = {"objective": "full-sequence next-token prediction",
                "architecture": "random-init byte Transformer",
                "pretrained_weights": False, "external_inference_api": False,
                "speaker_role_objective": False, "seed": a.seed,
                "documents": len(docs), "unique_languages": sorted({d["language"] for d in docs}),
                "sources": sorted({d["source"] for d in docs}),
                "licenses": sorted({d["license"] for d in docs}), "audit": audit}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corpus_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"event": "corpus_audit", **{k:v for k,v in manifest.items() if k!="audit"}}, ensure_ascii=False), flush=True)
    if a.audit_only:
        print(f"Full audit: {OUT / 'corpus_audit.json'}")
        return
    train_docs, val_docs, test_docs = split_docs(docs, a.seed)
    train = examples(train_docs, a.ctx)
    valid = examples(val_docs, a.ctx)
    test = examples(test_docs, a.ctx)
    if not train or not valid or not test:
        raise SystemExit("Train/validation/test split produced an empty set.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    d, layers, heads, ctx, batch_size, steps = a.d, a.layers, a.heads, a.ctx, a.batch, a.steps
    if a.smoke:
        d, layers, heads, ctx, batch_size, steps = 192, 4, 6, 64, 2, 2
        train, valid, test = examples(train_docs, ctx), examples(val_docs, ctx), examples(test_docs, ctx)
    model = Rook(d, layers, heads, ctx).to(device)  # fresh random initialization unless a compatible resume exists
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    config = {"d":d,"layers":layers,"heads":heads,"ctx":ctx,"vocab":V,"position_scheme":"prefix-right-pad-v2"}
    corpus_fingerprint = hashlib.sha256(
        "".join(sorted(doc["sha256"] for doc in docs)).encode("ascii")
    ).hexdigest()
    resume_path = OUT / "resume.pt"
    best_path = OUT / "best.pt"
    start_step = 0
    best = float("inf")
    if a.smoke:
        a.fresh = True
    if resume_path.exists() and not a.fresh:
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved.get("config") != config:
            raise SystemExit("Saved checkpoint configuration differs. Use --fresh to intentionally start a new model.")
        saved_fingerprint = saved.get("corpus_fingerprint")
        if saved_fingerprint and saved_fingerprint != corpus_fingerprint:
            raise SystemExit("Corpus changed since the saved checkpoint. Review the new corpus, then use --fresh for a new run.")
        if not saved_fingerprint and saved.get("objective") != "full-sequence next-token prediction":
            raise SystemExit("Saved checkpoint lacks verified full-sequence objective metadata; refusing to resume it.")
        model.load_state_dict(saved["model"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        if "python_rng" in saved:
            random.setstate(saved["python_rng"])
        if "torch_rng" in saved:
            torch.set_rng_state(saved["torch_rng"].cpu())
        start_step = int(saved.get("step", 0))
        if "best_validation_loss" in saved:
            best = float(saved["best_validation_loss"])
        elif best_path.exists():
            previous_best = torch.load(best_path, map_location="cpu", weights_only=False)
            best = float(previous_best.get("val_loss", float("inf")))
        print(json.dumps({"event":"resumed","step":start_step,"best_validation_loss":best,"corpus_fingerprint":corpus_fingerprint,"legacy_checkpoint_metadata":not bool(saved_fingerprint)}), flush=True)
    else:
        print(json.dumps({"event":"fresh_initialization","pretrained_weights":False,"corpus_fingerprint":corpus_fingerprint}), flush=True)
    start = time.time()
    report = {"architecture": "random-init byte Transformer", "pretrained_weights": False,
              "external_inference_api": False, "objective": "full-sequence next-token prediction",
              "speaker_role_objective": False, "config": config,
              "corpus_fingerprint": corpus_fingerprint,
              "parameters": sum(q.numel() for q in model.parameters()), "corpus": manifest}
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for step in range(start_step + 1, start_step + steps + 1):
        model.train()
        batch = random.choices(train, k=batch_size)
        x = torch.tensor([r[0] for r in batch], dtype=torch.long, device=device)
        y = torch.tensor([r[1] for r in batch], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=PAD)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == start_step + 1 or step % 100 == 0 or step == start_step + steps:
            val = evaluate(model, valid, device)
            print(json.dumps({"step":step,"train_loss":float(loss.item()),"validation":val,
                              "elapsed_seconds":round(time.time()-start,1)}), flush=True)
            if val["loss"] < best:
                best = val["loss"]
                torch.save({"model":model.state_dict(),"config":report["config"],"step":step,
                            "val_loss":best,"objective":report["objective"],
                            "corpus_fingerprint":corpus_fingerprint}, best_path)
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,
                        "best_validation_loss":best,"config":report["config"],
                        "corpus_fingerprint":corpus_fingerprint,
                        "python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),
                        "objective":report["objective"]}, resume_path)
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    report.update({"best_step":ck["step"],"best_validation":evaluate(model,valid,device),
                   "heldout_test":evaluate(model,test,device),
                   "elapsed_seconds":round(time.time()-start,1),
                   "note":"Loss is a language-modeling metric, not proof of semantic understanding."})
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)
    print(json.dumps({"event":"training_stage_complete","objective":"full-sequence next-token prediction","pretrained_weights":False,"speaker_role_objective":False,"note":"Assistant-only conversation fine-tuning is intentionally not launched."}), flush=True)

if __name__ == "__main__":
    main()
