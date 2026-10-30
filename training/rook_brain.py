#!/usr/bin/env python3
"""Rook Brain: random-init language-model pretraining on documented local text.

Text becomes tokens with a byte-level BPE tokenizer learned only from Rook's own
training documents (rook_tokenizer.py). This is deliberately NOT instruction tuning.
Every non-padding token is a next-token target; no speaker role or answer span is
privileged, and no identity or instruction text is added anywhere.
"""
import argparse, hashlib, json, math, random, time, sys
from array import array
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from train_rook_next import Rook
from rook_tokenizer import ByteBPE, byte_tokenizer, train_bpe

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
MANIFEST = CORPUS / "manifest.jsonl"
OUT = ROOT / "artifacts" / "rook-brain"

def load_documents(corpus_dir=None):
    corpus = Path(corpus_dir).resolve() if corpus_dir else CORPUS
    manifest_path = corpus / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing {manifest_path}. Add one JSON object per line with path, source, "
            "license, url, and language. Only include text you have the right to use."
        )
    docs, seen, audit = [], set(), []
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
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
        path = (corpus / row["path"]).resolve()
        if corpus.resolve() not in path.parents:
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
        weight = int(row.get("weight", 1) or 1)
        if not 1 <= weight <= 8:
            raise SystemExit(f"Manifest line {line_no}: weight must be between 1 and 8, got {weight}")
        docs.append({"path": row["path"], "text": text, "source": row["source"], "weight": weight,
                     "license": row["license"], "url": row["url"], "language": row["language"],
                     "sha256": digest})
        audit.append({"path": row["path"], "status": "included", "bytes": len(text.encode("utf-8")),
                      "license": row["license"], "language": row["language"], "weight": weight, "sha256": digest})
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

def expand_weights(docs):
    """Repeat each training document `weight` times (validation and test are never repeated)."""
    return [doc for doc in docs for _ in range(doc.get("weight", 1))]

def source_mixture(docs):
    """Share of training text per source family, counting weights. Used to check the data mix."""
    totals = {}
    for doc in docs:
        family = doc["source"].split(":")[0].replace(" dialogue shard", "").replace(" story shard", "")
        totals[family] = totals.get(family, 0) + len(doc["text"].encode("utf-8")) * doc.get("weight", 1)
    whole = sum(totals.values()) or 1
    return {family: round(size / whole, 4) for family, size in sorted(totals.items(), key=lambda kv: -kv[1])}

class Windows(Sequence):
    """Lazy (x, y) training windows over compact token arrays (2 bytes per token).

    Adjacent windows keep every token as a next-token target, including the first
    token after each window boundary. Short last windows are right-padded with PAD.
    """
    def __init__(self, arrays, ctx, pad):
        self.arrays, self.ctx, self.pad = arrays, ctx, pad
        self.index = []
        for ai, ids in enumerate(arrays):
            for start in range(0, max(1, len(ids) - 1), ctx):
                if min(ctx + 1, len(ids) - start) >= 2:
                    self.index.append((ai, start))
        self.tokens = sum(len(ids) for ids in arrays)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self.index)))]
        ai, start = self.index[i]
        segment = self.arrays[ai][start:start + self.ctx + 1].tolist()
        x, y = segment[:-1], segment[1:]
        if len(x) < self.ctx:
            x += [self.pad] * (self.ctx - len(x))
            y += [self.pad] * (self.ctx - len(y))
        return x, y

def examples(docs, ctx, tokenizer=None):
    """Tokenize whole documents as BOS + text + EOS and cut them into windows.

    With no tokenizer this is the original plain-byte vocabulary (259 tokens).
    """
    tok = tokenizer or byte_tokenizer()
    if tok.vocab_size > 65535:
        raise ValueError("Vocabulary too large for 16-bit token storage.")
    arrays = [array("H", [tok.bos] + tok.encode(doc["text"]) + [tok.eos]) for doc in docs]
    return Windows(arrays, ctx, tok.pad)

@torch.no_grad()
def evaluate(model, batches, device, limit=500, pad=None, chunk=16):
    """Average next-token loss over up to `limit` windows spread evenly across the set.

    Spreading the windows matters: the held-out set mixes several sources, and taking
    only the first windows would measure just the first source.
    """
    model.eval()
    pad = model.vocab - 3 if pad is None else pad
    stride = max(1, len(batches) // max(1, limit))
    items = batches[::stride][:limit]
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        xt = torch.tensor([r[0] for r in part], dtype=torch.long, device=device)
        yt = torch.tensor([r[1] for r in part], dtype=torch.long, device=device)
        logits = model(xt)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yt.reshape(-1), ignore_index=pad, reduction="sum")
        total_loss += float(loss.item())
        total_tokens += int((yt != pad).sum().item())
    mean = total_loss / max(1, total_tokens)
    return {"loss": mean, "perplexity": float(torch.exp(torch.tensor(mean)).item()),
            "tokens": total_tokens, "windows": len(items)}

def lr_at(step, base_lr, warmup, total_steps, min_ratio):
    """Linear warmup, then cosine decay to base_lr * min_ratio. Uses the global step."""
    if warmup > 0 and step <= warmup:
        return base_lr * step / warmup
    span = max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / span))
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

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
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout for training only; 0 disables it.")
    p.add_argument("--accum", type=int, default=4, help="Micro-batches per optimizer step (effective batch = batch * accum).")
    p.add_argument("--warmup", type=int, default=200, help="Linear warmup steps, counted from global step 0 (skipped on resume).")
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="Final learning rate as a fraction of --lr.")
    p.add_argument("--total-steps", type=int, default=0, help="Global step where cosine decay ends. Default: start step + --steps.")
    p.add_argument("--vocab-size", type=int, default=4096, help="Byte-level BPE vocabulary size learned from the training split. 259 means plain bytes.")
    p.add_argument("--tokenizer-sample-mb", type=int, default=30, help="Training text (MB) used to learn the BPE merges.")
    p.add_argument("--corpus-dir", default=None, help="Corpus folder containing manifest.jsonl (default: training/corpus).")
    p.add_argument("--out-dir", default=None, help="Where checkpoints and reports go (default: training/artifacts/rook-brain).")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Run two tiny steps to validate the pipeline.")
    p.add_argument("--fresh", action="store_true", help="Ignore saved Rook Brain state and start from random weights.")
    a = p.parse_args()
    out = Path(a.out_dir).resolve() if a.out_dir else OUT
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    docs, audit = load_documents(a.corpus_dir)
    manifest = {"objective": "full-sequence next-token prediction",
                "architecture": "random-init Transformer",
                "pretrained_weights": False, "external_inference_api": False,
                "speaker_role_objective": False, "seed": a.seed,
                "documents": len(docs), "unique_languages": sorted({d["language"] for d in docs}),
                "sources": sorted({d["source"] for d in docs}),
                "licenses": sorted({d["license"] for d in docs}), "audit": audit}
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"event": "corpus_audit", **{k:v for k,v in manifest.items() if k!="audit"}}, ensure_ascii=False), flush=True)
    if a.audit_only:
        print(f"Full audit: {out / 'corpus_audit.json'}")
        return
    train_docs, val_docs, test_docs = split_docs(docs, a.seed)
    if a.smoke:
        # A small slice keeps the pipeline check quick; it still runs every real code path.
        train_docs, val_docs, test_docs = train_docs[:150], val_docs[:20], test_docs[:20]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    d, layers, heads, ctx, batch_size, steps = a.d, a.layers, a.heads, a.ctx, a.batch, a.steps
    if a.smoke:
        d, layers, heads, ctx, batch_size, steps = 192, 4, 6, 64, 2, 2
    resume_path = out / "resume.pt"
    best_path = out / "best.pt"
    if a.smoke:
        a.fresh = True
    saved = None
    if resume_path.exists() and not a.fresh:
        saved = torch.load(resume_path, map_location=device, weights_only=False)

    # Tokenizer: a resumed run must keep the exact tokenizer its weights were trained with.
    if saved is not None:
        tokenizer = ByteBPE.from_json(saved["tokenizer"]) if saved.get("tokenizer") else byte_tokenizer()
        tokenizer_source = "checkpoint"
    elif a.vocab_size > 259:
        began = time.time()
        # Learned from the training split only, so validation and test text never shape it.
        tokenizer = train_bpe((doc["text"] for doc in train_docs), a.vocab_size,
                              max_bytes=a.tokenizer_sample_mb * 1_000_000)
        tokenizer_source = f"learned from training split in {time.time() - began:.1f}s"
    else:
        tokenizer = byte_tokenizer()
        tokenizer_source = "plain bytes"
    vocab, pad = tokenizer.vocab_size, tokenizer.pad
    mixture = source_mixture(train_docs)
    print(json.dumps({"event": "training_mixture", "share_of_training_text_by_source": mixture}), flush=True)
    train = examples(expand_weights(train_docs), ctx, tokenizer)
    valid = examples(val_docs, ctx, tokenizer)
    test = examples(test_docs, ctx, tokenizer)
    if not train or not valid or not test:
        raise SystemExit("Train/validation/test split produced an empty set.")
    train_bytes = sum(len(doc["text"].encode("utf-8")) * doc.get("weight", 1) for doc in train_docs)
    tokenizer_info = {"format": "rook-bytebpe-v1" if tokenizer.n else "plain-bytes", "vocab_size": vocab,
                      "merges": tokenizer.n, "sha256": tokenizer.digest(), "source": tokenizer_source,
                      "train_tokens": train.tokens, "train_bytes": train_bytes,
                      "bytes_per_token": round(train_bytes / max(1, train.tokens), 3)}
    print(json.dumps({"event": "tokenizer", **tokenizer_info}), flush=True)
    tokenizer_extra = {"tokenizer": tokenizer.to_json()} if tokenizer.n else {}
    if tokenizer.n:
        (out / "tokenizer.json").write_text(json.dumps(tokenizer.to_json(), separators=(",", ":")), encoding="utf-8")

    model = Rook(d, layers, heads, ctx, dropout=a.dropout, vocab=vocab).to(device)  # fresh random initialization unless a compatible resume exists
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    config = {"d":d,"layers":layers,"heads":heads,"ctx":ctx,"vocab":vocab,"position_scheme":"prefix-right-pad-v2"}
    if tokenizer.n:
        config["tokenizer_sha256"] = tokenizer.digest()
    corpus_fingerprint = hashlib.sha256(
        "".join(sorted(doc["sha256"] for doc in docs)).encode("ascii")
    ).hexdigest()
    start_step = 0
    best = float("inf")
    if saved is not None:
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
    report = {"architecture": "random-init byte-level BPE Transformer" if tokenizer.n else "random-init byte Transformer",
              "pretrained_weights": False,
              "external_inference_api": False, "objective": "full-sequence next-token prediction",
              "speaker_role_objective": False, "config": config, "tokenizer": tokenizer_info, "training_mixture": mixture,
              "corpus_fingerprint": corpus_fingerprint,
              "training": {"lr": a.lr, "dropout": a.dropout, "batch": batch_size, "accum": a.accum,
                           "warmup": a.warmup, "min_lr_ratio": a.min_lr_ratio},
              "parameters": sum(q.numel() for q in model.parameters()), "corpus": manifest}
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total_steps = a.total_steps if a.total_steps > 0 else start_step + steps
    accum = 1 if a.smoke else max(1, a.accum)
    for step in range(start_step + 1, start_step + steps + 1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, a.lr, a.warmup, total_steps, a.min_lr_ratio)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accum):
            batch = random.choices(train, k=batch_size)
            x = torch.tensor([r[0] for r in batch], dtype=torch.long, device=device)
            y = torch.tensor([r[1] for r in batch], dtype=torch.long, device=device)
            logits = model(x)
            micro = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), ignore_index=pad)
            if not torch.isfinite(micro):
                raise RuntimeError("Non-finite training loss.")
            (micro / accum).backward()
            loss_sum += float(micro.item())
        loss = torch.tensor(loss_sum / accum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == start_step + 1 or step % 100 == 0 or step == start_step + steps:
            val = evaluate(model, valid, device, pad=pad)
            print(json.dumps({"step":step,"train_loss":float(loss.item()),"validation":val,
                              "elapsed_seconds":round(time.time()-start,1)}), flush=True)
            if val["loss"] < best:
                best = val["loss"]
                torch.save({"model":model.state_dict(),"config":report["config"],"step":step,
                            "val_loss":best,"objective":report["objective"],
                            "corpus_fingerprint":corpus_fingerprint, **tokenizer_extra}, best_path)
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,
                        "best_validation_loss":best,"config":report["config"],
                        "corpus_fingerprint":corpus_fingerprint,
                        "python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),
                        "objective":report["objective"], **tokenizer_extra}, resume_path)
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    report.update({"best_step":ck["step"],"best_validation":evaluate(model,valid,device,pad=pad),
                   "heldout_test":evaluate(model,test,device,pad=pad),
                   "elapsed_seconds":round(time.time()-start,1),
                   "note":"Loss is a language-modeling metric, not proof of semantic understanding."})
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)
    print(json.dumps({"event":"training_stage_complete","objective":"full-sequence next-token prediction","pretrained_weights":False,"speaker_role_objective":False,"note":"Assistant-only conversation fine-tuning is intentionally not launched."}), flush=True)

if __name__ == "__main__":
    main()
