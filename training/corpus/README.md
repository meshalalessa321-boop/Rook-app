# Rook Brain corpus

This directory is intentionally empty until sources are selected and documented. The new training script will not download arbitrary datasets or quietly fall back to assistant-response training.

## Add text

1. Add plain UTF-8 `.txt` files under this directory.
2. Add one JSON object per line to `manifest.jsonl`, with all required fields:

```json
{"path":"texts/example.txt","source":"Exact collection or work name","license":"Public domain or exact license identifier","url":"https://example.org/source","language":"en"}
```

3. Confirm the license really permits the intended use. Record the source URL, exact license, and language accurately. Do not include private conversations, personal data, or material you do not have rights to use.
4. Run the audit before training:

```bash
python training/rook_brain.py --audit-only
```

The audit records SHA-256 hashes, exact source/license metadata, duplicate removal, and document counts. Exact duplicate documents are removed. The training script splits whole documents into train/validation/test, then optimizes next-token loss across the full text sequence. It does not use User/Assistant labels or mask loss to favor one speaker.

At least three distinct, non-duplicate documents are required. For meaningful evaluation, use many documents from several sources and avoid putting near-duplicates in separate splits. A small corpus is only a pipeline test, not a brain upgrade.

## Important

- Fresh random initialization; no pretrained weights.
- No external AI/API generates model output.
- Do not resume a legacy assistant-objective checkpoint in this pipeline.
- Loss/perplexity measures prediction of held-out text, not proof of human-like understanding.
- Training outputs go to `training/artifacts/rook-brain/`; they do not automatically replace the public ONNX model. Validate and export deliberately after training.

## Identity rule

Do not add prompt templates or curated training labels intended to tell Rook what he is. Never inject an identity statement into the model's context. Ordinary licensed text discussing AI may exist in a broad corpus as a topic, but it is not to be framed as a statement about Rook.
