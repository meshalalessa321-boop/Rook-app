# Rook Next training experiment

This is a **new research branch**, not the deployed Rook model. It does not replace the current app or load its weights.

## Why this branch exists

The existing Rook dialogue corpus contains only a few hundred supervised response turns. Repeating optimizer steps over it was not a credible route to conversational ability. This experiment instead builds examples from the public OpenAssistant conversation tree, splits by conversation root to prevent related turns crossing splits, and evaluates on held-out validation and test roots.

## Model

- Decoder-only Transformer initialized from random weights.
- Default size: 384 hidden width, 8 layers, 6 attention heads, 512-byte context, approximately 14.5 million parameters.
- Byte-level encoding avoids training a tokenizer on the same small corpus.
- Only assistant-response bytes are included in the supervised loss; prompt bytes are masked.
- No pretrained weights and no external inference API.
- Dataset source: `OpenAssistant/oasst1`, licensed CC BY 4.0. Preserve attribution and review the source dataset's terms before redistributing derived data.

## Reproducible commands

Install PyTorch and Hugging Face Datasets, then run:

```bash
python training/train_rook_next.py --steps 5000
```

The run saves `artifacts/rook-next/dataset_manifest.json`, best checkpoint, resumable checkpoint, and `report.json`. The final test loss is computed after selecting the checkpoint by validation loss.

Unit tests (from the training directory):

```bash
python -m unittest test_train_rook_next.py
```

## Promotion gate

A falling loss is necessary but not sufficient. Before any candidate can replace the live app, evaluate generated conversations on a fixed unseen prompt set for relevance across turns, repetition, instruction-following, and context retention. Inspect actual generations manually. Do not promote a model solely because its loss improves.

## Resource note

The 14.5M default model should be trained on a compatible GPU. The initial attempt to launch a Hugging Face compute job returned **HTTP 402 Payment Required**, so no remote training job started and no training results are claimed. Use a compute environment with available credits, or reduce model/context/batch size for a pipeline smoke test.
