# Rook training status

**The new Rook Brain training workflow is enabled.** It trains with full-sequence language modeling, not assistant-only answer training. The old worker remains retired.

Read [ROOK_TRAINING_OBJECTIVE.md](ROOK_TRAINING_OBJECTIVE.md) before changing the pipeline.

## Why the old experiment is retired

The previous experiment used OpenAssistant conversation trees, formatted examples as `User: ...\nAssistant:`, and computed loss only over assistant-response bytes. This directly optimizes a designated assistant role, which conflicts with Meshal's project constraints. The old checkpoints are preserved as historical artifacts, not as the intended final direction.

The old script and worker require an explicit `--allow-legacy-assistant-objective` opt-in. The GitHub Actions workflow now runs `rook_brain.py` instead; it does not launch the retired worker.

## Replacement requirements

- Start model parameters from random initialization. Do not load pretrained weights.
- Use full-sequence next-token prediction, not a loss mask that privileges a responder role.
- Build a broader, documented and appropriately licensed text mixture; conversation should be one source among many, not an assistant-only foundation.
- Do not inject a hidden identity prompt, prescribed personality, worldview, beliefs, opinions, emotions, or canned responses.
- Track source licenses, deduplication, language distribution, and held-out splits.
- Evaluate general text continuation, context coherence, language drift, and multi-turn continuity without using “assistant helpfulness” as the target identity.

The replacement pipeline is implemented in `rook_brain.py`. The workflow prepares a documented WikiText-2 corpus plus a bounded English OpenAssistant conversation subset licensed under Apache-2.0. Both user and assistant turns are ordinary next-token targets in full transcripts; no answer-only loss mask is used. The workflow runs tests and a corpus audit, then trains for 500 additional steps per run while resuming compatible Rook Brain checkpoints. This is still a small experimental corpus and model, not evidence of coherent conversation by itself.

## Identity rule

Never place a hidden or explicit instruction in Rook's model context telling him that he is AI, a machine, a chatbot, or any other predefined identity. This applies to training, inference, evaluation, and tools. Ordinary source texts may discuss AI as a subject, but they must not be turned into an instruction about Rook's identity.


## Current training workflow

The GitHub Actions workflow `.github/workflows/rook-training.yml` runs on a daily schedule and can also be started manually. It prepares WikiText-2 and English OpenAssistant conversation transcripts, then predicts the next token across every non-padding token in every document. Compatible checkpoints are cached between runs. On a successful run, the workflow exports the exact full-sequence checkpoint, runs ONNX and autoregressive smoke tests, uploads the report/checkpoint, and publishes the verified model to the stable browser-model branch and release. Runtime checks establish that inference works, not that the model is conversationally coherent.

## Tokenizer: byte-level BPE (`rook_tokenizer.py`)

Rook now reads text as learned sub-word tokens instead of single bytes, so the same 256-token context covers roughly four times as much text and the small model spends less effort learning to spell.

- The merges are learned **only from Rook's own training split** (`rook_brain.py --vocab-size 4096`). No pretrained tokenizer or vocabulary is used, and validation/test text never shapes it.
- Every byte sequence round-trips exactly. The tokenizer adds no identity, role, instruction or canned text.
- Conversation transcripts use neutral `Person A` / `Person B` labels, in training and in the chat page.
- The tokenizer is stored inside every checkpoint, so a resumed run always keeps the tokenizer its weights were trained with. It is published as `tokenizer.json` next to the model, and `rook-tokenizer.js` runs the same rules in the browser (`test_tokenizer_js.py` checks Python and JavaScript agree).
- Changing the tokenizer means a new model: the workflow cache key moved to `v2`, so the next run starts from random initialization.
- `rook_brain.py --smoke --out-dir ...` plus `export_rook_brain_onnx.py --checkpoint ...` is run at the start of every training workflow, so a broken export fails in minutes instead of after hours of training.

## Data mix: stories, Wikipedia text, and human dialogue

OpenAssistant is no longer used. Its conversations are people asking an AI to do things, which teaches an assistant role even with neutral labels. The workflow now builds the corpus from:

- WikiText-2 (Wikipedia-derived text),
- TinyStories (simple English written for very small models),
- DailyDialog (everyday conversations between people, CC BY-NC-SA 4.0: non-commercial, share-alike; check before commercial use or redistribution).

Dialogue is labelled `Person A` / `Person B` in alternation, both sides are ordinary next-token targets, and nothing tells Rook what he is. Each manifest row can carry a `weight` (1 to 8): the training split repeats that source's text that many times, so a small source of real conversation is not drowned out by stories. Validation and test text are never repeated. `rook_brain.py` prints and saves `training_mixture` so the actual share of each source is visible. `rook-data-check.yml` downloads a slice of every source on pull requests that touch data preparation.

