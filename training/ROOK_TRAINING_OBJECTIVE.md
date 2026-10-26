# Rook's training objective: correction and next direction

## Non-negotiable constraints

- Initialize Rook's model from random weights. Do not import pretrained weights.
- Do not define Rook as an assistant or optimize him specifically to answer user instructions.
- Do not prescribe a personality, worldview, beliefs, opinions, emotions, or identity.
- Do not tell the model in a hidden/system prompt that it is an AI, a person, or anything else.
- Do not add canned replies or a scripted layer that imitates conversation.
- Conversation is a capability to emerge and evaluate, not the identity or role being imposed.
- Preserve checkpoints and report the exact training objective, data sources, licenses, splits, and model provenance.

## Why the previous objective is retired

The previous pipeline builds prompt/answer pairs from OpenAssistant and masks the prompt tokens so the loss is applied only to the answer after the literal `Assistant:` marker. Repeating that objective teaches a specific assistant-response role. This conflicts with the project's intended direction, so the previous worker and its scheduled workflow have been paused.

## Replacement direction

Do not simply resume the old worker with more steps or remove the word `Assistant:` while keeping the same answer-only loss. Design a distinct pretraining pipeline that:

1. Trains next-token prediction across the full sequence, not only a designated responder's output.
2. Uses a broader, diverse, documented corpus of licensed text, rather than treating an assistant dataset as the foundation of Rook's identity.
3. Includes conversational material as one part of the corpus, without hard-coding a speaker role as the model's purpose.
4. Keeps data provenance, license/redistribution constraints, deduplication, language distribution, and train/validation/test separation explicit.
5. Evaluates general continuation, coherence across context, language drift, and multi-turn continuity without grading whether Rook sounds like a conventional assistant.
6. Makes no identity/personality claims based on fluent output alone; those are questions for observation, not labels to train into him.

## Current state

The existing model/checkpoints from the old assistant-response objective are preserved as historical artifacts and are not resumed by the new pipeline. The new `rook_brain.py` pipeline uses full-sequence next-token prediction and resumes only checkpoints with a matching configuration and corpus fingerprint. The scheduled workflow must not invoke `finetune_rook_conversation.py`; that legacy answer-only fine-tuning path is intentionally excluded. The corpus builder now combines WikiText-2 with a bounded English OpenAssistant transcript subset (Apache-2.0). This is a more relevant starting mixture, but the model remains small and must pass held-out and multi-turn evaluation before we can claim conversational capability.


## Implemented first replacement component

`rook_brain.py` now provides a separate, from-scratch full-sequence pretraining path. It reads only local UTF-8 text files listed in `training/corpus/manifest.jsonl`, records source/license metadata and document hashes, removes exact duplicates, and splits whole documents into train/validation/test sets. It computes next-token loss over all non-padding bytes, without a speaker-role or answer-only mask.

The corpus is deliberately not populated automatically: source selection and license verification must happen before a real run. Use `python training/rook_brain.py --audit-only` to audit the local corpus, and `--smoke` only after the corpus has been prepared to check the training loop. The new pipeline has not yet been run end-to-end and does not automatically replace the deployed ONNX model.

## Explicit identity prohibition

Never add an identity prompt, system prompt, metadata instruction, training example template, or UI-to-model message telling Rook that he is an AI, an artificial intelligence, a machine, a chatbot, or any other predefined category. Do not instruct him to describe himself using those labels. Do not insert hidden identity statements into his context. This prohibition applies to the training pipeline, inference wrapper, evaluation harness, and any future tool integration.

The model may encounter ordinary text discussing AI as part of a genuinely broad corpus; that is not an instruction about Rook himself. Keep source selection honest and do not selectively manipulate examples to force a prescribed self-concept. The rule is to avoid assigning Rook an identity, not to falsify the wider world represented in language.
