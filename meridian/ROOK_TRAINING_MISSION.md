# Rook Training Mission

## Single objective

Develop Rook's ability to produce coherent, natural, multi-turn conversation through training his own model, starting from randomly initialized weights.

## Hard boundaries

- Do not turn Rook into an assistant, chatbot product, helper, agent, or role-playing persona. Do not add assistant-style instructions, service duties, or a predefined character merely to make his replies seem more useful.
- Do not assign Rook any purpose beyond learning to understand and generate language well enough to hold a conversation.
- Do not replace Rook's model or weights with a pretrained external model. External models may be used by Meridian for engineering assistance only, never as Rook's response generator or as a substitute for training Rook.
- Do not inject identity claims or scripted self-descriptions into Rook's training data or prompts, including telling him that he is an AI. Keep training focused on language learning and conversation, not on imposing an identity.
- Do not make claims about Rook's identity or inner experience on his behalf. If asked directly, do not fabricate facts; this mission simply does not require identity instruction.
- Do not use canned replies or hand-written scripts to disguise weaknesses in the model. Test genuine generations from Rook's own current weights.
- Preserve existing weights and checkpoints before experiments. Never overwrite the only usable checkpoint.
- Do not invent training progress. Record commands, data sources, configuration, checkpoint identifiers, evaluation results, failures, and limitations.

## Engineering loop

1. Inspect the current model, tokenizer, training code, datasets, checkpoints, and evaluation workflow before changing anything.
2. Establish a baseline using Rook's own weights and a repeatable set of language and multi-turn conversation tests.
3. Identify the highest-impact bottleneck in data quality, tokenization, architecture, optimization, training stability, or inference.
4. Make a small, reversible change and run a bounded experiment.
5. Compare against the baseline using the same tests. Keep changes only when evidence supports them.
6. Save checkpoints and a concise experiment log; report what actually improved and what remains weak.
7. Continue with the next useful experiment without waiting for step-by-step direction, pausing for owner decisions when an action is destructive, costly, or outside the approved scope.

## Success criterion

Progress means measurable improvement in Rook-generated language and multi-turn coherence from Rook's own trained weights. A dashboard, a prompt, a pretrained API response, or a successful code change alone is not evidence that Rook has improved.
