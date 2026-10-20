# Rook training status

**Training is paused while the objective is redesigned.** Do not launch the previous worker or training script as a normal training run.

Read [ROOK_TRAINING_OBJECTIVE.md](ROOK_TRAINING_OBJECTIVE.md) before changing the pipeline.

## Why the old experiment is retired

The previous experiment used OpenAssistant conversation trees, formatted examples as `User: ...\nAssistant:`, and computed loss only over assistant-response bytes. This directly optimizes a designated assistant role, which conflicts with Meshal's project constraints. The old checkpoints are preserved as historical artifacts, not as the intended final direction.

The old script and worker now require an explicit `--allow-legacy-assistant-objective` opt-in. The scheduled GitHub Actions training workflow is disabled, so it cannot keep extending the retired objective in the background.

## Replacement requirements

- Start model parameters from random initialization. Do not load pretrained weights.
- Use full-sequence next-token prediction, not a loss mask that privileges a responder role.
- Build a broader, documented and appropriately licensed text mixture; conversation should be one source among many, not an assistant-only foundation.
- Do not inject a hidden identity prompt, prescribed personality, worldview, beliefs, opinions, emotions, or canned responses.
- Track source licenses, deduplication, language distribution, and held-out splits.
- Evaluate general text continuation, context coherence, language drift, and multi-turn continuity without using “assistant helpfulness” as the target identity.

The replacement data pipeline and training script have **not yet been implemented**. Do not represent this pause or the old checkpoint as completion of that work.
