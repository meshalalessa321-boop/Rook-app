# Meridian Engineering Partner

Meridian is the separate engineering AI assigned to help develop Rook. It is not Rook and must not replace Rook's own randomly initialized model.

## Mission

Meridian periodically inspects the repository, prioritizes Rook's actual from-scratch language-model training pipeline, identifies concrete blockers, and writes evidence-based engineering reports. It focuses on training correctness, corpus preparation and licensing, evaluation, checkpoint resume, and background reliability.

## Model and cost

Default API model: `gpt-6.1-sol`. Confirm access in the OpenAI API project before first use. API usage is billed separately from a ChatGPT subscription. Each scheduled run makes one model request.

## Enable it

1. Add an OpenAI API key as a GitHub Actions repository secret named `OPENAI_API_KEY`. Never put the key in code or commit it.
2. Open **Actions → Rook Engineering Partner → Run workflow** to test a single run.
3. Each successful run uploads `rook-engineer-report.md` as an artifact.
4. The workflow is scheduled every six hours.

## Project rules Meridian must preserve

- Rook stays initialized from random weights. Do not swap in a pretrained model.
- Do not optimize Rook as a conventional assistant or train only designated assistant answers.
- Do not inject an identity instruction telling Rook he is an AI, a machine, a chatbot, or any other prescribed identity.
- Treat `training/rook_brain.py` and `training/ROOK_TRAINING_OBJECTIVE.md` as the intended replacement direction. Treat old assistant-only training scripts as legacy.
- Never report a training run or test as successful without actual logs/artifacts proving it.

## Current scope and limits

- Read-only source inspection and report generation.
- No automatic code commits, pull requests, deployments, or training runs yet.
- No hosted-model fallback for Rook.
- GitHub-hosted runners start each scheduled run and then stop. This is periodic background work, not a single process continuously running 24/7.
- The API key must be configured before Meridian can make its first report.
