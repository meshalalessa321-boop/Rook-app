# Meridian Engineering Partner

Meridian is the separate engineering AI assigned to help develop Rook. It is not Rook and must not replace Rook's own randomly initialized model.

## Mission

Every six hours, Meridian inspects the repository, prioritizes Rook's from-scratch training pipeline, and chooses one narrow engineering fix. When justified, it proposes a patch, runs syntax checks and the training unit tests, and includes the diff in the engineering report artifact. It does not push or merge code, deploy, or directly start a Rook training run.

## Model and cost

Default API model: gpt-6.1-sol. Confirm API access before first use. API usage is billed separately from a ChatGPT subscription. Each run makes one model request, reads a bounded set of source files, and may install the training dependencies to run tests.

## Enable it

1. Add an OpenAI API key as a GitHub Actions repository secret named OPENAI_API_KEY. Never put the key in code or commit it.
2. Merge the reviewed Meridian workflow into main.
3. Open Actions, choose Meridian Engineering Partner, and run the workflow once.
4. Each successful run uploads rook-engineer-report.md as an artifact. The report includes the proposed diff when Meridian's patch passes validation.

## Rules

- Rook stays initialized from random weights. Do not swap in a pretrained model.
- Do not optimize Rook as a conventional assistant or train only designated assistant answers.
- Do not inject an identity instruction telling Rook he is an AI, machine, chatbot, or any other prescribed identity.
- Treat training/rook_brain.py and training/ROOK_TRAINING_OBJECTIVE.md as the intended direction. Treat old assistant-only training scripts as legacy.
- Never report training or tests as successful without logs proving it.
- Never modify source corpus/licensing metadata, private correction data, checkpoints, or model binaries automatically.

## Limits

GitHub-hosted runners start each scheduled run and then stop. This is recurring background work, not one process continuously running 24/7. The worker requires the OPENAI_API_KEY secret and a reviewed/merged workflow. It creates reviewable fixes but does not auto-apply them to main.
