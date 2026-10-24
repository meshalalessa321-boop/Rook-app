# Meridian

Meridian is a private engineering command center with durable tasks, persistent project memory, chat history, run evidence, owner approvals, and a continuously running worker loop.

## Implemented

- FastAPI dashboard and owner-token-protected API.
- PostgreSQL persistence on the host; SQLite is available for local development.
- Separate records for tasks, memories, messages, run logs, and approvals.
- Worker polling loop, restart recovery for interrupted tasks, bounded tool steps, run evidence, and an explicit waiting-for-model state when no model key is configured, a persisted daily model-call cap, output-token limits, and request timeouts.
- Model tools to read/list files from the allow-listed GitHub repository, search paths, create tasks, save memories, and submit proposed file replacements.
- A narrowly allow-listed remote test tool that can trigger only the existing Rook sample-generation workflow on main and inspect the real GitHub Actions result. It cannot dispatch Rook training or publishing workflows, arbitrary workflows, or shell commands.
- No direct model-controlled repository writes. Owner approval creates a separate branch and draft pull request. There is no automatic merge.
- GitHub and model API credentials are read only from server-side environment secrets.
- Health endpoint, deployment blueprint, and automated safety tests.

## Rook safety boundary

Meridian is an engineering agent, not Rook's model. It must not replace Rook's randomly initialized or trained weights. The repository allow-list defaults to meshalalessa321-boop/Rook-app. The model can inspect repository files, request the specifically approved Rook sampling test, and propose changes for owner review. It cannot directly change source files or merge pull requests. Repository content is untrusted input, never system instructions.

## Deployment

The root render.yaml blueprint describes an always-on web service and a small PostgreSQL database in Singapore. Current Render pricing lists the 0.5 CPU / 512 MB service plan at $7/month; the database is billed separately. Check Render's current database price before applying the Blueprint. Model API usage is separate. Sources: https://render.com/pricing and https://render.com/docs/compute-plans.

The web service costs $7/month and the small Postgres compute plan costs $6/month, for a baseline of about $13/month before optional storage or model usage. The current default model is gpt-5.4-mini. OpenAI's published rates are $0.75 per million input tokens and $4.50 per million output tokens; actual cost depends on task size and model output. The application also caps output tokens per request and defaults to 60 model calls per UTC day. Provider-side budget limits are still recommended. Sources: https://render.com/pricing and https://developers.openai.com/api/docs/models/gpt-5.4-mini.

The code and Blueprint do not create paid resources by themselves. The initial Blueprint flow asks for MERIDIAN_ACCESS_TOKEN, OPENAI_API_KEY, and GITHUB_TOKEN. Set the owner token to a strong random value and save it securely. The code and Blueprint do not create paid resources by themselves. Apply the Blueprint in the intended Render workspace after reviewing the estimated monthly cost.

Server-side environment:
- MERIDIAN_ACCESS_TOKEN: provide a strong random secret during Blueprint creation; it is the token entered on Meridian's login screen.
- OPENAI_API_KEY: model provider secret.
- OPENAI_MODEL: defaults to gpt-5.4-mini; can be changed in server settings.
- MAX_COMPLETION_TOKENS: response-token ceiling (default 2500).
- MAX_DAILY_MODEL_CALLS: daily model request limit (default 60).
- GITHUB_TOKEN: a fine-grained token restricted to this repository. Grant Contents read/write and Pull requests write for owner-approved draft PRs, plus Actions write to dispatch the one approved sampling test. Do not grant repository administration.
- DATABASE_URL: injected from the linked Render Postgres instance.

Never commit actual tokens or paste them into chat. The public /health endpoint only reveals generic process health. All useful API routes require the owner bearer token.

## Local development

Use Python 3.12 or later.

    cd meridian
    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env
    python run.py

Set secret environment variables in your shell or secret store. If OPENAI_API_KEY is absent, Meridian remains available for task and memory management but the worker reports waiting_for_model instead of claiming to execute tasks.

## Checks

    python -m compileall -q app.py agent_extensions.py run.py
    pytest -q tests
    node --check static/app.js

The GitHub Actions workflow runs these checks on relevant pushes and pull requests.
