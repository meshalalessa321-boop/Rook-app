# Rook Engineering Partner

A scheduled engineering agent that uses a separate OpenAI model to inspect the repository and create a focused engineering report for Rook's development. It does not replace Rook's model or modify source code automatically.

## Model choice

Default: `gpt-6.1-sol`, chosen for strong coding/reasoning at substantially lower API rates than the flagship model. Set `ROOK_ENGINEER_MODEL` to another model only after verifying account access and cost.

## Enable it

1. Add an OpenAI API key as a GitHub Actions repository secret named `OPENAI_API_KEY`. Never put the key in code or commit it.
2. Open **Actions → Rook Engineering Partner → Run workflow** to test a single run.
3. Each successful run uploads `rook-engineer-report.md` as an artifact. The workflow is scheduled every six hours.
4. Review the report and any suggested changes before applying them.

The API is billed separately from a ChatGPT subscription. The agent makes one model request per scheduled run, reads a bounded selection of source files, and does not browse the web or execute generated commands. The workflow uses GitHub-hosted runners and ends after the report is produced, so it is a recurring worker, not a single process running continuously.

## Safety and current scope

- Read-only source inspection.
- No automatic commits, pull requests, deployments, or training runs.
- No use of a hosted model as Rook's own inference model.
- The report must distinguish repository evidence from assumptions.
