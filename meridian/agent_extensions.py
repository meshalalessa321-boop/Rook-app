"""Narrowly allow-listed remote execution tools for the Meridian agent."""
import time
from datetime import datetime, timezone, timedelta

WORKFLOW = ".github/workflows/rook-generation-test.yml"
WORKFLOW_ID = "rook-generation-test.yml"

def install(namespace):
    base_specs = namespace["specs"]
    base_execute = namespace["execute"]

    def extended_specs():
        result = base_specs()
        result.extend([
            {
                "type": "function",
                "function": {
                    "name": "run_rook_sample_tests",
                    "description": "Trigger the one approved Rook sampling test workflow on main. It generates sample replies using the current published Rook weights. This is a test run, not training. Requires GitHub Actions write permission.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string"}},
                        "required": ["repo"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_rook_sample_test",
                    "description": "Check the status and actual step results of a run ID returned by run_rook_sample_tests. Only permits the allow-listed Rook sampling workflow on main.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string"},
                            "run_id": {"type": "integer"}
                        },
                        "required": ["repo", "run_id"],
                        "additionalProperties": False
                    }
                }
            }
        ])
        return result

    def extended_execute(name, args, task_id=""):
        if name == "run_rook_sample_tests":
            repo = args.get("repo", "")
            if repo not in namespace["ALLOWED_REPOS"]:
                raise RuntimeError("Repository is not allow-listed")
            started = datetime.now(timezone.utc)
            namespace["gh"](
                "POST",
                "/repos/" + repo + "/actions/workflows/" + WORKFLOW_ID + "/dispatches",
                repo,
                json={"ref": "main"}
            )
            for _ in range(10):
                data = namespace["gh"](
                    "GET",
                    "/repos/" + repo + "/actions/workflows/" + WORKFLOW_ID + "/runs",
                    repo,
                    params={"branch": "main", "event": "workflow_dispatch", "per_page": 10}
                )
                runs = data.get("workflow_runs", [])
                for run in runs:
                    if run.get("path") != WORKFLOW or run.get("head_branch") != "main":
                        continue
                    created_text = run.get("created_at", "")
                    try:
                        created = datetime.fromisoformat(created_text.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if created >= started - timedelta(seconds=5):
                        return {
                            "run_id": run.get("id"),
                            "status": run.get("status"),
                            "conclusion": run.get("conclusion"),
                            "url": run.get("html_url"),
                            "workflow": WORKFLOW,
                            "message": "GitHub accepted the workflow dispatch. Use check_rook_sample_test to verify completion and outcome."
                        }
                time.sleep(2)
            return {
                "dispatched": True,
                "status": "accepted_waiting_for_run_record",
                "url": "https://github.com/" + repo + "/actions/workflows/" + WORKFLOW_ID,
                "workflow": WORKFLOW,
                "message": "GitHub accepted the dispatch, but the run record was not visible yet. Do not claim the test completed; check the workflow page or retry lookup shortly."
            }

        if name == "check_rook_sample_test":
            repo = args.get("repo", "")
            if repo not in namespace["ALLOWED_REPOS"]:
                raise RuntimeError("Repository is not allow-listed")
            run_id = int(args["run_id"])
            run = namespace["gh"]("GET", "/repos/" + repo + "/actions/runs/" + str(run_id), repo)
            if run.get("path") != WORKFLOW or run.get("head_branch") != "main" or run.get("head_repository", {}).get("full_name") != repo:
                raise RuntimeError("Run is not from the allow-listed Rook sampling workflow on main")
            jobs_data = namespace["gh"](
                "GET",
                "/repos/" + repo + "/actions/runs/" + str(run_id) + "/jobs",
                repo,
                params={"per_page": 100}
            )
            jobs = []
            for job in jobs_data.get("jobs", []):
                jobs.append({
                    "name": job.get("name"),
                    "status": job.get("status"),
                    "conclusion": job.get("conclusion"),
                    "steps": [
                        {"name": step.get("name"), "status": step.get("status"), "conclusion": step.get("conclusion")}
                        for step in job.get("steps", [])
                    ]
                })
            return {
                "run_id": run_id,
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "url": run.get("html_url"),
                "created_at": run.get("created_at"),
                "updated_at": run.get("updated_at"),
                "jobs": jobs,
                "message": "Only a completed run with conclusion success proves the workflow passed."
            }

        return base_execute(name, args, task_id)

    namespace["specs"] = extended_specs
    namespace["execute"] = extended_execute
