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
                    "name": "run_rook_training",
                    "description": "Dispatch only the repository's approved Rook Brain from-scratch full-sequence language-model training workflow on main. This runs a bounded training job and may consume GitHub Actions minutes. It must not be used before checking that the training corpus and mission constraints are satisfied.",
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
                    "name": "check_rook_training",
                    "description": "Check the status and steps of a run from the allow-listed Rook Brain training workflow on main.",
                    "parameters": {
                        "type": "object",
                        "properties": {"repo": {"type": "string"}, "run_id": {"type": "integer"}},
                        "required": ["repo", "run_id"],
                        "additionalProperties": False
                    }
                }
            },
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
        if name == "run_rook_training":
            repo = args.get("repo", "")
            if repo not in namespace["ALLOWED_REPOS"]:
                raise RuntimeError("Repository is not allow-listed")
            started = datetime.now(timezone.utc)
            namespace["gh"](
                "POST",
                "/repos/" + repo + "/actions/workflows/rook-training.yml/dispatches",
                repo,
                json={"ref": "main"}
            )
            for _ in range(10):
                data = namespace["gh"](
                    "GET",
                    "/repos/" + repo + "/actions/workflows/rook-training.yml/runs",
                    repo,
                    params={"branch": "main", "event": "workflow_dispatch", "per_page": 10}
                )
                for run in data.get("workflow_runs", []):
                    if run.get("path") != ".github/workflows/rook-training.yml" or run.get("head_branch") != "main":
                        continue
                    try:
                        created = datetime.fromisoformat(run.get("created_at", "").replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if created >= started - timedelta(seconds=5):
                        return {"run_id": run.get("id"), "status": run.get("status"), "conclusion": run.get("conclusion"), "url": run.get("html_url"), "workflow": ".github/workflows/rook-training.yml", "message": "Training workflow dispatch accepted; use check_rook_training to verify the run and artifacts. This does not prove the model improved."}
                time.sleep(2)
            return {"dispatched": True, "status": "accepted_waiting_for_run_record", "url": "https://github.com/" + repo + "/actions/workflows/rook-training.yml", "workflow": ".github/workflows/rook-training.yml", "message": "Dispatch accepted but run record is not visible yet. Do not claim training started or finished until verified."}

        if name == "check_rook_training":
            repo = args.get("repo", "")
            if repo not in namespace["ALLOWED_REPOS"]:
                raise RuntimeError("Repository is not allow-listed")
            run_id = int(args["run_id"])
            run = namespace["gh"]("GET", "/repos/" + repo + "/actions/runs/" + str(run_id), repo)
            if run.get("path") != ".github/workflows/rook-training.yml" or run.get("head_branch") != "main" or run.get("head_repository", {}).get("full_name") != repo:
                raise RuntimeError("Run is not from the allow-listed Rook Brain training workflow on main")
            jobs_data = namespace["gh"]("GET", "/repos/" + repo + "/actions/runs/" + str(run_id) + "/jobs", repo, params={"per_page": 100})
            jobs = [{"name": job.get("name"), "status": job.get("status"), "conclusion": job.get("conclusion"), "steps": [{"name": step.get("name"), "status": step.get("status"), "conclusion": step.get("conclusion")} for step in job.get("steps", [])]} for job in jobs_data.get("jobs", [])]
            return {"run_id": run_id, "status": run.get("status"), "conclusion": run.get("conclusion"), "url": run.get("html_url"), "created_at": run.get("created_at"), "updated_at": run.get("updated_at"), "jobs": jobs, "message": "A completed successful run proves only that the workflow completed; inspect report metrics and checkpoint metadata before claiming improvement."}

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
