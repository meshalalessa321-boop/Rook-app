import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime, timezone
import pytest
from agent_extensions import install

REPO = "meshalalessa321-boop/Rook-app"
WORKFLOW = ".github/workflows/rook-generation-test.yml"

def namespace_with(fake_gh):
    namespace = {
        "specs": lambda: [],
        "execute": lambda name, args, task_id="": {"fallback": name},
        "ALLOWED_REPOS": {REPO},
        "gh": fake_gh,
    }
    install(namespace)
    return namespace

def test_extension_registers_only_bounded_remote_test_tools():
    ns = namespace_with(lambda *args, **kwargs: {})
    names = {x["function"]["name"] for x in ns["specs"]()}
    assert "run_rook_sample_tests" in names
    assert "check_rook_sample_test" in names

def test_remote_test_dispatch_rejects_other_repositories():
    calls = []
    ns = namespace_with(lambda *args, **kwargs: calls.append(args) or {})
    with pytest.raises(RuntimeError, match="not allow-listed"):
        ns["execute"]("run_rook_sample_tests", {"repo": "attacker/example"})
    assert calls == []

def test_dispatch_returns_real_workflow_run_record():
    def fake_gh(method, path, repo, **kwargs):
        if method == "POST" and path.endswith("/dispatches"):
            return {}
        if path.endswith("/runs"):
            return {"workflow_runs": [{
                "id": 1234,
                "path": WORKFLOW,
                "head_branch": "main",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "queued",
                "html_url": "https://github.com/" + REPO + "/actions/runs/1234",
            }]}
        raise AssertionError("Unexpected API request: " + method + " " + path)
    ns = namespace_with(fake_gh)
    result = ns["execute"]("run_rook_sample_tests", {"repo": REPO})
    assert result["run_id"] == 1234
    assert result["status"] == "queued"
    assert result["url"].endswith("/1234")

def test_run_checker_refuses_arbitrary_workflow_ids():
    def fake_gh(method, path, repo, **kwargs):
        if path.endswith("/actions/runs/5678"):
            return {"path": ".github/workflows/rook-training.yml", "head_branch": "main", "head_repository": {"full_name": REPO}}
        raise AssertionError("It must reject the wrong workflow before querying jobs")
    ns = namespace_with(fake_gh)
    with pytest.raises(RuntimeError, match="not from the allow-listed"):
        ns["execute"]("check_rook_sample_test", {"repo": REPO, "run_id": 5678})
