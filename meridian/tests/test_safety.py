import os
import sys
from pathlib import Path

os.environ.setdefault("MERIDIAN_ACCESS_TOKEN", "test-token-that-is-long-enough-for-the-owner")
os.environ.setdefault("WORKER_ENABLED", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import app as meridian

def test_safe_path_rejects_path_traversal():
    with pytest.raises(RuntimeError):
        meridian.safe_path("../secrets.env")
    with pytest.raises(RuntimeError):
        meridian.safe_path("/etc/passwd")

def test_safe_path_rejects_automation_and_deployment_files():
    with pytest.raises(RuntimeError):
        meridian.safe_path(".github/workflows/steal-token.yml")
    with pytest.raises(RuntimeError):
        meridian.safe_path("render.yaml")

def test_safe_path_allows_normal_source():
    assert meridian.safe_path("training/test_onnx_sampling.py") == "training/test_onnx_sampling.py"

def test_agent_tools_do_not_offer_direct_write_or_merge():
    names = {x["function"]["name"] for x in meridian.specs()}
    assert "read_repo_file" in names
    assert "propose_code_changes" in names
    assert "write_repo_file" not in names
    assert "merge_pull_request" not in names

def test_proposal_requires_allowed_repository():
    with pytest.raises(RuntimeError, match="not allow-listed"):
        meridian.execute("propose_code_changes", {
            "repo":"attacker/example",
            "title":"change",
            "summary":"summary",
            "files":[{"path":"README.md","content":"no"}]
        })
