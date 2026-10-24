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


def test_persistent_memory_is_loaded_from_database(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app import Base, Memory, recent_memories
    db = create_engine("sqlite:///" + str(tmp_path / "memory.db"))
    Base.metadata.create_all(db)
    local_session = sessionmaker(db, expire_on_commit=False)
    monkeypatch.setattr(meridian, "Session", local_session)
    with local_session() as session:
        session.add(Memory(title="Model decision", content="Rook keeps his own weights.", category="architecture", verified=True))
        session.commit()
    saved = recent_memories(5)
    assert saved[0]["title"] == "Model decision"
    assert saved[0]["verified"] is True
    db.dispose()

def test_daily_model_call_limit_is_persistent_and_checked_before_api(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app import Base, DailyUsage, model_call
    db = create_engine("sqlite:///" + str(tmp_path / "usage.db"))
    Base.metadata.create_all(db)
    local_session = sessionmaker(db, expire_on_commit=False)
    monkeypatch.setattr(meridian, "Session", local_session)
    monkeypatch.setattr(meridian, "MAX_DAILY_MODEL_CALLS", 1)
    with local_session() as session:
        session.add(DailyUsage(day=meridian.now().date().isoformat(), calls=1, input_tokens=0, output_tokens=0))
        session.commit()
    with pytest.raises(RuntimeError, match="daily model-call limit reached"):
        model_call([{"role": "user", "content": "This must not reach the API."}])
    db.dispose()
