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


def test_approved_changes_are_committed_atomically_without_contents_put(monkeypatch, tmp_path):
    import json
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app import Base, Approval, publish_approved
    repo = "meshalalessa321-boop/Rook-app"
    approval_id = "12345678-1234-5678-1234-567812345678"
    db = create_engine("sqlite:///" + str(tmp_path / "approval.db"))
    Base.metadata.create_all(db)
    local_session = sessionmaker(db, expire_on_commit=False)
    monkeypatch.setattr(meridian, "Session", local_session)
    with local_session() as session:
        session.add(Approval(
            id=approval_id,
            task_id="task-1",
            title="Improve sampler test",
            details=json.dumps({"repo": repo, "summary": "A safe proposal", "files": [
                {"path": "training/test_onnx_sampling.py", "content": "print('test')\n"},
                {"path": "README.md", "content": "# proposal\n"}
            ]}),
            status="publishing"
        ))
        session.commit()
    calls=[]
    def fake_gh(method,path,repo_arg,**kwargs):
        calls.append((method,path,kwargs.get("json")))
        if "/git/ref/heads/meridian/" in path:
            raise RuntimeError("GitHub API 404: Not Found")
        if path.endswith("/git/ref/heads/main"):
            return {"object":{"sha":"base-sha"}}
        if path.endswith("/git/commits/base-sha"):
            return {"tree":{"sha":"base-tree"}}
        if path.endswith("/git/blobs"):
            return {"sha":"blob-"+str(len([c for c in calls if c[1].endswith("/git/blobs")]))}
        if path.endswith("/git/trees"):
            assert len(kwargs["json"]["tree"]) == 2
            return {"sha":"new-tree"}
        if path.endswith("/git/commits"):
            assert "Meridian-Approval: "+approval_id in kwargs["json"]["message"]
            return {"sha":"new-commit"}
        if path.endswith("/git/refs"):
            return {}
        if path.endswith("/pulls") and method=="GET":
            return []
        if path.endswith("/pulls") and method=="POST":
            return {"html_url":"https://github.com/"+repo+"/pull/999"}
        raise AssertionError("Unexpected GitHub request: "+method+" "+path)
    monkeypatch.setattr(meridian, "gh", fake_gh)
    url=publish_approved(approval_id)
    assert url.endswith("/pull/999")
    assert not any(method=="PUT" or "/contents/" in path for method,path,_ in calls)
    assert sum(path.endswith("/git/blobs") for _,path,_ in calls) == 2
    with local_session() as session:
        assert session.get(Approval,approval_id).status == "submitted"
    db.dispose()


def test_memory_source_urls_reject_script_schemes():
    assert meridian.safe_source_url("https://example.com/docs") == "https://example.com/docs"
    assert meridian.safe_source_url("http://example.com") == "http://example.com"
    assert meridian.safe_source_url("javascript:alert(1)") == ""
    assert meridian.safe_source_url("//example.com/path") == ""
    assert meridian.safe_source_url("not a url") == ""


def test_restart_recovery_requeues_tasks_and_unlocks_approvals(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app import Base, Task, Run, Approval, recover_interrupted_work
    db = create_engine("sqlite:///" + str(tmp_path / "recovery.db"))
    Base.metadata.create_all(db)
    local_session = sessionmaker(db, expire_on_commit=False)
    monkeypatch.setattr(meridian, "Session", local_session)
    with local_session() as session:
        task = Task(id="task-running", title="Interrupted task", goal="resume me", status="running")
        run = Run(id="run-running", task_id="task-running", status="running", summary="Was working")
        approval = Approval(id="approval-publishing", task_id="task-running", title="Publishing", details="{}", status="publishing")
        session.add_all([task, run, approval])
        session.commit()
    recover_interrupted_work()
    with local_session() as session:
        assert session.get(Task, "task-running").status == "queued"
        assert session.get(Run, "run-running").status == "interrupted"
        assert session.get(Run, "run-running").finished_at is not None
        assert session.get(Approval, "approval-publishing").status == "pending"
    db.dispose()


def test_task_pause_and_resume_is_persistent(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app import Base, Task, set_task_control
    db = create_engine("sqlite:///" + str(tmp_path / "task-control.db"))
    Base.metadata.create_all(db)
    local_session = sessionmaker(db, expire_on_commit=False)
    monkeypatch.setattr(meridian, "Session", local_session)
    with local_session() as session:
        session.add(Task(id="pause-me", title="Pause me", goal="test", status="queued"))
        session.commit()
    paused=set_task_control("pause-me","pause")
    assert paused["status"]=="paused"
    with pytest.raises(ValueError, match="Only queued tasks"):
        set_task_control("pause-me","pause")
    resumed=set_task_control("pause-me","resume")
    assert resumed["status"]=="queued"
    with local_session() as session:
        assert session.get(Task,"pause-me").status=="queued"
    db.dispose()


def test_chat_agent_uses_tools_and_records_evidence(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app import Base, Task, Run, run_chat_agent
    db = create_engine("sqlite:///" + str(tmp_path / "chat-tools.db"))
    Base.metadata.create_all(db)
    local_session = sessionmaker(db, expire_on_commit=False)
    monkeypatch.setattr(meridian, "Session", local_session)
    call = SimpleNamespace(
        id="call-create-task",
        function=SimpleNamespace(name="create_task", arguments=json.dumps({
            "title":"Inspect tokenizer",
            "goal":"Find whether Rook's tokenizer handles unknown characters.",
            "priority":"normal"
        }))
    )
    first = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))])
    second = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="I queued the tokenizer investigation.", tool_calls=None))])
    responses = iter([first, second])
    monkeypatch.setattr(meridian, "model_call", lambda messages, tools=None: next(responses))
    answer = run_chat_agent([{"role":"user","content":"Inspect the tokenizer"}], "message-1")
    assert "queued" in answer
    with local_session() as session:
        task=session.query(Task).filter_by(title="Inspect tokenizer").one()
        assert task.status=="queued"
        run=session.query(Run).filter_by(task_id="chat:message-1").one()
        assert run.status=="done"
        assert "create_task" in run.evidence
    db.dispose()
