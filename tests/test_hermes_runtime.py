from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import harnesshop.hermes_runtime as hermes_runtime
from harnesshop.hermes_runtime import import_sessions


def test_runtime_imports_through_target_sessiondb_and_verifies_readback(
    tmp_path: Path,
) -> None:
    install = tmp_path / "hermes-agent"
    install.mkdir()
    (install / "hermes_state.py").write_text(
        """
import json

class SessionDB:
    def __init__(self, db_path):
        self.path = db_path
        self.data = json.loads(db_path.read_text()) if db_path.exists() else {}

    def import_sessions(self, sessions):
        imported, skipped = [], []
        for session in sessions:
            if session[\"id\"] in self.data:
                skipped.append(session[\"id\"])
            else:
                self.data[session[\"id\"]] = session
                imported.append(session[\"id\"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))
        return {
            \"ok\": True,
            \"imported\": len(imported),
            \"skipped\": len(skipped),
            \"detached\": 0,
            \"imported_ids\": imported,
            \"skipped_ids\": skipped,
            \"errors\": [],
        }

    def get_session(self, session_id):
        return self.data.get(session_id)

    def get_messages(self, session_id):
        return self.data[session_id][\"messages\"]

    def get_resume_conversations(self, session_id):
        messages = self.get_messages(session_id)
        return messages, messages

    def close(self):
        pass
""".strip()
        + "\n",
        encoding="utf-8",
    )
    home = tmp_path / "profile"
    sessions = [
        {
            "id": "codex_demo",
            "source": "codex",
            "started_at": 1.0,
            "messages": [
                {"role": "user", "content": "hello", "timestamp": 1.0},
                {"role": "assistant", "content": "hi", "timestamp": 2.0},
            ],
        }
    ]

    result = import_sessions(
        sessions,
        hermes_home=home,
        hermes_install=install,
        hermes_python=Path(sys.executable),
    )

    assert result["ok"] is True
    assert result["imported"] == 1
    assert result["verified"] is True
    assert result["resumable"] is True
    assert result["verified_ids"] == ["codex_demo"]
    stored = json.loads((home / "state.db").read_text(encoding="utf-8"))
    assert stored["codex_demo"]["messages"][0]["content"] == "hello"


def test_runtime_batches_parent_before_child_and_aggregates_verification(
    tmp_path: Path, monkeypatch
) -> None:
    install = tmp_path / "hermes-agent"
    install.mkdir()
    (install / "hermes_state.py").write_text(
        """
import json

class SessionDB:
    def __init__(self, db_path):
        self.path = db_path
        self.data = json.loads(db_path.read_text()) if db_path.exists() else {}

    def import_sessions(self, sessions):
        imported = []
        incoming = {session["id"] for session in sessions}
        detached = 0
        for raw in sessions:
            session = dict(raw)
            parent = session.get("parent_session_id")
            if parent and parent not in self.data and parent not in incoming:
                session["parent_session_id"] = None
                detached += 1
            self.data[session["id"]] = session
            imported.append(session["id"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))
        return {"ok": True, "imported": len(imported), "skipped": 0,
                "detached": detached, "imported_ids": imported,
                "skipped_ids": [], "errors": []}

    def get_session(self, session_id):
        return self.data.get(session_id)

    def get_messages(self, session_id):
        return self.data[session_id]["messages"]

    def get_resume_conversations(self, session_id):
        messages = self.get_messages(session_id)
        return messages, messages

    def close(self):
        pass
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(hermes_runtime, "_IMPORT_BATCH_MAX_BYTES", 170)
    home = tmp_path / "profile"
    parent = {
        "id": "parent",
        "source": "codex",
        "started_at": 1.0,
        "messages": [{"role": "user", "content": "parent"}],
    }
    child = {
        "id": "child",
        "source": "codex",
        "parent_session_id": "parent",
        "started_at": 2.0,
        "messages": [{"role": "user", "content": "child"}],
    }

    result = import_sessions(
        [child, parent],
        hermes_home=home,
        hermes_install=install,
        hermes_python=Path(sys.executable),
    )

    assert result["ok"] is True
    assert result["batches"] == 2
    assert result["imported"] == 2
    assert result["verified_ids"] == ["parent", "child"]
    stored = json.loads((home / "state.db").read_text(encoding="utf-8"))
    assert stored["child"]["parent_session_id"] == "parent"


def test_explicit_hermes_install_is_authoritative_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured-hermes"
    configured.mkdir()
    (configured / "hermes_state.py").write_text("# valid fallback\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_INSTALL_DIR", str(configured))
    explicit = tmp_path / "misspelled-hermes"

    with pytest.raises(
        ValueError,
        match=f"invalid explicit Hermes source installation: {explicit}",
    ):
        hermes_runtime._find_install(explicit)


def test_runtime_rejects_duplicate_generated_session_ids() -> None:
    sessions = [
        {"id": "codex_duplicate", "messages": []},
        {"id": "codex_duplicate", "messages": []},
    ]

    with pytest.raises(ValueError, match="duplicate generated Hermes session id: codex_duplicate"):
        hermes_runtime._parent_first(sessions)


def test_runtime_preserves_structured_prior_batches_when_a_later_batch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install = tmp_path / "hermes-agent"
    install.mkdir()
    (install / "hermes_state.py").write_text("# worker is mocked\n", encoding="utf-8")
    sessions = [
        {"id": "first", "messages": [{"role": "user", "content": "one"}]},
        {"id": "second", "messages": [{"role": "user", "content": "two"}]},
    ]
    monkeypatch.setattr(
        hermes_runtime,
        "_batch_sessions",
        lambda ordered: [[item] for item in ordered],
    )
    calls = 0

    def run_worker(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second batch failed")
        return {
            "ok": True,
            "verified": True,
            "resumable": True,
            "imported": 1,
            "skipped": 0,
            "detached": 0,
            "imported_ids": ["first"],
            "skipped_ids": [],
            "verified_ids": ["first"],
            "errors": [],
        }

    monkeypatch.setattr(hermes_runtime, "_run_worker", run_worker)

    result = import_sessions(
        sessions,
        hermes_home=tmp_path / "home",
        hermes_install=install,
        hermes_python=Path(sys.executable),
    )

    assert result["ok"] is False
    assert result["imported_ids"] == ["first"]
    assert result["verified_ids"] == ["first"]
    assert result["failed_ids"] == ["second"]
    assert result["completed_batches"] == 1
    assert result["batch_results"][0]["imported_ids"] == ["first"]
    assert result["batch_results"][1]["session_ids"] == ["second"]
    assert result["errors"] == ["second batch failed"]
