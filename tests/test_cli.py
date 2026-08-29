from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
from pathlib import Path

import pytest
from atif import Trajectory

from harnesshop.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "codex_rollout.jsonl"


def test_cli_converts_codex_rollout_to_valid_atif(tmp_path: Path) -> None:
    output = tmp_path / "session.atif.json"

    code = main(
        [
            "convert",
            "--from",
            "codex",
            "--to",
            "atif",
            str(FIXTURE),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    trajectory = Trajectory.model_validate_json(output.read_text(encoding="utf-8"))
    assert trajectory.session_id == "codex-session-1"
    assert len(trajectory.steps) == 4


def test_cli_can_export_full_audit_history(tmp_path: Path) -> None:
    output = tmp_path / "audit.atif.json"

    code = main(
        [
            "convert",
            "--from",
            "codex",
            "--to",
            "atif",
            str(FIXTURE),
            "--history",
            "audit",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    trajectory = Trajectory.model_validate_json(output.read_text(encoding="utf-8"))
    assert trajectory.extra["harnesshop"]["history_mode"] == "audit"


def test_cli_emits_hermes_dashboard_import_payload(tmp_path: Path) -> None:
    output = tmp_path / "hermes.json"

    code = main(
        [
            "convert",
            "--from",
            "codex",
            "--to",
            "hermes",
            str(FIXTURE),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert list(payload) == ["sessions"]
    assert payload["sessions"][0]["id"] == "codex_codex-session-1-5e369a9076953f97"
    assert len(payload["sessions"][0]["messages"]) == 5


@pytest.mark.parametrize("target", ["atif", "hermes"])
def test_cli_export_atomically_replaces_with_private_permissions(
    tmp_path: Path, target: str
) -> None:
    output = tmp_path / f"session.{target}.json"
    output.write_text("old private transcript", encoding="utf-8")
    output.chmod(0o644)
    previous_inode = output.stat().st_ino

    code = main(
        [
            "convert",
            "--from",
            "codex",
            "--to",
            target,
            str(FIXTURE),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert output.stat().st_ino != previous_inode


def test_cli_export_preserves_existing_file_when_atomic_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    output = tmp_path / "session.atif.json"
    output.write_text("old private transcript", encoding="utf-8")

    def fail_replace(_source: os.PathLike[str], _destination: os.PathLike[str]) -> None:
        raise PermissionError("replacement denied")

    monkeypatch.setattr(os, "replace", fail_replace)

    code = main(
        [
            "convert",
            "--from",
            "codex",
            "--to",
            "atif",
            str(FIXTURE),
            "--output",
            str(output),
        ]
    )

    assert code == 2
    assert "replacement denied" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "old private transcript"
    assert list(tmp_path.iterdir()) == [output]


def test_hermes_import_is_dry_run_unless_apply_is_explicit(
    tmp_path: Path, capsys
) -> None:
    hermes_home = tmp_path / "hermes-home"

    code = main(
        [
            "import",
            "--from",
            "codex",
            "--to",
            "hermes",
            str(FIXTURE),
            "--hermes-home",
            str(hermes_home),
        ]
    )

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "apply": False,
        "messages": 5,
        "sessions": 1,
        "target": str(hermes_home / "state.db"),
    }
    assert not (hermes_home / "state.db").exists()


def test_hermes_import_defaults_to_active_hermes_home_environment(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    hermes_home = tmp_path / "active-profile"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    code = main(
        [
            "import",
            "--from",
            "codex",
            "--to",
            "hermes",
            str(FIXTURE),
        ]
    )

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["target"] == str(hermes_home / "state.db")


def test_cli_can_preserve_source_model_in_hermes_projection(tmp_path: Path) -> None:
    output = tmp_path / "hermes-source-model.json"

    code = main(
        [
            "convert",
            "--from",
            "codex",
            "--to",
            "hermes",
            str(FIXTURE),
            "--preserve-source-model",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sessions"][0]["model"] == "gpt-5-codex"


def test_active_directory_conversion_uses_codex_current_rollout_index(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex"
    sessions_dir = codex_home / "sessions" / "2026" / "08" / "01"
    sessions_dir.mkdir(parents=True)
    old_path = sessions_dir / "rollout-old.jsonl"
    current_path = sessions_dir / "rollout-current.jsonl"
    old_path.write_text(_minimal_rollout("same-thread", "old message"), encoding="utf-8")
    current_path.write_text(
        _minimal_rollout("same-thread", "current message"), encoding="utf-8"
    )
    db = sqlite3.connect(codex_home / "state_5.sqlite")
    db.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL)")
    db.execute("INSERT INTO threads VALUES (?, ?)", ("same-thread", str(current_path)))
    db.commit()
    db.close()
    output = tmp_path / "active.atif.json"

    code = main(
        [
            "convert",
            "--from",
            "codex",
            "--to",
            "atif",
            str(codex_home),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    trajectory = Trajectory.model_validate_json(output.read_text(encoding="utf-8"))
    assert [step.message for step in trajectory.steps] == ["current message"]


@pytest.mark.parametrize("index_state", ["corrupt", "empty", "unmatched"])
def test_active_directory_conversion_fails_closed_for_unusable_index(
    tmp_path: Path, index_state: str
) -> None:
    codex_home = tmp_path / ".codex"
    sessions_dir = codex_home / "sessions"
    sessions_dir.mkdir(parents=True)
    rollout = sessions_dir / "rollout-current.jsonl"
    rollout.write_text(_minimal_rollout("thread", "message"), encoding="utf-8")
    state_db = codex_home / "state_9.sqlite"
    if index_state == "corrupt":
        state_db.write_bytes(b"not a SQLite database")
    else:
        db = sqlite3.connect(state_db)
        db.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL)")
        if index_state == "unmatched":
            db.execute("INSERT INTO threads VALUES (?, ?)", ("other", "missing.jsonl"))
        db.commit()
        db.close()

    code = main(
        [
            "convert",
            "--from",
            "codex",
            "--to",
            "atif",
            str(codex_home),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )

    assert code == 2
    assert not (tmp_path / "out.json").exists()


def test_current_rollout_index_uses_uri_safe_sqlite_path(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex?home#profile"
    sessions_dir = codex_home / "sessions"
    sessions_dir.mkdir(parents=True)
    rollout = sessions_dir / "rollout-current.jsonl"
    rollout.write_text(_minimal_rollout("thread", "message"), encoding="utf-8")
    db = sqlite3.connect(codex_home / "state_1.sqlite")
    db.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL)")
    db.execute("INSERT INTO threads VALUES (?, ?)", ("thread", str(rollout)))
    db.commit()
    db.close()

    output = tmp_path / "out.json"
    code = main(
        [
            "convert",
            "--from",
            "codex",
            "--to",
            "atif",
            str(codex_home),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert output.exists()


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("worker failed"), subprocess.TimeoutExpired("worker", 1.0)],
)
def test_cli_reports_runtime_and_timeout_import_failures_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, failure: Exception
) -> None:
    import harnesshop.hermes_runtime as hermes_runtime

    monkeypatch.setattr(
        hermes_runtime,
        "import_sessions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    code = main(
        [
            "import",
            "--from",
            "codex",
            "--to",
            "hermes",
            str(FIXTURE),
            "--hermes-home",
            str(tmp_path / "home"),
            "--apply",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "harnesshop:" in captured.err
    assert "Traceback" not in captured.err


def _minimal_rollout(session_id: str, message: str) -> str:
    records = [
        {
            "timestamp": "2026-08-01T00:00:00Z",
            "type": "session_meta",
            "payload": {"id": session_id, "cli_version": "test"},
        },
        {
            "timestamp": "2026-08-01T00:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": message}],
            },
        },
    ]
    return "\n".join(json.dumps(record) for record in records) + "\n"
