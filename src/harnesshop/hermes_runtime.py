from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

_IMPORT_BATCH_MAX_SESSIONS = 400
_IMPORT_BATCH_MAX_MESSAGES = 40_000
_IMPORT_BATCH_MAX_BYTES = 24 * 1024 * 1024
_IMPORT_MAX_SESSION_BYTES = 5 * 1024 * 1024


def import_sessions(
    sessions: list[dict[str, Any]],
    *,
    hermes_home: str | Path,
    hermes_install: str | Path | None = None,
    hermes_python: str | Path | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Import through the target Hermes installation and verify readback.

    The payload is sent over stdin and is never placed in a temporary file.
    Hermes commits each batch independently; structured results preserve prior
    commits when a later batch fails so callers can report and resume safely.
    """
    if not sessions:
        return {
            "ok": True,
            "batches": 0,
            "completed_batches": 0,
            "batch_results": [],
            "imported": 0,
            "skipped": 0,
            "detached": 0,
            "imported_ids": [],
            "skipped_ids": [],
            "failed_ids": [],
            "errors": [],
            "verified": True,
            "resumable": True,
            "verified_ids": [],
        }
    install = _find_install(Path(hermes_install).expanduser() if hermes_install else None)
    python = (
        Path(hermes_python).expanduser()
        if hermes_python
        else _python_for_install(install)
    )
    if not python.exists():
        raise ValueError(f"Hermes Python does not exist: {python}")

    home = Path(hermes_home).expanduser()
    worker = Path(__file__).with_name("hermes_worker.py")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["HERMES_HOME"] = str(home)
    batches = _batch_sessions(_parent_first(sessions))
    aggregate: dict[str, Any] = {
        "ok": True,
        "batches": len(batches),
        "completed_batches": 0,
        "batch_results": [],
        "imported": 0,
        "skipped": 0,
        "detached": 0,
        "imported_ids": [],
        "skipped_ids": [],
        "failed_ids": [],
        "errors": [],
        "verified": True,
        "resumable": True,
        "verified_ids": [],
    }
    for batch in batches:
        batch_ids = [str(session.get("id")) for session in batch]
        try:
            result = _run_worker(
                batch,
                python=python,
                worker=worker,
                install=install,
                db_path=home / "state.db",
                env=env,
                timeout=timeout,
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            detail = str(exc)
            aggregate["ok"] = False
            aggregate["verified"] = False
            aggregate["resumable"] = False
            aggregate["failed_ids"].extend(batch_ids)
            aggregate["errors"].append(detail)
            aggregate["batch_results"].append(
                {
                    "ok": False,
                    "verified": False,
                    "resumable": False,
                    "session_ids": batch_ids,
                    "errors": [detail],
                }
            )
            break
        aggregate["batch_results"].append(result)
        aggregate["completed_batches"] += 1
        aggregate["ok"] = aggregate["ok"] and bool(result.get("ok"))
        aggregate["verified"] = aggregate["verified"] and bool(result.get("verified"))
        aggregate["resumable"] = aggregate["resumable"] and bool(result.get("resumable"))
        for key in ("imported", "skipped", "detached"):
            aggregate[key] += int(result.get(key) or 0)
        for key in ("imported_ids", "skipped_ids", "errors", "verified_ids"):
            aggregate[key].extend(result.get(key) or [])
        if not (result.get("ok") and result.get("verified") and result.get("resumable")):
            accounted = {
                str(session_id)
                for key in ("imported_ids", "skipped_ids", "verified_ids")
                for session_id in (result.get(key) or [])
            }
            aggregate["failed_ids"].extend(
                session_id for session_id in batch_ids if session_id not in accounted
            )
            break
    return aggregate


def _run_worker(
    sessions: list[dict[str, Any]],
    *,
    python: Path,
    worker: Path,
    install: Path,
    db_path: Path,
    env: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    payload = json.dumps(sessions, ensure_ascii=False, separators=(",", ":"))
    completed = subprocess.run(
        [
            str(python),
            str(worker),
            "--hermes-source",
            str(install),
            "--db-path",
            str(db_path),
        ],
        input=payload,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()[-1:] or ["unknown worker error"]
            raise RuntimeError(f"Hermes import worker failed: {detail[0]}")
        raise RuntimeError("Hermes import worker returned no result")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Hermes import worker returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Hermes import worker returned a non-object result")
    return result


def _parent_first(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for session in sessions:
        session_id = str(session.get("id"))
        if session_id in by_id:
            raise ValueError(f"duplicate generated Hermes session id: {session_id}")
        by_id[session_id] = session
    ordered: list[dict[str, Any]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(session_id: str) -> None:
        if session_id in visited:
            return
        if session_id in visiting:
            raise ValueError(f"session parent cycle includes {session_id}")
        visiting.add(session_id)
        session = by_id[session_id]
        parent = session.get("parent_session_id")
        if isinstance(parent, str) and parent in by_id:
            visit(parent)
        visiting.remove(session_id)
        visited.add(session_id)
        ordered.append(session)

    for session_id in by_id:
        visit(session_id)
    return ordered


def _batch_sessions(sessions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    current_messages = 0
    for session in sessions:
        encoded_bytes = len(
            json.dumps(session, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        message_count = len(session.get("messages") or [])
        if encoded_bytes > _IMPORT_MAX_SESSION_BYTES:
            raise ValueError(
                f"session {session.get('id')} exceeds Hermes's 5 MiB per-session import limit"
            )
        if message_count > 10_000:
            raise ValueError(
                f"session {session.get('id')} exceeds Hermes's 10,000-message import limit"
            )
        would_overflow = current and (
            len(current) + 1 > _IMPORT_BATCH_MAX_SESSIONS
            or current_bytes + encoded_bytes > _IMPORT_BATCH_MAX_BYTES
            or current_messages + message_count > _IMPORT_BATCH_MAX_MESSAGES
        )
        if would_overflow:
            batches.append(current)
            current = []
            current_bytes = 0
            current_messages = 0
        current.append(session)
        current_bytes += encoded_bytes
        current_messages += message_count
    if current:
        batches.append(current)
    return batches


def _find_install(explicit: Path | None) -> Path:
    if explicit is not None:
        if (explicit / "hermes_state.py").is_file():
            return explicit.resolve()
        raise ValueError(f"invalid explicit Hermes source installation: {explicit}")

    candidates: list[Path] = []
    configured = os.environ.get("HERMES_INSTALL_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path.home() / ".hermes" / "hermes-agent")

    executable = shutil.which("hermes")
    if executable:
        inferred = _install_from_launcher(Path(executable))
        if inferred is not None:
            candidates.append(inferred)

    for candidate in candidates:
        if (candidate / "hermes_state.py").is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise ValueError(f"could not locate a Hermes source installation; searched: {searched}")


def _install_from_launcher(launcher: Path) -> Path | None:
    try:
        text = launcher.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r'["\']([^"\']+/hermes-agent)/venv/(?:bin|Scripts)/python', text)
    return Path(match.group(1)).expanduser() if match else None


def _python_for_install(install: Path) -> Path:
    candidates = [
        install / "venv" / "bin" / "python",
        install / "venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"could not locate the Hermes Python environment under {install}")
