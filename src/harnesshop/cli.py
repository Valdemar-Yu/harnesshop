from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from atif import Trajectory

from harnesshop.adapters.codex import parse_codex_rollout
from harnesshop.adapters.hermes import to_hermes_session


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "convert":
            return _convert(args)
        if args.command == "import":
            return _import(args)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"harnesshop: {exc}", file=sys.stderr)
        return 2
    parser.print_help(sys.stderr)
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harnesshop",
        description="Move coding-agent sessions through ATIF without executing historical tools.",
    )
    parser.add_argument("--version", action="version", version="harnesshop 0.1.0")
    subparsers = parser.add_subparsers(dest="command")

    convert = subparsers.add_parser(
        "convert", help="Convert session records to ATIF or target JSON"
    )
    _source_target_args(convert, targets=("atif", "hermes"))
    convert.add_argument(
        "--history",
        choices=("active", "audit"),
        default="active",
        help="active applies Codex rollback/compaction; audit keeps every durable response item",
    )
    convert.add_argument(
        "--preserve-source-model",
        action="store_true",
        help="store the Codex model in Hermes instead of using the target's current model",
    )
    convert.add_argument("--output", "-o", required=True, type=Path)

    import_parser = subparsers.add_parser(
        "import", help="Import sessions through a supported target API"
    )
    _source_target_args(import_parser, targets=("hermes",))
    import_parser.add_argument(
        "--hermes-home",
        type=Path,
        default=Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes"),
        help="Target Hermes home (default: ~/.hermes or the active profile home)",
    )
    import_parser.add_argument(
        "--hermes-install",
        type=Path,
        help="Hermes source installation containing hermes_state.py",
    )
    import_parser.add_argument(
        "--preserve-source-model",
        action="store_true",
        help="store the Codex model in Hermes instead of using the target's current model",
    )
    import_parser.add_argument(
        "--apply",
        action="store_true",
        help="Write through Hermes SessionDB.import_sessions (default: dry run)",
    )
    return parser


def _source_target_args(parser: argparse.ArgumentParser, *, targets: tuple[str, ...]) -> None:
    parser.add_argument("--from", dest="source", choices=("codex",), required=True)
    parser.add_argument("--to", dest="target", choices=targets, required=True)
    parser.add_argument(
        "input",
        type=Path,
        help="A rollout JSONL file or directory containing rollouts",
    )


def _convert(args: argparse.Namespace) -> int:
    if args.target == "hermes" and args.history != "active":
        raise ValueError("Hermes output requires --history active")
    trajectories = _load_trajectories(
        args.source,
        args.input,
        history_mode=args.history,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.target == "atif":
        _write_atif(args.output, trajectories)
    else:
        sessions = [
            _hermes_payload(
                trajectory,
                preserve_source_model=args.preserve_source_model,
            )
            for trajectory in trajectories
        ]
        _atomic_write_text(
            args.output,
            json.dumps({"sessions": sessions}, ensure_ascii=False, indent=2) + "\n",
        )
    print(
        f"Wrote {len(trajectories)} session(s) to {args.output}",
        file=sys.stderr,
    )
    return 0


def _import(args: argparse.Namespace) -> int:
    trajectories = _load_trajectories(args.source, args.input)
    sessions = [
        _hermes_payload(
            trajectory,
            preserve_source_model=args.preserve_source_model,
        )
        for trajectory in trajectories
    ]
    report = {
        "apply": bool(args.apply),
        "messages": sum(len(session["messages"]) for session in sessions),
        "sessions": len(sessions),
        "target": str(args.hermes_home.expanduser() / "state.db"),
    }
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0

    from harnesshop.hermes_runtime import import_sessions

    result = import_sessions(
        sessions,
        hermes_home=args.hermes_home,
        hermes_install=args.hermes_install,
    )
    print(json.dumps({**report, "result": result}, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("ok") and result.get("verified") and result.get("resumable") else 1


def _load_trajectories(
    source: str,
    input_path: Path,
    *,
    history_mode: str = "active",
) -> list[Trajectory]:
    if source != "codex":
        raise ValueError(f"unsupported source: {source}")
    paths = _rollout_paths(input_path, history_mode=history_mode)
    if not paths:
        raise ValueError(f"no Codex rollout JSONL files found under {input_path}")
    return [
        parse_codex_rollout(path, history_mode=history_mode)
        for path in paths
    ]


def _rollout_paths(input_path: Path, *, history_mode: str) -> list[Path]:
    path = input_path.expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        candidates = sorted(path.rglob("rollout-*.jsonl"))
        if history_mode == "audit":
            return candidates
        indexed = _indexed_current_rollouts(path, candidates)
        if indexed is not None:
            return indexed
        _reject_duplicate_thread_rollouts(candidates)
        return candidates
    raise ValueError(f"input does not exist: {path}")


def _indexed_current_rollouts(
    input_path: Path,
    candidates: list[Path],
) -> list[Path] | None:
    codex_home = _find_codex_home(input_path)
    if codex_home is None:
        return None
    state_dbs = list(codex_home.glob("state_*.sqlite"))
    if not state_dbs:
        return None
    state_db = max(state_dbs, key=_state_db_version)
    candidate_map = {path.resolve(): path for path in candidates}
    try:
        connection = sqlite3.connect(state_db.resolve().as_uri() + "?mode=ro", uri=True)
        rows = connection.execute("SELECT rollout_path FROM threads").fetchall()
    except sqlite3.Error as exc:
        raise ValueError(f"could not read Codex rollout index {state_db}: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    if not rows:
        raise ValueError(f"Codex rollout index {state_db} contains no current threads")
    selected: list[Path] = []
    seen: set[Path] = set()
    for (raw_path,) in rows:
        if not isinstance(raw_path, str) or not raw_path:
            continue
        rollout = Path(raw_path).expanduser()
        if not rollout.is_absolute():
            rollout = codex_home / rollout
        resolved = rollout.resolve()
        candidate = candidate_map.get(resolved)
        if candidate is not None and resolved not in seen:
            selected.append(candidate)
            seen.add(resolved)
    if not selected:
        raise ValueError(
            f"Codex rollout index {state_db} does not match any rollout files under {input_path}"
        )
    return sorted(selected)


def _find_codex_home(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if any(candidate.glob("state_*.sqlite")):
            return candidate
    return None


def _state_db_version(path: Path) -> tuple[int, int]:
    match = re.search(r"state_(\d+)\.sqlite$", path.name)
    return (int(match.group(1)) if match else -1, path.stat().st_mtime_ns)


def _reject_duplicate_thread_rollouts(paths: list[Path]) -> None:
    seen: dict[str, Path] = {}
    for path in paths:
        session_id = _rollout_session_id(path)
        if not session_id:
            continue
        previous = seen.get(session_id)
        if previous is not None:
            raise ValueError(
                "multiple rollout files share Codex thread id "
                f"{session_id}; pass the Codex home with state_*.sqlite, a single "
                "rollout file, or use --history audit"
            )
        seen[session_id] = path


def _rollout_session_id(path: Path) -> str | None:
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            payload = record.get("payload")
            if record.get("type") == "session_meta" and isinstance(payload, dict):
                value = payload.get("id") or payload.get("session_id")
                return str(value) if value else None
    return None


def _write_atif(output: Path, trajectories: list[Trajectory]) -> None:
    if len(trajectories) == 1:
        _atomic_write_text(
            output,
            trajectories[0].model_dump_json(indent=2, exclude_none=True) + "\n",
        )
        return
    _atomic_write_text(
        output,
        "".join(
            json.dumps(
                trajectory.model_dump(mode="json", exclude_none=True),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for trajectory in trajectories
        ),
    )


def _atomic_write_text(output: Path, content: str) -> None:
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _hermes_payload(
    trajectory: Trajectory,
    *,
    preserve_source_model: bool = False,
) -> dict:
    return to_hermes_session(
        trajectory,
        title=_first_user_text(trajectory),
        preserve_source_model=preserve_source_model,
    )


def _first_user_text(trajectory: Trajectory) -> str | None:
    for step in trajectory.steps:
        if step.source != "user":
            continue
        if isinstance(step.message, str):
            return step.message[:80]
        text = " ".join(part.text or "" for part in step.message if part.type == "text").strip()
        if text:
            return text[:80]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
