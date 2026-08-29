from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-source", required=True, type=Path)
    parser.add_argument("--db-path", required=True, type=Path)
    args = parser.parse_args()

    payload = json.load(sys.stdin)
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError("session payload must be a list of objects")

    sys.path.insert(0, str(args.hermes_source))
    from hermes_state import SessionDB

    db = SessionDB(db_path=args.db_path)
    try:
        result = db.import_sessions(payload)
        if not result.get("ok"):
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 1
        verified_ids, resumable_ids = _verify(db, payload)
        expected_ids = [str(session["id"]) for session in payload]
        result["verified_ids"] = verified_ids
        result["verified"] = verified_ids == expected_ids
        result["resumable"] = resumable_ids == expected_ids
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["verified"] and result["resumable"] else 1
    finally:
        db.close()


def _verify(db: Any, sessions: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    verified: list[str] = []
    resumable: list[str] = []
    for expected in sessions:
        session_id = str(expected["id"])
        stored = db.get_session(session_id)
        messages = db.get_messages(session_id) if stored else None
        matches = bool(
            stored
            and isinstance(messages, list)
            and _canonical_session(stored, expected) == _canonical_session(expected, expected)
            and _canonical_messages(messages, expected.get("messages") or [])
            == _canonical_messages(expected.get("messages") or [], expected.get("messages") or [])
        )
        if matches:
            verified.append(session_id)
        if not matches:
            continue
        if hasattr(db, "get_resume_conversations"):
            model_history, _display_history = db.get_resume_conversations(session_id)
            if isinstance(model_history, list):
                resumable.append(session_id)
        elif hasattr(db, "get_messages_as_conversation"):
            history = db.get_messages_as_conversation(
                session_id,
                repair_alternation=True,
            )
            if isinstance(history, list):
                resumable.append(session_id)
        elif isinstance(messages, list):
            resumable.append(session_id)
    return verified, resumable


_SESSION_FIELDS = (
    "id",
    "source",
    "model",
    "model_config",
    "system_prompt",
    "parent_session_id",
    "started_at",
    "ended_at",
    "end_reason",
    "cwd",
    "git_branch",
    "git_repo_root",
    "title",
)
_MESSAGE_FIELDS = (
    "role",
    "content",
    "timestamp",
    "reasoning_content",
    "tool_calls",
    "tool_call_id",
    "tool_name",
    "display_metadata",
)


def _canonical_session(actual: dict[str, Any], expected: dict[str, Any]) -> str:
    return _canonical_subset(actual, expected, _SESSION_FIELDS)


def _canonical_messages(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> str:
    if len(actual) != len(expected):
        return ""
    projected = [
        {
            field: message.get(field)
            for field in _MESSAGE_FIELDS
            if field in expected[index]
        }
        for index, message in enumerate(actual)
    ]
    return json.dumps(projected, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _canonical_subset(
    actual: dict[str, Any], expected: dict[str, Any], fields: tuple[str, ...]
) -> str:
    projected = {field: actual.get(field) for field in fields if field in expected}
    return json.dumps(projected, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
