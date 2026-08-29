from __future__ import annotations

import json
from pathlib import Path

from atif import Trajectory

from harnesshop.adapters.codex import parse_codex_rollout

FIXTURE = Path(__file__).parent / "fixtures" / "codex_rollout.jsonl"


def test_codex_rollout_maps_messages_tools_and_metadata_to_atif() -> None:
    trajectory = parse_codex_rollout(FIXTURE)

    assert isinstance(trajectory, Trajectory)
    assert trajectory.schema_version.startswith("ATIF-v1.")
    assert trajectory.session_id == "codex-session-1"
    assert trajectory.trajectory_id == "codex:codex-session-1:codex_rollout"
    assert trajectory.agent.name == "openai-codex"
    assert trajectory.agent.version == "0.142.4"
    assert trajectory.agent.model_name == "gpt-5-codex"

    assert [step.source for step in trajectory.steps] == ["system", "user", "agent", "agent"]
    assert trajectory.steps[0].message == "Follow project rules."

    user_message = trajectory.steps[1].message
    assert isinstance(user_message, list)
    assert [part.type for part in user_message] == ["text", "image"]
    assert user_message[0].text == "Run pwd."
    assert user_message[1].source.path == "images/prompt.png"

    action = trajectory.steps[2]
    assert action.reasoning_content == "I should inspect the current directory."
    assert len(action.tool_calls or []) == 1
    assert action.tool_calls[0].tool_call_id == "call_1"
    assert action.tool_calls[0].function_name == "terminal"
    assert action.tool_calls[0].arguments == {"command": "pwd"}
    assert action.observation.results[0].source_call_id == "call_1"
    assert action.observation.results[0].content == "/workspace/demo"

    assert trajectory.steps[3].message == "The cwd is /workspace/demo."
    extension = trajectory.extra["harnesshop"]
    assert extension["workspace"]["cwd"] == "/workspace/demo"
    assert extension["workspace"]["repository"]["branch"] == "main"
    assert extension["source"]["thread_id"] == "codex-session-1"
    assert extension["source"]["run_session_id"] == "codex-session-1"
    assert extension["fidelity"]["duplicate_events_skipped"] == 2
    assert extension["fidelity"]["orphaned_tool_results"] == 0


def test_active_history_applies_rollbacks_while_audit_history_keeps_all_turns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rollback.jsonl"
    records = [
        _record("session_meta", {"id": "rollback", "cli_version": "test"}, 0),
        _response_message("user", "first", 1),
        _response_message("assistant", "first answer", 2),
        _response_message("user", "withdrawn", 3),
        _response_message("assistant", "withdrawn answer", 4),
        _record("event_msg", {"type": "thread_rolled_back", "num_turns": 1}, 5),
        _response_message("user", "replacement", 6),
        _response_message("assistant", "replacement answer", 7),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    active = parse_codex_rollout(path)
    audit = parse_codex_rollout(path, history_mode="audit")

    assert [step.message for step in active.steps] == [
        "first",
        "first answer",
        "replacement",
        "replacement answer",
    ]
    assert active.extra["harnesshop"]["fidelity"]["rolled_back_turns_removed"] == 1
    assert [step.message for step in audit.steps] == [
        "first",
        "first answer",
        "withdrawn",
        "withdrawn answer",
        "replacement",
        "replacement answer",
    ]


def test_active_history_uses_compaction_replacement_history(tmp_path: Path) -> None:
    path = tmp_path / "compacted.jsonl"
    records = [
        _record("session_meta", {"id": "compacted", "cli_version": "test"}, 0),
        _response_message("user", "old prompt", 1),
        _response_message("assistant", "old answer", 2),
        _record(
            "compacted",
            {
                "message": "legacy summary",
                "replacement_history": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "compacted context"}],
                    }
                ],
            },
            3,
        ),
        _response_message("user", "new prompt", 4),
        _response_message("assistant", "new answer", 5),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    trajectory = parse_codex_rollout(path)

    assert [step.message for step in trajectory.steps] == [
        "compacted context",
        "new prompt",
        "new answer",
    ]
    fidelity = trajectory.extra["harnesshop"]["fidelity"]
    assert fidelity["compactions_applied"] == 1
    assert fidelity["legacy_compactions_approximated"] == 0


def test_event_only_dialogue_is_not_dropped_by_other_durable_messages(
    tmp_path: Path,
) -> None:
    path = tmp_path / "event-only.jsonl"
    records = [
        _record("session_meta", {"id": "event-only", "cli_version": "test"}, 0),
        _response_message("user", "durable prompt", 1),
        _event_message("user_message", "event-only follow-up", 2),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    trajectory = parse_codex_rollout(path)

    assert [step.message for step in trajectory.steps] == [
        "durable prompt",
        "event-only follow-up",
    ]


def test_rollback_does_not_resurrect_event_projections(tmp_path: Path) -> None:
    path = tmp_path / "rollback-events.jsonl"
    records = [
        _record("session_meta", {"id": "rollback-events", "cli_version": "test"}, 0),
        _response_message("user", "withdrawn", 1),
        _event_message("user_message", "withdrawn", 1),
        _response_message("assistant", "withdrawn answer", 2),
        _event_message("agent_message", "withdrawn answer", 2),
        _record("event_msg", {"type": "thread_rolled_back", "num_turns": 1}, 3),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    trajectory = parse_codex_rollout(path)

    assert trajectory.steps == []
    fidelity = trajectory.extra["harnesshop"]["fidelity"]
    assert fidelity["rolled_back_turns_removed"] == 1
    assert fidelity["duplicate_events_skipped"] == 2


def test_rollback_deduplicates_event_projections_with_timestamp_skew(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rollback-events-skew.jsonl"
    user_response = _response_message("user", "withdrawn", 1)
    user_response["timestamp"] = "2026-08-01T10:00:01.100Z"
    assistant_response = _response_message("assistant", "withdrawn answer", 2)
    assistant_response["timestamp"] = "2026-08-01T10:00:02.100Z"
    records = [
        _record("session_meta", {"id": "rollback-skew", "cli_version": "test"}, 0),
        user_response,
        _event_message("user_message", "withdrawn", 1),
        assistant_response,
        _event_message("agent_message", "withdrawn answer", 2),
        _record("event_msg", {"type": "thread_rolled_back", "num_turns": 1}, 3),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    trajectory = parse_codex_rollout(path)

    assert trajectory.steps == []
    fidelity = trajectory.extra["harnesshop"]["fidelity"]
    assert fidelity["rolled_back_turns_removed"] == 1
    assert fidelity["duplicate_events_skipped"] == 2


def test_compaction_does_not_resurrect_superseded_event_projections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compaction-events.jsonl"
    records = [
        _record("session_meta", {"id": "compaction-events", "cli_version": "test"}, 0),
        _response_message("user", "superseded", 1),
        _event_message("user_message", "superseded", 1),
        _response_message("assistant", "superseded answer", 2),
        _event_message("agent_message", "superseded answer", 2),
        _record("compacted", {"replacement_history": []}, 3),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    trajectory = parse_codex_rollout(path)

    assert trajectory.steps == []
    assert trajectory.extra["harnesshop"]["fidelity"]["compactions_applied"] == 1


def test_each_agent_step_uses_the_applicable_timestamped_turn_context(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-switch.jsonl"
    records = [
        _record("session_meta", {"id": "model-switch", "cli_version": "test"}, 0),
        _record("turn_context", {"model": "model-a", "effort": "low"}, 1),
        _response_message("assistant", "first answer", 2),
        _record("turn_context", {"model": "model-b", "effort": "high"}, 3),
        _response_message("assistant", "second answer", 4),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    trajectory = parse_codex_rollout(path)

    assert [(step.model_name, step.reasoning_effort) for step in trajectory.steps] == [
        ("model-a", "low"),
        ("model-b", "high"),
    ]


def test_event_only_agent_step_uses_applicable_turn_context(tmp_path: Path) -> None:
    path = tmp_path / "event-model.jsonl"
    records = [
        _record("session_meta", {"id": "event-model", "cli_version": "test"}, 0),
        _record("turn_context", {"model": "model-a", "effort": "low"}, 1),
        _event_message("agent_message", "event answer", 2),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    trajectory = parse_codex_rollout(path)

    assert [(step.model_name, step.reasoning_effort) for step in trajectory.steps] == [
        ("model-a", "low")
    ]


def test_first_session_meta_owns_thread_identity_when_prefix_contains_parent_meta(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fork.jsonl"
    records = [
        _record(
            "session_meta",
            {"id": "child-thread", "session_id": "shared-run", "cli_version": "test"},
            0,
        ),
        _response_message("user", "copied context", 1),
        _record(
            "session_meta",
            {"id": "parent-thread", "session_id": "shared-run", "cli_version": "test"},
            2,
        ),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    trajectory = parse_codex_rollout(path)

    source = trajectory.extra["harnesshop"]["source"]
    assert source["thread_id"] == "child-thread"
    assert source["run_session_id"] == "shared-run"


def _record(record_type: str, payload: dict, second: int) -> dict:
    return {
        "timestamp": f"2026-08-01T10:00:{second:02d}Z",
        "type": record_type,
        "payload": payload,
    }


def _response_message(role: str, text: str, second: int) -> dict:
    content_type = "output_text" if role == "assistant" else "input_text"
    return _record(
        "response_item",
        {
            "type": "message",
            "role": role,
            "content": [{"type": content_type, "text": text}],
        },
        second,
    )


def _event_message(event_type: str, text: str, second: int) -> dict:
    return _record("event_msg", {"type": event_type, "message": text}, second)
