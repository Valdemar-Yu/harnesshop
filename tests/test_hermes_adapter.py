from __future__ import annotations

import hashlib
import json
from pathlib import Path

from harnesshop.adapters.codex import parse_codex_rollout
from harnesshop.adapters.hermes import to_hermes_session

FIXTURE = Path(__file__).parent / "fixtures" / "codex_rollout.jsonl"


def test_atif_trajectory_maps_to_hermes_import_payload() -> None:
    trajectory = parse_codex_rollout(FIXTURE)

    payload = to_hermes_session(trajectory, title="Inspect cwd")

    assert payload["id"] == "codex_codex-session-1-5e369a9076953f97"
    assert payload["source"] == "codex"
    assert payload["model"] is None
    assert payload["_harnesshop"]["source_model"] == "gpt-5-codex"
    assert payload["model_config"] is None
    assert payload["system_prompt"] is None
    assert payload["cwd"] == "/workspace/demo"
    assert payload["git_branch"] == "main"
    assert payload["title"] == "[Codex] Inspect cwd (5e369a90)"
    assert payload["ended_at"] >= payload["started_at"]

    messages = payload["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[0]["content"] == "Follow project rules."
    assert messages[1]["content"] == [
        {"type": "text", "text": "Run pwd."},
        {"type": "image_url", "image_url": {"url": "images/prompt.png"}},
    ]

    action = messages[2]
    assert action["content"] == ""
    assert action["reasoning_content"] == "I should inspect the current directory."
    assert action["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "terminal",
                "arguments": '{"command":"pwd"}',
            },
        }
    ]
    assert messages[3]["tool_call_id"] == "call_1"
    assert messages[3]["tool_name"] == "terminal"
    assert messages[3]["content"] == "/workspace/demo"
    assert messages[4]["content"] == "The cwd is /workspace/demo."


def test_hermes_payload_truncates_oversized_tool_output_with_explicit_loss_metadata() -> None:
    trajectory = parse_codex_rollout(FIXTURE)
    action = trajectory.steps[2]
    action.observation.results[0].content = "start-" + ("x" * (6 * 1024 * 1024)) + "-end"

    payload = to_hermes_session(trajectory)

    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(encoded) <= 5 * 1024 * 1024
    report = payload["_harnesshop"]
    assert report["truncated_messages"] == 1
    assert report["original_bytes"] > report["final_bytes"]
    tool_message = next(message for message in payload["messages"] if message["role"] == "tool")
    assert "[HarnessHop truncated" in tool_message["content"]
    assert tool_message["content"].startswith("[HarnessHop truncated")
    assert tool_message["content"].endswith("-end")
    assert tool_message["display_metadata"]["harnesshop"]["truncated"] is True


def test_hermes_session_id_uses_unique_codex_thread_not_shared_run_id() -> None:
    trajectory = parse_codex_rollout(FIXTURE)
    trajectory.session_id = "shared-run"
    trajectory.extra["harnesshop"]["source"]["thread_id"] = "unique-thread"

    payload = to_hermes_session(trajectory)

    assert payload["id"] == "codex_unique-thread-14df7bce92bdbbef"


def test_hermes_target_drops_codex_self_parent_edge() -> None:
    trajectory = parse_codex_rollout(FIXTURE)
    source = trajectory.extra["harnesshop"]["source"]
    source["parent_thread_id"] = source["thread_id"]

    payload = to_hermes_session(trajectory)

    assert payload["parent_session_id"] is None


def test_hermes_target_can_explicitly_preserve_codex_source_model() -> None:
    trajectory = parse_codex_rollout(FIXTURE)

    payload = to_hermes_session(trajectory, preserve_source_model=True)

    assert payload["model"] == "gpt-5-codex"
    assert payload["_harnesshop"]["model_policy"] == "source"


def test_hermes_ids_disambiguate_sanitizer_collisions_and_remain_bounded() -> None:
    first = parse_codex_rollout(FIXTURE)
    second = parse_codex_rollout(FIXTURE)
    first_source = "thread/a"
    first.extra["harnesshop"]["source"]["thread_id"] = first_source
    digest = hashlib.sha256(first_source.encode()).hexdigest()[:16]
    second.extra["harnesshop"]["source"]["thread_id"] = f"thread-a-{digest}"

    first_id = to_hermes_session(first)["id"]
    second_id = to_hermes_session(second)["id"]

    assert first_id != second_id
    assert len(first_id) <= 134
    assert len(second_id) <= 134
