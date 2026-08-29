from __future__ import annotations

from copy import deepcopy

import pytest

from harnesshop.hermes_worker import _verify


class FakeSessionDB:
    def __init__(self, stored: dict) -> None:
        self.stored = stored

    def get_session(self, _session_id: str) -> dict:
        return self.stored

    def get_messages(self, _session_id: str) -> list[dict]:
        return self.stored["messages"]

    def get_resume_conversations(self, _session_id: str) -> tuple[list[dict], list[dict]]:
        return self.stored["messages"], self.stored["messages"]


@pytest.mark.parametrize("mutation", ["content", "metadata"])
def test_worker_verification_rejects_same_count_stale_sessions(mutation: str) -> None:
    expected = {
        "id": "codex_demo",
        "source": "codex",
        "model": "model-a",
        "parent_session_id": None,
        "started_at": 1.0,
        "cwd": "/workspace/project",
        "messages": [
            {"role": "user", "content": {"text": "hello", "tags": ["a", "b"]}},
            {"role": "assistant", "content": "hi", "reasoning_content": "because"},
        ],
    }
    stored = deepcopy(expected)
    if mutation == "content":
        stored["messages"][1]["content"] = "different"
    else:
        stored["cwd"] = "/other/project"

    verified, resumable = _verify(FakeSessionDB(stored), [expected])

    assert verified == []
    assert resumable == []