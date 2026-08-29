from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from typing import Any

from atif import ContentPart, Step, Trajectory

_HERMES_IMPORT_MAX_SESSION_BYTES = 5 * 1024 * 1024
_HERMES_IMPORT_TARGET_BYTES = _HERMES_IMPORT_MAX_SESSION_BYTES - 4096


def to_hermes_session(
    trajectory: Trajectory,
    *,
    title: str | None = None,
    preserve_source_model: bool = False,
) -> dict[str, Any]:
    """Map an ATIF trajectory to Hermes ``SessionDB.import_sessions`` input."""
    source_id = (
        _source_thread_id(trajectory)
        or trajectory.session_id
        or trajectory.trajectory_id
        or "imported"
    )
    session_id = "codex_" + _safe_id(source_id)
    workspace = _workspace(trajectory)
    timestamps = [
        value
        for value in (_epoch(step.timestamp) for step in trajectory.steps)
        if value is not None
    ]
    now = time.time()
    started_at = min(timestamps) if timestamps else now
    ended_at = max(timestamps) if timestamps else started_at

    payload = {
        "id": session_id,
        "source": "codex" if trajectory.agent.name == "openai-codex" else "import",
        "user_id": None,
        "model": trajectory.agent.model_name if preserve_source_model else None,
        "model_config": None,
        "system_prompt": None,
        "parent_session_id": _parent_session_id(trajectory),
        "started_at": started_at,
        "ended_at": ended_at,
        "end_reason": "imported",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "cwd": workspace.get("cwd"),
        "git_branch": workspace.get("repository", {}).get("branch"),
        "git_repo_root": workspace.get("repository", {}).get("root"),
        "billing_provider": None,
        "billing_base_url": None,
        "billing_mode": None,
        "estimated_cost_usd": None,
        "actual_cost_usd": None,
        "cost_status": None,
        "cost_source": None,
        "pricing_version": None,
        "title": _title(title, source_id),
        "api_call_count": 0,
        "archived": False,
        "messages": _messages(trajectory),
    }
    _fit_hermes_import_limit(
        payload,
        source_model=trajectory.agent.model_name,
        model_policy="source" if preserve_source_model else "target_default",
    )
    return payload


def _fit_hermes_import_limit(
    payload: dict[str, Any],
    *,
    source_model: str | None,
    model_policy: str,
) -> None:
    """Make one session fit Hermes's 5 MiB importer cap, with loss metadata."""
    original_bytes = _json_bytes(payload)
    report = {
        "target": "hermes-session-import",
        "truncated_messages": 0,
        "original_bytes": original_bytes,
        "final_bytes": original_bytes,
        "source_model": source_model,
        "model_policy": model_policy,
    }
    payload["_harnesshop"] = report
    truncated_indexes: set[int] = set()

    while _json_bytes(payload) > _HERMES_IMPORT_TARGET_BYTES:
        candidates: list[tuple[int, int, int]] = []
        priorities = {"tool": 0, "system": 1, "assistant": 2, "user": 3}
        for index, message in enumerate(payload["messages"]):
            content_bytes = _content_bytes(message.get("content"))
            if content_bytes <= 2048:
                continue
            candidates.append(
                (priorities.get(str(message.get("role")), 4), -content_bytes, index)
            )
        if not candidates:
            raise ValueError(
                "Hermes session metadata exceeds the 5 MiB import limit even "
                "after content truncation"
            )

        _priority, negative_size, index = min(candidates)
        message = payload["messages"][index]
        current_bytes = -negative_size
        target_bytes = max(2048, current_bytes // 2)
        metadata = message.get("display_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        harnesshop = metadata.get("harnesshop")
        if not isinstance(harnesshop, dict):
            harnesshop = {}
        original_content_bytes = int(harnesshop.get("original_content_bytes") or current_bytes)
        harnesshop.update(
            {
                "truncated": True,
                "reason": "hermes_import_session_size_limit",
                "original_content_bytes": original_content_bytes,
            }
        )
        metadata["harnesshop"] = harnesshop
        message["display_metadata"] = metadata
        message["content"] = _truncated_content(
            message.get("content"),
            original_content_bytes=original_content_bytes,
            target_bytes=target_bytes,
        )
        truncated_indexes.add(index)

    report["truncated_messages"] = len(truncated_indexes)
    # The decimal width of final_bytes can change its own encoded size. Iterate
    # to a fixed point; the 4 KiB target margin covers any remaining difference.
    for _ in range(3):
        final_bytes = _json_bytes(payload)
        if report["final_bytes"] == final_bytes:
            break
        report["final_bytes"] = final_bytes


def _json_bytes(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _content_bytes(value: Any) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return _json_bytes(value)


def _truncated_content(
    value: Any,
    *,
    original_content_bytes: int,
    target_bytes: int,
) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )
    marker = (
        f"[HarnessHop truncated {original_content_bytes} bytes to satisfy "
        "Hermes's 5 MiB session import limit]"
    )
    body_budget = max(256, target_bytes - len(marker.encode("utf-8")) - 8)
    head_budget = body_budget // 2
    tail_budget = body_budget - head_budget
    head = _utf8_prefix(text, head_budget)
    tail = _utf8_suffix(text, tail_budget)
    return f"{marker}\n{head}\n…\n{tail}"


def _utf8_prefix(value: str, max_bytes: int) -> str:
    return value.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _utf8_suffix(value: str, max_bytes: int) -> str:
    return value.encode("utf-8")[-max_bytes:].decode("utf-8", errors="ignore")


def _messages(trajectory: Trajectory) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for step in trajectory.steps:
        timestamp = _epoch(step.timestamp)
        if step.source in {"system", "user"}:
            message: dict[str, Any] = {
                "role": step.source,
                "content": _hermes_content(step.message),
            }
            if timestamp is not None:
                message["timestamp"] = timestamp
            messages.append(message)
            continue

        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": _hermes_content(step.message),
        }
        if timestamp is not None:
            assistant["timestamp"] = timestamp
        if step.reasoning_content is not None:
            assistant["reasoning_content"] = step.reasoning_content
        if step.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": call.function_name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    },
                }
                for call in step.tool_calls
            ]
        messages.append(assistant)
        _append_observations(messages, step, timestamp)
    return messages


def _append_observations(
    messages: list[dict[str, Any]], step: Step, timestamp: float | None
) -> None:
    if step.observation is None:
        return
    names = {
        call.tool_call_id: call.function_name
        for call in (step.tool_calls or [])
    }
    for result in step.observation.results:
        message: dict[str, Any] = {
            "role": "tool",
            "content": _hermes_content(result.content or ""),
            "tool_call_id": result.source_call_id,
            "tool_name": names.get(result.source_call_id or ""),
        }
        if timestamp is not None:
            message["timestamp"] = timestamp
        messages.append(message)


def _hermes_content(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return value
    parts: list[dict[str, Any]] = []
    for raw in value:
        part = raw if isinstance(raw, ContentPart) else ContentPart.model_validate(raw)
        if part.type == "text":
            parts.append({"type": "text", "text": part.text or ""})
        elif part.type == "image" and part.source is not None:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": part.source.path},
                }
            )
    return parts


def _workspace(trajectory: Trajectory) -> dict[str, Any]:
    extra = trajectory.extra or {}
    harnesshop = extra.get("harnesshop") if isinstance(extra, dict) else None
    workspace = harnesshop.get("workspace") if isinstance(harnesshop, dict) else None
    return workspace if isinstance(workspace, dict) else {}


def _source_thread_id(trajectory: Trajectory) -> str | None:
    extra = trajectory.extra or {}
    harnesshop = extra.get("harnesshop") if isinstance(extra, dict) else None
    source = harnesshop.get("source") if isinstance(harnesshop, dict) else None
    thread_id = source.get("thread_id") if isinstance(source, dict) else None
    return str(thread_id) if thread_id else None


def _parent_session_id(trajectory: Trajectory) -> str | None:
    extra = trajectory.extra or {}
    harnesshop = extra.get("harnesshop") if isinstance(extra, dict) else None
    source = harnesshop.get("source") if isinstance(harnesshop, dict) else None
    parent = (
        source.get("parent_thread_id") or source.get("forked_from_id")
        if isinstance(source, dict)
        else None
    )
    current = source.get("thread_id") if isinstance(source, dict) else None
    if parent == current:
        return None
    return "codex_" + _safe_id(parent) if isinstance(parent, str) and parent else None


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    prefix = (safe or "imported")[:111].rstrip("-.") or "imported"
    return f"{prefix}-{digest}"


def _title(title: str | None, source_id: str) -> str | None:
    if not title:
        return None
    clean = " ".join(title.split())
    suffix = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8]
    return f"[Codex] {clean} ({suffix})"


def _epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
