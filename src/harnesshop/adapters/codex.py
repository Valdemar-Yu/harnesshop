from __future__ import annotations

import json
import mimetypes
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from atif import Agent, ContentPart, Observation, ObservationResult, Step, ToolCall, Trajectory


@dataclass
class FidelityReport:
    source_records_preserved: int = 0
    tool_calls_preserved: int = 0
    observation_results_preserved: int = 0
    reasoning_items_preserved: int = 0
    encrypted_reasoning_items: int = 0
    encrypted_content_items: int = 0
    duplicate_events_skipped: int = 0
    unsupported_source_records: int = 0
    unsupported_source_items: int = 0
    orphaned_tool_results: int = 0
    compactions_applied: int = 0
    legacy_compactions_approximated: int = 0
    rolled_back_turns_removed: int = 0
    transformations: list[str] = field(default_factory=list)


def parse_codex_rollout(
    path: str | Path,
    *,
    history_mode: str = "active",
) -> Trajectory:
    """Parse one OpenAI Codex rollout JSONL file into an ATIF trajectory.

    The parser treats historical tool calls as data. It never executes them.
    Response items are the primary source; duplicate ``event_msg`` dialogue
    records are used only as a fallback.
    """
    source_path = Path(path)
    if source_path.name.endswith(".jsonl.zst"):
        raise ValueError(
            "Zstandard-compressed Codex rollouts are not supported yet; "
            "decompress this file or select an uncompressed current rollout"
        )
    records, malformed = _load_records(source_path)
    fidelity = FidelityReport(unsupported_source_records=malformed)
    if history_mode not in {"active", "audit"}:
        raise ValueError("history_mode must be 'active' or 'audit'")

    session_meta: dict[str, Any] = {}
    turn_contexts: list[dict[str, Any]] = []
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if record.get("type") == "session_meta" and not session_meta:
            session_meta = dict(payload)
        elif record.get("type") == "turn_context":
            context = dict(payload)
            context["_harnesshop_timestamp"] = _timestamp(record)
            turn_contexts.append(context)

    if not session_meta:
        raise ValueError(f"Codex rollout {source_path} has no usable session_meta record")
    source_history_mode = session_meta.get("history_mode", "legacy")
    if not isinstance(source_history_mode, str) or source_history_mode not in {
        "legacy",
        "paginated",
    }:
        raise ValueError(f"unknown Codex history_mode: {source_history_mode!r}")
    if session_meta.get("history_base") is not None:
        raise ValueError(
            "paginated Codex lineage in session_meta.history_base is not supported yet"
        )
    if any(
        session_meta.get(field_name) is not None
        for field_name in (
            "forked_from_ordinal_exclusive",
            "subagent_history_start_ordinal",
        )
    ):
        raise ValueError(
            "ordinal-bounded Codex fork or subagent history is not supported yet"
        )
    ordinals = [record.get("ordinal") for record in records]
    if source_history_mode == "paginated":
        if any(type(ordinal) is not int for ordinal in ordinals) or ordinals != list(
            range(len(records))
        ):
            raise ValueError(
                "self-contained paginated Codex rollouts require contiguous ordinals "
                "starting at zero"
            )
    elif any(ordinal is not None for ordinal in ordinals):
        raise ValueError(
            "ordinal-bearing Codex rollouts require paginated history reconstruction"
        )

    active_records = _materialize_response_records(records, fidelity, history_mode)
    response_records = [
        record for record in active_records if record.get("type") == "response_item"
    ]
    event_records = [record for record in active_records if record.get("type") == "event_msg"]
    steps: list[Step] = []
    call_steps: dict[str, Step] = {}
    call_names: dict[str, str] = {}
    pending_reasoning: list[str] = []

    # Durable response items are authoritative. Event records are secondary
    # projections and run afterwards so early completion events can attach to
    # their durable tool calls without depending on file order.
    for record in [*response_records, *event_records]:
        record_type = record.get("type")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            if record_type not in {
                "compacted",
                "inter_agent_communication_metadata",
                "world_state",
            }:
                fidelity.unsupported_source_records += 1
            continue
        timestamp = _timestamp(record)

        if record_type == "response_item":
            item_type = payload.get("type")
            if item_type == "message":
                role = payload.get("role")
                source = {
                    "developer": "system",
                    "system": "system",
                    "user": "user",
                    "assistant": "agent",
                }.get(role)
                if source is None:
                    fidelity.unsupported_source_items += 1
                    continue
                message = _message_content(payload.get("content"), fidelity)
                step_kwargs: dict[str, Any] = {
                    "step_id": len(steps) + 1,
                    "timestamp": timestamp,
                    "source": source,
                    "message": message,
                    "extra": _step_extra(payload, record),
                }
                if source == "agent":
                    reasoning = "\n".join(pending_reasoning).strip()
                    if reasoning:
                        step_kwargs["reasoning_content"] = reasoning
                    pending_reasoning.clear()
                    model, effort = _model_for_timestamp(turn_contexts, timestamp)
                    if model:
                        step_kwargs["model_name"] = model
                    if effort is not None:
                        step_kwargs["reasoning_effort"] = effort
                steps.append(Step(**step_kwargs))
                fidelity.source_records_preserved += 1
                continue

            if item_type == "reasoning":
                summaries = _reasoning_text(payload)
                if summaries:
                    pending_reasoning.extend(summaries)
                    fidelity.reasoning_items_preserved += 1
                if payload.get("encrypted_content"):
                    fidelity.encrypted_reasoning_items += 1
                fidelity.source_records_preserved += 1
                continue

            if item_type in {
                "function_call",
                "custom_tool_call",
                "tool_search_call",
                "web_search_call",
            }:
                call_id, name, arguments = _tool_call(payload, fidelity)
                if not call_id:
                    fidelity.unsupported_source_items += 1
                    continue
                reasoning = "\n".join(pending_reasoning).strip() or None
                pending_reasoning.clear()
                model, effort = _model_for_timestamp(turn_contexts, timestamp)
                step = Step(
                    step_id=len(steps) + 1,
                    timestamp=timestamp,
                    source="agent",
                    model_name=model,
                    reasoning_effort=effort,
                    message="",
                    reasoning_content=reasoning,
                    tool_calls=[
                        ToolCall(
                            tool_call_id=call_id,
                            function_name=name,
                            arguments=arguments,
                            extra={"codex_item_type": item_type},
                        )
                    ],
                    extra=_step_extra(payload, record),
                )
                steps.append(step)
                call_steps[call_id] = step
                call_names[call_id] = name
                fidelity.tool_calls_preserved += 1
                fidelity.source_records_preserved += 1
                continue

            if item_type in {
                "function_call_output",
                "custom_tool_call_output",
                "tool_search_output",
            }:
                call_id = str(payload.get("call_id") or "").strip()
                content = _output_text(payload)
                _attach_observation(
                    call_id,
                    content,
                    call_steps,
                    fidelity,
                    {"codex_item_type": item_type, "tool_name": call_names.get(call_id)},
                )
                fidelity.source_records_preserved += 1
                continue

            if item_type == "agent_message":
                # Current Codex uses this item for routed multi-agent traffic,
                # not a user-visible assistant answer. ATIF v1.7 has no stream
                # graph to represent it without flattening agent boundaries.
                fidelity.unsupported_source_items += 1
                continue

            fidelity.unsupported_source_items += 1
            continue

        if record_type == "event_msg":
            event_type = payload.get("type")
            if event_type in {"user_message", "agent_message"}:
                source = "user" if event_type == "user_message" else "agent"
                message = payload.get("message")
                if not isinstance(message, str):
                    fidelity.unsupported_source_items += 1
                    continue
                step_kwargs: dict[str, Any] = {
                    "step_id": len(steps) + 1,
                    "timestamp": timestamp,
                    "source": source,
                    "message": message,
                    "extra": _event_step_extra(event_type, record),
                }
                if source == "agent":
                    model, effort = _model_for_timestamp(turn_contexts, timestamp)
                    step_kwargs["model_name"] = model
                    step_kwargs["reasoning_effort"] = effort
                steps.append(Step(**step_kwargs))
                fidelity.source_records_preserved += 1
                continue
            if event_type == "web_search_end":
                call_id = str(payload.get("call_id") or "").strip()
                content = _json_text(
                    payload.get("results")
                    if payload.get("results") is not None
                    else {
                        key: payload.get(key)
                        for key in ("query", "action")
                        if payload.get(key) is not None
                    }
                )
                if call_id in call_steps:
                    _attach_observation(
                        call_id,
                        content,
                        call_steps,
                        fidelity,
                        {"codex_event_type": event_type},
                    )
                elif call_id:
                    arguments = {
                        key: payload[key]
                        for key in ("query", "action")
                        if payload.get(key) is not None
                    }
                    model, effort = _model_for_timestamp(turn_contexts, timestamp)
                    step = Step(
                        step_id=len(steps) + 1,
                        timestamp=timestamp,
                        source="agent",
                        model_name=model,
                        reasoning_effort=effort,
                        message="",
                        tool_calls=[
                            ToolCall(
                                tool_call_id=call_id,
                                function_name="web_search",
                                arguments=arguments,
                                extra={"reconstructed_from": "web_search_end"},
                            )
                        ],
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    source_call_id=call_id,
                                    content=content,
                                    extra={"codex_event_type": event_type},
                                )
                            ]
                        ),
                        extra=_event_step_extra(event_type, record),
                    )
                    steps.append(step)
                    call_steps[call_id] = step
                    call_names[call_id] = "web_search"
                    fidelity.tool_calls_preserved += 1
                    fidelity.observation_results_preserved += 1
                fidelity.source_records_preserved += 1
                continue
            if event_type == "item_completed":
                item = payload.get("item")
                item_type = item.get("type") if isinstance(item, dict) else None
                if item_type in {
                    "AgentMessage",
                    "ContextCompaction",
                    "FunctionCallOutput",
                    "Reasoning",
                    "UserMessage",
                }:
                    fidelity.duplicate_events_skipped += 1
                else:
                    fidelity.unsupported_source_items += 1
                continue
            if event_type in {
                "task_started",
                "task_complete",
                "token_count",
                "context_compacted",
                "thread_settings_applied",
                "thread_name_updated",
                "agent_reasoning",
                "mcp_tool_call_end",
                "patch_apply_end",
                "image_generation_end",
                "turn_aborted",
                "thread_rolled_back",
            }:
                if event_type in {
                    "agent_reasoning",
                    "mcp_tool_call_end",
                    "patch_apply_end",
                    "image_generation_end",
                }:
                    fidelity.duplicate_events_skipped += 1
                continue
            fidelity.unsupported_source_items += 1
            continue

        if record_type in {
            "session_meta",
            "turn_context",
            "world_state",
            "compacted",
            "inter_agent_communication_metadata",
        }:
            continue
        fidelity.unsupported_source_records += 1

    if source_history_mode == "paginated":
        steps.sort(
            key=lambda step: (
                _source_ordinal(step) is None,
                _source_ordinal(step),
                step.step_id,
            )
        )
    else:
        steps.sort(key=lambda step: (step.timestamp is None, step.timestamp or "", step.step_id))
    for index, step in enumerate(steps, start=1):
        step.step_id = index

    session_id = str(session_meta.get("session_id") or session_meta.get("id") or source_path.stem)
    version = str(session_meta.get("cli_version") or "unknown")
    model = _latest_model(turn_contexts)
    extension = _extension(source_path, session_meta, fidelity)
    extension["history_mode"] = history_mode
    notes = None
    if fidelity.unsupported_source_records or fidelity.unsupported_source_items:
        notes = "Some Codex records were not representable; inspect extra.harnesshop.fidelity."

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        trajectory_id=f"codex:{session_id}:{source_path.stem}",
        agent=Agent(
            name="openai-codex",
            version=version,
            model_name=model,
            extra={"provider": session_meta.get("model_provider") or "openai"},
        ),
        steps=steps,
        notes=notes,
        extra={"harnesshop": extension},
    )


def _load_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                malformed += 1
    return records, malformed


def _materialize_response_records(
    records: list[dict[str, Any]],
    fidelity: FidelityReport,
    history_mode: str,
) -> list[dict[str, Any]]:
    """Apply Codex's forward history semantics to durable response items.

    A compaction replacement is a complete new history base. A thread rollback
    removes the newest N user-message boundaries and every response item after
    each boundary. ``audit`` mode intentionally bypasses both transforms.
    """
    durable_dialogue = Counter(
        key
        for record in records
        if record.get("type") == "response_item"
        for key in [_dialogue_projection_key(record)]
        if key is not None
    )
    active: list[dict[str, Any]] = []
    for record in records:
        record_type = record.get("type")
        payload = record.get("payload")
        if record_type == "response_item" and isinstance(payload, dict):
            active.append(record)
            continue

        if history_mode == "active" and record_type == "compacted" and isinstance(payload, dict):
            replacement = payload.get("replacement_history")
            if isinstance(replacement, list):
                active = []
                for item in replacement:
                    if not isinstance(item, dict):
                        fidelity.unsupported_source_items += 1
                        continue
                    if item.get("type") == "compaction":
                        if item.get("encrypted_content"):
                            fidelity.encrypted_reasoning_items += 1
                        continue
                    active.append(
                        {
                            "timestamp": record.get("timestamp"),
                            "ordinal": record.get("ordinal"),
                            "type": "response_item",
                            "payload": item,
                        }
                    )
                fidelity.compactions_applied += 1
                _add_transformation_once(
                    fidelity,
                    "Applied Codex replacement_history as the active history base.",
                )
                continue

            # Older Codex rollouts did not persist replacement_history. Mirror
            # their intent without pretending to reproduce opaque prompt bytes:
            # retain user boundaries and add the persisted compaction summary.
            retained_users = [
                item
                for item in active
                if _is_user_history_record(item)
            ]
            summary = payload.get("message")
            active = retained_users
            if isinstance(summary, str) and summary:
                active.append(
                    {
                        "timestamp": record.get("timestamp"),
                        "ordinal": record.get("ordinal"),
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "phase": "compaction_summary",
                            "content": [{"type": "output_text", "text": summary}],
                        },
                    }
                )
            fidelity.compactions_applied += 1
            fidelity.legacy_compactions_approximated += 1
            _add_transformation_once(
                fidelity,
                "Approximated a legacy Codex compaction from retained user "
                "messages and its summary.",
            )
            continue

        if record_type != "event_msg" or not isinstance(payload, dict):
            continue
        if history_mode != "active" or payload.get("type") != "thread_rolled_back":
            projection_key = _dialogue_projection_key(record)
            if projection_key is not None and durable_dialogue[projection_key] > 0:
                durable_dialogue[projection_key] -= 1
                fidelity.duplicate_events_skipped += 1
                continue
            active.append(record)
            continue
        try:
            count = max(0, int(payload.get("num_turns") or 0))
        except (TypeError, ValueError):
            fidelity.unsupported_source_items += 1
            continue
        for _ in range(count):
            boundary = next(
                (
                    index
                    for index in range(len(active) - 1, -1, -1)
                    if _is_user_history_record(active[index])
                ),
                None,
            )
            if boundary is None:
                break
            del active[boundary:]
            fidelity.rolled_back_turns_removed += 1
    return active


def _is_user_history_record(record: dict[str, Any]) -> bool:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return False
    return (
        record.get("type") == "response_item"
        and payload.get("type") == "message"
        and payload.get("role") == "user"
    ) or (
        record.get("type") == "event_msg" and payload.get("type") == "user_message"
    )


def _dialogue_projection_key(record: dict[str, Any]) -> tuple[str, str] | None:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    if record.get("type") == "event_msg":
        source = {"user_message": "user", "agent_message": "agent"}.get(
            payload.get("type")
        )
        message = payload.get("message")
        return (source, message) if source and isinstance(message, str) else None
    if record.get("type") != "response_item" or payload.get("type") != "message":
        return None
    source = {"user": "user", "assistant": "agent"}.get(payload.get("role"))
    content = payload.get("content")
    if not source:
        return None
    if isinstance(content, str):
        return source, content
    if not isinstance(content, list):
        return None
    text = "\n".join(
        str(item.get("text") or "")
        for item in content
        if isinstance(item, dict) and item.get("type") in {"input_text", "output_text", "text"}
    )
    return source, text


def _add_transformation_once(fidelity: FidelityReport, message: str) -> None:
    if message not in fidelity.transformations:
        fidelity.transformations.append(message)


def _message_content(content: Any, fidelity: FidelityReport) -> str | list[ContentPart]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        fidelity.unsupported_source_items += 1
        return ""
    parts: list[ContentPart] = []
    for item in content:
        if not isinstance(item, dict):
            fidelity.unsupported_source_items += 1
            continue
        item_type = item.get("type")
        if item_type in {"input_text", "output_text", "text"}:
            parts.append(ContentPart(type="text", text=str(item.get("text") or "")))
        elif item_type in {"input_image", "image"}:
            image_path = item.get("image_url") or item.get("url") or item.get("path")
            if not isinstance(image_path, str) or not image_path:
                fidelity.unsupported_source_items += 1
                continue
            media_type = mimetypes.guess_type(image_path)[0]
            if media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
                media_type = "image/png"
                fidelity.transformations.append(
                    "Defaulted an image with an unknown MIME type to image/png."
                )
            parts.append(
                ContentPart(
                    type="image",
                    source={"media_type": media_type, "path": image_path},
                )
            )
        elif item_type == "encrypted_content":
            fidelity.encrypted_content_items += 1
        else:
            fidelity.unsupported_source_items += 1
    if not parts:
        return ""
    if all(part.type == "text" for part in parts):
        return "\n".join(part.text or "" for part in parts)
    return parts


def _reasoning_text(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field_name in ("summary", "content"):
        field_value = payload.get(field_name)
        if not isinstance(field_value, list):
            continue
        for item in field_value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                values.append(item["text"])
    return values


def _tool_call(
    payload: dict[str, Any], fidelity: FidelityReport
) -> tuple[str, str, dict[str, Any]]:
    item_type = payload.get("type")
    call_id = str(payload.get("call_id") or payload.get("id") or "").strip()
    if item_type == "web_search_call":
        return call_id, "web_search", _arguments_object(payload.get("action"), "action", fidelity)
    if item_type == "tool_search_call":
        return call_id, "tool_search", _arguments_object(
            payload.get("arguments"), "arguments", fidelity
        )
    name = str(payload.get("name") or "unknown_tool")
    field_name = "input" if item_type == "custom_tool_call" else "arguments"
    return call_id, name, _arguments_object(payload.get(field_name), field_name, fidelity)


def _arguments_object(value: Any, field_name: str, fidelity: FidelityReport) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        fidelity.transformations.append(
            f"Wrapped non-object Codex tool {field_name} in a JSON object."
        )
        return {field_name: value}
    if value is None:
        return {}
    fidelity.transformations.append(
        f"Wrapped non-object Codex tool {field_name} in a JSON object."
    )
    return {field_name: value}


def _output_text(payload: dict[str, Any]) -> str:
    if payload.get("type") == "tool_search_output":
        value = payload.get("tools")
    else:
        value = payload.get("output")
    return value if isinstance(value, str) else _json_text(value)


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _attach_observation(
    call_id: str,
    content: str,
    call_steps: dict[str, Step],
    fidelity: FidelityReport,
    extra: dict[str, Any] | None = None,
) -> None:
    step = call_steps.get(call_id)
    if step is None:
        fidelity.orphaned_tool_results += 1
        return
    if step.observation is None:
        step.observation = Observation(results=[])
    step.observation.results.append(
        ObservationResult(source_call_id=call_id, content=content, extra=extra)
    )
    fidelity.observation_results_preserved += 1


def _timestamp(record: dict[str, Any]) -> str | None:
    value = record.get("timestamp")
    return value if isinstance(value, str) and value else None


def _model_for_timestamp(
    contexts: list[dict[str, Any]], timestamp_value: str | None
) -> tuple[str | None, str | float | int | None]:
    if not contexts:
        return None, None
    context = contexts[-1]
    target = _timestamp_epoch(timestamp_value)
    if target is not None:
        applicable = [
            (context_timestamp, candidate)
            for candidate in contexts
            if (context_timestamp := _timestamp_epoch(candidate.get("_harnesshop_timestamp")))
            is not None
            and context_timestamp <= target
        ]
        if not applicable:
            return None, None
        _, context = max(applicable, key=lambda item: item[0])
    model = context.get("model")
    effort = context.get("effort")
    return (
        str(model) if model else None,
        effort if isinstance(effort, (str, float, int)) else None,
    )


def _timestamp_epoch(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _latest_model(contexts: list[dict[str, Any]]) -> str | None:
    model, _ = _model_for_timestamp(contexts, None)
    return model


def _step_extra(
    payload: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any] | None:
    values = {
        key: payload.get(key)
        for key in ("id", "phase", "status", "author", "recipient")
        if payload.get(key) is not None
    }
    if type(record.get("ordinal")) is int:
        values["source_ordinal"] = record["ordinal"]
    if payload.get("encrypted_content"):
        values["has_encrypted_content"] = True
    return {"harnesshop": {"codex": values}} if values else None


def _event_step_extra(event_type: str, record: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {"event_type": event_type}
    if type(record.get("ordinal")) is int:
        values["source_ordinal"] = record["ordinal"]
    return {"harnesshop": {"codex": values}}


def _source_ordinal(step: Step) -> int | None:
    extra = step.extra
    if not isinstance(extra, dict):
        return None
    harnesshop = extra.get("harnesshop")
    if not isinstance(harnesshop, dict):
        return None
    codex = harnesshop.get("codex")
    if not isinstance(codex, dict):
        return None
    ordinal = codex.get("source_ordinal")
    return ordinal if type(ordinal) is int else None


def _extension(
    source_path: Path,
    session_meta: dict[str, Any],
    fidelity: FidelityReport,
) -> dict[str, Any]:
    extension: dict[str, Any] = {
        "provenance": {
            "original_format": "openai-codex-rollout-jsonl",
            "source_file": source_path.name,
            "converted_by": "harnesshop",
        },
        "fidelity": asdict(fidelity),
    }
    cwd = session_meta.get("cwd")
    git = session_meta.get("git")
    workspace: dict[str, Any] = {}
    if isinstance(cwd, str) and cwd:
        workspace["cwd"] = cwd
    if isinstance(git, dict) and git:
        workspace["repository"] = {
            key: git[key]
            for key in ("branch", "commit_hash", "repository_url")
            if git.get(key) is not None
        }
    if workspace:
        extension["workspace"] = workspace
    extension["source"] = {
        key: session_meta[key]
        for key in (
            "originator",
            "model_provider",
            "source",
            "thread_source",
            "history_mode",
            "parent_thread_id",
            "forked_from_id",
            "agent_nickname",
            "agent_path",
        )
        if session_meta.get(key) is not None
    }
    if session_meta.get("id") is not None:
        extension["source"]["thread_id"] = session_meta["id"]
    if session_meta.get("session_id") is not None:
        extension["source"]["run_session_id"] = session_meta["session_id"]
    return extension
