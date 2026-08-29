# HarnessHop format profile

HarnessHop uses ATIF v1.7 as its portable layer and owns only the `extra.harnesshop` extension namespace.

## Root extension

```json
{
  "extra": {
    "harnesshop": {
      "provenance": {
        "original_format": "openai-codex-rollout-jsonl",
        "source_file": "rollout-....jsonl",
        "converted_by": "harnesshop"
      },
      "history_mode": "active",
      "workspace": {
        "cwd": "/workspace",
        "repository": {
          "branch": "main",
          "commit_hash": "...",
          "repository_url": "..."
        }
      },
      "source": {
        "thread_id": "...",
        "run_session_id": "...",
        "parent_thread_id": "..."
      },
      "fidelity": {}
    }
  }
}
```

`thread_id` is the unique Codex thread identity. `run_session_id` can be shared by child or sibling threads and is not used as a Hermes primary key.

## Codex to ATIF mapping

| Codex durable item | ATIF representation |
| --- | --- |
| `response_item.message` role `developer` / `system` | `Step(source="system")` |
| `response_item.message` role `user` | `Step(source="user")` |
| `response_item.message` role `assistant` | `Step(source="agent")` |
| `response_item.reasoning.summary` | Next agent step's `reasoning_content` |
| `function_call` / `custom_tool_call` | Agent `tool_calls[]` |
| `function_call_output` / `custom_tool_call_output` | Originating call's `observation.results[]` |
| `web_search_call` / `web_search_end` | `web_search` tool call plus correlated observation |
| `tool_search_call` / `tool_search_output` | `tool_search` call plus correlated observation |
| input images | ATIF `ContentPart(type="image")` |
| `turn_context.model` / `effort` | Agent step `model_name` / `reasoning_effort` |
| first `session_meta` | Root agent, workspace, Git, and source metadata |

Codex UI/projection events that duplicate durable response items are counted in `duplicate_events_skipped`.

## Active history

Active mode mirrors Codex's forward model-history semantics:

1. Response items append to active history.
2. A `compacted.replacement_history` list replaces the complete earlier base.
3. Legacy compactions without replacement history retain user boundaries and add the persisted summary; this is explicitly counted as an approximation.
4. `thread_rolled_back.num_turns` removes the newest N user boundaries and every item after each boundary.
5. The first session meta owns the current thread identity; later metas can belong to copied fork/prefix history.

When a Codex home is supplied, `state_*.sqlite` is opened read-only and `threads.rollout_path` selects the current file for each thread.

## Audit history

Audit mode preserves all durable response items and includes superseded rollout files. Each ATIF document has a rollout-specific `trajectory_id`; related documents can share a run-level `session_id`.

Audit mode is intentionally not accepted by the Hermes writer because abandoned and superseded turns should not become live context.

## Fidelity fields

| Field | Meaning |
| --- | --- |
| `source_records_preserved` | Source records represented in ATIF |
| `tool_calls_preserved` | Structured tool calls created |
| `observation_results_preserved` | Results correlated to calls |
| `reasoning_items_preserved` | Plaintext reasoning summaries retained |
| `encrypted_reasoning_items` | Encrypted reasoning items observed but not decrypted |
| `encrypted_content_items` | Other encrypted content observed but not decrypted |
| `duplicate_events_skipped` | Secondary Codex events skipped in favor of durable items |
| `unsupported_source_records` | Unknown top-level records |
| `unsupported_source_items` | Known records with unrepresentable item content |
| `orphaned_tool_results` | Tool results with no durable or reconstructed call |
| `compactions_applied` | Active-history base replacements applied |
| `legacy_compactions_approximated` | Compactions lacking exact replacement history |
| `rolled_back_turns_removed` | User turns removed by rollback |
| `transformations` | Human-readable normalization notes |

## ATIF to Hermes mapping

| ATIF | Hermes import message |
| --- | --- |
| system step | `role="system"` |
| user step | `role="user"` |
| agent step | `role="assistant"` |
| `tool_calls[]` | OpenAI-style `assistant.tool_calls` JSON |
| observation result | `role="tool"`, `tool_call_id`, and `tool_name` |
| `reasoning_content` | Hermes assistant `reasoning_content` |
| text/image parts | OpenAI-style text and `image_url` content parts |

The Hermes payload uses deterministic, bounded
`codex_<sanitized-thread-id>-<sha256-prefix>` session IDs. Parent IDs use the
same mapping. The digest prevents distinct source IDs that sanitize to the same
prefix from collapsing. Invalid self-parent edges are omitted while the
original Codex metadata remains in ATIF.

The original source model is recorded in `_harnesshop.source_model`. The
Hermes `model` column is `NULL` by default so resume uses the target profile's
current model. `--preserve-source-model` opts into storing the Codex model and
sets `_harnesshop.model_policy` to `source`.

## Hermes target size profile

Hermes v0.20.x limits one imported session to 5 MiB and one import transaction to 25 MiB / 50,000 messages. HarnessHop:

- targets slightly below 5 MiB per session;
- truncates the largest tool content first, then system, assistant, and user content only if necessary;
- retains both the head and tail;
- inserts a visible truncation marker;
- writes loss metadata to `display_metadata.harnesshop`;
- batches sessions parent-first below the transaction limits.

The ATIF output is the preservation artifact; the Hermes payload is the target-compatible projection.
