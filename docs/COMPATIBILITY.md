# Compatibility matrix

HarnessHop treats coding-agent history formats as version-sensitive internals.
A successful parse means the declared profile was reconstructed; it does not
mean every future record type is automatically supported.

## Verified targets

| Component | Verified version/profile | Status |
| --- | --- | --- |
| OpenAI Codex CLI | 0.142.4, legacy rollout JSONL | End-to-end parse, ATIF export, and Hermes import verified |
| OpenAI Codex CLI | 0.149.0-alpha.4, self-contained paginated JSONL | Contiguous ordinal projection and Hermes import verified |
| ATIF Python package | 1.7.0 / `ATIF-v1.7` | Current portable output profile |
| Hermes Agent | 0.20.1, `state.db` schema 25 | Import, exact readback, FTS search, and resume verified through `SessionDB.import_sessions()` |
| Python | 3.11, 3.12, 3.13 | Tested in GitHub Actions |

The Codex verifications used local current-rollout selection through the
read-only `state_*.sqlite` index. No real transcript is committed to this
repository.

## Codex schema boundary

Codex rollout JSONL has no explicit schema-version field. Consumers must infer
compatibility from `session_meta.cli_version`, optional fields, aliases, and
`session_meta.history_mode`.

HarnessHop 0.1.1 supports the legacy profile used by Codex 0.142.4 and
self-contained paginated rollouts with a complete ordinal sequence. Supported
behavior includes:

- first-`session_meta` thread identity;
- response messages, developer/system context, images, reasoning summaries;
- function/custom/web/tool-search calls and correlated results;
- timestamp-applicable model and reasoning-effort metadata;
- `replacement_history` compaction and legacy compaction approximation;
- `thread_rolled_back` active-history semantics;
- read-only current-rollout selection from `threads.rollout_path`;
- explicit loss counts for unknown or unrepresentable records.
- ordinal-preserving projection for self-contained paginated files.

An audit of upstream Codex main at commit
[`6478a751fde8884b2fdc76486fe23175a8e795d4`](https://github.com/openai/codex/commit/6478a751fde8884b2fdc76486fe23175a8e795d4)
(latest stable release at that audit: 0.150.1) found a newer paginated profile.
That profile adds ordinals, recursively bounded `history_base` lineage,
revert/fork rollout IDs, inherited subagent prefixes, compressed cold rollouts,
paginated `item_completed` projections, realtime records, and richer
multi-agent data.

HarnessHop supports a paginated rollout when it is self-contained: it has no
external/bounded lineage and every record has a unique contiguous ordinal from
zero. HarnessHop rejects these cases before producing output or writing Hermes:

- unknown source history modes;
- `history_base` lineage;
- ordinal-bounded fork or subagent metadata;
- missing, non-integer, non-zero-based, or noncontiguous paginated ordinals;
- ordinal-bearing legacy rollouts;
- Zstandard-compressed `*.jsonl.zst` files;
- files without a usable `session_meta`.

This is intentional. Parsing only the child segment or silently ignoring a
compressed parent would create a plausible-looking but incomplete resumable
conversation.

## Newer legacy records

Common newer legacy response and event records are parsed tolerantly. Unknown
outer records and unsupported response/event variants increment
`extra.harnesshop.fidelity` counters.

Current `response_item.agent_message` means routed agent-to-agent traffic, not a
normal assistant reply. HarnessHop does not promote it into the main Hermes
conversation. Multi-agent streams require a graph-aware projection before they
can be represented without changing meaning.

Paginated `item_completed` TurnItems are classified by type. Simple display
projections backed by durable history count as duplicates; rich UI records
whose extra metadata is not mapped count as unsupported loss instead of being
misreported as fully duplicated.

## Planned Codex lineage and compression support

External/bounded paginated lineage requires one coherent implementation rather
than independent field parsers:

1. read plain and `*.jsonl.zst` rollout representations;
2. resolve the authoritative selected rollout from the Codex thread index;
3. recursively resolve `history_base` with ordinal and byte-offset bounds;
4. detect missing parents, cycles, invalid offsets, and copied session metadata;
5. apply revert/fork bounds and `subagent_history_start_ordinal`;
6. project paginated `item_completed` records without duplicating response items;
7. group regression fixtures by `cli_version` and history mode.

Until those invariants are implemented and tested together, the adapter remains
fail-closed for paginated rollouts that depend on external or bounded lineage.

## Other harnesses

| Harness | Planned portability mode |
| --- | --- |
| OpenCode | Native round trip through its first-party export/import envelope |
| Claude Code | High-fidelity source archive; branch/sidechain reconstruction; target handoff unless a supported native importer appears |
| Gemini CLI | Source/export adapter after a field-level recording-schema audit |
| Aider | Explicitly low-fidelity Markdown transcript plus repository provenance |

HarnessHop distinguishes native continuation, context handoff, and archive-only
conversion. An adapter must not label a handoff or rendered transcript as a
lossless native resume.
