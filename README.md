# HarnessHop

Move coding-agent sessions between harnesses without flattening them into a prose summary.

HarnessHop currently converts OpenAI Codex rollout history to the official Agent Trajectory Interchange Format (ATIF v1.7) and imports it into Hermes Agent through Hermes's own `SessionDB.import_sessions()` API. It is local-first: historical tool calls are parsed as data and are never executed.

[![CI](https://github.com/Valdemar-Yu/harnesshop/actions/workflows/ci.yml/badge.svg)](https://github.com/Valdemar-Yu/harnesshop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

中文简介：HarnessHop 把 Codex 的本地会话解析为标准 ATIF，或安全导入 Hermes。默认只预检，不写 Hermes；只有显式传入 `--apply` 才会调用 Hermes 官方 SessionDB。

## Why

Coding harnesses persist rich context—user and assistant messages, tool calls and results, reasoning summaries, attachments, timestamps, model metadata, Git state, compaction checkpoints, and branches—in incompatible formats. Copying a final summary loses the structure required for search, inspection, and continuation.

HarnessHop uses a portable intermediate representation:

```text
Codex rollout JSONL
        │
        ▼
ATIF v1.7 trajectory
        │
        ├── portable JSON / JSONL
        └── Hermes session import payload
                    │
                    ▼
          SessionDB.import_sessions()
```

ATIF is an existing public specification maintained by the Harbor project; HarnessHop does not invent a competing interchange format.

## What works

| Capability | Status | Notes |
| --- | --- | --- |
| Codex legacy rollout JSONL → ATIF v1.7 | Supported | Verified end to end with Codex 0.142.4; common newer legacy records use explicit loss accounting |
| Codex legacy active-history reconstruction | Supported | Applies `replacement_history` compactions and `thread_rolled_back` events |
| Codex audit export | Supported | Keeps every durable response item from every selected rollout |
| Codex current-rollout selection | Supported | Reads `state_*.sqlite` read-only and honors `threads.rollout_path` |
| Codex self-contained paginated rollout | Supported | Requires complete contiguous ordinals from zero and preserves source ordering |
| Codex paginated `history_base` / bounded lineage | Fail-closed | Rejected until recursive lineage reconstruction is implemented |
| Zstandard-compressed `*.jsonl.zst` rollout | Fail-closed | Detected rather than silently omitted; decompression support is on the roadmap |
| ATIF → Hermes JSON | Supported | Dashboard-compatible `{ "sessions": [...] }` payload |
| Codex → live Hermes import | Supported | Uses the target Hermes installation's own Python and `SessionDB` |
| Claude Code / OpenCode / Gemini / Aider adapters | Roadmap | Contributions welcome |

See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for the versioned support
boundary. Codex rollout JSONL has no schema-version field and evolves between
CLI releases, so HarnessHop distinguishes self-contained paginated files from
lineage segments that would be incomplete on their own.

## Install

Python 3.11 or newer is required.

```bash
# uv tool install from GitHub
uv tool install git+https://github.com/Valdemar-Yu/harnesshop.git

# or develop from source
git clone https://github.com/Valdemar-Yu/harnesshop.git
cd harnesshop
uv sync --extra dev
```

## Quick start

### 1. Export current Codex threads to ATIF

Pass the Codex home, not only the sessions directory, so HarnessHop can use Codex's current-rollout SQLite index:

```bash
harnesshop convert \
  --from codex \
  --to atif \
  ~/.codex \
  --output codex-current.atif.jsonl
```

For one rollout, the output is one formatted ATIF JSON document. For multiple rollouts, it is JSONL with one ATIF trajectory per line.

### 2. Export an audit view

Audit mode does not apply rollback or compaction replacement and includes superseded rollout files:

```bash
harnesshop convert \
  --from codex \
  --to atif \
  ~/.codex \
  --history audit \
  --output codex-audit.atif.jsonl
```

Audit history is portable evidence, not the context you should resume from. Hermes output therefore requires the default `--history active`.

### 3. Generate Hermes-compatible JSON

```bash
harnesshop convert \
  --from codex \
  --to hermes \
  ~/.codex \
  --output hermes-sessions.json
```

The resulting object can be submitted to Hermes Dashboard's session import endpoint or inspected before any write.

### 4. Import into Hermes

Dry-run is the default:

```bash
harnesshop import \
  --from codex \
  --to hermes \
  ~/.codex
```

Apply only after reviewing the dry-run summary:

```bash
harnesshop import \
  --from codex \
  --to hermes \
  ~/.codex \
  --apply
```

By default, imported sessions use the target Hermes profile's current model.
This avoids forcing a Codex-only model onto a different provider. Add
`--preserve-source-model` only when the target Hermes installation can run the
original Codex model.

Import into a named Hermes profile by targeting its home directly:

```bash
harnesshop import \
  --from codex \
  --to hermes \
  ~/.codex \
  --hermes-home ~/.hermes/profiles/work \
  --apply
```

If Hermes is installed in a non-standard location, add `--hermes-install /path/to/hermes-agent`.
An explicit path is authoritative: HarnessHop fails instead of falling back to another installation.

## Safety and fidelity

- Source files and Codex SQLite databases are opened read-only.
- Export files are atomically replaced with owner-only `0600` permissions.
- Historical shell commands, patches, web searches, and tool calls are never executed.
- Hermes writes require the explicit `--apply` flag.
- Hermes writes go through `SessionDB.import_sessions()`; HarnessHop never hand-edits `state.db`.
- Deterministic, bounded `codex_<sanitized-thread-id>-<digest>` IDs make repeated imports idempotent without collapsing distinct source identities; Hermes skips existing IDs.
- The original Codex model stays in target loss metadata; Hermes uses its current model unless `--preserve-source-model` is explicit.
- Parent sessions are ordered before children, and invalid self-parent edges are dropped at the Hermes boundary.
- Imports are split below Hermes's 25 MiB / 50,000-message transaction limits.
- Import batches commit independently through Hermes. If a later batch fails, earlier commits are not rolled back; the JSON result reports each batch plus imported, skipped, verified, and failed session IDs so the operation can be resumed safely.
- Hermes limits one imported session to 5 MiB. ATIF keeps the complete representable content, while the Hermes target truncates oversized message bodies head-and-tail. Every truncation contains a visible marker and `display_metadata.harnesshop` loss metadata.
- Encrypted Codex content cannot be decrypted by HarnessHop. Its presence is counted in `extra.harnesshop.fidelity` rather than misrepresented as plaintext.
- Unsupported records are counted; they are not silently claimed as preserved.
- Unsupported paginated lineage, invalid ordinal sequences, and compressed rollouts stop conversion before any output or Hermes write.

See [docs/FORMAT.md](docs/FORMAT.md) for the exact mapping and loss-accounting fields.

## Privacy warning

Converted transcripts can contain credentials, personal information, proprietary source, local paths, remote URLs, prompts, and complete tool output. HarnessHop does not automatically redact them because silent mutation would undermine migration fidelity.

Do not publish generated transcripts without manual review. Do not attach real private rollout files to public bug reports; create a synthetic minimal fixture instead.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv build
```

All committed fixtures are synthetic. New behavior should be developed test-first, including a failing regression fixture before parser changes.

## Design references

- [ATIF specification](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md)
- [OpenAI Codex rollout persistence](https://github.com/openai/codex/tree/main/codex-rs/rollout)
- [OpenAI Codex persisted history types](https://github.com/openai/codex/tree/main/codex-rs/history)
- [Hermes Agent session storage](https://hermes-agent.nousresearch.com/docs/developer-guide/session-storage)
- [Hermes Agent sessions](https://hermes-agent.nousresearch.com/docs/user-guide/sessions)

## Roadmap

1. Codex paginated lineage reconstruction: ordinals, `history_base`, fork/revert bounds, subagent inherited-history cutoff, and `*.jsonl.zst`.
2. OpenCode import/export adapter, using its first-party round-trip schema.
3. Claude Code JSONL reader with branch/sidechain reconstruction.
4. Gemini CLI recording adapter after a field-level compatibility audit.
5. Aider Markdown reader, explicitly marked low fidelity.
6. Additional native writers only where the target exposes a supported import boundary.

## License

MIT. See [LICENSE](LICENSE).
