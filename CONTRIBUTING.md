# Contributing

Contributions are welcome, especially source fixtures, adapter implementations, and documented target import boundaries.

## Rules

- Never commit real private transcripts, credentials, local source code, or copied tool output.
- Build synthetic minimal fixtures that reproduce one schema behavior.
- Treat historical tool calls as inert data. Parser tests must never execute them.
- Do not claim native resume support unless the target provides a verified import boundary.
- Every lossy mapping must update a machine-readable fidelity or target-loss report.
- Follow test-driven development: add a failing behavior test, implement the smallest fix, then refactor.

## Setup

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv build
```

## Adapter checklist

A source adapter should cover:

- stable session/thread identity and parent relationships;
- user, assistant, system/developer, and tool roles;
- structured calls and call-correlated results;
- timestamps and ordering;
- compaction, rollback, branches, and copied context;
- model/provider and workspace/Git metadata;
- multimodal parts and attachments;
- encrypted or opaque content;
- unknown-version behavior and explicit loss accounting.

A target adapter must use a supported target-owned API where one exists. Direct writes to opaque or versioned application databases are not accepted.

## Pull requests

Include:

1. the behavior and source format version;
2. the synthetic fixture or upstream public fixture;
3. tests proving preservation and declared loss;
4. links to first-party documentation or source;
5. the exact verification command and result.
