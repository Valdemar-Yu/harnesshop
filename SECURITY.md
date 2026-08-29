# Security policy

## Data sensitivity

Harness transcripts routinely contain secrets, personal data, proprietary source, filesystem paths, prompts, and complete command output. HarnessHop processes that data locally but does not automatically make it safe to publish.

- Review generated files before sharing them.
- Do not attach real transcripts to public issues.
- Prefer synthetic fixtures with example.invalid URLs and fake tokens.
- Store exports with permissions appropriate for the source transcript.

## Execution boundary

Historical tools are data only. HarnessHop must never execute recorded commands, patches, MCP calls, URLs, or scripts. Live Hermes writes require `--apply` and use Hermes's own `SessionDB.import_sessions()` implementation.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Include a minimal synthetic reproduction and avoid sending real transcript content or credentials.

Supported security fixes target the latest release on `main`.
