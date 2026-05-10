# 0002 Use provider CLIs as the v1 execution boundary

- Status: Accepted
- Date: 2026-05-10

## Context

The workflow needs to coordinate Codex, Claude, Cursor-style, and future provider models. Direct API integrations would require secret handling and separate auth setup, while local CLIs let users keep provider authentication in each provider's own tooling.

## Decision

Use installed provider CLIs as the v1 execution boundary. The orchestrator calls provider commands from local configuration, captures stdout/stderr, records usage when available, and keeps a dry-run provider for tests and demos.

## Consequences

The implementation avoids storing provider API keys. Provider support depends on each CLI's stability and automation behavior, so adapters must isolate provider differences and keep failures reportable. Exact usage capture may be unavailable for some CLIs.
