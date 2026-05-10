# 0006. Use provider-specific command adapters

## Status

Accepted

## Context

Provider CLIs do not share a stable argument contract. A generic command such as `<provider> --model <model>` cannot safely represent Codex, Cursor Agent, Claude, or future tools.

## Decision

Use one adapter per provider name. Codex runs through `codex exec --model <model> --cd <workspace> <prompt>`. Cursor Agent runs through `cursor-agent --print --output-format text --model <model> --workspace <workspace> --trust <prompt>`. Claude remains a disabled config shape until its local CLI contract is validated.

## Consequences

Real provider calls are explicit and testable. Adding or changing a provider now requires a narrow adapter update plus command-construction tests.
