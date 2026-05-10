# 0008. Validate local provider config

## Status

Accepted

## Context

The CLI depends on local JSON configuration for provider names, commands, models, role assignments, and score metadata. Invalid config previously failed late during model assignment or provider execution.

## Decision

Validate config shape before loading it into runtime objects. The validator checks supported providers, duplicate model ids, role names, score ranges, scalar types, and provider command availability through `llm-plan-execute config validate`.

## Consequences

Users get direct configuration errors before a run starts. Runtime code can assume parsed config objects are structurally valid.
