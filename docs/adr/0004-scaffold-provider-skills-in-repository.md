# 0004 Scaffold provider skills in the repository

- Status: Accepted
- Date: 2026-05-10

## Context

The project should support Codex, Claude, Cursor-style agents, and future providers. Installing directly into user home directories during early development would make review harder and could overwrite local user configuration.

## Decision

Scaffold provider skill/instruction files in this repository first. The skill files tell host agents to call the local `llm-plan-execute` CLI and preserve generated run artifacts.

## Consequences

The skill pack is easy to review and version. Users must copy or install the files into provider-specific locations later. A future installer can automate that once the provider layouts are stable.
