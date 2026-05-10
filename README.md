# llm-plan-execute

Multi-provider planning, build, review, and reporting orchestration for LLM coding agents.

The project provides:

- A Python CLI that assigns available provider models to planning, review, arbitration, build, and reporting roles.
- Repo-local skill templates for Codex, Claude, and Cursor-style agents.
- Run artifacts with draft plans, independent reviews, arbiter decisions, accepted plans, build review summaries, and reports.
- Exact-or-estimated usage and cost reporting, depending on what each provider CLI exposes.

## Quick Start

Run the workflow with simulated providers:

```bash
uv run llm-plan-execute --dry-run models
uv run llm-plan-execute --dry-run plan --prompt "Add a small feature" --yes
uv run llm-plan-execute --dry-run build --run-dir .llm-plan-execute/runs/<run-id>
```

Create a local provider config:

```bash
uv run llm-plan-execute init-config
```

Provider configs live at `.llm-plan-execute/config.json` and are intentionally local.

## Workflow

Planning uses one planning model, two independent reviewer models, and an arbiter model that decides which suggestions enter the accepted plan. Build then uses a speed/accuracy oriented model and repeats review with two separate reviewers plus a summarizing arbiter.

The final report includes model assignments, fallback warnings, token usage, estimated cost, and prompt improvement advice.

## Current Status

The current implementation is a working v1 scaffold:

- `llm-plan-execute` CLI supports `init-config`, `models`, `plan`, `build`, and `report`.
- Dry-run providers allow the full workflow to run without live provider calls.
- Provider CLI adapters are in place for local command execution through configuration.
- Model assignment uses deterministic role scoring with diversity fallback warnings.
- Run artifacts are written as Markdown and JSON under `.llm-plan-execute/runs/`.
- Repo-local skill templates exist for Codex, Claude, and Cursor-style agents.
- Usage/cost reporting is exact when available and estimated otherwise.

Next implementation priorities are provider-specific CLI argument handling, exact usage parsing where providers expose it, interactive plan Q&A, and an installer for provider skill locations.

## Architecture Decisions

Architecture decisions are recorded in [docs/adr](docs/adr/README.md). Add a new ADR whenever a decision changes dependencies, provider boundaries, workflow behavior, security posture, or long-term maintenance expectations.

## Development

Use `uv` for the local environment:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pre-commit install
uv run pre-commit run --all-files
```
