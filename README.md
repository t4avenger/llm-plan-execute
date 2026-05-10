# llm-plan-execute

Multi-provider planning, build, review, and reporting orchestration for LLM coding agents.

The project provides:

- A Python CLI that assigns available provider models to planning, review, arbitration, build, and reporting roles.
- Repo-local skill templates for Codex, Claude, and Cursor-style agents.
- Run artifacts with draft plans, independent reviews, arbiter decisions, accepted plans, build review summaries, and reports.
- Exact-or-estimated usage and cost reporting, depending on what each provider CLI exposes.

## Quick Start

Install dependencies:

```bash
uv sync --dev
```

Run the workflow with simulated providers:

```bash
uv run llm-plan-execute --dry-run models
uv run llm-plan-execute --dry-run plan --prompt "Add a small feature" --yes
uv run llm-plan-execute --dry-run build --run-dir .llm-plan-execute/runs/<run-id>
```

Dry-run mode is deterministic and never shells out to provider CLIs.

Create a local provider config:

```bash
uv run llm-plan-execute init-config
uv run llm-plan-execute config validate
```

Provider configs live at `.llm-plan-execute/config.json` and are intentionally local.

## Provider Setup

The real-provider path uses explicit adapters for each supported CLI:

```json
{
  "dry_run": false,
  "workspace": ".",
  "runs_dir": ".llm-plan-execute/runs",
  "providers": [
    {
      "name": "codex",
      "command": "codex",
      "enabled": true,
      "models": [
        {
          "name": "gpt-5.4",
          "roles": ["planner", "builder"],
          "reasoning": 4,
          "speed": 4,
          "cost": 4,
          "context": 4,
          "exact_usage": false
        }
      ]
    },
    {
      "name": "cursor",
      "command": "cursor-agent",
      "enabled": true,
      "models": [
        {
          "name": "auto",
          "roles": ["build_reviewer_a"],
          "reasoning": 3,
          "speed": 4,
          "cost": 3,
          "context": 4,
          "exact_usage": false
        }
      ]
    },
    {
      "name": "claude",
      "command": "claude",
      "enabled": false,
      "models": []
    }
  ]
}
```

Codex is invoked as `codex exec --model <model> --cd <workspace> <prompt>`. Cursor Agent is invoked as `cursor-agent --print --output-format text --model <model> --workspace <workspace> --trust <prompt>`. Claude remains a documented extension point, but it should stay disabled unless the `claude` command is installed and an adapter has been validated for the local CLI version.

Common failure modes:

- `config validate` reports an unsupported provider, duplicate model id, invalid role, or score outside `1..5`: fix `.llm-plan-execute/config.json`.
- `config validate` reports a missing command: install the provider CLI or set that provider to `"enabled": false`.
- `models` reports no available models: use `--dry-run`, enable at least one installed provider, or add models with roles that match the workflow.
- Provider output is empty with stderr: inspect the run report and the provider's authentication or workspace trust status.

## Workflow

Planning uses one planning model, two independent reviewer models, and an arbiter model that decides which suggestions enter the accepted plan. Build then uses a speed/accuracy oriented model and repeats review with two separate reviewers plus a summarizing arbiter.

The final report includes model assignments, fallback warnings, token usage, estimated cost, and prompt improvement advice.

## Production Readiness

The current implementation is provider-ready for local Codex and Cursor Agent execution with estimated usage reporting:

- `llm-plan-execute` CLI supports `init-config`, `models`, `plan`, `build`, and `report`.
- Dry-run providers allow the full workflow to run without live provider calls.
- Provider-specific CLI adapters are in place for Codex and Cursor Agent.
- Config validation checks provider names, roles, score ranges, duplicate model ids, shape errors, and provider command availability.
- Model assignment uses deterministic role scoring with diversity fallback warnings.
- Run artifacts are written as Markdown and JSON under `.llm-plan-execute/runs/`.
- Repo-local skill templates exist for Codex, Claude, and Cursor-style agents.
- Usage/cost reporting is exact when available and estimated otherwise.

Known follow-ups:

- Add exact usage parsing per provider when CLIs expose structured usage.
- Add structured JSON output mode for provider calls where available.
- Add release workflow and versioning policy.
- Add coverage reporting and publish coverage into Sonar.
- Add Dependabot or Renovate for pinned actions and Python tooling.
- Add a security policy, contribution guide, and changelog once external contributors are expected.
- Add real integration tests gated behind opt-in environment variables for authenticated provider CLIs.

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

CI is split into `.github/workflows/quality.yml` for lint, format, tests, and pre-commit, and `.github/workflows/build.yml` for Sonar scanning.
