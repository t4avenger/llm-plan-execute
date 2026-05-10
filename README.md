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
uv run llm-plan-execute --dry-run run --prompt "Add a small feature"
uv run llm-plan-execute --dry-run report --run-dir .llm-plan-execute/runs/<run-id>
```

The `run` command handles clarification, planning, plan review, arbitration, inline approval, build, build review, and reporting in one terminal flow. Dry-run mode is deterministic, completes quickly, and never shells out to provider CLIs. Use it only for validation and demos.

Create a local provider config:

```bash
uv run llm-plan-execute init-config
uv run llm-plan-execute config validate
```

Provider configs live at `.llm-plan-execute/config.json` and are intentionally local.
`init-config` enables provider entries whose configured command is found on `PATH` and disables missing commands. For example, a machine with `codex` and `cursor-agent` installed will enable Codex and Cursor, while leaving Claude disabled if `claude` is absent. Cursor's default model is builder-only unless you explicitly add planning or review roles.

Normal CLI usage requires a config file. If `.llm-plan-execute/config.json` is missing, commands such as `models`, `run`, `plan`, and `build` fail with guidance to run `init-config`, pass `--config`, or opt into `--dry-run`.

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

- `plan` completes in milliseconds and produces generic output: you are using `--dry-run` or a config with `"dry_run": true`; remove dry-run mode and validate provider CLI setup.
- `models` reports a missing config: run `llm-plan-execute init-config`, edit `.llm-plan-execute/config.json`, then run `llm-plan-execute config validate`.
- `config validate` reports an unsupported provider, duplicate model id, invalid role, or score outside `1..5`: fix `.llm-plan-execute/config.json`.
- `config validate` reports a missing command: install the provider CLI or set that provider to `"enabled": false`.
- `models` reports no available models: use `--dry-run`, enable at least one installed provider, or add models with roles that match the workflow.
- Provider output is empty with stderr: inspect the run report and the provider's authentication or workspace trust status.

## Workflow

For interactive use, start with:

```bash
uv run llm-plan-execute run --prompt "Add a small feature"
```

Planning first checks whether the request needs clarification. In an interactive terminal, `run` and `plan` ask those questions before drafting. In non-interactive use, they write `00-clarification.md` and exit so the caller can provide a clearer prompt or rerun with `--no-clarify`.

After clarification, planning uses one planning model, two independent reviewer models, and an arbiter model. By default, the arbiter output is written as `04-proposed-plan.md` for review. Accept that exact reviewed plan with:

```bash
uv run llm-plan-execute accept --run-dir .llm-plan-execute/runs/<run-id>
```

Build then uses a speed/accuracy oriented model and repeats review with two separate reviewers plus a summarizing arbiter. The `run` command prompts you to approve, cancel, or save the reviewed plan before build. Use separate `plan`, `accept`, `build`, and `report` commands for scripting or advanced control. Use `plan --yes` only for deliberate non-interactive automation that should accept the reviewed plan immediately.

Progress is written to stderr so stdout stays usable for command output. Pass `--quiet` to suppress progress, or `--verbose` to include provider error details. The final report includes model assignments, fallback warnings, build status, token usage, estimated cost, and prompt improvement advice.

## Using Skills From An Agent

The repository includes optional skill/rule files that tell host agents how to use this CLI instead of trying to simulate the workflow in one response:

- Codex: `skills/codex/SKILL.md`
- Claude: `skills/claude/CLAUDE.md`
- Cursor-style agents: `skills/cursor/llm-plan-execute.md`

Install or reference the file that matches your agent environment according to that agent's local skill/rule mechanism:

- For Codex, copy `skills/codex` into your Codex skills directory as `llm-plan-execute`, or otherwise register `skills/codex/SKILL.md` with the session.
- For Claude, copy the guidance from `skills/claude/CLAUDE.md` into the project or user `CLAUDE.md` that your Claude environment reads.
- For Cursor-style agents, copy `skills/cursor/llm-plan-execute.md` into the rules location your Cursor environment reads, such as a project rules directory.

Once loaded, the agent should clarify ambiguous work, run `llm-plan-execute plan`, show you the proposed plan and report, run `llm-plan-execute accept` only after you approve the plan, then run `build`.

The skills are optional. The CLI works directly from a terminal without any agent.

## Production Readiness

The current implementation is provider-ready for local Codex and Cursor Agent execution with estimated usage reporting:

- `llm-plan-execute` CLI supports `init-config`, `models`, `run`, `plan`, `accept`, `build`, and `report`.
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
