---
name: llm-plan-execute
description: Multi-provider planning, implementation, review, and reporting workflow for coding tasks. Use when a user asks Codex to coordinate multiple models/providers, develop a plan, review plans, arbitrate reviewer suggestions, build from an accepted plan, review implementation output, or report token usage, estimated cost, and prompt improvement advice.
---

# LLM Plan Execute

Use the repo-local `llm-plan-execute` CLI to orchestrate multi-provider work instead of manually simulating the workflow in one model.

## Workflow

1. Run `llm-plan-execute models --dry-run` to verify the orchestrator when provider CLIs are not configured.
2. Ask clarifying questions before invoking the CLI when the request is ambiguous.
3. Run `llm-plan-execute plan --prompt "<request>"` to create clarification, draft plan, plan reviews, arbiter proposal, and report artifacts.
4. Present the proposed plan and report to the user. After the user approves it, run `llm-plan-execute accept --run-dir <run-dir>`.
5. Run `llm-plan-execute build --run-dir <run-dir>` after the plan is accepted.
6. Read `report.md` before responding to the user. Summarize model choices, warnings, costs, and next options.

## Rules

- Prefer real provider CLIs from `.llm-plan-execute/config.json` when available.
- Use `--dry-run` only for testing, demos, or when no provider CLIs are installed.
- Use `plan --yes` only for deliberate non-interactive automation that should bypass manual plan review.
- Preserve run artifacts under `.llm-plan-execute/runs/`.
- If reviewer findings exist after build, present the user with: fix findings, accept as-is, or return to planning.
