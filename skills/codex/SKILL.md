---
name: llm-plan-execute
description: Multi-provider planning, implementation, review, and reporting workflow for coding tasks. Use when a user asks Codex to coordinate multiple models/providers, develop a plan, review plans, arbitrate reviewer suggestions, build from an accepted plan, review implementation output, or report token usage, estimated cost, and prompt improvement advice.
---

# LLM Plan Execute

Use the repo-local `llm-plan-execute` CLI to orchestrate multi-provider work instead of manually simulating the workflow in one model.

## Workflow

1. Run `llm-plan-execute models --dry-run` to verify the orchestrator when provider CLIs are not configured.
2. Run `llm-plan-execute plan --prompt "<request>" --yes` to create the initial plan, two plan reviews, arbiter decision, accepted plan, and report.
3. Run `llm-plan-execute build --run-dir <run-dir>` after the user accepts the plan.
4. Read `report.md` before responding to the user. Summarize model choices, warnings, costs, and next options.

## Rules

- Prefer real provider CLIs from `.llm-plan-execute/config.json` when available.
- Use `--dry-run` only for testing, demos, or when no provider CLIs are installed.
- Preserve run artifacts under `.llm-plan-execute/runs/`.
- If reviewer findings exist after build, present the user with: fix findings, accept as-is, or return to planning.
