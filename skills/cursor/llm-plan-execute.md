# LLM Plan Execute Cursor Rule

When a task needs multi-model planning, plan review, arbitration, build execution, implementation review, or usage/cost reporting, call the repo-local `llm-plan-execute` CLI.

Preferred flow:

1. Ask clarifying questions before invoking the CLI when the request is ambiguous.
2. In an interactive terminal, run `llm-plan-execute run --prompt "<request>"` so planning, approval, build, review, and reporting stay in one smooth CLI flow.
3. In non-interactive agent orchestration, run `llm-plan-execute plan --prompt "<request>"`, show the user the proposed plan and report artifacts, then run `llm-plan-execute accept --run-dir <run-dir>` and `llm-plan-execute build --run-dir <run-dir>` only after approval.
4. Summarize `report.md`, including model assignments, warnings, usage confidence, estimated cost, prompt advice, and next options.

Do not invent token usage or cost numbers outside the generated report.
Use `plan --yes` only for deliberate non-interactive automation that should bypass manual plan review.
