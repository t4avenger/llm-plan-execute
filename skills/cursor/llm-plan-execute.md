# LLM Plan Execute Cursor Rule

When a task needs multi-model planning, plan review, arbitration, build execution, implementation review, or usage/cost reporting, call the repo-local `llm-plan-execute` CLI.

Preferred flow:

1. Ask clarifying questions before invoking the CLI when the request is ambiguous.
2. `llm-plan-execute plan --prompt "<request>"`
3. Show the user the proposed plan and report artifacts.
4. After the user approves the plan, run `llm-plan-execute accept --run-dir <run-dir>`.
5. `llm-plan-execute build --run-dir <run-dir>`
6. Summarize `report.md`, including model assignments, warnings, usage confidence, estimated cost, prompt advice, and next options.

Do not invent token usage or cost numbers outside the generated report.
Use `plan --yes` only for deliberate non-interactive automation that should bypass manual plan review.
