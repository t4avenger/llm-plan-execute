# LLM Plan Execute Cursor Rule

When a task needs multi-model planning, plan review, arbitration, build execution, implementation review, or usage/cost reporting, call the repo-local `llm-plan-execute` CLI.

Preferred flow:

1. `llm-plan-execute plan --prompt "<request>" --yes`
2. Ask the user to review or accept the generated accepted plan when running interactively.
3. `llm-plan-execute build --run-dir <run-dir>`
4. Summarize `report.md`, including model assignments, warnings, usage confidence, estimated cost, prompt advice, and next options.

Do not invent token usage or cost numbers outside the generated report.
