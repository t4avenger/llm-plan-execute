# LLM Plan Execute

For multi-provider planning and build workflows, use the repository CLI rather than running the whole process in one response.

Commands:

- `llm-plan-execute models --dry-run`
- `llm-plan-execute plan --prompt "<request>" --yes`
- `llm-plan-execute build --run-dir <run-dir>`
- `llm-plan-execute report --run-dir <run-dir>`

Use real provider CLI configuration from `.llm-plan-execute/config.json` when available. Use dry-run mode only for validation or when providers are unavailable.

After build review, offer the user these options: fix findings, accept as-is, or return to planning with review feedback.
