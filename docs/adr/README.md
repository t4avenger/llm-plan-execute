# Architecture Decision Records

This project uses Architecture Decision Records (ADRs) to capture decisions that materially affect design, workflow, dependencies, security, or long-term maintenance.

## Process

1. Create a new file named `NNNN-short-title.md`.
2. Use the template in `template.md`.
3. Keep each ADR focused on one decision.
4. Mark the status as `Proposed`, `Accepted`, `Superseded`, or `Deprecated`.
5. If a decision changes, add a new ADR and link it from the older one instead of rewriting history.

## Index

- [0001 Use uv for Python project management](0001-use-uv-for-python-project-management.md)
- [0002 Use provider CLIs as the v1 execution boundary](0002-use-provider-clis-as-execution-boundary.md)
- [0003 Use deterministic model selection from a capability registry](0003-use-deterministic-model-selection.md)
- [0004 Scaffold provider skills in the repository](0004-scaffold-provider-skills-in-repository.md)
- [0005 Report exact usage when available and estimates otherwise](0005-exact-or-estimated-usage-reporting.md)
- [0006 Use provider-specific command adapters](0006-use-provider-specific-command-adapters.md)
- [0007 Split quality and Sonar workflows](0007-split-quality-and-sonar-workflows.md)
- [0008 Validate local provider config](0008-validate-local-provider-config.md)
