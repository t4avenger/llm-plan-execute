# 0003 Use deterministic model selection from a capability registry

- Status: Accepted
- Date: 2026-05-10

## Context

The workflow assigns models to planning, review, arbitration, build, and reporting roles. Letting a model choose the role assignments live would be flexible but harder to audit, test, and reproduce.

## Decision

Use a deterministic capability registry and scoring function for v1 model selection. Models declare provider, name, suitable roles, reasoning, speed, cost, context, and usage capabilities. The selector scores each role and prefers diversity while allowing fallback reuse with warnings.

## Consequences

Assignments are predictable and unit-testable. The registry needs maintenance as provider model lists change. Future versions can add live discovery or model-assisted selection behind the same interface without changing the workflow contract.
