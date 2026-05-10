# 0007. Split quality and Sonar workflows

## Status

Accepted

## Context

Local quality gates and Sonar analysis have different failure modes, permissions, and secrets. Combining them makes routine test failures harder to separate from scanner or token issues.

## Decision

Keep `.github/workflows/build.yml` focused on Sonar scanning. Add `.github/workflows/quality.yml` for dependency sync, Ruff linting, Ruff formatting checks, pytest, and pre-commit.

## Consequences

Pull requests get faster, clearer feedback for code quality. Sonar remains independently configurable and can depend on its own secret setup.
