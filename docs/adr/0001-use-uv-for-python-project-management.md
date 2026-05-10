# 0001 Use uv for Python project management

- Status: Accepted
- Date: 2026-05-10

## Context

The project needs a repeatable Python development environment, fast dependency installation, a lockfile, and a straightforward way to run the CLI, tests, linting, formatting, and pre-commit hooks.

## Decision

Use `uv` as the preferred package and environment manager. Store dependencies in `pyproject.toml`, commit `uv.lock`, and document development commands using `uv run`.

## Consequences

`uv` becomes the default entry point for local development and CI-style checks. Contributors need `uv` installed, but the project gets a reproducible lockfile and a single command surface for tests, linting, formatting, and pre-commit.
