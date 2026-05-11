# Changelog

## 0.2.0

- Document Claude Code CLI prerequisites, `PATH` / `config.command` discovery, and the subprocess invocation contract.
- Add automated tests for Claude provider command construction, subprocess arguments (including timeout and working directory), timeout error mapping, and missing-binary errors without requiring a real `claude` install.
- Enforce synchronized versions across `pyproject.toml`, `llm_plan_execute.__version__`, and the `llm-plan-execute` package entry in `uv.lock`.
