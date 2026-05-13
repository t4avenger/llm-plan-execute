# Changelog

## 1.0.0

- Release notes pending.

## Unreleased

- Bare `llm-plan-execute` on a TTY now launches an interactive wizard that confirms the workspace, manages project-local `.llm-plan-execute/config.json` (use existing / reconfigure / validate / create), detects paused runs to resume, and prompts for the task text (inline or via `$EDITOR`). On non-TTY or with `--non-interactive`, the bare command prints help and exits nonzero. Existing subcommands and automation are unchanged.
- Add `--reconfigure` flag (wizard-only) to rerun provider/model setup even when a config already exists.

## 0.3.0

- Fix `_cmd_build` to acquire the session lock **before** mutating workflow state, preventing race conditions and stage clobbers on resume.
- Preserve workflow stage when `build --run-dir` resumes at `build_review`; skip re-running the build instead of resetting to `pre_build`.
- Add stale-PID detection to the session lock: a lock left by a crashed process is now cleared automatically on next run.
- Add `--force-session` flag to `build` to override any stuck lock.
- Reach the `complete` workflow stage after a successful build+review+report cycle.
- Remove raw `wf.stage = ...` assignments in `execute_build_through_completion`; all transitions now go through `transition_stage`.
- Ctrl-C at an interactive prompt now warns on the first press and cancels only on the second, preventing accidental workflow termination.
- Heartbeat progress events are now emitted on a 30-second interval (daemon thread) while a provider is running, replacing the single pre-call event.

## 0.2.0

- Document Claude Code CLI prerequisites, `PATH` / `config.command` discovery, and the subprocess invocation contract.
- Add automated tests for Claude provider command construction, subprocess arguments (including timeout and working directory), timeout error mapping, and missing-binary errors without requiring a real `claude` install.
- Enforce synchronized versions across `pyproject.toml`, `llm_plan_execute.__version__`, and the `llm-plan-execute` package entry in `uv.lock`.
