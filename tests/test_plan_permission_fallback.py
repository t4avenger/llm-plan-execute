"""Tests for plan-command permission failure detection and workspace-write retry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_plan_execute.cli import ProgressReporter, _plan_run_planning_phase, _run_planning_only
from llm_plan_execute.config import AppConfig, ExecutionConfig
from llm_plan_execute.interactive import InteractiveSession
from llm_plan_execute.providers import ProviderRouter
from llm_plan_execute.types import ModelAssignment, ModelInfo, ProviderResult, RunState, Usage
from llm_plan_execute.workflow_runner import (
    PLAN_PERMISSION_FALLBACK_WARNING,
    is_planning_permission_failure_message,
    is_run_planning_permission_failure,
    plan_permission_workspace_write_fallback_applies,
)
from llm_plan_execute.workflow_state import WorkflowState


def _model() -> ModelInfo:
    return ModelInfo("p", "m", ("planner",), 3, 3, 3, 3)


def _failed_run(runs_root: Path, *, error: str) -> RunState:
    run = RunState.create("x", runs_root)
    run.assignments["planner"] = ModelAssignment("planner", _model(), False, "")
    run.results.append(
        ProviderResult("planner", _model(), "p", "", Usage(), 0.0, error=error),
    )
    return run


def _success_run(runs_root: Path) -> RunState:
    run = RunState.create("ok", runs_root)
    run.assignments["planner"] = ModelAssignment("planner", _model(), False, "")
    run.results.append(
        ProviderResult("planner", _model(), "p", "# ok\n", Usage(), 0.0, error=None),
    )
    return run


def test_is_planning_permission_failure_message_file_read() -> None:
    assert is_planning_permission_failure_message("Permission denied opening src/foo.py")


def test_is_planning_permission_failure_message_command() -> None:
    assert is_planning_permission_failure_message("not allowed to run shell commands in this mode")


def test_is_planning_permission_failure_message_sandbox() -> None:
    assert is_planning_permission_failure_message("blocked by sandbox policy: read denied")


def test_is_planning_permission_failure_message_rejects_unrelated() -> None:
    assert not is_planning_permission_failure_message("model returned invalid JSON")
    assert not is_planning_permission_failure_message("connection refused")


def test_plan_fallback_applies_implicit_only() -> None:
    ex = ExecutionConfig(planning_mode="read-only", review_mode="read-only")
    assert plan_permission_workspace_write_fallback_applies(ex, permission_mode_cli=None, no_clarify=True)
    assert not plan_permission_workspace_write_fallback_applies(ex, permission_mode_cli="read-only", no_clarify=True)


def test_plan_fallback_not_when_all_workspace_write() -> None:
    ex = ExecutionConfig(
        planning_mode="workspace-write",
        review_mode="workspace-write",
    )
    assert not plan_permission_workspace_write_fallback_applies(ex, permission_mode_cli=None, no_clarify=True)


def test_is_run_planning_permission_failure_uses_planning_roles(tmp_path: Path) -> None:
    run = RunState.create("x", tmp_path)
    run.results.append(
        ProviderResult("builder", _model(), "p", "", Usage(), 0.0, error="Permission denied"),
    )
    assert not is_run_planning_permission_failure(run)


def test_plan_retries_after_file_read_denial(monkeypatch, tmp_path: Path, capsys) -> None:
    calls: list[str | None] = []

    def fake_once(*_a, permission_mode: str | None, **_k):
        calls.append(permission_mode)
        if len(calls) == 1:
            return _failed_run(tmp_path, error="Permission denied reading ./README.md")
        return _success_run(tmp_path)

    monkeypatch.setattr("llm_plan_execute.cli._plan_run_planning_once", fake_once)

    args = argparse.Namespace(
        no_clarify=True,
        yes=False,
        permission_mode=None,
    )
    execution = ExecutionConfig(planning_mode="read-only", review_mode="read-only")
    out = _plan_run_planning_phase(
        args,
        "prompt",
        ProviderRouter([]),
        AppConfig(providers=()),
        execution,
        ProgressReporter(enabled=False, verbose=False, stream=sys.stderr),
        WorkflowState(),
        InteractiveSession(non_interactive=True),
    )
    assert isinstance(out, RunState)
    assert calls == [None, "workspace-write"]
    assert out.warnings == [PLAN_PERMISSION_FALLBACK_WARNING]
    assert json.loads((out.run_dir / "run.json").read_text(encoding="utf-8"))["warnings"] == [
        PLAN_PERMISSION_FALLBACK_WARNING
    ]
    assert PLAN_PERMISSION_FALLBACK_WARNING in (out.run_dir / "report.md").read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert err.count(PLAN_PERMISSION_FALLBACK_WARNING) == 1


def test_plan_retries_after_command_denial(monkeypatch, tmp_path: Path) -> None:
    calls: list[str | None] = []

    def fake_once(*_a, permission_mode: str | None, **_k):
        calls.append(permission_mode)
        if len(calls) == 1:
            return _failed_run(tmp_path, error="not allowed to run command: git status")
        return _success_run(tmp_path)

    monkeypatch.setattr("llm_plan_execute.cli._plan_run_planning_once", fake_once)

    args = argparse.Namespace(no_clarify=True, yes=False, permission_mode=None)
    execution = ExecutionConfig(planning_mode="read-only", review_mode="read-only")
    _plan_run_planning_phase(
        args,
        "p",
        ProviderRouter([]),
        AppConfig(providers=()),
        execution,
        ProgressReporter(enabled=False, verbose=False, stream=sys.stderr),
        WorkflowState(),
        InteractiveSession(non_interactive=True),
    )
    assert calls == [None, "workspace-write"]


def test_non_permission_failure_no_retry(monkeypatch, tmp_path: Path, capsys) -> None:
    calls: list[str | None] = []

    def fake_once(*_a, permission_mode: str | None, **_k):
        calls.append(permission_mode)
        return _failed_run(tmp_path, error="upstream rate limit exceeded")

    monkeypatch.setattr("llm_plan_execute.cli._plan_run_planning_once", fake_once)

    args = argparse.Namespace(no_clarify=True, yes=False, permission_mode=None)
    execution = ExecutionConfig(planning_mode="read-only", review_mode="read-only")
    _plan_run_planning_phase(
        args,
        "p",
        ProviderRouter([]),
        AppConfig(providers=()),
        execution,
        ProgressReporter(enabled=False, verbose=False, stream=sys.stderr),
        WorkflowState(),
        InteractiveSession(non_interactive=True),
    )
    assert calls == [None]
    assert PLAN_PERMISSION_FALLBACK_WARNING not in capsys.readouterr().err


def test_explicit_permission_mode_respected(monkeypatch, tmp_path: Path, capsys) -> None:
    calls: list[str | None] = []

    def fake_once(*_a, permission_mode: str | None, **_k):
        calls.append(permission_mode)
        return _failed_run(tmp_path, error="Permission denied")

    monkeypatch.setattr("llm_plan_execute.cli._plan_run_planning_once", fake_once)

    args = argparse.Namespace(no_clarify=True, yes=False, permission_mode="read-only")
    execution = ExecutionConfig(planning_mode="read-only", review_mode="read-only")
    _plan_run_planning_phase(
        args,
        "p",
        ProviderRouter([]),
        AppConfig(providers=()),
        execution,
        ProgressReporter(enabled=False, verbose=False, stream=sys.stderr),
        WorkflowState(),
        InteractiveSession(non_interactive=True),
    )
    assert calls == ["read-only"]
    assert PLAN_PERMISSION_FALLBACK_WARNING not in capsys.readouterr().err


def test_retry_at_most_once(monkeypatch, tmp_path: Path) -> None:
    calls: list[str | None] = []

    def fake_once(*_a, permission_mode: str | None, **_k):
        calls.append(permission_mode)
        return _failed_run(tmp_path, error="sandbox blocked read")

    monkeypatch.setattr("llm_plan_execute.cli._plan_run_planning_once", fake_once)

    args = argparse.Namespace(no_clarify=True, yes=False, permission_mode=None)
    execution = ExecutionConfig(planning_mode="read-only", review_mode="read-only")
    _plan_run_planning_phase(
        args,
        "p",
        ProviderRouter([]),
        AppConfig(providers=()),
        execution,
        ProgressReporter(enabled=False, verbose=False, stream=sys.stderr),
        WorkflowState(),
        InteractiveSession(non_interactive=True),
    )
    assert calls == [None, "workspace-write"]


def test_run_retries_after_file_read_denial(monkeypatch, tmp_path: Path, capsys) -> None:
    calls: list[str | None] = []

    def fake_once(*_a, permission_mode: str | None, **_k):
        calls.append(permission_mode)
        if len(calls) == 1:
            return _failed_run(tmp_path, error="Permission denied reading ./README.md")
        return _success_run(tmp_path)

    monkeypatch.setattr("llm_plan_execute.cli._run_planning_once", fake_once)

    args = argparse.Namespace(no_clarify=True, permission_mode=None)
    execution = ExecutionConfig(planning_mode="read-only", review_mode="read-only")
    out = _run_planning_only(
        args,
        "prompt",
        ProviderRouter([]),
        AppConfig(providers=()),
        execution,
        ProgressReporter(enabled=False, verbose=False, stream=sys.stderr),
        WorkflowState(),
        InteractiveSession(non_interactive=True),
    )
    assert isinstance(out, RunState)
    assert calls == [None, "workspace-write"]
    assert out.warnings == [PLAN_PERMISSION_FALLBACK_WARNING]
    assert PLAN_PERMISSION_FALLBACK_WARNING in capsys.readouterr().err
