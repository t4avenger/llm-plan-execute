"""Unit coverage for workflow_runner helpers (complements CLI integration tests)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from llm_plan_execute.config import ExecutionConfig
from llm_plan_execute.interactive import InteractiveSession
from llm_plan_execute.types import RunState
from llm_plan_execute.workflow import BuildFailedError
from llm_plan_execute.workflow_runner import (
    execute_build_through_completion,
    finalize_completion_reports,
    interactive_build_review_loop,
    merge_execution_dirs,
)
from llm_plan_execute.workflow_state import WorkflowState


def _noop_progress(*_args, **_kwargs) -> None:
    return None


def test_merge_execution_dirs_returns_same_when_no_extra_dirs(tmp_path: Path):
    ex = ExecutionConfig()
    assert merge_execution_dirs(tmp_path, ex, []) is ex


def test_interactive_build_review_loop_without_summary_returns_none(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run = RunState.create("prompt", runs_root)
    run.run_dir.mkdir(parents=True)
    wf = WorkflowState()
    assert (
        interactive_build_review_loop(
            session=InteractiveSession(non_interactive=True),
            run=run,
            router=MagicMock(),
            execution=ExecutionConfig(),
            permission_mode=None,
            progress=_noop_progress,
            wf=wf,
        )
        is None
    )


def test_finalize_completion_reports_non_interactive_skips(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run = RunState.create("prompt", runs_root)
    run.run_dir.mkdir(parents=True)
    wf = WorkflowState()
    session = InteractiveSession(non_interactive=True)
    assert finalize_completion_reports(session=session, run=run, wf=wf) is None
    assert wf.lifecycle_status == "completed"


def _stub_run_build(run, _router, **_kw):
    return run


def _stub_none(**_kw):
    return None


def test_execute_build_through_completion_exit_zero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("llm_plan_execute.workflow_runner.run_build", _stub_run_build)
    monkeypatch.setattr(
        "llm_plan_execute.workflow_runner.interactive_build_review_loop",
        _stub_none,
    )
    monkeypatch.setattr(
        "llm_plan_execute.workflow_runner.finalize_completion_reports",
        _stub_none,
    )

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run = RunState.create("prompt", runs_root)
    run.run_dir.mkdir(parents=True)
    run.accepted_plan = "plan body"
    wf = WorkflowState()

    _, code = execute_build_through_completion(
        run=run,
        wf=wf,
        router=MagicMock(),
        execution=ExecutionConfig(),
        session=InteractiveSession(non_interactive=True),
        permission_mode_cli=None,
        progress=_noop_progress,
        runs_root=runs_root,
    )
    assert code == 0
    assert wf.stage == "build_review"


def test_execute_build_through_completion_propagates_build_failed(monkeypatch, tmp_path: Path) -> None:
    def boom(run, _router, **_kw):
        raise BuildFailedError("fail", run)

    monkeypatch.setattr("llm_plan_execute.workflow_runner.run_build", boom)

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run = RunState.create("prompt", runs_root)
    run.run_dir.mkdir(parents=True)
    run.accepted_plan = "plan"
    wf = WorkflowState()

    _, code = execute_build_through_completion(
        run=run,
        wf=wf,
        router=MagicMock(),
        execution=ExecutionConfig(),
        session=InteractiveSession(non_interactive=True),
        permission_mode_cli=None,
        progress=_noop_progress,
        runs_root=runs_root,
    )
    assert code == 1
    assert wf.lifecycle_status == "failed"
