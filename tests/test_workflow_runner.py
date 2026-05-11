"""Unit coverage for workflow_runner helpers (complements CLI integration tests)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from llm_plan_execute.config import ExecutionConfig
from llm_plan_execute.interactive import (
    BuildReviewDecision,
    CompletionReportDecision,
    InteractiveSession,
    PlanReviewDecision,
    StageTransitionDecision,
    StepThroughOutcome,
    session_with_mock_stdin,
)
from llm_plan_execute.types import RunState
from llm_plan_execute.workflow import BuildFailedError
from llm_plan_execute.workflow_runner import (
    execute_build_through_completion,
    finalize_completion_reports,
    gate_stage_transition,
    interactive_build_review_loop,
    interactive_plan_review,
    merge_execution_dirs,
    orchestrate_clarification,
    resolve_build_permission,
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


def test_orchestrate_clarification_no_clarify_runs_planning(monkeypatch, tmp_path: Path) -> None:
    planned = RunState.create("prompt", tmp_path)
    planned.run_dir.mkdir(parents=True)

    monkeypatch.setattr("llm_plan_execute.workflow_runner.run_planning", lambda *_args, **_kwargs: planned)
    wf = WorkflowState()

    result = orchestrate_clarification(
        prompt="prompt",
        router=MagicMock(),
        runs_dir=tmp_path,
        execution=ExecutionConfig(),
        permission_mode=None,
        progress=_noop_progress,
        session=InteractiveSession(non_interactive=True),
        no_clarify=True,
        wf=wf,
    )

    assert result is planned
    assert wf.stage == "plan_review"


def test_interactive_plan_review_accept_modify_and_step_cancel(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = RunState.create("prompt", runs_root)
    run.run_dir.mkdir(parents=True)
    (run.run_dir / "04-proposed-plan.md").write_text("## Plan\nDo it", encoding="utf-8")
    wf = WorkflowState()
    session = _PlanReviewSession(["modify", "accept"], feedback="Tighten scope.")
    monkeypatch.setattr("llm_plan_execute.workflow_runner.revise_proposed_plan", lambda *_args, **_kwargs: run)

    reviewed = interactive_plan_review(
        session=session,
        run=run,
        router=MagicMock(),
        execution=ExecutionConfig(),
        permission_mode=None,
        progress=_noop_progress,
        wf=wf,
    )

    assert reviewed is run
    assert wf.plan_feedback_history == ["Tighten scope."]
    assert wf.accepted_plan_version == 1

    cancel_run = RunState.create("prompt", runs_root)
    cancel_run.run_dir.mkdir(parents=True)
    (cancel_run.run_dir / "04-proposed-plan.md").write_text("## Plan\nCancel it", encoding="utf-8")
    cancel_wf = WorkflowState()

    assert (
        interactive_plan_review(
            session=_PlanReviewSession(["stepThrough"], step_result="cancel"),
            run=cancel_run,
            router=MagicMock(),
            execution=ExecutionConfig(),
            permission_mode=None,
            progress=_noop_progress,
            wf=cancel_wf,
        )
        is None
    )
    assert cancel_wf.lifecycle_status == "canceled"


def test_gate_stage_transition_updates_pause_and_cancel(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = RunState.create("prompt", runs_root)
    run.run_dir.mkdir(parents=True)

    pause_wf = WorkflowState()
    assert gate_stage_transition(session=_StageSession("pause"), wf=pause_wf, run=run) == "pause"
    assert pause_wf.lifecycle_status == "paused"
    assert pause_wf.stage == "pre_build"

    cancel_wf = WorkflowState()
    assert gate_stage_transition(session=_StageSession("cancel"), wf=cancel_wf, run=run) == "cancel"
    assert cancel_wf.lifecycle_status == "canceled"

    assert gate_stage_transition(session=_StageSession("proceed"), wf=WorkflowState(), run=run) == "proceed"


def test_resolve_build_permission_prompts_when_interactive() -> None:
    session, _stdout, _stderr = session_with_mock_stdin(["3"])

    assert resolve_build_permission("read-only", session) == "full-access"


def test_finalize_completion_reports_non_interactive_skips(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run = RunState.create("prompt", runs_root)
    run.run_dir.mkdir(parents=True)
    wf = WorkflowState()
    session = InteractiveSession(non_interactive=True)
    assert finalize_completion_reports(session=session, run=run, wf=wf) is None
    assert wf.lifecycle_status == "completed"


def test_finalize_completion_reports_terminal_html_and_cancel(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = RunState.create("prompt", runs_root)
    run.run_dir.mkdir(parents=True)
    wf = WorkflowState()

    html_path = run.run_dir / "report.html"
    monkeypatch.setattr("llm_plan_execute.workflow_runner.write_html_report", lambda _run: html_path)

    assert finalize_completion_reports(session=_CompletionSession("both"), run=run, wf=wf) is None
    assert wf.lifecycle_status == "completed"
    assert wf.terminal_report_printed is True
    assert wf.report_html_path == "report.html"

    cancel_wf = WorkflowState()
    assert finalize_completion_reports(session=_CompletionSession("cancel"), run=run, wf=cancel_wf) == "cancel"
    assert cancel_wf.lifecycle_status == "canceled"


def test_interactive_build_review_loop_records_apply_selection(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = RunState.create("prompt", runs_root)
    run.run_dir.mkdir(parents=True)
    (run.run_dir / "08-build-review-summary.md").write_text("- finding\n", encoding="utf-8")
    wf = WorkflowState()
    calls: list[list[str]] = []
    monkeypatch.setattr("llm_plan_execute.workflow_runner.record_build_recommendation_application", lambda *_args: None)
    monkeypatch.setattr("llm_plan_execute.workflow_runner._write_apply_follow_up_notes", lambda *_args: None)

    def fake_expand(selected, _recommendations):
        calls.append(list(selected))
        return list(selected)

    monkeypatch.setattr("llm_plan_execute.workflow_runner.expand_with_dependencies", fake_expand)

    assert (
        interactive_build_review_loop(
            session=_BuildReviewSession("applyAll"),
            run=run,
            router=MagicMock(),
            execution=ExecutionConfig(),
            permission_mode=None,
            progress=_noop_progress,
            wf=wf,
        )
        is None
    )
    assert wf.build_review_selected_action == "applyAll"
    assert wf.build_review_selected_ids == ["finding-1"]
    assert wf.build_review_applied_ids == ["finding-1"]
    assert calls == [["finding-1"]]


def test_interactive_build_review_loop_feedback_then_select(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = RunState.create("prompt", runs_root)
    run.run_dir.mkdir(parents=True)
    (run.run_dir / "08-build-review-summary.md").write_text("- first\n- second\n", encoding="utf-8")
    wf = WorkflowState()
    monkeypatch.setattr("llm_plan_execute.workflow_runner.record_build_recommendation_application", lambda *_args: None)
    monkeypatch.setattr("llm_plan_execute.workflow_runner._write_apply_follow_up_notes", lambda *_args: None)
    monkeypatch.setattr("llm_plan_execute.workflow_runner.rerun_build_review", lambda run, *_args, **_kwargs: run)

    assert (
        interactive_build_review_loop(
            session=_BuildReviewSession("feedback", "select", selection=("2",)),
            run=run,
            router=MagicMock(),
            execution=ExecutionConfig(),
            permission_mode=None,
            progress=_noop_progress,
            wf=wf,
        )
        is None
    )
    assert wf.build_review_feedback_history == ["rerun review"]
    assert wf.build_review_selected_ids == ["finding-2"]
    assert wf.build_review_applied_ids == ["finding-2"]


def test_interactive_build_review_loop_cancel_and_continue(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = RunState.create("prompt", runs_root)
    run.run_dir.mkdir(parents=True)
    (run.run_dir / "08-build-review-summary.md").write_text("- finding\n", encoding="utf-8")

    cancel_wf = WorkflowState()
    assert (
        interactive_build_review_loop(
            session=_BuildReviewSession("cancel"),
            run=run,
            router=MagicMock(),
            execution=ExecutionConfig(),
            permission_mode=None,
            progress=_noop_progress,
            wf=cancel_wf,
        )
        == "cancel"
    )
    assert cancel_wf.lifecycle_status == "canceled"

    continue_wf = WorkflowState()
    assert (
        interactive_build_review_loop(
            session=_BuildReviewSession("continueWithoutApplying"),
            run=run,
            router=MagicMock(),
            execution=ExecutionConfig(),
            permission_mode=None,
            progress=_noop_progress,
            wf=continue_wf,
        )
        is None
    )
    assert continue_wf.build_review_selected_ids == []


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
    wf = WorkflowState(stage="pre_build")  # valid precondition for execute_build_through_completion

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
    assert wf.stage == "complete"  # stage advances through build_review to complete


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


class _StageSession:
    non_interactive = False

    def __init__(self, decision_type: str) -> None:
        self._decision_type = decision_type

    def ask_stage_transition(self) -> StageTransitionDecision:
        return StageTransitionDecision(type=self._decision_type)


class _CompletionSession:
    def __init__(self, decision_type: str) -> None:
        self._decision_type = decision_type

    def ask_completion_report(self) -> CompletionReportDecision:
        return CompletionReportDecision(type=self._decision_type)


class _BuildReviewSession:
    non_interactive = False

    def __init__(self, *decision_types: str, selection: tuple[str, ...] = ()) -> None:
        self._decision_types = list(decision_types)
        self._selection = selection

    def ask_build_review(self) -> BuildReviewDecision:
        return BuildReviewDecision(type=self._decision_types.pop(0))

    def read_build_feedback(self) -> str:
        return "rerun review"

    def read_recommendation_selection(self, _count: int) -> tuple[str, ...]:
        return self._selection


class _PlanReviewSession:
    non_interactive = False

    def __init__(self, decision_types: list[str], *, feedback: str = "", step_result: str = "stop") -> None:
        self._decision_types = decision_types
        self._feedback = feedback
        self._step_result = step_result

    def ask_plan_review(self) -> PlanReviewDecision:
        return PlanReviewDecision(type=self._decision_types.pop(0))

    def read_plan_feedback(self) -> str:
        return self._feedback

    def step_through_sections(self, _title: str, _sections) -> StepThroughOutcome:
        return StepThroughOutcome(type=self._step_result)
