import pytest

from llm_plan_execute.session_runner import WorkflowSessionRunner
from llm_plan_execute.workflow_state import WorkflowState, WorkflowStateError


def test_start_sets_lifecycle_active():
    wf = WorkflowState()
    runner = WorkflowSessionRunner(wf)
    status = runner.start()
    assert status.lifecycle_status == "active"
    assert status.stage == "clarification"
    assert status.retry_available is False


def test_resume_from_paused_sets_active():
    wf = WorkflowState(lifecycle_status="paused")
    runner = WorkflowSessionRunner(wf)
    status = runner.resume()
    assert status.lifecycle_status == "active"


def test_resume_from_non_paused_keeps_status():
    wf = WorkflowState(lifecycle_status="failed")
    runner = WorkflowSessionRunner(wf)
    status = runner.resume()
    assert status.lifecycle_status == "failed"


def test_advance_transitions_stage_and_clears_retry():
    wf = WorkflowState(stage="clarification")
    runner = WorkflowSessionRunner(wf)
    runner.mark_retry_available()
    assert runner.status().retry_available is True
    status = runner.advance("planning")
    assert status.stage == "planning"
    assert status.retry_available is False


def test_advance_raises_on_invalid_transition():
    wf = WorkflowState(stage="clarification")
    runner = WorkflowSessionRunner(wf)
    with pytest.raises(WorkflowStateError):
        runner.advance("build")


def test_retry_when_available_sets_active():
    wf = WorkflowState(lifecycle_status="failed")
    runner = WorkflowSessionRunner(wf)
    runner.mark_retry_available()
    status = runner.retry()
    assert status.lifecycle_status == "active"
    assert status.retry_available is False


def test_retry_when_not_available_is_noop():
    wf = WorkflowState(lifecycle_status="failed")
    runner = WorkflowSessionRunner(wf)
    status = runner.retry()
    assert status.lifecycle_status == "failed"
    assert status.retry_available is False


def test_cancel_current_step_sets_canceled():
    wf = WorkflowState()
    runner = WorkflowSessionRunner(wf)
    status = runner.cancel_current_step()
    assert status.lifecycle_status == "canceled"


def test_exit_when_active_sets_paused():
    wf = WorkflowState(lifecycle_status="active")
    runner = WorkflowSessionRunner(wf)
    status = runner.exit()
    assert status.lifecycle_status == "paused"


def test_exit_when_not_active_no_change():
    wf = WorkflowState(lifecycle_status="completed")
    runner = WorkflowSessionRunner(wf)
    status = runner.exit()
    assert status.lifecycle_status == "completed"


def test_status_reflects_current_state():
    wf = WorkflowState(stage="build", lifecycle_status="paused")
    runner = WorkflowSessionRunner(wf)
    status = runner.status()
    assert status.stage == "build"
    assert status.lifecycle_status == "paused"
    assert status.retry_available is False


def test_mark_retry_available_then_retry_clears_flag():
    wf = WorkflowState()
    runner = WorkflowSessionRunner(wf)
    runner.mark_retry_available()
    assert runner.status().retry_available is True
    runner.retry()
    assert runner.status().retry_available is False
