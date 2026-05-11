import json
import os

import pytest

from llm_plan_execute.workflow_state import (
    CURRENT_WORKFLOW_SCHEMA_VERSION,
    WorkflowState,
    WorkflowStateCorruptError,
    WorkflowStateError,
    WorkflowStateLockError,
    WorkflowStateVersionError,
    acquire_workflow_lock,
    load_workflow_state,
    release_workflow_lock,
    save_workflow_state,
    transition_stage,
    workflow_lock_path,
)


def test_workflow_state_v2_missing_fields_default_none(tmp_path):
    """Schema v2 files deserialize new v3-only fields as ``None``."""
    legacy_schema_v2 = 2
    legacy = {
        "schema_version": legacy_schema_v2,
        "stage": "plan_review",
        "lifecycle_status": "active",
    }
    path = tmp_path / "workflow-state.json"
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    loaded = load_workflow_state(tmp_path)
    assert loaded.schema_version == legacy_schema_v2
    assert loaded.task_id is None
    assert loaded.branch is None
    assert loaded.base_branch is None
    assert loaded.context_db_path is None
    save_workflow_state(tmp_path, loaded)
    upgraded = load_workflow_state(tmp_path)
    assert upgraded.schema_version == CURRENT_WORKFLOW_SCHEMA_VERSION


def test_terminal_report_printed_requires_literal_true():
    assert WorkflowState.from_json_dict({"terminal_report_printed": True}).terminal_report_printed is True
    assert WorkflowState.from_json_dict({"terminal_report_printed": False}).terminal_report_printed is False
    assert WorkflowState.from_json_dict({"terminal_report_printed": "false"}).terminal_report_printed is False
    assert WorkflowState.from_json_dict({}).terminal_report_printed is False


def test_workflow_state_roundtrip_persists_schema(tmp_path):
    state = WorkflowState(stage="build_review", lifecycle_status="paused")
    save_workflow_state(tmp_path, state)
    loaded = load_workflow_state(tmp_path)
    assert loaded.stage == "build_review"
    assert loaded.schema_version == CURRENT_WORKFLOW_SCHEMA_VERSION


def test_workflow_state_rejects_future_schema(tmp_path):
    path = tmp_path / "workflow-state.json"
    path.write_text(json.dumps({"schema_version": CURRENT_WORKFLOW_SCHEMA_VERSION + 1}) + "\n", encoding="utf-8")
    with pytest.raises(WorkflowStateVersionError):
        load_workflow_state(tmp_path)


def test_workflow_state_corrupt_json_raises(tmp_path):
    path = tmp_path / "workflow-state.json"
    path.write_text("{bad-json", encoding="utf-8")
    with pytest.raises(WorkflowStateCorruptError):
        load_workflow_state(tmp_path)


def test_workflow_state_lock_prevents_second_session(tmp_path):
    acquire_workflow_lock(tmp_path)
    with pytest.raises(WorkflowStateLockError):
        acquire_workflow_lock(tmp_path)
    release_workflow_lock(tmp_path)


def test_transition_stage_rejects_invalid_jump():
    state = WorkflowState(stage="clarification")
    with pytest.raises(WorkflowStateError):
        transition_stage(state, "build")


def test_stale_lock_is_cleared_automatically(tmp_path):
    """A lock file containing a dead PID is treated as stale and replaced."""
    lock_path = workflow_lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Write a PID that will never belong to a live process (PID 0 is invalid).
    lock_path.write_text("0\n", encoding="utf-8")
    # Should not raise — stale lock is removed and re-acquired.
    acquire_workflow_lock(tmp_path)
    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())
    release_workflow_lock(tmp_path)


def test_force_session_overrides_live_lock(tmp_path):
    """force=True removes an active lock (for --force-session escalation)."""
    acquire_workflow_lock(tmp_path)
    # With force=True, even a live lock is overridden.
    acquire_workflow_lock(tmp_path, force=True)
    release_workflow_lock(tmp_path)


def test_lock_error_message_references_force_session(tmp_path):
    """Error raised by a live lock names --force-session so the user knows the fix."""
    acquire_workflow_lock(tmp_path)
    with pytest.raises(WorkflowStateLockError, match="--force-session"):
        acquire_workflow_lock(tmp_path)
    release_workflow_lock(tmp_path)


def test_is_lock_stale_process_lookup_error_clears_lock(tmp_path, monkeypatch):
    """A lock whose PID raises ProcessLookupError (ESRCH) is treated as stale."""
    lock_path = workflow_lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("99999\n", encoding="utf-8")

    def fake_kill(_pid, _sig):
        raise ProcessLookupError("no such process")

    monkeypatch.setattr("llm_plan_execute.workflow_state.os.kill", fake_kill)

    acquire_workflow_lock(tmp_path)
    assert int(lock_path.read_text(encoding="utf-8").strip()) == os.getpid()
    release_workflow_lock(tmp_path)


def test_is_lock_stale_permission_error_treats_lock_as_live(tmp_path, monkeypatch):
    """A lock whose PID raises PermissionError (EPERM) is treated as a live process."""
    lock_path = workflow_lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("99999\n", encoding="utf-8")

    def fake_kill(_pid, _sig):
        raise PermissionError("operation not permitted")

    monkeypatch.setattr("llm_plan_execute.workflow_state.os.kill", fake_kill)

    with pytest.raises(WorkflowStateLockError):
        acquire_workflow_lock(tmp_path)
    assert lock_path.read_text(encoding="utf-8").strip() == "99999"
