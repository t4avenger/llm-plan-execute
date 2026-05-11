"""Coverage for implementation_hooks orchestration."""

from __future__ import annotations

from pathlib import Path

from llm_plan_execute.implementation_hooks import prepare_implementation_entry
from llm_plan_execute.types import RunState
from llm_plan_execute.workflow_state import WorkflowState


def test_prepare_implementation_entry_adds_branch_checkpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = RunState.create("hello world\nsecond line", runs_root)
    run.run_dir.mkdir(parents=True)
    wf = WorkflowState()

    def fake_guard(_workspace: Path, wfx: WorkflowState, **_kwargs: object) -> None:
        wfx.branch = "feat/task-branch"
        wfx.base_branch = "main"

    monkeypatch.setattr(
        "llm_plan_execute.implementation_hooks.guard_build_start",
        fake_guard,
    )

    store = prepare_implementation_entry(tmp_path, wf, run, base_branch_override=None)
    conn = store._connect()
    try:
        row = conn.execute(
            "SELECT kind, content FROM context_items WHERE kind = 'checkpoint' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert "feat/task-branch" in str(row["content"])
    finally:
        conn.close()
