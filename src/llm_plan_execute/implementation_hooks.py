"""Orchestration glue between workflow transitions, git guardrails, and the context store."""

from __future__ import annotations

from pathlib import Path

from .context_store import ContextStore, context_db_path, ensure_context_gitignore
from .git_flow import guard_build_start
from .types import RunState
from .workflow_state import WorkflowState


def prepare_implementation_entry(
    workspace: Path,
    wf: WorkflowState,
    run: RunState,
    *,
    base_branch_override: str | None,
) -> ContextStore:
    """Initialize workspace context storage and enforce git-flow before build work."""
    workspace = workspace.resolve()
    ensure_context_gitignore(workspace)
    db_path = context_db_path(workspace)
    wf.context_db_path = str(db_path)
    store = ContextStore(workspace)
    store.init()
    title_line = run.prompt.strip().splitlines()[0][:200] if run.prompt.strip() else run.run_id
    store.upsert_task(run.run_id, title=title_line or run.run_id)
    before_branch = wf.branch
    guard_build_start(workspace, wf, base_branch_override=base_branch_override, task_id=run.run_id)
    if wf.branch and wf.branch != before_branch:
        store.add_context_item(
            task_id=run.run_id,
            kind="checkpoint",
            path=None,
            content=f"branch-created: {wf.branch}\n",
            metadata={"label": "branch-created", "branch": wf.branch, "base_branch": wf.base_branch},
        )
    return store
