"""Persisted workflow orchestration state (separate from provider run.json)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

WorkflowStage = Literal[
    "clarification",
    "planning",
    "plan_review",
    "pre_build",
    "build",
    "build_review",
    "complete",
]

WorkflowLifecycleStatus = Literal["active", "paused", "completed", "failed", "canceled"]
CURRENT_WORKFLOW_SCHEMA_VERSION = 3
WORKFLOW_STATE_LOCK_FILENAME = ".workflow-state.lock"


class WorkflowStateError(ValueError):
    """Base workflow-state persistence error."""


class WorkflowStateVersionError(WorkflowStateError):
    """Raised when reading an unsupported future schema."""


class WorkflowStateCorruptError(WorkflowStateError):
    """Raised when workflow-state.json is invalid JSON."""


class WorkflowStateLockError(WorkflowStateError):
    """Raised when another active session already holds the lock."""


@dataclass
class WorkflowState:
    """Contract for interactive workflow persistence under each run directory."""

    schema_version: int = CURRENT_WORKFLOW_SCHEMA_VERSION
    stage: WorkflowStage = "clarification"
    lifecycle_status: WorkflowLifecycleStatus = "active"
    accepted_plan_version: int | None = None
    accepted_plan_artifact: str | None = None
    proposed_plan_artifact: str | None = "04-proposed-plan.md"
    plan_feedback_history: list[str] = field(default_factory=list)
    build_review_feedback_history: list[str] = field(default_factory=list)
    build_review_selected_action: str | None = None
    build_review_selected_ids: list[str] = field(default_factory=list)
    build_review_applied_ids: list[str] = field(default_factory=list)
    report_markdown_path: str | None = "report.md"
    report_html_path: str | None = None
    terminal_report_printed: bool = False
    task_id: str | None = None
    branch: str | None = None
    base_branch: str | None = None
    context_db_path: str | None = None

    def touch_accepted_plan(self, *, artifact_name: str = "04-accepted-plan.md") -> None:
        self.accepted_plan_artifact = artifact_name
        self.accepted_plan_version = (self.accepted_plan_version or 0) + 1

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, raw: dict[str, Any]) -> WorkflowState:
        schema_version = _read_schema_version(raw.get("schema_version"))
        return cls(
            schema_version=schema_version,
            stage=_read_stage(raw.get("stage")),
            lifecycle_status=_read_lifecycle(raw.get("lifecycle_status")),
            accepted_plan_version=_optional_int(raw.get("accepted_plan_version")),
            accepted_plan_artifact=_optional_str(raw.get("accepted_plan_artifact")),
            proposed_plan_artifact=_optional_str(raw.get("proposed_plan_artifact")) or "04-proposed-plan.md",
            plan_feedback_history=_string_list(raw.get("plan_feedback_history")),
            build_review_feedback_history=_string_list(raw.get("build_review_feedback_history")),
            build_review_selected_action=_optional_str(raw.get("build_review_selected_action")),
            build_review_selected_ids=_string_list(raw.get("build_review_selected_ids")),
            build_review_applied_ids=_string_list(raw.get("build_review_applied_ids")),
            report_markdown_path=_optional_str(raw.get("report_markdown_path")) or "report.md",
            report_html_path=_optional_str(raw.get("report_html_path")),
            terminal_report_printed=_read_bool(raw.get("terminal_report_printed")),
            task_id=_optional_str(raw.get("task_id")),
            branch=_optional_str(raw.get("branch")),
            base_branch=_optional_str(raw.get("base_branch")),
            context_db_path=_optional_str(raw.get("context_db_path")),
        )


WORKFLOW_STATE_FILENAME = "workflow-state.json"
_WORKFLOW_TRANSITIONS: dict[WorkflowStage, set[WorkflowStage]] = {
    "clarification": {"planning", "plan_review"},
    "planning": {"plan_review"},
    "plan_review": {"pre_build"},
    "pre_build": {"build"},
    "build": {"build_review"},
    "build_review": {"complete"},
    "complete": set(),
}


def workflow_state_path(run_dir: Path) -> Path:
    return run_dir / WORKFLOW_STATE_FILENAME


def load_workflow_state(run_dir: Path) -> WorkflowState:
    path = workflow_state_path(run_dir)
    if not path.exists():
        return WorkflowState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowStateCorruptError(f"Corrupt workflow state at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        return WorkflowState()
    _check_supported_schema(raw)
    return WorkflowState.from_json_dict(raw)


def save_workflow_state(run_dir: Path, state: WorkflowState) -> Path:
    path = workflow_state_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_json_dict()
    payload["schema_version"] = CURRENT_WORKFLOW_SCHEMA_VERSION
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def workflow_lock_path(run_dir: Path) -> Path:
    return run_dir / WORKFLOW_STATE_LOCK_FILENAME


def _is_lock_stale(lock_path: Path) -> bool:
    """Return True when the PID in the lock file is no longer running."""
    try:
        pid = int(lock_path.read_text(encoding="utf-8").strip())
        if pid <= 0:
            return True
        os.kill(pid, 0)
        return False  # process alive
    except ProcessLookupError:
        return True  # ESRCH: process gone
    except PermissionError:
        return False  # EPERM: process exists, we just can't signal it
    except (ValueError, OSError):
        return True  # unreadable / other error


def acquire_workflow_lock(run_dir: Path, *, force: bool = False) -> Path | None:
    path = workflow_lock_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        stale = _is_lock_stale(path)
        if not stale and not force:
            raise WorkflowStateLockError(
                f"Another active session is using {run_dir}. Use --force-session to override a stale lock."
            ) from exc
        path.unlink(missing_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc2:
            raise WorkflowStateLockError(
                f"Another session grabbed {run_dir} during lock recovery. Use --force-session to override."
            ) from exc2
    with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
        lock_file.write(str(os.getpid()) + "\n")
    return path


def release_workflow_lock(run_dir: Path) -> None:
    workflow_lock_path(run_dir).unlink(missing_ok=True)


def _read_stage(value: object) -> WorkflowStage:
    allowed: tuple[WorkflowStage, ...] = (
        "clarification",
        "planning",
        "plan_review",
        "pre_build",
        "build",
        "build_review",
        "complete",
    )
    if isinstance(value, str) and value in allowed:
        return value  # type: ignore[return-value]
    return "clarification"


def _read_schema_version(value: object) -> int:
    if value is None:
        return CURRENT_WORKFLOW_SCHEMA_VERSION
    if isinstance(value, int) and value > 0:
        return value
    return CURRENT_WORKFLOW_SCHEMA_VERSION


def _check_supported_schema(raw: dict[str, Any]) -> None:
    version = raw.get("schema_version")
    if version is None:
        return
    if not isinstance(version, int):
        raise WorkflowStateVersionError("workflow-state schema_version must be an integer.")
    if version > CURRENT_WORKFLOW_SCHEMA_VERSION:
        raise WorkflowStateVersionError(
            f"workflow-state schema_version={version} is newer than supported "
            f"version {CURRENT_WORKFLOW_SCHEMA_VERSION}."
        )


def can_transition(current: WorkflowStage, nxt: WorkflowStage) -> bool:
    if current == nxt:
        return True
    return nxt in _WORKFLOW_TRANSITIONS[current]


def transition_stage(state: WorkflowState, nxt: WorkflowStage) -> None:
    if not can_transition(state.stage, nxt):
        raise WorkflowStateError(f"Invalid workflow transition: {state.stage} -> {nxt}")
    state.stage = nxt


def _read_lifecycle(value: object) -> WorkflowLifecycleStatus:
    allowed: tuple[WorkflowLifecycleStatus, ...] = ("active", "paused", "completed", "failed", "canceled")
    if isinstance(value, str) and value in allowed:
        return value  # type: ignore[return-value]
    return "active"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _read_bool(value: object) -> bool:
    return value is True


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
