"""Persisted workflow orchestration state (separate from provider run.json)."""

from __future__ import annotations

import json
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


@dataclass
class WorkflowState:
    """Contract for interactive workflow persistence under each run directory."""

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

    def touch_accepted_plan(self, *, artifact_name: str = "04-accepted-plan.md") -> None:
        self.accepted_plan_artifact = artifact_name
        self.accepted_plan_version = (self.accepted_plan_version or 0) + 1

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, raw: dict[str, Any]) -> WorkflowState:
        return cls(
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
            terminal_report_printed=bool(raw.get("terminal_report_printed", False)),
        )


WORKFLOW_STATE_FILENAME = "workflow-state.json"


def workflow_state_path(run_dir: Path) -> Path:
    return run_dir / WORKFLOW_STATE_FILENAME


def load_workflow_state(run_dir: Path) -> WorkflowState:
    path = workflow_state_path(run_dir)
    if not path.exists():
        return WorkflowState()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return WorkflowState()
    return WorkflowState.from_json_dict(raw)


def save_workflow_state(run_dir: Path, state: WorkflowState) -> Path:
    path = workflow_state_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    return path


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


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
