from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .workflow_state import WorkflowLifecycleStatus, WorkflowStage, WorkflowState, transition_stage


@dataclass(frozen=True)
class SessionStatus:
    stage: WorkflowStage
    lifecycle_status: WorkflowLifecycleStatus
    retry_available: bool


class SessionRunner(Protocol):
    def start(self) -> SessionStatus: ...

    def resume(self) -> SessionStatus: ...

    def advance(self, stage: WorkflowStage) -> SessionStatus: ...

    def retry(self) -> SessionStatus: ...

    def cancel_current_step(self) -> SessionStatus: ...

    def status(self) -> SessionStatus: ...

    def exit(self) -> SessionStatus: ...


class WorkflowSessionRunner:
    """Stage-transition authority for interactive orchestration."""

    def __init__(self, state: WorkflowState) -> None:
        self._state = state
        self._retry_available = False

    def start(self) -> SessionStatus:
        self._state.lifecycle_status = "active"
        return self.status()

    def resume(self) -> SessionStatus:
        if self._state.lifecycle_status == "paused":
            self._state.lifecycle_status = "active"
        return self.status()

    def advance(self, stage: WorkflowStage) -> SessionStatus:
        transition_stage(self._state, stage)
        self._retry_available = False
        return self.status()

    def retry(self) -> SessionStatus:
        if not self._retry_available:
            return self.status()
        self._state.lifecycle_status = "active"
        self._retry_available = False
        return self.status()

    def cancel_current_step(self) -> SessionStatus:
        self._state.lifecycle_status = "canceled"
        return self.status()

    def status(self) -> SessionStatus:
        return SessionStatus(
            stage=self._state.stage,
            lifecycle_status=self._state.lifecycle_status,
            retry_available=self._retry_available,
        )

    def exit(self) -> SessionStatus:
        if self._state.lifecycle_status == "active":
            self._state.lifecycle_status = "paused"
        return self.status()

    def mark_retry_available(self) -> None:
        self._retry_available = True
