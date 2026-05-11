"""Shared types for workspace context storage and search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class ContextItem:
    id: str
    task_id: str | None
    kind: str
    path: str | None
    content: str
    metadata: dict[str, Any]
    created_at: str


@dataclass
class SearchResult:
    item: ContextItem
    score: float
    provider: str


@dataclass
class HandoffPayload:
    v: int
    current_goal: str
    current_branch: str | None
    important_files: list[str]
    decisions_made: list[str]
    work_completed: list[str]
    tests_run: list[str]
    remaining_work: list[str]
    risks_blockers: list[str]
    suggested_next_agent_role: str | None


@dataclass(frozen=True)
class EmbeddingRecord:
    context_item_id: str
    provider: str
    model: str
    dim: int
    vector_blob: bytes
    created_at: str


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    model: str

    def available(self) -> bool: ...

    def embed(self, text: str) -> list[float]: ...
