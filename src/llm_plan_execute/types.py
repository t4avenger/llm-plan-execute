from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PERMISSION_MODES = ("read-only", "workspace-write", "full-access")
ROLES = (
    "planner",
    "plan_reviewer_a",
    "plan_reviewer_b",
    "plan_arbiter",
    "builder",
    "build_reviewer_a",
    "build_reviewer_b",
    "build_arbiter",
)


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    name: str
    roles: tuple[str, ...] = ()
    reasoning: int = 3
    speed: int = 3
    cost: int = 3
    context: int = 3
    exact_usage: bool = False

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.name}"


@dataclass(frozen=True)
class ModelAssignment:
    role: str
    model: ModelInfo
    reused: bool = False
    reason: str = ""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    exact: bool = False
    confidence: str = "estimated"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ProviderResult:
    role: str
    model: ModelInfo
    prompt: str
    output: str
    usage: Usage
    elapsed_seconds: float
    error: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class ExecutionPolicy:
    mode: str = "workspace-write"
    writable_dirs: tuple[Path, ...] = ()


@dataclass
class Clarification:
    status: str = "skipped"
    questions: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class RunState:
    run_id: str
    prompt: str
    run_dir: Path
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    accepted_plan: str | None = None
    build_output: str | None = None
    build_status: str | None = None
    build_failure: str | None = None
    assignments: dict[str, ModelAssignment] = field(default_factory=dict)
    results: list[ProviderResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_options: list[str] = field(default_factory=list)
    clarification: Clarification | None = None
    execution_policies: dict[str, ExecutionPolicy] = field(default_factory=dict)

    @classmethod
    def create(cls, prompt: str, runs_root: Path) -> RunState:
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        return cls(run_id=run_id, prompt=prompt, run_dir=runs_root / run_id)


def model_to_dict(model: ModelInfo) -> dict[str, Any]:
    return {
        "provider": model.provider,
        "name": model.name,
        "id": model.id,
        "roles": list(model.roles),
        "reasoning": model.reasoning,
        "speed": model.speed,
        "cost": model.cost,
        "context": model.context,
        "exact_usage": model.exact_usage,
    }


def assignment_to_dict(assignment: ModelAssignment) -> dict[str, Any]:
    return {
        "role": assignment.role,
        "model": model_to_dict(assignment.model),
        "reused": assignment.reused,
        "reason": assignment.reason,
    }


def usage_to_dict(usage: Usage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cost_usd": usage.cost_usd,
        "exact": usage.exact,
        "confidence": usage.confidence,
    }


def provider_result_to_dict(result: ProviderResult) -> dict[str, Any]:
    item = {
        "role": result.role,
        "model": model_to_dict(result.model),
        "prompt": result.prompt,
        "output": result.output,
        "usage": usage_to_dict(result.usage),
        "elapsed_seconds": result.elapsed_seconds,
        "error": result.error,
    }
    if result.warning:
        item["warning"] = result.warning
    return item


def clarification_to_dict(clarification: Clarification) -> dict[str, Any]:
    return {
        "status": clarification.status,
        "questions": clarification.questions,
        "assumptions": clarification.assumptions,
        "answers": clarification.answers,
        "raw_output": clarification.raw_output,
    }


def execution_policy_to_dict(policy: ExecutionPolicy) -> dict[str, Any]:
    return {
        "mode": policy.mode,
        "writable_dirs": [str(path) for path in policy.writable_dirs],
    }
