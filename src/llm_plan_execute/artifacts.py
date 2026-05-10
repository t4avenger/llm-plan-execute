from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import (
    ProviderResult,
    RunState,
    assignment_to_dict,
    clarification_to_dict,
    execution_policy_to_dict,
    provider_result_to_dict,
)


def ensure_run_dir(run: RunState) -> None:
    run.run_dir.mkdir(parents=True, exist_ok=True)


def write_text(run: RunState, name: str, content: str) -> Path:
    ensure_run_dir(run)
    path = run.run_dir / name
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path


def write_json(run: RunState, name: str, payload: dict[str, Any]) -> Path:
    ensure_run_dir(run)
    path = run.run_dir / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def result_to_dict(result: ProviderResult) -> dict[str, Any]:
    item = provider_result_to_dict(result)
    item["elapsed_seconds"] = round(result.elapsed_seconds, 3)
    return item


def write_state(run: RunState) -> Path:
    return write_json(
        run,
        "run.json",
        {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "prompt": run.prompt,
            "accepted_plan": run.accepted_plan,
            "build_output": run.build_output,
            "build_status": run.build_status,
            "build_failure": run.build_failure,
            "assignments": {role: assignment_to_dict(item) for role, item in run.assignments.items()},
            "results": [result_to_dict(result) for result in run.results],
            "warnings": run.warnings,
            "next_options": run.next_options,
            "clarification": clarification_to_dict(run.clarification) if run.clarification else None,
            "execution_policies": {
                role: execution_policy_to_dict(policy) for role, policy in run.execution_policies.items()
            },
        },
    )


def load_state(path: Path) -> dict[str, Any]:
    run_file = path / "run.json" if path.is_dir() else path
    return json.loads(run_file.read_text(encoding="utf-8"))
