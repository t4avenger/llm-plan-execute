from __future__ import annotations

from collections import Counter

from .types import ROLES, ModelAssignment, ModelInfo

ROLE_WEIGHTS: dict[str, dict[str, int]] = {
    "planner": {"reasoning": 5, "context": 3, "speed": 1, "cost": -1},
    "plan_reviewer_a": {"reasoning": 4, "context": 2, "speed": 1, "cost": -1},
    "plan_reviewer_b": {"reasoning": 4, "context": 2, "speed": 1, "cost": -1},
    "plan_arbiter": {"reasoning": 5, "context": 3, "speed": 1, "cost": -1},
    "builder": {"reasoning": 3, "context": 3, "speed": 4, "cost": -2},
    "build_reviewer_a": {"reasoning": 4, "context": 2, "speed": 2, "cost": -1},
    "build_reviewer_b": {"reasoning": 4, "context": 2, "speed": 2, "cost": -1},
    "build_arbiter": {"reasoning": 4, "context": 3, "speed": 2, "cost": -1},
}


def score_model(role: str, model: ModelInfo, used: Counter[str]) -> int:
    weights = ROLE_WEIGHTS[role]
    score = (
        weights["reasoning"] * model.reasoning
        + weights["context"] * model.context
        + weights["speed"] * model.speed
        + weights["cost"] * model.cost
    )
    if role in model.roles:
        score += 20
    elif model.roles:
        score -= 5
    score -= used[model.id] * 15
    score -= used[f"provider:{model.provider}"] * 4
    return score


def assign_models(models: list[ModelInfo]) -> tuple[dict[str, ModelAssignment], list[str]]:
    if not models:
        raise ValueError("No models are available. Add provider models to config or run with --dry-run.")

    warnings: list[str] = []
    assignments: dict[str, ModelAssignment] = {}
    used: Counter[str] = Counter()

    for role in ROLES:
        ranked = sorted(models, key=lambda model: score_model(role, model, used), reverse=True)
        selected = ranked[0]
        reused = used[selected.id] > 0
        reason = "highest deterministic score for role"
        if reused:
            warning = f"Role {role} reused {selected.id}; not enough distinct suitable models were available."
            warnings.append(warning)
            reason = "best available model after diversity fallback"
        assignments[role] = ModelAssignment(role=role, model=selected, reused=reused, reason=reason)
        used[selected.id] += 1
        used[f"provider:{selected.provider}"] += 1

    return assignments, warnings
