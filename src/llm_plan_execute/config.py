from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import ModelInfo

DEFAULT_ROOT = Path(".llm-plan-execute")
DEFAULT_CONFIG = DEFAULT_ROOT / "config.json"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    command: str
    enabled: bool
    models: tuple[ModelInfo, ...]


@dataclass(frozen=True)
class AppConfig:
    providers: tuple[ProviderConfig, ...]
    runs_dir: Path = DEFAULT_ROOT / "runs"
    dry_run: bool = False


def sample_config() -> dict[str, Any]:
    return {
        "dry_run": False,
        "runs_dir": ".llm-plan-execute/runs",
        "providers": [
            {
                "name": "codex",
                "command": "codex",
                "enabled": True,
                "models": [
                    {
                        "name": "gpt-5.5",
                        "roles": ["planner", "plan_arbiter", "build_arbiter"],
                        "reasoning": 5,
                        "speed": 3,
                        "cost": 5,
                        "context": 5,
                        "exact_usage": False,
                    },
                    {
                        "name": "gpt-5.4",
                        "roles": ["builder", "plan_reviewer_a", "build_reviewer_a"],
                        "reasoning": 4,
                        "speed": 4,
                        "cost": 4,
                        "context": 4,
                        "exact_usage": False,
                    },
                ],
            },
            {
                "name": "claude",
                "command": "claude",
                "enabled": True,
                "models": [
                    {
                        "name": "opus",
                        "roles": ["plan_reviewer_b", "build_reviewer_b"],
                        "reasoning": 5,
                        "speed": 3,
                        "cost": 5,
                        "context": 5,
                        "exact_usage": False,
                    },
                    {
                        "name": "sonnet",
                        "roles": ["builder", "plan_reviewer_a", "build_reviewer_a"],
                        "reasoning": 4,
                        "speed": 4,
                        "cost": 3,
                        "context": 5,
                        "exact_usage": False,
                    },
                ],
            },
            {
                "name": "cursor",
                "command": "cursor-agent",
                "enabled": False,
                "models": [
                    {
                        "name": "auto",
                        "roles": ["builder"],
                        "reasoning": 3,
                        "speed": 4,
                        "cost": 3,
                        "context": 4,
                        "exact_usage": False,
                    }
                ],
            },
        ],
    }


def write_sample_config(path: Path = DEFAULT_CONFIG) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample_config(), indent=2) + "\n", encoding="utf-8")
    return path


def load_config(path: Path | None, *, dry_run: bool = False) -> AppConfig:
    config_path = path or DEFAULT_CONFIG
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        raw = sample_config()
        raw["dry_run"] = True

    providers: list[ProviderConfig] = []
    for provider in raw.get("providers", []):
        models: list[ModelInfo] = []
        for model in provider.get("models", []):
            models.append(
                ModelInfo(
                    provider=provider["name"],
                    name=model["name"],
                    roles=tuple(model.get("roles", ())),
                    reasoning=int(model.get("reasoning", 3)),
                    speed=int(model.get("speed", 3)),
                    cost=int(model.get("cost", 3)),
                    context=int(model.get("context", 3)),
                    exact_usage=bool(model.get("exact_usage", False)),
                )
            )
        providers.append(
            ProviderConfig(
                name=provider["name"],
                command=provider.get("command", provider["name"]),
                enabled=bool(provider.get("enabled", True)),
                models=tuple(models),
            )
        )

    return AppConfig(
        providers=tuple(providers),
        runs_dir=Path(raw.get("runs_dir", ".llm-plan-execute/runs")),
        dry_run=bool(raw.get("dry_run", False) or dry_run),
    )
