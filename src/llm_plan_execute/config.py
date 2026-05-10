from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import ROLES, ModelInfo

DEFAULT_ROOT = Path(".llm-plan-execute")
DEFAULT_CONFIG = DEFAULT_ROOT / "config.json"
SUPPORTED_PROVIDERS = frozenset({"codex", "cursor", "claude"})
SCORE_FIELDS = ("reasoning", "speed", "cost", "context")
MIN_SCORE = 1
MAX_SCORE = 5


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
    workspace: Path = Path(".")


@dataclass(frozen=True)
class ConfigIssue:
    severity: str
    path: str
    message: str


@dataclass(frozen=True)
class ConfigValidation:
    errors: tuple[ConfigIssue, ...]
    warnings: tuple[ConfigIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def sample_config() -> dict[str, Any]:
    return {
        "dry_run": False,
        "runs_dir": ".llm-plan-execute/runs",
        "workspace": ".",
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
                "enabled": False,
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

    validation = validate_config_data(raw, require_providers=not bool(raw.get("dry_run", False) or dry_run))
    if validation.errors:
        raise ValueError(format_validation(validation))

    return parse_config(raw, dry_run=dry_run)


def parse_config(raw: dict[str, Any], *, dry_run: bool = False) -> AppConfig:
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
        workspace=Path(raw.get("workspace", ".")),
    )


def validate_config_file(path: Path | None, *, dry_run: bool = False) -> ConfigValidation:
    config_path = path or DEFAULT_CONFIG
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return ConfigValidation(
                errors=(ConfigIssue("error", str(config_path), f"invalid JSON: {exc}"),),
                warnings=(),
            )
    else:
        raw = sample_config()
        raw["dry_run"] = True
    return validate_config_data(raw, require_providers=not bool(raw.get("dry_run", False) or dry_run))


def validate_config_data(raw: object, *, require_providers: bool | None = None) -> ConfigValidation:
    errors: list[ConfigIssue] = []
    warnings: list[ConfigIssue] = []
    seen_model_ids: set[str] = set()

    if not isinstance(raw, dict):
        return ConfigValidation((ConfigIssue("error", "$", "config must be a JSON object"),), ())
    if require_providers is None:
        require_providers = not bool(raw.get("dry_run", False))

    _validate_root_fields(raw, errors)

    providers = raw.get("providers")
    if providers is None and not require_providers:
        providers = []
    if not isinstance(providers, list):
        errors.append(ConfigIssue("error", "providers", "must be a list"))
        return ConfigValidation(tuple(errors), tuple(warnings))
    if require_providers and not providers:
        errors.append(ConfigIssue("error", "providers", "must include at least one provider"))

    for index, provider in enumerate(providers):
        _validate_provider(index, provider, seen_model_ids, errors, warnings)

    return ConfigValidation(tuple(errors), tuple(warnings))


def _validate_root_fields(raw: dict[str, Any], errors: list[ConfigIssue]) -> None:
    if "dry_run" in raw and not isinstance(raw["dry_run"], bool):
        errors.append(ConfigIssue("error", "dry_run", "must be a boolean"))
    if "runs_dir" in raw and not isinstance(raw["runs_dir"], str):
        errors.append(ConfigIssue("error", "runs_dir", "must be a string path"))
    if "workspace" in raw and not isinstance(raw["workspace"], str):
        errors.append(ConfigIssue("error", "workspace", "must be a string path"))


def _validate_provider(
    index: int,
    provider: object,
    seen_model_ids: set[str],
    errors: list[ConfigIssue],
    warnings: list[ConfigIssue],
) -> None:
    provider_path = f"providers[{index}]"
    if not isinstance(provider, dict):
        errors.append(ConfigIssue("error", provider_path, "provider must be an object"))
        return

    name = _validate_provider_name(provider_path, provider.get("name"), errors)
    command = provider.get("command", name)
    if not isinstance(command, str) or not command:
        errors.append(ConfigIssue("error", f"{provider_path}.command", "must be a non-empty string"))
    if "enabled" in provider and not isinstance(provider["enabled"], bool):
        errors.append(ConfigIssue("error", f"{provider_path}.enabled", "must be a boolean"))

    models = provider.get("models")
    if not isinstance(models, list):
        errors.append(ConfigIssue("error", f"{provider_path}.models", "must be a list"))
        return
    if provider.get("enabled", True) and not models:
        warnings.append(ConfigIssue("warning", f"{provider_path}.models", "enabled provider has no models"))

    for model_index, model in enumerate(models):
        _validate_model(provider_path, model_index, model, name, seen_model_ids, errors)


def _validate_provider_name(provider_path: str, name: object, errors: list[ConfigIssue]) -> str:
    if not isinstance(name, str) or not name:
        errors.append(ConfigIssue("error", f"{provider_path}.name", "must be a non-empty string"))
        return "<unknown>"
    if name not in SUPPORTED_PROVIDERS:
        errors.append(
            ConfigIssue(
                "error",
                f"{provider_path}.name",
                f"unsupported provider {name!r}; expected one of {sorted(SUPPORTED_PROVIDERS)}",
            )
        )
    return name


def _validate_model(
    provider_path: str,
    model_index: int,
    model: object,
    provider_name: str,
    seen_model_ids: set[str],
    errors: list[ConfigIssue],
) -> None:
    model_path = f"{provider_path}.models[{model_index}]"
    if not isinstance(model, dict):
        errors.append(ConfigIssue("error", model_path, "model must be an object"))
        return

    model_name = model.get("name")
    if not isinstance(model_name, str) or not model_name:
        errors.append(ConfigIssue("error", f"{model_path}.name", "must be a non-empty string"))
        return
    model_id = f"{provider_name}:{model_name}"
    if model_id in seen_model_ids:
        errors.append(ConfigIssue("error", f"{model_path}.name", f"duplicate model id {model_id!r}"))
    seen_model_ids.add(model_id)

    _validate_roles(model_path, model.get("roles", []), errors)
    _validate_scores(model_path, model, errors)
    if "exact_usage" in model and not isinstance(model["exact_usage"], bool):
        errors.append(ConfigIssue("error", f"{model_path}.exact_usage", "must be a boolean"))


def _validate_roles(model_path: str, roles: object, errors: list[ConfigIssue]) -> None:
    if not isinstance(roles, list):
        errors.append(ConfigIssue("error", f"{model_path}.roles", "must be a list"))
        return
    for role_index, role in enumerate(roles):
        role_path = f"{model_path}.roles[{role_index}]"
        if role not in ROLES:
            errors.append(ConfigIssue("error", role_path, f"unsupported role {role!r}"))


def _validate_scores(model_path: str, model: dict[str, Any], errors: list[ConfigIssue]) -> None:
    for score in SCORE_FIELDS:
        value = model.get(score, 3)
        if not isinstance(value, int) or isinstance(value, bool) or not MIN_SCORE <= value <= MAX_SCORE:
            errors.append(
                ConfigIssue("error", f"{model_path}.{score}", f"must be an integer from {MIN_SCORE} to {MAX_SCORE}")
            )


def format_validation(validation: ConfigValidation) -> str:
    lines = ["Invalid config:" if validation.errors else "Config warnings:"]
    for issue in (*validation.errors, *validation.warnings):
        lines.append(f"- {issue.severity}: {issue.path}: {issue.message}")
    return "\n".join(lines)
