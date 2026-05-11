from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import PERMISSION_MODES, ROLES, ExecutionPolicy, ModelInfo

DEFAULT_ROOT = Path(".llm-plan-execute")
DEFAULT_CONFIG = DEFAULT_ROOT / "config.json"
_MSG_JSON_VALUE_MUST_BE_OBJECT = "must be an object"
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
class ExecutionConfig:
    default_mode: str = "workspace-write"
    planning_mode: str = "read-only"
    review_mode: str = "read-only"
    build_mode: str = "workspace-write"
    writable_dirs: tuple[Path, ...] = ()

    def policy_for_role(self, role: str, *, mode_override: str | None = None) -> ExecutionPolicy:
        mode = mode_override or self.mode_for_role(role)
        return ExecutionPolicy(mode, self.writable_dirs)

    def mode_for_role(self, role: str) -> str:
        if role == "builder":
            return self.build_mode
        if "reviewer" in role or role.endswith("arbiter"):
            return self.review_mode
        if role in {"planner", "clarifier"}:
            return self.planning_mode
        return self.default_mode


@dataclass(frozen=True)
class BuildConfig:
    """Git/build orchestration defaults (see docs)."""

    base_branch: str | None = None
    create_pr: bool = False


@dataclass(frozen=True)
class AppConfig:
    providers: tuple[ProviderConfig, ...]
    runs_dir: Path = DEFAULT_ROOT / "runs"
    dry_run: bool = False
    workspace: Path = Path(".")
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    build: BuildConfig = field(default_factory=BuildConfig)


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
    codex_enabled = shutil.which("codex") is not None
    claude_enabled = shutil.which("claude") is not None
    cursor_enabled = shutil.which("cursor-agent") is not None
    return {
        "dry_run": False,
        "runs_dir": ".llm-plan-execute/runs",
        "workspace": ".",
        "execution": {
            "default_mode": "workspace-write",
            "phases": {
                "planning": "read-only",
                "review": "read-only",
                "build": "workspace-write",
            },
            "writable_dirs": [],
        },
        "build": {"base_branch": None, "create_pr": False},
        "providers": [
            {
                "name": "codex",
                "command": "codex",
                "enabled": codex_enabled,
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
                "enabled": claude_enabled,
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
                "enabled": cursor_enabled,
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


def resolve_repo(repo: Path | None) -> Path:
    """Resolve the workspace directory; defaults to the current working directory."""
    candidate = (repo if repo is not None else Path.cwd()).expanduser()
    resolved = candidate.resolve()
    if not resolved.exists():
        raise ValueError(f"Workspace {resolved} does not exist.")
    if not resolved.is_dir():
        raise ValueError(f"Workspace {resolved} is not a directory.")
    return resolved


def resolve_config_path(config_arg: Path | None, workspace: Path) -> Path:
    """Resolve which config file to load: explicit --config wins, else workspace-local default."""
    workspace = workspace.resolve()
    if config_arg is not None:
        config_arg = config_arg.expanduser()
        return config_arg.resolve() if config_arg.is_absolute() else (workspace / config_arg).resolve()
    return (workspace / DEFAULT_ROOT / "config.json").resolve()


def resolve_workspace_relative_path(workspace: Path, path: Path) -> Path:
    """Resolve a path that may be absolute or relative to workspace."""
    workspace = workspace.resolve()
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _must_live_under_workspace(workspace: Path, resolved: Path, *, field: str, original: Path) -> Path:
    ws = workspace.resolve()
    try:
        resolved.relative_to(ws)
    except ValueError as exc:
        raise ValueError(
            f"{field}: path {original} resolves outside workspace {ws}; choose a path inside the workspace."
        ) from exc
    return resolved


def normalize_runs_dir(workspace: Path, runs_dir: Path) -> Path:
    workspace = workspace.resolve()
    runs_dir = runs_dir.expanduser()
    candidate = runs_dir if runs_dir.is_absolute() else workspace / runs_dir
    resolved = candidate.resolve()  # NOSONAR - validated against the resolved workspace before use.
    return _must_live_under_workspace(workspace, resolved, field="runs_dir", original=runs_dir)


def normalize_writable_dirs(workspace: Path, paths: tuple[Path, ...], *, field: str) -> tuple[Path, ...]:
    normalized: list[Path] = []
    workspace = workspace.resolve()
    for index, path in enumerate(paths):
        expanded = path.expanduser()
        if expanded.is_absolute():
            normalized.append(expanded.resolve())
            continue
        resolved = (workspace / expanded).resolve()
        normalized.append(_must_live_under_workspace(workspace, resolved, field=f"{field}[{index}]", original=expanded))
    return tuple(normalized)


def load_config(config_arg: Path | None, *, workspace: Path, dry_run: bool = False) -> AppConfig:
    workspace = workspace.resolve()
    config_path = resolve_config_path(config_arg, workspace)
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    elif dry_run:
        raw = sample_config()
        raw["dry_run"] = True
    elif config_arg is not None:
        raise ValueError(
            f"Config file {config_path} was not found. Run init-config, pass --config, or use --dry-run explicitly."
        )
    else:
        raise ValueError(
            f"Config file {config_path} was not found for workspace {workspace}. "
            "Run init-config, pass --config, or use --dry-run explicitly."
        )

    validation = validate_config_data(raw, require_providers=not bool(raw.get("dry_run", False) or dry_run))
    if validation.errors:
        raise ValueError(format_validation(validation))

    return parse_config(raw, workspace=workspace, dry_run=dry_run)


def parse_config(raw: dict[str, Any], *, workspace: Path, dry_run: bool = False) -> AppConfig:
    workspace = workspace.resolve()
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

    runs_dir = normalize_runs_dir(workspace, Path(raw.get("runs_dir", ".llm-plan-execute/runs")))
    execution = _parse_execution(raw.get("execution", {}), workspace)
    build = _parse_build(raw.get("build", {}))

    return AppConfig(
        providers=tuple(providers),
        runs_dir=runs_dir,
        dry_run=bool(raw.get("dry_run", False) or dry_run),
        workspace=workspace,
        execution=execution,
        build=build,
    )


def _parse_build(raw: object) -> BuildConfig:
    if raw in ({}, None):
        return BuildConfig()
    if not isinstance(raw, dict):
        return BuildConfig()
    bb = raw.get("base_branch")
    base_branch = bb if isinstance(bb, str) and bb.strip() else None
    return BuildConfig(base_branch=base_branch, create_pr=bool(raw.get("create_pr", False)))


def _parse_execution(raw: object, workspace: Path) -> ExecutionConfig:
    if not isinstance(raw, dict):
        return ExecutionConfig()
    phases = raw.get("phases", {})
    if not isinstance(phases, dict):
        phases = {}
    writable_dirs = raw.get("writable_dirs", [])
    if not isinstance(writable_dirs, list):
        writable_dirs = []
    default_mode = str(raw.get("default_mode", "workspace-write"))
    paths = tuple(Path(path) for path in writable_dirs if isinstance(path, str))
    normalized = normalize_writable_dirs(workspace, paths, field="execution.writable_dirs")
    return ExecutionConfig(
        default_mode=default_mode,
        planning_mode=str(phases.get("planning", "read-only")),
        review_mode=str(phases.get("review", "read-only")),
        build_mode=str(phases.get("build", default_mode)),
        writable_dirs=normalized,
    )


def validate_config_file(config_arg: Path | None, *, workspace: Path, dry_run: bool = False) -> ConfigValidation:
    workspace = workspace.resolve()
    config_path = resolve_config_path(config_arg, workspace)
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return ConfigValidation(
                errors=(ConfigIssue("error", str(config_path), f"invalid JSON: {exc}"),),
                warnings=(),
            )
    elif dry_run:
        raw = sample_config()
        raw["dry_run"] = True
    else:
        return ConfigValidation(
            errors=(
                ConfigIssue(
                    "error",
                    str(config_path),
                    "config file was not found; run init-config, pass --config, or use --dry-run explicitly",
                ),
            ),
            warnings=(),
        )
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
    _validate_execution(raw.get("execution", {}), errors)
    _validate_build(raw.get("build"), errors)

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


def _validate_execution(raw: object, errors: list[ConfigIssue]) -> None:
    if raw in ({}, None):
        return
    if not isinstance(raw, dict):
        errors.append(ConfigIssue("error", "execution", _MSG_JSON_VALUE_MUST_BE_OBJECT))
        return
    _validate_permission_mode("execution.default_mode", raw.get("default_mode", "workspace-write"), errors)
    _validate_execution_phases(raw.get("phases", {}), errors)
    _validate_writable_dirs(raw.get("writable_dirs", []), errors)


def _validate_execution_phases(phases: object, errors: list[ConfigIssue]) -> None:
    if not isinstance(phases, dict):
        errors.append(ConfigIssue("error", "execution.phases", _MSG_JSON_VALUE_MUST_BE_OBJECT))
        return
    for phase in ("planning", "review", "build"):
        if phase in phases:
            _validate_permission_mode(f"execution.phases.{phase}", phases[phase], errors)


def _validate_build(raw: object, errors: list[ConfigIssue]) -> None:
    if raw in (None, {}):
        return
    if not isinstance(raw, dict):
        errors.append(ConfigIssue("error", "build", _MSG_JSON_VALUE_MUST_BE_OBJECT))
        return
    if "base_branch" in raw and raw["base_branch"] is not None and not isinstance(raw["base_branch"], str):
        errors.append(ConfigIssue("error", "build.base_branch", "must be a string or null"))
    if "create_pr" in raw and not isinstance(raw["create_pr"], bool):
        errors.append(ConfigIssue("error", "build.create_pr", "must be a boolean"))


def _validate_writable_dirs(writable_dirs: object, errors: list[ConfigIssue]) -> None:
    if not isinstance(writable_dirs, list):
        errors.append(ConfigIssue("error", "execution.writable_dirs", "must be a list of string paths"))
        return
    for index, item in enumerate(writable_dirs):
        if not isinstance(item, str) or not item:
            errors.append(ConfigIssue("error", f"execution.writable_dirs[{index}]", "must be a non-empty string path"))


def _validate_permission_mode(path: str, value: object, errors: list[ConfigIssue]) -> None:
    if value not in PERMISSION_MODES:
        errors.append(ConfigIssue("error", path, f"must be one of {list(PERMISSION_MODES)}"))


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
