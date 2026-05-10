from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .artifacts import load_state
from .config import (
    DEFAULT_CONFIG,
    ConfigIssue,
    ConfigValidation,
    format_validation,
    load_config,
    validate_config_file,
    write_sample_config,
)
from .providers import ProviderRouter
from .reporting import render_report
from .selection import assign_models
from .types import ModelAssignment, ModelInfo, ProviderResult, RunState, Usage
from .workflow import run_build, run_planning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-plan-execute")
    parser.add_argument("--config", type=Path, default=None, help="Path to config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Use simulated providers and models.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-config", help="Write a sample local config.")
    init.add_argument("--path", type=Path, default=DEFAULT_CONFIG)

    config = sub.add_parser("config", help="Inspect and validate configuration.")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("validate", help="Validate config shape and provider command availability.")

    sub.add_parser("models", help="List available models and role assignments.")

    plan = sub.add_parser("plan", help="Create and review an implementation plan.")
    plan.add_argument("--prompt", default=None)
    plan.add_argument("--prompt-file", type=Path, default=None)
    plan.add_argument("--yes", action="store_true", help="Accept the arbiter-revised plan non-interactively.")

    build = sub.add_parser("build", help="Run build and review from an accepted plan.")
    build.add_argument("--run-dir", type=Path, required=True)

    report = sub.add_parser("report", help="Render a report for a run.")
    report.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "init-config":
        return _cmd_init_config(args)

    if args.command == "config":
        return _cmd_config(args)

    app_config = load_config(args.config, dry_run=args.dry_run)
    router = ProviderRouter.from_config(app_config)

    return _dispatch_command(args, router, app_config.runs_dir)


def _dispatch_command(args: argparse.Namespace, router: ProviderRouter, runs_dir: Path) -> int:
    if args.command == "models":
        return _cmd_models(router)

    if args.command == "plan":
        return _cmd_plan(args, router, runs_dir)

    if args.command == "build":
        return _cmd_build(args, router)

    if args.command == "report":
        return _cmd_report(args)

    return 1


def _cmd_init_config(args: argparse.Namespace) -> int:
    path = write_sample_config(args.path)
    print(f"Wrote {path}")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    if args.config_command != "validate":
        return 1

    validation = validate_config_file(args.config, dry_run=args.dry_run)
    if validation.errors:
        print(format_validation(validation), file=sys.stderr)
        return 1

    config = load_config(args.config, dry_run=args.dry_run)
    command_errors: list[ConfigIssue] = []
    if not config.dry_run:
        command_errors = [
            ConfigIssue(
                "error",
                f"providers[{index}].command",
                f"enabled provider {provider.name!r} command {provider.command!r} was not found on PATH",
            )
            for index, provider in enumerate(config.providers)
            if provider.enabled and shutil.which(provider.command) is None
        ]
    command_warnings = list(validation.warnings)
    for index, provider in enumerate(config.providers):
        if not provider.enabled:
            command_warnings.append(
                ConfigIssue("warning", f"providers[{index}]", f"provider {provider.name!r} is disabled")
            )

    if command_errors:
        print(format_validation(ConfigValidation(tuple(command_errors), tuple(command_warnings))), file=sys.stderr)
        return 1
    if command_warnings:
        print(format_validation(ConfigValidation((), tuple(command_warnings))))
    else:
        print("Config is valid.")
    return 0


def _cmd_models(router: ProviderRouter) -> int:
    models = router.available_models()
    if not models:
        print("No available models. Run init-config, install provider CLIs, or use --dry-run.")
        return 1
    assignments, warnings = assign_models(models)
    print(
        json.dumps(
            {
                "models": [model.__dict__ | {"id": model.id} for model in models],
                "assignments": _assignment_json(assignments),
                "warnings": warnings,
            },
            indent=2,
        )
    )
    return 0


def _cmd_plan(args: argparse.Namespace, router: ProviderRouter, runs_dir: Path) -> int:
    prompt = _read_prompt(args.prompt, args.prompt_file)
    run = run_planning(prompt, runs_dir, router, auto_accept=args.yes)
    print(f"Run: {run.run_id}")
    if args.yes:
        print(f"Accepted plan: {run.run_dir / '04-accepted-plan.md'}")
    else:
        print(f"Proposed plan: {run.run_dir / '04-proposed-plan.md'}")
        print("Rerun planning with --yes when you want this run to be buildable.")
    print(f"Report: {run.run_dir / 'report.md'}")
    return 0


def _cmd_build(args: argparse.Namespace, router: ProviderRouter) -> int:
    raw = load_state(args.run_dir)
    run = _state_from_json(raw, args.run_dir)
    run = run_build(run, router)
    print(f"Build output: {run.run_dir / '05-build-output.md'}")
    print(f"Review summary: {run.run_dir / '08-build-review-summary.md'}")
    print(f"Report: {run.run_dir / 'report.md'}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    raw = load_state(args.run_dir)
    run = _state_from_json(raw, args.run_dir)
    print(render_report(run), end="")
    return 0


def _read_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt:
        return prompt
    if prompt_file:
        return prompt_file.read_text(encoding="utf-8")
    raise SystemExit("Provide --prompt or --prompt-file.")


def _assignment_json(assignments: dict[str, ModelAssignment]) -> dict[str, object]:
    return {
        role: {
            "model": assignment.model.id,
            "reused": assignment.reused,
            "reason": assignment.reason,
        }
        for role, assignment in assignments.items()
    }


def _state_from_json(raw: dict[str, object], run_dir: Path) -> RunState:
    run = RunState(
        run_id=str(raw["run_id"]),
        prompt=str(raw["prompt"]),
        run_dir=run_dir if run_dir.name != "run.json" else run_dir.parent,
        created_at=str(raw.get("created_at", "")),
        accepted_plan=raw.get("accepted_plan") if isinstance(raw.get("accepted_plan"), str) else None,
        build_output=raw.get("build_output") if isinstance(raw.get("build_output"), str) else None,
        warnings=_string_list(raw.get("warnings")),
        next_options=_string_list(raw.get("next_options")),
    )
    for role, item in dict(raw.get("assignments", {})).items():
        if not isinstance(item, dict):
            continue
        model_raw = item["model"]
        model = _model_from_json(model_raw)
        run.assignments[role] = ModelAssignment(
            role=role,
            model=model,
            reused=bool(item.get("reused", False)),
            reason=str(item.get("reason", "loaded from run")),
        )
    for item in list(raw.get("results", [])):
        if not isinstance(item, dict):
            continue
        usage_raw = item.get("usage", {})
        if not isinstance(usage_raw, dict):
            usage_raw = {}
        run.results.append(
            ProviderResult(
                role=str(item["role"]),
                model=_model_from_json(item["model"]),
                prompt=str(item.get("prompt", "")),
                output=str(item.get("output", "")),
                usage=Usage(
                    input_tokens=int(usage_raw.get("input_tokens", 0)),
                    output_tokens=int(usage_raw.get("output_tokens", 0)),
                    cost_usd=_optional_float(usage_raw.get("cost_usd")),
                    exact=bool(usage_raw.get("exact", False)),
                    confidence=str(usage_raw.get("confidence", "estimated")),
                ),
                elapsed_seconds=float(item.get("elapsed_seconds", 0.0)),
                error=item.get("error") if isinstance(item.get("error"), str) else None,
            )
        )
    return run


def _model_from_json(model_raw: Any) -> ModelInfo:
    if not isinstance(model_raw, dict):
        provider, _, name = str(model_raw).partition(":")
        return ModelInfo(provider=provider or "unknown", name=name or str(model_raw))
    return ModelInfo(
        provider=str(model_raw["provider"]),
        name=str(model_raw["name"]),
        roles=tuple(str(role) for role in model_raw.get("roles", ())),
        reasoning=int(model_raw.get("reasoning", 3)),
        speed=int(model_raw.get("speed", 3)),
        cost=int(model_raw.get("cost", 3)),
        context=int(model_raw.get("context", 3)),
        exact_usage=bool(model_raw.get("exact_usage", False)),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
