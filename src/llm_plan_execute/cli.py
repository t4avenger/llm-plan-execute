from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import load_state, write_state, write_text
from .config import (
    DEFAULT_CONFIG,
    ConfigIssue,
    ConfigValidation,
    ExecutionConfig,
    format_validation,
    load_config,
    validate_config_file,
    write_sample_config,
)
from .providers import ProviderRouter
from .reporting import render_report
from .selection import assign_models
from .types import (
    PERMISSION_MODES,
    Clarification,
    ExecutionPolicy,
    ModelAssignment,
    ModelInfo,
    ProviderResult,
    RunState,
    Usage,
)
from .workflow import (
    BuildFailedError,
    accept_plan,
    complete_planning,
    render_clarification,
    run_build,
    run_clarification,
    run_planning,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-plan-execute")
    parser.add_argument("--config", type=Path, default=None, help="Path to config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Use simulated providers and models.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument("--verbose", action="store_true", help="Show provider error details in progress output.")
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
    plan.add_argument("--no-clarify", action="store_true", help="Skip the clarification phase.")
    _add_permission_args(plan)

    run = sub.add_parser("run", help="Plan, approve, build, review, and report in one interactive flow.")
    run.add_argument("--prompt", default=None)
    run.add_argument("--prompt-file", type=Path, default=None)
    run.add_argument("--no-clarify", action="store_true", help="Skip the clarification phase.")
    _add_permission_args(run)

    accept = sub.add_parser("accept", help="Accept a reviewed proposed plan.")
    accept.add_argument("--run-dir", type=Path, required=True)

    build = sub.add_parser("build", help="Run build and review from an accepted plan.")
    build.add_argument("--run-dir", type=Path, required=True)
    _add_permission_args(build)

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
    progress = ProgressReporter(enabled=not args.quiet, verbose=args.verbose, stream=sys.stderr)

    return _dispatch_command(args, router, app_config, progress)


def _dispatch_command(
    args: argparse.Namespace,
    router: ProviderRouter,
    app_config: Any,
    progress: ProgressReporter,
) -> int:
    handlers = {
        "models": lambda: _cmd_models(router),
        "plan": lambda: _cmd_plan(args, router, app_config.runs_dir, app_config.execution, progress),
        "run": lambda: _cmd_run(args, router, app_config.runs_dir, app_config.execution, progress),
        "build": lambda: _cmd_build(args, router, app_config.execution, progress),
        "accept": lambda: _cmd_accept(args),
        "report": lambda: _cmd_report(args),
    }
    handler = handlers.get(args.command)
    return handler() if handler else 1


def _add_permission_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--permission-mode",
        choices=PERMISSION_MODES,
        default=None,
        help="Override provider execution permissions for this command.",
    )
    parser.add_argument(
        "--writable-dir",
        action="append",
        default=[],
        type=Path,
        help="Additional directory the provider may write when supported by the provider CLI.",
    )


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


def _cmd_plan(
    args: argparse.Namespace,
    router: ProviderRouter,
    runs_dir: Path,
    execution: ExecutionConfig,
    progress: ProgressReporter,
) -> int:
    prompt = _read_prompt(args.prompt, args.prompt_file)
    execution = _execution_with_cli_dirs(execution, args.writable_dir)
    if args.no_clarify:
        run = run_planning(
            prompt,
            runs_dir,
            router,
            auto_accept=args.yes,
            execution=execution,
            permission_mode=args.permission_mode,
            progress=progress.update,
        )
    else:
        run = run_clarification(
            prompt,
            runs_dir,
            router,
            execution=execution,
            permission_mode=args.permission_mode,
            progress=progress.update,
        )
        clarification = run.clarification
        if clarification and clarification.status == "needs_questions":
            if not sys.stdin.isatty():
                print(f"Run: {run.run_id}")
                print(f"Clarification needed: {run.run_dir / '00-clarification.md'}")
                print("Answer the questions and rerun planning, or use --no-clarify to plan with assumptions.")
                print(f"Report: {run.run_dir / 'report.md'}")
                return 2
            clarification.answers = [_ask_question(question) for question in clarification.questions]
            clarification.status = "clear"
            write_text(run, "00-clarification.md", render_clarification(clarification))
            write_state(run)
            print(f"Clarification: answered {len(clarification.answers)} question(s).")
        else:
            print(f"Clarification: no questions required ({run.run_dir / '00-clarification.md'}).")
        run = complete_planning(
            run,
            router,
            auto_accept=args.yes,
            execution=execution,
            permission_mode=args.permission_mode,
            progress=progress.update,
        )
    print(f"Run: {run.run_id}")
    if args.yes:
        print(f"Accepted plan: {run.run_dir / '04-accepted-plan.md'}")
    else:
        print(f"Proposed plan: {run.run_dir / '04-proposed-plan.md'}")
        print(f"Accept with: llm-plan-execute accept --run-dir {run.run_dir}")
    print(f"Report: {run.run_dir / 'report.md'}")
    return 0


def _cmd_run(
    args: argparse.Namespace,
    router: ProviderRouter,
    runs_dir: Path,
    execution: ExecutionConfig,
    progress: ProgressReporter,
) -> int:
    prompt = _read_prompt(args.prompt, args.prompt_file)
    execution = _execution_with_cli_dirs(execution, args.writable_dir)
    if args.no_clarify:
        run = run_planning(
            prompt,
            runs_dir,
            router,
            auto_accept=False,
            execution=execution,
            permission_mode=args.permission_mode,
            progress=progress.update,
        )
    else:
        run = run_clarification(
            prompt,
            runs_dir,
            router,
            execution=execution,
            permission_mode=args.permission_mode,
            progress=progress.update,
        )
        clarification = run.clarification
        if clarification and clarification.status == "needs_questions":
            if not sys.stdin.isatty():
                print(f"Run: {run.run_id}")
                print(f"Clarification needed: {run.run_dir / '00-clarification.md'}")
                print("Run requires an interactive terminal to answer clarification questions.")
                print(f"Report: {run.run_dir / 'report.md'}")
                return 2
            clarification.answers = [_ask_question(question) for question in clarification.questions]
            clarification.status = "clear"
            write_text(run, "00-clarification.md", render_clarification(clarification))
            write_state(run)
            print(f"Clarification: answered {len(clarification.answers)} question(s).")
        else:
            print(f"Clarification: no questions required ({run.run_dir / '00-clarification.md'}).")
        run = complete_planning(
            run,
            router,
            auto_accept=False,
            execution=execution,
            permission_mode=args.permission_mode,
            progress=progress.update,
        )

    proposed_path = run.run_dir / "04-proposed-plan.md"
    proposed_plan = proposed_path.read_text(encoding="utf-8")
    print(f"Run: {run.run_id}")
    print(f"Proposed plan: {proposed_path}")
    print("")
    print(proposed_plan.rstrip())
    print("")
    decision, build_permission_mode = _run_approval_decision(args.permission_mode or execution.build_mode)
    if decision == "cancel":
        print(f"Canceled before build. Report: {run.run_dir / 'report.md'}")
        return 130
    if decision == "save-only":
        print(f"Saved proposed plan. Accept later with: llm-plan-execute accept --run-dir {run.run_dir}")
        print(f"Report: {run.run_dir / 'report.md'}")
        return 0

    run = accept_plan(run)
    try:
        run = run_build(
            run,
            router,
            execution=execution,
            permission_mode=build_permission_mode,
            progress=progress.update,
        )
    except BuildFailedError as exc:
        run = exc.run
        print(f"Build failed: {run.run_dir / '05-build-output.md'}")
        print(f"Report: {run.run_dir / 'report.md'}")
        return 1
    print(f"Build output: {run.run_dir / '05-build-output.md'}")
    print(f"Review summary: {run.run_dir / '08-build-review-summary.md'}")
    print(f"Report: {run.run_dir / 'report.md'}")
    print("Next options: inspect the report, accept the build as-is, or return to planning with review feedback.")
    return 0


def _cmd_accept(args: argparse.Namespace) -> int:
    raw = load_state(args.run_dir)
    run = _state_from_json(raw, args.run_dir)
    run = accept_plan(run)
    print(f"Accepted plan: {run.run_dir / '04-accepted-plan.md'}")
    print(f"Build with: llm-plan-execute build --run-dir {run.run_dir}")
    print(f"Report: {run.run_dir / 'report.md'}")
    return 0


def _cmd_build(
    args: argparse.Namespace,
    router: ProviderRouter,
    execution: ExecutionConfig,
    progress: ProgressReporter,
) -> int:
    raw = load_state(args.run_dir)
    run = _state_from_json(raw, args.run_dir)
    execution = _execution_with_cli_dirs(execution, args.writable_dir)
    try:
        run = run_build(
            run,
            router,
            execution=execution,
            permission_mode=args.permission_mode,
            progress=progress.update,
        )
    except BuildFailedError as exc:
        run = exc.run
        print(f"Build failed: {run.run_dir / '05-build-output.md'}")
        print(f"Report: {run.run_dir / 'report.md'}")
        return 1
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
        build_status=raw.get("build_status") if isinstance(raw.get("build_status"), str) else None,
        build_failure=raw.get("build_failure") if isinstance(raw.get("build_failure"), str) else None,
        warnings=_string_list(raw.get("warnings")),
        next_options=_string_list(raw.get("next_options")),
        clarification=_clarification_from_json(raw.get("clarification")),
        execution_policies=_execution_policies_from_json(raw.get("execution_policies")),
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


def _clarification_from_json(value: object) -> Clarification | None:
    if not isinstance(value, dict):
        return None
    return Clarification(
        status=str(value.get("status", "skipped")),
        questions=_string_list(value.get("questions")),
        assumptions=_string_list(value.get("assumptions")),
        answers=_string_list(value.get("answers")),
        raw_output=str(value.get("raw_output", "")),
    )


def _execution_policies_from_json(value: object) -> dict[str, ExecutionPolicy]:
    if not isinstance(value, dict):
        return {}
    policies: dict[str, ExecutionPolicy] = {}
    for role, item in value.items():
        if not isinstance(role, str) or not isinstance(item, dict):
            continue
        mode = item.get("mode")
        dirs = item.get("writable_dirs", [])
        if not isinstance(mode, str) or mode not in PERMISSION_MODES:
            continue
        policies[role] = ExecutionPolicy(
            mode,
            tuple(Path(path) for path in dirs if isinstance(path, str)),
        )
    return policies


def _execution_with_cli_dirs(execution: ExecutionConfig, writable_dirs: list[Path]) -> ExecutionConfig:
    if not writable_dirs:
        return execution
    return ExecutionConfig(
        default_mode=execution.default_mode,
        planning_mode=execution.planning_mode,
        review_mode=execution.review_mode,
        build_mode=execution.build_mode,
        writable_dirs=(*execution.writable_dirs, *writable_dirs),
    )


def _ask_question(question: str) -> str:
    print(question)
    return input("> ").strip()


def _approval_decision() -> str:
    if not sys.stdin.isatty():
        return "save-only"
    while True:
        answer = input("Approve plan? [approve/cancel/save-only] > ").strip().lower()
        if answer in {"approve", "a", "yes", "y"}:
            return "approve"
        if answer in {"cancel", "c", "no", "n"}:
            return "cancel"
        if answer in {"save-only", "save", "s"}:
            return "save-only"
        print("Enter approve, cancel, or save-only.")


def _run_approval_decision(default_permission: str) -> tuple[str, str | None]:
    if not sys.stdin.isatty():
        return "save-only", None
    while True:
        answer = (
            input(
                "Approve plan? [approve/cancel/save-only/read-only/workspace-write/full-access] "
                f"(permission default: {default_permission}) > "
            )
            .strip()
            .lower()
        )
        if answer in {"approve", "a", "yes", "y", ""}:
            return "approve", default_permission
        if answer in {"cancel", "c", "no", "n"}:
            return "cancel", None
        if answer in {"save-only", "save", "s"}:
            return "save-only", None
        if answer in PERMISSION_MODES:
            return "approve", answer
        print("Enter approve, cancel, save-only, or a permission mode.")


class ProgressReporter:
    def __init__(self, *, enabled: bool, verbose: bool, stream: Any) -> None:
        self.enabled = enabled
        self.verbose = verbose
        self.stream = stream
        self.starts: dict[str, float] = {}

    def update(
        self,
        event: str,
        role: str,
        run: RunState,
        model: ModelInfo | None,
        result: ProviderResult | None,
        artifact: Path | None,
    ) -> None:
        if not self.enabled:
            return
        label = role.replace("_", " ")
        if event == "start":
            self.starts[role] = time.monotonic()
            model_id = model.id if model else "unassigned"
            print(f"[{run.run_id}] {label}: starting with {model_id}", file=self.stream)
            return
        if event == "finish":
            elapsed = result.elapsed_seconds if result else time.monotonic() - self.starts.get(role, time.monotonic())
            model_id = result.model.id if result else (model.id if model else "unassigned")
            suffix = " failed" if result and result.error else " done"
            print(f"[{run.run_id}] {label}: {model_id}{suffix} in {elapsed:.1f}s", file=self.stream)
            if self.verbose and result and result.error:
                print(f"[{run.run_id}] {label}: {result.error}", file=self.stream)
            return
        if event == "artifact" and artifact:
            print(f"[{run.run_id}] {label}: wrote {artifact}", file=self.stream)
