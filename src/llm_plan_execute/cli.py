from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import load_state
from .config import (
    DEFAULT_CONFIG,
    AppConfig,
    ConfigIssue,
    ConfigValidation,
    format_validation,
    load_config,
    resolve_repo,
    resolve_workspace_relative_path,
    validate_config_file,
    write_sample_config,
)
from .interactive import InteractiveCanceledError, InteractiveSession
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
from .workflow import accept_plan, run_planning
from .workflow_runner import (
    execute_build_through_completion,
    gate_stage_transition,
    interactive_plan_review,
    merge_execution_dirs,
    orchestrate_clarification,
)
from .workflow_state import (
    WorkflowState,
    acquire_workflow_lock,
    load_workflow_state,
    release_workflow_lock,
    save_workflow_state,
    transition_stage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-plan-execute")
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        metavar="PATH",
        help="Workspace root to plan, build, or configure (default: current directory).",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Use simulated providers and models.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument("--verbose", action="store_true", help="Show provider error details in progress output.")
    parser.add_argument(
        "--non-interactive",
        "--ci",
        action="store_true",
        help="Use deterministic defaults for menus (also implied when stdin is not a TTY).",
    )
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

    run = sub.add_parser(
        "run",
        help="Plan, approve, build, review, and report in one interactive flow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="If you pause before build, resume with:\n  llm-plan-execute build --run-dir <run-dir>",
    )
    run.add_argument("--prompt", default=None)
    run.add_argument("--prompt-file", type=Path, default=None)
    run.add_argument("--no-clarify", action="store_true", help="Skip the clarification phase.")
    _add_permission_args(run)

    accept = sub.add_parser("accept", help="Accept a reviewed proposed plan.")
    accept.add_argument("--run-dir", type=Path, required=True)

    build = sub.add_parser(
        "build",
        help="Run build and review from an accepted plan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Loads workflow-state.json when present. After a paused `run`, continue from the same "
            "--run-dir so persisted workflow history is kept."
        ),
    )
    build.add_argument("--run-dir", type=Path, required=True)
    build.add_argument(
        "--force-session",
        action="store_true",
        help="Override a stale session lock from a previously crashed run.",
    )
    _add_permission_args(build)

    report = sub.add_parser("report", help="Render a report for a run.")
    report.add_argument("--run-dir", type=Path, required=True)
    return parser


PAUSED_EXIT_CODE = 3


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except InteractiveCanceledError:
        print("Workflow canceled.", file=sys.stderr)
        return 130
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = resolve_repo(args.repo)

    if args.command == "init-config":
        return _cmd_init_config(args, workspace)

    if args.command == "config":
        return _cmd_config(args, workspace)

    app_config = load_config(args.config, workspace=workspace, dry_run=args.dry_run)
    router = ProviderRouter.from_config(app_config)
    progress = ProgressReporter(enabled=not args.quiet, verbose=args.verbose, stream=sys.stderr)

    return _dispatch_command(args, router, app_config, progress)


def _dispatch_command(
    args: argparse.Namespace,
    router: ProviderRouter,
    app_config: AppConfig,
    progress: ProgressReporter,
) -> int:
    handlers = {
        "models": lambda: _cmd_models(router),
        "plan": lambda: _cmd_plan(args, router, app_config, progress),
        "run": lambda: _cmd_run(args, router, app_config, progress),
        "build": lambda: _cmd_build(args, router, app_config, progress),
        "accept": lambda: _cmd_accept(args, app_config),
        "report": lambda: _cmd_report(args, app_config),
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


def _cmd_init_config(args: argparse.Namespace, workspace: Path) -> int:
    output = resolve_workspace_relative_path(workspace, args.path)
    path = write_sample_config(output)
    print(f"Wrote {path}")
    return 0


def _cmd_config(args: argparse.Namespace, workspace: Path) -> int:
    if args.config_command != "validate":
        return 1

    validation = validate_config_file(args.config, workspace=workspace, dry_run=args.dry_run)
    if validation.errors:
        print(format_validation(validation), file=sys.stderr)
        return 1

    config = load_config(args.config, workspace=workspace, dry_run=args.dry_run)
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


def _plan_run_planning_phase(
    args: argparse.Namespace,
    prompt: str,
    router: ProviderRouter,
    app_config: AppConfig,
    execution,
    progress: ProgressReporter,
    wf: WorkflowState,
    session: InteractiveSession,
) -> RunState | int:
    if args.no_clarify:
        run = run_planning(
            prompt,
            app_config.runs_dir,
            router,
            auto_accept=args.yes,
            execution=execution,
            permission_mode=args.permission_mode,
            progress=progress.update,
        )
        transition_stage(wf, "plan_review")
        save_workflow_state(run.run_dir, wf)
        if args.yes:
            wf.touch_accepted_plan()
            wf.lifecycle_status = "completed"
            save_workflow_state(run.run_dir, wf)
        return run
    outcome = orchestrate_clarification(
        prompt=prompt,
        router=router,
        runs_dir=app_config.runs_dir,
        execution=execution,
        permission_mode=args.permission_mode,
        progress=progress.update,
        session=session,
        no_clarify=False,
        wf=wf,
    )
    if isinstance(outcome, int):
        return outcome
    run = outcome
    if args.yes:
        accept_plan(run)
        wf.touch_accepted_plan()
        wf.lifecycle_status = "completed"
        save_workflow_state(run.run_dir, wf)
    return run


def _plan_print_run_and_proposed_hints(
    args: argparse.Namespace,
    session: InteractiveSession,
    run: RunState,
) -> None:
    print(f"Run: {run.run_id}")
    if not (args.yes or session.non_interactive):
        proposed_hint = run.run_dir / "04-proposed-plan.md"
        if proposed_hint.exists():
            print(f"Proposed plan: {proposed_hint}")
            print(f"Accept with: llm-plan-execute accept --run-dir {run.run_dir}")


def _plan_finish_accept_or_review(
    args: argparse.Namespace,
    session: InteractiveSession,
    run: RunState,
    wf: WorkflowState,
    router: ProviderRouter,
    execution,
    progress: ProgressReporter,
) -> int:
    if args.yes or session.non_interactive:
        if not args.yes and session.non_interactive:
            accept_plan(run)
            wf.touch_accepted_plan()
            wf.lifecycle_status = "completed"
            save_workflow_state(run.run_dir, wf)
        print(f"Accepted plan: {run.run_dir / '04-accepted-plan.md'}")
        print(f"Report: {run.run_dir / 'report.md'}")
        return 0
    reviewed = interactive_plan_review(
        session=session,
        run=run,
        router=router,
        execution=execution,
        permission_mode=args.permission_mode,
        progress=progress.update,
        wf=wf,
    )
    if reviewed is None:
        print(f"Report: {run.run_dir / 'report.md'}")
        return 130
    print(f"Accepted plan: {reviewed.run_dir / '04-accepted-plan.md'}")
    print(f"Report: {reviewed.run_dir / 'report.md'}")
    return 0


def _cmd_plan(
    args: argparse.Namespace,
    router: ProviderRouter,
    app_config: AppConfig,
    progress: ProgressReporter,
) -> int:
    prompt = _read_prompt(app_config.workspace, args.prompt, args.prompt_file)
    execution = merge_execution_dirs(app_config.workspace, app_config.execution, args.writable_dir)
    session = InteractiveSession(
        non_interactive=args.non_interactive or not sys.stdin.isatty(),
        on_verbose_change=lambda enabled: setattr(progress, "verbose", enabled),
    )
    wf = WorkflowState()
    planned = _plan_run_planning_phase(args, prompt, router, app_config, execution, progress, wf, session)
    if isinstance(planned, int):
        return planned
    run = planned
    _plan_print_run_and_proposed_hints(args, session, run)
    return _plan_finish_accept_or_review(args, session, run, wf, router, execution, progress)


def _run_planning_only(
    args: argparse.Namespace,
    prompt: str,
    router: ProviderRouter,
    app_config: AppConfig,
    execution,
    progress: ProgressReporter,
    wf: WorkflowState,
    session: InteractiveSession,
) -> RunState | int:
    if args.no_clarify:
        run = run_planning(
            prompt,
            app_config.runs_dir,
            router,
            auto_accept=False,
            execution=execution,
            permission_mode=args.permission_mode,
            progress=progress.update,
        )
        transition_stage(wf, "plan_review")
        save_workflow_state(run.run_dir, wf)
        return run
    outcome = orchestrate_clarification(
        prompt=prompt,
        router=router,
        runs_dir=app_config.runs_dir,
        execution=execution,
        permission_mode=args.permission_mode,
        progress=progress.update,
        session=session,
        no_clarify=False,
        wf=wf,
    )
    if isinstance(outcome, int):
        return outcome
    return outcome


def _run_print_proposed_plan_banner(run: RunState) -> None:
    proposed_path = run.run_dir / "04-proposed-plan.md"
    if proposed_path.exists():
        proposed_plan = proposed_path.read_text(encoding="utf-8")
        print(f"Run: {run.run_id}")
        print(f"Proposed plan: {proposed_path}")
        print("")
        print(proposed_plan.rstrip())
        print("")
    else:
        print(f"Run: {run.run_id}")


def _run_accept_plan_phase(
    session: InteractiveSession,
    run: RunState,
    wf: WorkflowState,
    router: ProviderRouter,
    execution,
    permission_mode: str | None,
    progress: ProgressReporter,
) -> RunState | int:
    if session.non_interactive:
        accept_plan(run)
        wf.touch_accepted_plan()
        transition_stage(wf, "pre_build")
        save_workflow_state(run.run_dir, wf)
        return run
    reviewed = interactive_plan_review(
        session=session,
        run=run,
        router=router,
        execution=execution,
        permission_mode=permission_mode,
        progress=progress.update,
        wf=wf,
    )
    if reviewed is None:
        print(f"Report: {run.run_dir / 'report.md'}")
        return 130
    transition_stage(wf, "pre_build")
    save_workflow_state(reviewed.run_dir, wf)
    return reviewed


def _run_gate_before_build(
    session: InteractiveSession,
    wf: WorkflowState,
    run: RunState,
) -> int | None:
    if session.non_interactive:
        return None
    transition = gate_stage_transition(session=session, wf=wf, run=run)
    if transition == "pause":
        state_path = save_workflow_state(run.run_dir, wf)
        print("Workflow paused; state preserved.")
        print(f"Workflow state: {state_path}")
        print(f"Continue with: llm-plan-execute build --run-dir {run.run_dir}")
        print(f"Report: {run.run_dir / 'report.md'}")
        return PAUSED_EXIT_CODE
    if transition == "cancel":
        print(f"Report: {run.run_dir / 'report.md'}")
        return 130
    return None


def _run_build_review_completion(
    args: argparse.Namespace,
    session: InteractiveSession,
    router: ProviderRouter,
    app_config: AppConfig,
    execution,
    wf: WorkflowState,
    run: RunState,
    progress: ProgressReporter,
) -> int:
    _, exit_code = execute_build_through_completion(
        run=run,
        wf=wf,
        router=router,
        execution=execution,
        session=session,
        permission_mode_cli=args.permission_mode,
        progress=progress.update,
        runs_root=app_config.runs_dir,
    )
    return exit_code


def _cmd_run(
    args: argparse.Namespace,
    router: ProviderRouter,
    app_config: AppConfig,
    progress: ProgressReporter,
) -> int:
    prompt = _read_prompt(app_config.workspace, args.prompt, args.prompt_file)
    execution = merge_execution_dirs(app_config.workspace, app_config.execution, args.writable_dir)
    session = InteractiveSession(
        non_interactive=args.non_interactive or not sys.stdin.isatty(),
        on_verbose_change=lambda enabled: setattr(progress, "verbose", enabled),
    )
    wf = WorkflowState()
    planned = _run_planning_only(args, prompt, router, app_config, execution, progress, wf, session)
    if isinstance(planned, int):
        return planned
    run = planned
    _run_print_proposed_plan_banner(run)
    accepted = _run_accept_plan_phase(
        session,
        run,
        wf,
        router,
        execution,
        args.permission_mode,
        progress,
    )
    if isinstance(accepted, int):
        return accepted
    run = accepted
    early = _run_gate_before_build(session, wf, run)
    if early is not None:
        return early
    return _run_build_review_completion(args, session, router, app_config, execution, wf, run, progress)


def _cmd_accept(args: argparse.Namespace, app_config: AppConfig) -> int:
    run_dir = resolve_workspace_relative_path(app_config.workspace, args.run_dir)
    raw = load_state(run_dir)
    run = _state_from_json(raw, run_dir)
    run = accept_plan(run)
    print(f"Accepted plan: {run.run_dir / '04-accepted-plan.md'}")
    print(f"Build with: llm-plan-execute build --run-dir {run.run_dir}")
    print(f"Report: {run.run_dir / 'report.md'}")
    return 0


def _cmd_build(
    args: argparse.Namespace,
    router: ProviderRouter,
    app_config: AppConfig,
    progress: ProgressReporter,
) -> int:
    run_dir = resolve_workspace_relative_path(app_config.workspace, args.run_dir)
    raw = load_state(run_dir)
    run = _state_from_json(raw, run_dir)
    execution = merge_execution_dirs(app_config.workspace, app_config.execution, args.writable_dir)
    session = InteractiveSession(
        non_interactive=args.non_interactive or not sys.stdin.isatty(),
        on_verbose_change=lambda enabled: setattr(progress, "verbose", enabled),
    )
    wf = load_workflow_state(run.run_dir)
    lock_acquired = False
    try:
        acquire_workflow_lock(run.run_dir, force=args.force_session)
        lock_acquired = True
        if wf.stage not in {"pre_build", "build", "build_review"}:
            transition_stage(wf, "pre_build")
            save_workflow_state(run.run_dir, wf)

        if not session.non_interactive and wf.stage != "build_review":
            transition = gate_stage_transition(session=session, wf=wf, run=run)
            if transition == "pause":
                state_path = save_workflow_state(run.run_dir, wf)
                print("Workflow paused before build; state preserved.")
                print(f"Workflow state: {state_path}")
                print(f"Continue with: llm-plan-execute build --run-dir {run.run_dir}")
                print(f"Report: {run.run_dir / 'report.md'}")
                return PAUSED_EXIT_CODE
            if transition == "cancel":
                print(f"Report: {run.run_dir / 'report.md'}")
                return 130

        _, exit_code = execute_build_through_completion(
            run=run,
            wf=wf,
            router=router,
            execution=execution,
            session=session,
            permission_mode_cli=args.permission_mode,
            progress=progress.update,
            runs_root=app_config.runs_dir,
        )
        return exit_code
    finally:
        if lock_acquired:
            release_workflow_lock(run.run_dir)


def _cmd_report(args: argparse.Namespace, app_config: AppConfig) -> int:
    run_dir = resolve_workspace_relative_path(app_config.workspace, args.run_dir)
    raw = load_state(run_dir)
    run = _state_from_json(raw, run_dir)
    print(render_report(run), end="")
    return 0


def _read_prompt(workspace: Path, prompt: str | None, prompt_file: Path | None) -> str:
    if prompt:
        return prompt
    if prompt_file:
        path = resolve_workspace_relative_path(workspace, prompt_file)
        return path.read_text(encoding="utf-8")
    raise ValueError("Provide --prompt or --prompt-file.")


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
    _load_assignments(run, raw.get("assignments", {}))
    _load_results(run, raw.get("results", []))
    return run


def _load_assignments(run: RunState, value: object) -> None:
    if not isinstance(value, dict):
        return
    for role, item in value.items():
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


def _load_results(run: RunState, value: object) -> None:
    if not isinstance(value, list):
        return
    for item in value:
        if not isinstance(item, dict):
            continue
        run.results.append(_provider_result_from_json(item))


def _provider_result_from_json(item: dict[str, object]) -> ProviderResult:
    return ProviderResult(
        role=str(item["role"]),
        model=_model_from_json(item["model"]),
        prompt=str(item.get("prompt", "")),
        output=str(item.get("output", "")),
        usage=_usage_from_json(item.get("usage", {})),
        elapsed_seconds=float(item.get("elapsed_seconds", 0.0)),
        error=item.get("error") if isinstance(item.get("error"), str) else None,
        warning=item.get("warning") if isinstance(item.get("warning"), str) else None,
    )


def _usage_from_json(value: object) -> Usage:
    raw = value if isinstance(value, dict) else {}
    return Usage(
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        cost_usd=_optional_float(raw.get("cost_usd")),
        exact=bool(raw.get("exact", False)),
        confidence=str(raw.get("confidence", "estimated")),
    )


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
        if not isinstance(dirs, list):
            dirs = []
        policies[role] = ExecutionPolicy(
            mode,
            tuple(Path(path) for path in dirs if isinstance(path, str)),
        )
    return policies


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
            self._start(role, run, label, model)
            return
        if event == "finish":
            self._finish(role, run, label, model, result)
            return
        if event == "heartbeat":
            print(f"[{run.run_id}] {label}: still running...", file=self.stream)
            return
        if event == "artifact" and artifact:
            print(f"[{run.run_id}] {label}: wrote {artifact}", file=self.stream)

    def _start(self, role: str, run: RunState, label: str, model: ModelInfo | None) -> None:
        self.starts[role] = time.monotonic()
        model_id = model.id if model else "unassigned"
        print(f"[{run.run_id}] {label}: starting with {model_id}", file=self.stream)

    def _finish(
        self,
        role: str,
        run: RunState,
        label: str,
        model: ModelInfo | None,
        result: ProviderResult | None,
    ) -> None:
        elapsed = result.elapsed_seconds if result else time.monotonic() - self.starts.get(role, time.monotonic())
        model_id = result.model.id if result else (model.id if model else "unassigned")
        suffix = " failed" if result and result.error else " done"
        print(f"[{run.run_id}] {label}: {model_id}{suffix} in {elapsed:.1f}s", file=self.stream)
        if self.verbose and result and result.error:
            print(f"[{run.run_id}] {label}: {result.error}", file=self.stream)
