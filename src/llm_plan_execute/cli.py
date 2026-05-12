from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console

from .artifacts import load_state, write_state, write_text
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
from .context_cli import register_context_parser, run_context_command
from .git_flow import GitFlowError
from .implementation_hooks import prepare_implementation_entry
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
    PLAN_PERMISSION_FALLBACK_WARNING,
    WORKFLOW_PAUSED_EXIT_CODE,
    execute_build_through_completion,
    gate_stage_transition,
    interactive_plan_review,
    is_run_planning_permission_failure,
    merge_execution_dirs,
    orchestrate_clarification,
    plan_permission_workspace_write_fallback_applies,
    run_accepted_plan,
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
        "--ui",
        choices=("auto", "rich", "plain", "jsonl"),
        default="auto",
        help="Progress renderer for provider activity (default: auto).",
    )
    parser.add_argument(
        "--non-interactive",
        "--ci",
        action="store_true",
        help="Use deterministic defaults for menus (also implied when stdin is not a TTY).",
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="In wizard mode, rerun provider/model setup even when a config exists.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

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
    accept.add_argument("--build", action="store_true", help="Continue into build after accepting the plan.")
    accept.add_argument("--no-build", action="store_true", help="Do not prompt to build after accepting the plan.")
    _add_permission_args(accept)

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

    register_context_parser(sub)
    return parser


PAUSED_EXIT_CODE = WORKFLOW_PAUSED_EXIT_CODE


def main(argv: list[str] | None = None) -> int:
    """CLI entry used by ``python -m`` and console scripts."""
    return _invoke_run_with_cli_exit_mapping(argv)


def _run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = resolve_repo(args.repo)

    if args.command is None:
        return _handle_no_command(args, workspace, parser)

    if args.command == "init-config":
        return _cmd_init_config(args, workspace)

    if args.command == "config":
        return _cmd_config(args, workspace)

    if args.command == "context":
        return run_context_command(args, workspace)

    app_config = load_config(args.config, workspace=workspace, dry_run=args.dry_run)
    router = ProviderRouter.from_config(app_config)
    progress = ProgressReporter(enabled=not args.quiet, verbose=args.verbose, stream=sys.stderr, ui=args.ui)

    return _dispatch_command(args, router, app_config, progress)


def _invoke_run_with_cli_exit_mapping(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except InteractiveCanceledError:
        print("Workflow canceled.", file=sys.stderr)
        return 130
    except GitFlowError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def dispatch_argv(argv: list[str] | None = None) -> int:
    """Run the CLI with ``argv``-style arguments (wizard hand-off, programmatic use, tests).

    Tests may replace this name on ``llm_plan_execute.cli`` to intercept wizard sub-invocations
    without affecting :func:`main`, which shares behavior via :func:`_invoke_run_with_cli_exit_mapping`.
    """
    return _invoke_run_with_cli_exit_mapping(argv)


def _handle_no_command(
    args: argparse.Namespace,
    workspace: Path,
    parser: argparse.ArgumentParser,
) -> int:
    """Launch the interactive wizard on a TTY, otherwise print help and exit nonzero."""
    from .wizard import is_tty, run_wizard

    if args.non_interactive or not is_tty(sys.stdin):
        parser.print_help(sys.stderr)
        print("\nNo subcommand provided; pass --help or a subcommand.", file=sys.stderr)
        return 1
    return run_wizard(args, workspace)


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
        "accept": lambda: _cmd_accept(args, router, app_config, progress),
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


def _plan_run_planning_once(
    args: argparse.Namespace,
    prompt: str,
    router: ProviderRouter,
    app_config: AppConfig,
    execution,
    progress: ProgressReporter,
    wf: WorkflowState,
    session: InteractiveSession,
    *,
    permission_mode: str | None,
) -> RunState | int:
    if args.no_clarify:
        run = run_planning(
            prompt,
            app_config.runs_dir,
            router,
            auto_accept=args.yes,
            execution=execution,
            permission_mode=permission_mode,
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
        permission_mode=permission_mode,
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
    permission_mode = args.permission_mode
    return _planning_with_permission_fallback(
        args=args,
        execution=execution,
        permission_mode=permission_mode,
        run_once=lambda retry_permission_mode: _plan_run_planning_once(
            args,
            prompt,
            router,
            app_config,
            execution,
            progress,
            wf,
            session,
            permission_mode=retry_permission_mode,
        ),
    )


def _planning_with_permission_fallback(
    *,
    args: argparse.Namespace,
    execution,
    permission_mode: str | None,
    run_once: Callable[[str | None], RunState | int],
) -> RunState | int:
    outcome = run_once(permission_mode)
    if isinstance(outcome, int):
        return outcome
    if plan_permission_workspace_write_fallback_applies(
        execution, permission_mode_cli=permission_mode, no_clarify=args.no_clarify
    ) and is_run_planning_permission_failure(outcome):
        print(PLAN_PERMISSION_FALLBACK_WARNING, file=sys.stderr)
        retried = run_once("workspace-write")
        if isinstance(retried, RunState):
            _record_plan_permission_fallback_warning(retried)
        return retried
    return outcome


def _record_plan_permission_fallback_warning(run: RunState) -> None:
    if PLAN_PERMISSION_FALLBACK_WARNING not in run.warnings:
        run.warnings.append(PLAN_PERMISSION_FALLBACK_WARNING)
    write_state(run)
    write_text(run, "report.md", render_report(run))


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


def _plan_finish_accept_or_review(
    args: argparse.Namespace,
    session: InteractiveSession,
    run: RunState,
    wf: WorkflowState,
    router: ProviderRouter,
    app_config: AppConfig,
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
    _, exit_code = run_accepted_plan(
        run=reviewed,
        wf=wf,
        router=router,
        workspace=app_config.workspace,
        runs_root=app_config.runs_dir,
        execution=execution,
        session=session,
        permission_mode_cli=args.permission_mode,
        progress=progress.update,
        build_create_pr=app_config.build.create_pr,
        base_branch_override=app_config.build.base_branch,
    )
    return exit_code


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
    return _plan_finish_accept_or_review(args, session, run, wf, router, app_config, execution, progress)


def _run_planning_once(
    args: argparse.Namespace,
    prompt: str,
    router: ProviderRouter,
    app_config: AppConfig,
    execution,
    progress: ProgressReporter,
    wf: WorkflowState,
    session: InteractiveSession,
    *,
    permission_mode: str | None,
) -> RunState | int:
    if args.no_clarify:
        run = run_planning(
            prompt,
            app_config.runs_dir,
            router,
            auto_accept=False,
            execution=execution,
            permission_mode=permission_mode,
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
        permission_mode=permission_mode,
        progress=progress.update,
        session=session,
        no_clarify=False,
        wf=wf,
    )
    if isinstance(outcome, int):
        return outcome
    return outcome


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
    permission_mode = args.permission_mode
    return _planning_with_permission_fallback(
        args=args,
        execution=execution,
        permission_mode=permission_mode,
        run_once=lambda retry_permission_mode: _run_planning_once(
            args,
            prompt,
            router,
            app_config,
            execution,
            progress,
            wf,
            session,
            permission_mode=retry_permission_mode,
        ),
    )


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
    return reviewed


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
    _, exit_code = run_accepted_plan(
        run=run,
        wf=wf,
        router=router,
        workspace=app_config.workspace,
        runs_root=app_config.runs_dir,
        execution=execution,
        session=session,
        permission_mode_cli=args.permission_mode,
        progress=progress.update,
        build_create_pr=app_config.build.create_pr,
        base_branch_override=app_config.build.base_branch,
    )
    return exit_code


def _cmd_accept(
    args: argparse.Namespace,
    router: ProviderRouter,
    app_config: AppConfig,
    progress: ProgressReporter,
) -> int:
    run_dir = resolve_workspace_relative_path(app_config.workspace, args.run_dir)
    raw = load_state(run_dir)
    run = _state_from_json(raw, run_dir)
    run = accept_plan(run)
    print(f"Accepted plan: {run.run_dir / '04-accepted-plan.md'}")
    if not _accept_should_build(args):
        print(f"Build with: llm-plan-execute build --run-dir {run.run_dir}")
        print(f"Report: {run.run_dir / 'report.md'}")
        return 0
    return _continue_build_after_accept(args, router, app_config, progress, run)


def _accept_should_build(args: argparse.Namespace) -> bool:
    if args.no_build:
        return False
    if args.build:
        return True
    if args.non_interactive or not sys.stdin.isatty():
        return False
    print("Build now? [Y/n]", flush=True)
    try:
        answer = input().strip().lower()
    except EOFError:
        answer = "n"
    return answer in {"", "y", "yes"}


def _continue_build_after_accept(
    args: argparse.Namespace,
    router: ProviderRouter,
    app_config: AppConfig,
    progress: ProgressReporter,
    run: RunState,
) -> int:
    execution = merge_execution_dirs(app_config.workspace, app_config.execution, args.writable_dir)
    session = InteractiveSession(
        non_interactive=args.non_interactive or not sys.stdin.isatty(),
        on_verbose_change=lambda enabled: setattr(progress, "verbose", enabled),
    )
    wf = load_workflow_state(run.run_dir)
    lock_acquired = False
    try:
        acquire_workflow_lock(run.run_dir, force=False)
        lock_acquired = True
        run._last_local_progress = "preparing workspace, context store, and task branch"  # type: ignore[attr-defined]
        progress.update("local", "workflow", run, None, None, None)
        _, exit_code = run_accepted_plan(
            run=run,
            wf=wf,
            router=router,
            workspace=app_config.workspace,
            runs_root=app_config.runs_dir,
            execution=execution,
            session=session,
            permission_mode_cli=args.permission_mode,
            progress=progress.update,
            build_create_pr=app_config.build.create_pr,
            base_branch_override=app_config.build.base_branch,
            pre_build_gate=False,
        )
        return exit_code
    finally:
        if lock_acquired:
            release_workflow_lock(run.run_dir)


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
        prepare_implementation_entry(
            app_config.workspace,
            wf,
            run,
            base_branch_override=app_config.build.base_branch,
        )
        save_workflow_state(run.run_dir, wf)
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
            build_create_pr=app_config.build.create_pr,
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
    def __init__(self, *, enabled: bool, verbose: bool, stream: Any, ui: str = "auto") -> None:
        self.enabled = enabled
        self.verbose = verbose
        self.stream = stream
        self.ui = self._resolve_ui(ui, stream)
        self.console = self._make_console(stream) if self.ui == "rich" else None
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
            self._heartbeat(role, run, label, model)
            return
        if event == "activity":
            self._activity(run)
            return
        if event == "local":
            self._local(run, label)
            return
        if event == "artifact" and artifact:
            self._emit(run.run_id, label, f"wrote {artifact}", status="artifact")

    def _resolve_ui(self, ui: str, stream: Any) -> str:
        if ui == "auto":
            return "rich" if hasattr(stream, "isatty") and stream.isatty() and self._rich_available() else "plain"
        if ui == "rich" and not self._rich_available():
            return "plain"
        return ui

    def _rich_available(self) -> bool:
        return True

    def _make_console(self, stream: Any) -> Any:
        return Console(file=stream, stderr=stream is sys.stderr)

    def _start(self, role: str, run: RunState, label: str, model: ModelInfo | None) -> None:
        self.starts[role] = time.monotonic()
        model_id = model.id if model else "unassigned"
        self._emit(run.run_id, label, f"starting with {model_id}; {_role_activity(role)}", status="start")

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
        self._emit(run.run_id, label, f"{model_id}{suffix} in {elapsed:.1f}s", status="finish")
        if self.verbose and result and result.error:
            self._emit(run.run_id, label, result.error, status="error")

    def _heartbeat(self, role: str, run: RunState, label: str, model: ModelInfo | None) -> None:
        elapsed = time.monotonic() - self.starts.get(role, time.monotonic())
        model_id = model.id if model else "unassigned"
        self._emit(
            run.run_id,
            label,
            f"waiting on {model_id} for {_format_elapsed(elapsed)}; {_role_activity(role)}",
            status="running",
        )

    def _activity(self, run: RunState) -> None:
        activity = getattr(run, "_last_provider_activity", None)
        if activity is None:
            return
        label = activity.role.replace("_", " ")
        self._emit(
            run.run_id,
            label,
            f"{activity.message} ({_format_elapsed(activity.elapsed_seconds)})",
            status="activity",
            extra={
                "kind": activity.kind,
                "tool_name": activity.tool_name,
                "workspace_path": activity.workspace_path,
                "command": activity.command,
                "model": activity.model.id,
            },
        )

    def _local(self, run: RunState, label: str) -> None:
        message = getattr(run, "_last_local_progress", None)
        if not message:
            return
        self._emit(run.run_id, label, str(message), status="local")

    def _emit(
        self,
        run_id: str,
        label: str,
        message: str,
        *,
        status: str,
        extra: dict[str, object | None] | None = None,
    ) -> None:
        if self.ui == "jsonl":
            payload = {"run_id": run_id, "role": label, "status": status, "message": message}
            if extra:
                payload.update({key: value for key, value in extra.items() if value is not None})
            print(json.dumps(payload), file=self.stream)
            return
        if self.console is not None:
            self.console.print(f"[dim]{run_id}[/dim] [bold]{label}[/bold]: {message}")
            return
        print(f"[{run_id}] {label}: {message}", file=self.stream)


def _format_elapsed(seconds: float) -> str:
    seconds_per_minute = 60
    if seconds < seconds_per_minute:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), seconds_per_minute)
    return f"{minutes}m{secs:02d}s"


def _role_activity(role: str) -> str:
    labels = {
        "clarifier": "checking whether the request needs clarification",
        "planner": "drafting the implementation plan",
        "plan_reviewer_a": "reviewing the draft plan",
        "plan_reviewer_b": "reviewing the draft plan",
        "plan_arbiter": "merging plan feedback into a proposed plan",
        "builder": "applying the accepted plan",
        "build_reviewer_a": "reviewing implementation output",
        "build_reviewer_b": "reviewing implementation output",
        "build_arbiter": "summarizing build review findings",
    }
    return labels.get(role, "working")
