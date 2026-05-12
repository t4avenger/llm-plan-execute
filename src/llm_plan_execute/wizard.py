"""Interactive wizard launched by bare ``llm-plan-execute`` on a TTY.

The wizard reuses ``InteractiveSession`` for typed prompts and existing
config helpers for project-local provider configuration.

It deliberately stays narrow:

- It does not introduce a new ``CliUI`` / ``InteractiveUI`` layer.
- It does not add a user-global config or new dependencies.
- It hands off to the existing ``run`` / ``build`` command paths so all
  permission, progress, and artifact behavior remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .config import (
    DEFAULT_ROOT,
    ConfigValidation,
    format_validation,
    resolve_config_path,
    sample_config,
    validate_config_data,
    validate_config_file,
)
from .interactive import ChoiceOption, InteractiveCanceledError, InteractiveSession
from .workflow_state import WorkflowState, load_workflow_state

# Roles documented in the wizard menus; mirrors ``types.ROLES``.
WIZARD_ROLE_HELP = (
    "Roles assigned by selection:\n"
    "  planner, plan_reviewer_a, plan_reviewer_b, plan_arbiter,\n"
    "  builder, build_reviewer_a, build_reviewer_b, build_arbiter"
)


def is_tty(stream: object) -> bool:
    """Return True only when ``stream`` is a real interactive terminal."""
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def run_wizard(args: argparse.Namespace, workspace: Path) -> int:
    """Entry point for the interactive wizard.

    Returns a CLI exit code. Workspace cancellation returns 130, the same
    code used by other interactive cancellations.
    """
    session = InteractiveSession(non_interactive=False)
    try:
        return _run_wizard(args, workspace, session)
    except InteractiveCanceledError:
        print("Workflow canceled.", file=sys.stderr)
        return 130


def _run_wizard(args: argparse.Namespace, workspace: Path, session: InteractiveSession) -> int:
    _print_title(workspace, dry_run=bool(args.dry_run))

    if not session.prompt_confirm(f"Use workspace {workspace}?", default_yes=True):
        print("Workspace declined; pass --repo to choose a different workspace.")
        return 1

    config_outcome = _configure(args, workspace, session)
    if isinstance(config_outcome, int):
        return config_outcome

    paused = _find_paused_runs(workspace, args.config)
    if paused:
        choice = _ask_paused_action(session, paused)
        if choice == "resume":
            run_dir = _ask_select_paused_run(session, paused)
            return _dispatch_build(args, run_dir)
        # "new" falls through to prompt entry.

    prompt = _ask_for_prompt(session)
    return _dispatch_run(args, prompt)


def _print_title(workspace: Path, *, dry_run: bool) -> None:
    suffix = " (dry-run)" if dry_run else ""
    print("llm-plan-execute wizard")
    print(f"Workspace: {workspace}{suffix}")


def _configure(args: argparse.Namespace, workspace: Path, session: InteractiveSession) -> int | None:
    """Resolve workspace config; return None on success, or an exit code."""
    config_path = resolve_config_path(args.config, workspace)
    reconfigure = bool(getattr(args, "reconfigure", False))
    if not config_path.exists():
        return _configure_new(args, session, config_path)
    if reconfigure:
        return _configure_overwrite(session, config_path)
    return _configure_existing(args, workspace, session, config_path)


def _configure_new(
    args: argparse.Namespace,
    session: InteractiveSession,
    config_path: Path,
) -> int | None:
    if args.dry_run:
        print(f"No config at {config_path}; continuing with --dry-run (simulated providers).")
        return None
    print(f"No config found at {config_path}.")
    options = (
        ChoiceOption("1", "Create default config (recommended)", "create"),
        ChoiceOption("2", "Continue with --dry-run (simulated providers)", "dry"),
        ChoiceOption("3", "Cancel", "cancel"),
    )
    choice = session.prompt_choice("What would you like to do?", options)
    if choice == "cancel":
        raise InteractiveCanceledError("Wizard canceled before config setup.")
    if choice == "dry":
        args.dry_run = True
        return None
    return _write_default_config(config_path)


def _configure_existing(
    args: argparse.Namespace,
    workspace: Path,
    session: InteractiveSession,
    config_path: Path,
) -> int | None:
    print(f"Using existing config at {config_path}.")
    options = (
        ChoiceOption("1", "Use existing config", "use"),
        ChoiceOption("2", "Reconfigure (overwrite)", "reconfigure"),
        ChoiceOption("3", "Validate config and exit", "validate"),
        ChoiceOption("4", "Cancel", "cancel"),
    )
    choice = session.prompt_choice("How should the wizard proceed?", options)
    if choice == "cancel":
        raise InteractiveCanceledError("Wizard canceled at config menu.")
    if choice == "validate":
        return _print_validation(args, workspace)
    if choice == "reconfigure":
        return _configure_overwrite(session, config_path)
    return None


def _configure_overwrite(
    session: InteractiveSession,
    config_path: Path,
) -> int | None:
    if config_path.exists() and not session.prompt_confirm(
        f"Overwrite existing config at {config_path}?", default_yes=False
    ):
        print("Keeping existing config.")
        return None
    return _write_default_config(config_path)


def _write_default_config(config_path: Path) -> int | None:
    sample = sample_config()
    validation = validate_config_data(sample, require_providers=False)
    if validation.errors:
        print(format_validation(validation), file=sys.stderr)
        return 1
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {config_path}")
    enabled = _enabled_provider_summary(sample)
    if enabled:
        print(f"Enabled providers: {enabled}")
    else:
        print(
            "No provider CLIs were found on PATH; edit the config to enable providers "
            "or run with --dry-run for simulated providers."
        )
    return None


def _enabled_provider_summary(raw: dict[str, object]) -> str:
    providers = raw.get("providers", [])
    if not isinstance(providers, list):
        return ""
    names: list[str] = []
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        if not provider.get("enabled"):
            continue
        name = provider.get("name")
        if isinstance(name, str):
            names.append(name)
    return ", ".join(sorted(names))


def _print_validation(args: argparse.Namespace, workspace: Path) -> int:
    validation: ConfigValidation = validate_config_file(args.config, workspace=workspace, dry_run=args.dry_run)
    if validation.errors:
        print(format_validation(validation), file=sys.stderr)
        return 1
    if validation.warnings:
        print(format_validation(validation))
    else:
        print("Config is valid.")
    return 0


def _find_paused_runs(workspace: Path, config_arg: Path | None) -> list[Path]:
    runs_dir = _resolve_runs_dir(workspace, config_arg)
    if not runs_dir.exists():
        return []
    paused: list[Path] = []
    for candidate in sorted(runs_dir.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        wf_path = candidate / "workflow-state.json"
        if not wf_path.exists():
            continue
        try:
            wf: WorkflowState = load_workflow_state(candidate)
        except (OSError, ValueError):
            continue
        if wf.lifecycle_status == "paused":
            paused.append(candidate)
    return paused


def _resolve_runs_dir(workspace: Path, config_arg: Path | None) -> Path:
    config_path = resolve_config_path(config_arg, workspace)
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        candidate = raw.get("runs_dir") if isinstance(raw, dict) else None
        if isinstance(candidate, str) and candidate:
            path = Path(candidate)
            return path if path.is_absolute() else (workspace / path)
    return workspace / DEFAULT_ROOT / "runs"


def _ask_paused_action(session: InteractiveSession, paused: Sequence[Path]) -> str:
    print(f"Detected {len(paused)} paused run(s) under the configured runs_dir.")
    options = (
        ChoiceOption("1", "Resume an existing paused run", "resume"),
        ChoiceOption("2", "Start a new run", "new"),
        ChoiceOption("3", "Cancel", "cancel"),
    )
    choice = session.prompt_choice("How would you like to proceed?", options)
    if choice == "cancel":
        raise InteractiveCanceledError("Wizard canceled at paused-run menu.")
    return choice


def _ask_select_paused_run(session: InteractiveSession, paused: Sequence[Path]) -> Path:
    if len(paused) == 1:
        return paused[0]
    options = tuple(ChoiceOption(str(index + 1), str(run_dir.name), run_dir) for index, run_dir in enumerate(paused))
    return session.prompt_choice("Select a paused run to resume:", options)


def _ask_for_prompt(session: InteractiveSession) -> str:
    print("")
    print("Describe the task you want planned and built.")
    options = (
        ChoiceOption("1", "Type the prompt now (single line)", "inline"),
        ChoiceOption("2", "Open $EDITOR for a multi-line prompt", "editor"),
        ChoiceOption("3", "Cancel", "cancel"),
    )
    while True:
        choice = session.prompt_choice("How would you like to enter the prompt?", options)
        if choice == "cancel":
            raise InteractiveCanceledError("Wizard canceled at prompt entry.")
        if choice == "inline":
            return session.prompt_free_text("Prompt:")
        text = _read_prompt_from_editor(session)
        if text:
            return text


def _read_prompt_from_editor(session: InteractiveSession) -> str:
    editor = os.environ.get("EDITOR")
    if not editor:
        print("$EDITOR is not set; choose another input method.", file=sys.stderr)
        return ""
    with tempfile.NamedTemporaryFile(
        prefix="llm-plan-execute-prompt-", suffix=".md", mode="w+", encoding="utf-8", delete=False
    ) as handle:
        path = Path(handle.name)
        handle.write("# Describe the task to plan and build. Lines starting with # are ignored.\n\n")
    try:
        argv = _editor_argv(editor, path)
        try:
            completed = subprocess.run(argv, check=False)  # noqa: S603 - argv form, no shell
        except OSError as exc:
            print(f"Failed to launch $EDITOR ({editor}): {exc}", file=sys.stderr)
            return ""
        if completed.returncode != 0:
            print(f"$EDITOR exited with code {completed.returncode}; cancelling editor entry.", file=sys.stderr)
            return ""
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        body = "\n".join(line for line in raw_lines if not line.lstrip().startswith("#")).strip()
        if not body:
            print("$EDITOR produced an empty prompt; choose another option.", file=sys.stderr)
            return ""
        if not session.prompt_confirm("Use this prompt?", default_yes=True):
            return ""
        return body
    finally:
        path.unlink(missing_ok=True)


def _editor_argv(editor: str, path: Path) -> list[str]:
    return [*shlex.split(editor), str(path)]


def _dispatch_run(args: argparse.Namespace, prompt: str) -> int:
    """Hand off to the existing ``run`` command with the wizard's choices."""
    from .cli import main as _cli_main

    forwarded = _forwarded_global_flags(args)
    argv = [*forwarded, "run", "--prompt", prompt]
    return _cli_main(argv)


def _dispatch_build(args: argparse.Namespace, run_dir: Path) -> int:
    from .cli import main as _cli_main

    forwarded = _forwarded_global_flags(args)
    argv = [*forwarded, "build", "--run-dir", str(run_dir)]
    return _cli_main(argv)


def _forwarded_global_flags(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    if args.repo is not None:
        forwarded += ["--repo", str(args.repo)]
    if args.config is not None:
        forwarded += ["--config", str(args.config)]
    if args.dry_run:
        forwarded += ["--dry-run"]
    if args.quiet:
        forwarded += ["--quiet"]
    if args.verbose:
        forwarded += ["--verbose"]
    if args.ui and args.ui != "auto":
        forwarded += ["--ui", args.ui]
    return forwarded


__all__ = [
    "WIZARD_ROLE_HELP",
    "is_tty",
    "run_wizard",
]
