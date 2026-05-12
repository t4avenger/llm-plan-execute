"""Central interactive workflow orchestration for plan/run/build commands.

Command inventory
---------------

- ``plan``: shared clarification path, planning, and typed plan review loop. Interactive acceptance
  continues in-process into build (same pre-build gate and execution path as ``run``). Non-interactive
  or ``--yes`` still stops after acceptance without building.

- ``run``: continuous flow — clarification (optional), planning, plan review, pre-build gate, build,
  build-review decisions, and completion reporting.

- ``build``: resumes from disk state (``workflow-state.json``) with pre-build gate, build execution,
  build-review decisions, and completion reporting. After pausing mid-``run``, continue with
  ``build --run-dir`` on the same directory—not a second ``run``.

- ``accept``, ``report``, ``models``, ``config``, ``init-config``: excluded from this runner (thin utilities).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from .artifacts import write_state, write_text
from .build_review_schema import (
    BuildRecommendation,
    expand_with_dependencies,
    map_numeric_selection_to_ids,
    parse_recommendations_from_summary,
    selection_requires_missing_dependency,
)
from .config import ExecutionConfig, normalize_writable_dirs
from .context_store import ContextStore
from .git_flow import GitFlowError, commit_checkpoint, find_git_root, git_changed_pathspecs, maybe_offer_github_pr
from .html_report import deterministic_html_report_path, write_html_report
from .implementation_hooks import prepare_implementation_entry
from .interactive import (
    BuildReviewDecision,
    ChoiceOption,
    InteractiveSession,
)
from .plan_sections import split_plan_sections
from .providers import ProviderRouter
from .reporting import render_report
from .types import PERMISSION_MODES, RunState
from .workflow import (
    BuildFailedError,
    accept_plan,
    apply_build_recommendation_fixes,
    complete_planning,
    record_build_recommendation_application,
    render_clarification,
    rerun_build_review,
    revise_proposed_plan,
    run_build,
    run_clarification,
    run_planning,
)
from .workflow_state import WorkflowState, can_transition, save_workflow_state, transition_stage

ProgressHook = Callable[..., None]
_REPORT_MARKDOWN_FILENAME = "report.md"

# Exit code when the user pauses before build (same as CLI ``PAUSED_EXIT_CODE``).
WORKFLOW_PAUSED_EXIT_CODE = 3


def merge_execution_dirs(workspace: Path, execution: ExecutionConfig, extra_dirs: list[Path]) -> ExecutionConfig:
    if not extra_dirs:
        return execution
    normalized = normalize_writable_dirs(workspace, tuple(extra_dirs), field="--writable-dir")
    return ExecutionConfig(
        default_mode=execution.default_mode,
        planning_mode=execution.planning_mode,
        review_mode=execution.review_mode,
        build_mode=execution.build_mode,
        writable_dirs=(*execution.writable_dirs, *normalized),
    )


_PLANNING_PERMISSION_PHASE_ROLES = frozenset(
    {"clarifier", "planner", "plan_reviewer_a", "plan_reviewer_b", "plan_arbiter"}
)

_PERMISSION_MODE_RANK: dict[str, int] = {"read-only": 0, "workspace-write": 1, "full-access": 2}

PLAN_PERMISSION_FALLBACK_WARNING = (
    "Warning: planning requires workspace file and command access. "
    "Retrying with minimum workspace-write permission enabled."
)


def is_planning_permission_failure_message(text: str) -> bool:
    """True when *text* looks like a sandbox or permission denial from a planning-phase provider."""
    lower = text.lower()
    direct_matches = (
        "permission denied",
        "eacces",
        "operation not permitted",
        "cannot execute",
        "unable to execute",
    )
    paired_matches = (
        ("eperm", ("operation",)),
        ("not allowed", ("read", "file", "path", "command", "shell", "exec")),
        ("sandbox", ("denied", "blocked", "forbidden", "restricted", "not allowed", "read-only")),
        ("read-only", ("sandbox", "filesystem")),
        ("blocked by", ("policy",)),
    )
    return any(token in lower for token in direct_matches) or any(
        lead in lower and any(token in lower for token in tokens) for lead, tokens in paired_matches
    )


def is_run_planning_permission_failure(run: RunState) -> bool:
    return any(
        result.role in _PLANNING_PERMISSION_PHASE_ROLES
        and result.error
        and is_planning_permission_failure_message(result.error)
        for result in run.results
    )


def plan_permission_workspace_write_fallback_applies(
    execution: ExecutionConfig, *, permission_mode_cli: str | None, no_clarify: bool
) -> bool:
    """Whether an automatic workspace-write retry is allowed (implicit modes only, not already elevated)."""
    if permission_mode_cli is not None:
        return False
    roles: tuple[str, ...] = ("planner", "plan_reviewer_a", "plan_reviewer_b", "plan_arbiter")
    if not no_clarify:
        roles = ("clarifier", *roles)
    write_rank = _PERMISSION_MODE_RANK["workspace-write"]
    for role in roles:
        mode = execution.mode_for_role(role)
        if _PERMISSION_MODE_RANK.get(mode, 0) < write_rank:
            return True
    return False


def orchestrate_clarification(
    *,
    prompt: str,
    router: ProviderRouter,
    runs_dir: Path,
    execution: ExecutionConfig,
    permission_mode: str | None,
    progress: ProgressHook,
    session: InteractiveSession,
    no_clarify: bool,
    wf: WorkflowState,
) -> RunState | int:
    """Return a run state or an exit code when clarification cannot proceed."""
    if no_clarify:
        run = run_planning(
            prompt,
            runs_dir,
            router,
            auto_accept=False,
            execution=execution,
            permission_mode=permission_mode,
            progress=progress,
        )
        transition_stage(wf, "plan_review")
        save_workflow_state(run.run_dir, wf)
        return run

    run = run_clarification(
        prompt,
        runs_dir,
        router,
        execution=execution,
        permission_mode=permission_mode,
        progress=progress,
    )
    clarification = run.clarification
    if clarification and clarification.status == "needs_questions":
        if session.non_interactive or not sys.stdin.isatty():
            print(f"Run: {run.run_id}")
            print(f"Clarification needed: {run.run_dir / '00-clarification.md'}")
            print("Answer the questions and rerun planning, or use --no-clarify to plan with assumptions.")
            print(f"Report: {run.run_dir / _REPORT_MARKDOWN_FILENAME}")
            return 2
        clarification.answers = [_ask_cli_question(question) for question in clarification.questions]
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
        permission_mode=permission_mode,
        progress=progress,
    )
    transition_stage(wf, "plan_review")
    save_workflow_state(run.run_dir, wf)
    return run


def _plan_review_mark_canceled(wf: WorkflowState, run: RunState) -> None:
    wf.lifecycle_status = "canceled"
    save_workflow_state(run.run_dir, wf)


def _plan_review_step_through_cancelled(
    session: InteractiveSession,
    wf: WorkflowState,
    run: RunState,
    proposed_path: Path,
) -> bool:
    plan_text = proposed_path.read_text(encoding="utf-8")
    sections = split_plan_sections(plan_text)
    outcome = session.step_through_sections("Step through plan sections", sections)
    if outcome.type == "cancel":
        _plan_review_mark_canceled(wf, run)
        return True
    return False


def _plan_review_maybe_revise(
    session: InteractiveSession,
    wf: WorkflowState,
    run: RunState,
    router: ProviderRouter,
    execution: ExecutionConfig,
    permission_mode: str | None,
    progress: ProgressHook,
) -> RunState | None:
    feedback = session.read_plan_feedback()
    if session.non_interactive and not feedback:
        return None
    wf.plan_feedback_history.append(feedback)
    save_workflow_state(run.run_dir, wf)
    revised = revise_proposed_plan(
        run,
        router,
        execution=execution,
        permission_mode=permission_mode,
        progress=progress,
        feedback_history=wf.plan_feedback_history,
    )
    save_workflow_state(run.run_dir, wf)
    return revised


def interactive_plan_review(
    *,
    session: InteractiveSession,
    run: RunState,
    router: ProviderRouter,
    execution: ExecutionConfig,
    permission_mode: str | None,
    progress: ProgressHook,
    wf: WorkflowState,
) -> RunState | None:
    """Return the run when a plan is accepted, or ``None`` when workflow should exit early."""
    while True:
        proposed_path = run.run_dir / "04-proposed-plan.md"
        if not proposed_path.exists():
            run = complete_planning(
                run,
                router,
                auto_accept=False,
                execution=execution,
                permission_mode=permission_mode,
                progress=progress,
            )
        decision = session.ask_plan_review()
        if decision.type == "cancel":
            _plan_review_mark_canceled(wf, run)
            return None
        if decision.type == "stepThrough":
            if _plan_review_step_through_cancelled(session, wf, run, proposed_path):
                return None
            continue
        if decision.type == "modify":
            revised = _plan_review_maybe_revise(session, wf, run, router, execution, permission_mode, progress)
            if revised is not None:
                run = revised
            continue

        accept_plan(run)
        wf.touch_accepted_plan()
        transition_stage(wf, "plan_review")
        save_workflow_state(run.run_dir, wf)
        return run


def gate_stage_transition(*, session: InteractiveSession, wf: WorkflowState, run: RunState) -> str:
    """Return ``proceed``, ``pause``, or ``cancel``."""
    decision = session.ask_stage_transition()
    if decision.type == "pause":
        wf.lifecycle_status = "paused"
        wf.stage = "pre_build"
        save_workflow_state(run.run_dir, wf)
        return "pause"
    if decision.type == "cancel":
        wf.lifecycle_status = "canceled"
        save_workflow_state(run.run_dir, wf)
        return "cancel"
    return "proceed"


def resolve_build_permission(default_mode: str, session: InteractiveSession) -> str:
    if session.non_interactive:
        return default_mode
    choices = [ChoiceOption(str(index + 1), mode, mode) for index, mode in enumerate(PERMISSION_MODES)]
    return session.prompt_choice(f"Choose builder permission mode (workspace default: {default_mode}):", choices)


def interactive_build_review_loop(
    *,
    session: InteractiveSession,
    run: RunState,
    router: ProviderRouter,
    execution: ExecutionConfig,
    permission_mode: str | None,
    progress: ProgressHook,
    wf: WorkflowState,
    runs_root: Path | None = None,
) -> str | None:
    """Return ``cancel`` when canceled; otherwise ``None``."""
    summary_path = run.run_dir / "08-build-review-summary.md"
    if not summary_path.exists():
        return None
    if session.non_interactive:
        wf.build_review_selected_action = "continueWithoutApplying"
        wf.build_review_selected_ids = []
        save_workflow_state(run.run_dir, wf)
        return None

    while True:
        recommendations = parse_recommendations_from_summary(summary_path.read_text(encoding="utf-8"))
        decision = session.ask_build_review()
        wf.build_review_selected_action = decision.type
        save_workflow_state(run.run_dir, wf)

        terminal = _handle_build_review_terminal_decision(decision, wf, run)
        if terminal is not False:
            return terminal
        if decision.type == "feedback":
            run = _handle_build_review_feedback(
                session,
                run,
                router,
                execution=execution,
                permission_mode=permission_mode,
                progress=progress,
                wf=wf,
            )
            continue

        outcome = _apply_build_review_decision(
            decision,
            recommendations,
            session=session,
            run=run,
            router=router,
            execution=execution,
            permission_mode=permission_mode,
            progress=progress,
            wf=wf,
            runs_root=runs_root,
        )
        if isinstance(outcome, str):
            if outcome == "retry":
                continue
            return outcome
        run = outcome


def _handle_build_review_terminal_decision(
    decision: BuildReviewDecision,
    wf: WorkflowState,
    run: RunState,
) -> str | None | bool:
    if decision.type == "cancel":
        wf.lifecycle_status = "canceled"
        wf.build_review_selected_ids = []
        save_workflow_state(run.run_dir, wf)
        return "cancel"
    if decision.type == "continueWithoutApplying":
        wf.build_review_selected_ids = []
        save_workflow_state(run.run_dir, wf)
        return None
    return False


def _handle_build_review_feedback(
    session: InteractiveSession,
    run: RunState,
    router: ProviderRouter,
    *,
    execution: ExecutionConfig,
    permission_mode: str | None,
    progress: ProgressHook,
    wf: WorkflowState,
) -> RunState:
    feedback = session.read_build_feedback()
    if session.non_interactive and not feedback:
        return run
    wf.build_review_feedback_history.append(feedback)
    save_workflow_state(run.run_dir, wf)
    rerun = rerun_build_review(
        run,
        router,
        execution=execution,
        permission_mode=permission_mode,
        progress=progress,
        feedback_history=wf.build_review_feedback_history,
    )
    save_workflow_state(run.run_dir, wf)
    return rerun


def _apply_build_review_decision(
    decision: BuildReviewDecision,
    recommendations: list[BuildRecommendation],
    *,
    session: InteractiveSession,
    run: RunState,
    router: ProviderRouter,
    execution: ExecutionConfig,
    permission_mode: str | None,
    progress: ProgressHook,
    wf: WorkflowState,
    runs_root: Path | None,
) -> RunState | str:
    selected_ids = _resolve_build_review_selection(decision, recommendations, session)
    wf.build_review_selected_ids = list(selected_ids)
    save_workflow_state(run.run_dir, wf)
    expanded = expand_with_dependencies(selected_ids, recommendations)
    missing = selection_requires_missing_dependency(expanded, recommendations)
    if missing:
        print(missing)
        return "retry"
    wf.build_review_applied_ids = list(expanded)
    save_workflow_state(run.run_dir, wf)
    record_build_recommendation_application(run, recommendations, expanded)
    recommendations_text = _write_apply_follow_up_notes(run, recommendations, expanded) or ""
    try:
        _emit_local_progress(progress, run, "applying selected build-review recommendations")
        rerun = apply_build_recommendation_fixes(
            run,
            router,
            recommendations_text,
            execution=execution,
            permission_mode=permission_mode,
            progress=progress,
            runs_root=runs_root,
        )
    except BuildFailedError as exc:
        wf.lifecycle_status = "failed"
        save_workflow_state(exc.run.run_dir, wf)
        print(f"Build review fix failed: {exc.run.run_dir / '11-build-review-fix-output.md'}")
        return "failed"
        _checkpoint_review_fixes(router.workspace, rerun, wf, runs_root=runs_root)
    return rerun


def execute_build_through_completion(
    *,
    run: RunState,
    wf: WorkflowState,
    router: ProviderRouter,
    execution: ExecutionConfig,
    session: InteractiveSession,
    permission_mode_cli: str | None,
    progress: ProgressHook,
    runs_root: Path,
    build_create_pr: bool = False,
) -> tuple[RunState, int]:
    """Run build, build-review loop, and completion reporting; return final run state and exit code."""
    build_permission = permission_mode_cli or execution.build_mode
    if wf.stage != "build_review":
        build_permission = permission_mode_cli or resolve_build_permission(execution.build_mode, session)
        build_result = _run_build_stage(
            run=run,
            wf=wf,
            router=router,
            execution=execution,
            build_permission=build_permission,
            progress=progress,
            runs_root=runs_root,
        )
        if isinstance(build_result, tuple):
            return build_result
        run = build_result

    review_code = _run_build_review_stage(
        run=run,
        wf=wf,
        router=router,
        execution=execution,
        build_permission=build_permission,
        session=session,
        progress=progress,
        runs_root=runs_root,
    )
    if review_code is not None:
        return run, review_code

    maybe_offer_github_pr(
        workspace=router.workspace,
        wf=wf,
        task_id=run.run_id,
        title_hint=run.prompt,
        interactive_tty=not session.non_interactive,
        create_pr_without_prompt=build_create_pr,
    )
    return _finish_build_completion(run=run, wf=wf, session=session)


def run_accepted_plan(
    *,
    run: RunState,
    wf: WorkflowState,
    router: ProviderRouter,
    workspace: Path,
    runs_root: Path,
    execution: ExecutionConfig,
    session: InteractiveSession,
    permission_mode_cli: str | None,
    progress: ProgressHook,
    build_create_pr: bool,
    base_branch_override: str | None,
    pre_build_gate: bool = True,
) -> tuple[RunState, int]:
    """Prepare workspace entry, optionally prompt before build, then run build through completion.

    Shared by ``run`` after plan acceptance, interactive ``plan`` after acceptance, and
    ``accept --build`` (with ``pre_build_gate=False`` when the user already chose to build).
    """
    prepare_implementation_entry(
        workspace,
        wf,
        run,
        base_branch_override=base_branch_override,
    )
    save_workflow_state(run.run_dir, wf)
    if wf.stage != "pre_build" and can_transition(wf.stage, "pre_build"):
        transition_stage(wf, "pre_build")
        save_workflow_state(run.run_dir, wf)

    if pre_build_gate and not session.non_interactive and wf.stage == "pre_build":
        transition = gate_stage_transition(session=session, wf=wf, run=run)
        if transition == "pause":
            state_path = save_workflow_state(run.run_dir, wf)
            print("Workflow paused; state preserved.")
            print(f"Workflow state: {state_path}")
            print(f"Continue with: llm-plan-execute build --run-dir {run.run_dir}")
            print(f"Report: {run.run_dir / _REPORT_MARKDOWN_FILENAME}")
            return run, WORKFLOW_PAUSED_EXIT_CODE
        if transition == "cancel":
            print(f"Report: {run.run_dir / _REPORT_MARKDOWN_FILENAME}")
            return run, 130

    return execute_build_through_completion(
        run=run,
        wf=wf,
        router=router,
        execution=execution,
        session=session,
        permission_mode_cli=permission_mode_cli,
        progress=progress,
        runs_root=runs_root,
        build_create_pr=build_create_pr,
    )


def _run_build_stage(
    *,
    run: RunState,
    wf: WorkflowState,
    router: ProviderRouter,
    execution: ExecutionConfig,
    build_permission: str,
    progress: ProgressHook,
    runs_root: Path,
) -> RunState | tuple[RunState, int]:
    try:
        run = run_build(
            run,
            router,
            execution=execution,
            permission_mode=build_permission,
            progress=progress,
            runs_root=runs_root,
        )
    except BuildFailedError as exc:
        return _handle_build_failure(exc, wf)
    _mark_build_review_stage(run, wf)
    checkpoint = _checkpoint_after_build(
        router.workspace,
        run,
        wf,
        router,
        execution,
        build_permission,
        progress,
        runs_root=runs_root,
    )
    return checkpoint or run


def _handle_build_failure(exc: BuildFailedError, wf: WorkflowState) -> tuple[RunState, int]:
    run = exc.run
    wf.lifecycle_status = "failed"
    save_workflow_state(run.run_dir, wf)
    print(f"Build failed: {run.run_dir / '05-build-output.md'}")
    print(f"Report: {run.run_dir / _REPORT_MARKDOWN_FILENAME}")
    return run, 1


def _mark_build_review_stage(run: RunState, wf: WorkflowState) -> None:
    if can_transition(wf.stage, "build") and wf.stage != "build":
        transition_stage(wf, "build")
    if can_transition(wf.stage, "build_review") and wf.stage != "build_review":
        transition_stage(wf, "build_review")
    save_workflow_state(run.run_dir, wf)


def _checkpoint_after_build(
    workspace: Path,
    run: RunState,
    wf: WorkflowState,
    router: ProviderRouter,
    execution: ExecutionConfig,
    build_permission: str,
    progress: ProgressHook,
    *,
    runs_root: Path,
) -> tuple[RunState, int] | None:
    git_root = find_git_root(workspace)
    if not git_root:
        return None
    paths = git_changed_pathspecs(git_root, excluded_roots=[runs_root])
    if not paths:
        return None
    try:
        _emit_local_progress(progress, run, "checkpointing implementation changes")
        _checkpoint_build_milestones(workspace, git_root, run, paths, runs_root=runs_root)
    except GitFlowError as exc:
        return _remediate_failed_build_checkpoint(
            workspace=workspace,
            git_root=git_root,
            run=run,
            wf=wf,
            router=router,
            execution=execution,
            build_permission=build_permission,
            progress=progress,
            runs_root=runs_root,
            changed_paths=paths,
            failure_output=str(exc),
        )
    return None


def _remediate_failed_build_checkpoint(
    *,
    workspace: Path,
    git_root: Path,
    run: RunState,
    wf: WorkflowState,
    router: ProviderRouter,
    execution: ExecutionConfig,
    build_permission: str,
    progress: ProgressHook,
    runs_root: Path,
    changed_paths: list[str],
    failure_output: str,
) -> tuple[RunState, int] | None:
    remediation_path = write_text(
        run,
        "12-commit-remediation.md",
        _commit_remediation_context(run, changed_paths, failure_output),
    )
    _emit_local_progress(progress, run, "asking builder to fix checkpoint commit failure")
    policy = execution.policy_for_role("builder", mode_override=build_permission)
    result = router.run(
        "builder",
        run.assignments["builder"].model,
        _commit_remediation_prompt(run, changed_paths, failure_output),
        policy,
        _build_checkpoint_activity_callback(progress, run),
    )
    run.results.append(result)
    run.build_output = result.output
    write_text(run, "13-commit-remediation-output.md", result.output)
    write_state(run)
    write_text(run, _REPORT_MARKDOWN_FILENAME, render_report(run))

    retry_paths = git_changed_pathspecs(git_root, excluded_roots=[runs_root])
    if not retry_paths:
        retry_paths = changed_paths
    try:
        _emit_local_progress(progress, run, "retrying checkpoint commit after remediation")
        _checkpoint_build_milestones(workspace, git_root, run, retry_paths, runs_root=runs_root)
    except GitFlowError as retry_exc:
        wf.lifecycle_status = "failed"
        save_workflow_state(run.run_dir, wf)
        print(f"Checkpoint commit retry failed: {retry_exc}")
        print(f"Commit remediation context: {remediation_path}")
        print(f"Report: {run.run_dir / _REPORT_MARKDOWN_FILENAME}")
        return run, 1
    return None


def _build_checkpoint_activity_callback(progress: ProgressHook, run: RunState):
    def activity(item) -> None:
        run._last_provider_activity = item  # type: ignore[attr-defined]
        progress("activity", item.role, run, item.model, None, None)

    return activity


def _commit_remediation_context(run: RunState, changed_paths: list[str], failure_output: str) -> str:
    return "\n\n".join(
        [
            "# Commit Remediation",
            "## Changed paths",
            "\n".join(f"- {path}" for path in changed_paths) or "- none",
            "## Failed commit/pre-commit output",
            "```text\n" + failure_output.strip() + "\n```",
            "## Accepted plan",
            run.accepted_plan or "",
            "## Build output",
            run.build_output or "",
        ]
    )


def _commit_remediation_prompt(run: RunState, changed_paths: list[str], failure_output: str) -> str:
    return "\n\n".join(
        [
            "The implementation build completed, but the checkpoint commit failed, usually because a git hook "
            "or pre-commit check rejected the workspace.",
            "Fix the workspace so the same checkpoint commit can succeed. Do not bypass hooks.",
            "Changed paths:",
            "\n".join(f"- {path}" for path in changed_paths) or "- none",
            "Exact failed commit/pre-commit output:",
            "```text\n" + failure_output.strip() + "\n```",
            "Accepted plan:",
            run.accepted_plan or "",
            "Build output:",
            run.build_output or "",
            "Make only the needed code, formatting, or test updates, then summarize what changed and what "
            "verification you ran.",
        ]
    )


def _run_build_review_stage(
    *,
    run: RunState,
    wf: WorkflowState,
    router: ProviderRouter,
    execution: ExecutionConfig,
    build_permission: str,
    session: InteractiveSession,
    progress: ProgressHook,
    runs_root: Path,
) -> int | None:
    review_outcome = interactive_build_review_loop(
        session=session,
        run=run,
        router=router,
        execution=execution,
        permission_mode=build_permission,
        progress=progress,
        wf=wf,
        runs_root=runs_root,
    )
    if review_outcome in {"cancel", "failed"}:
        print(f"Report: {run.run_dir / _REPORT_MARKDOWN_FILENAME}")
        return 130 if review_outcome == "cancel" else 1
    return None


def _finish_build_completion(
    *,
    run: RunState,
    wf: WorkflowState,
    session: InteractiveSession,
) -> tuple[RunState, int]:
    completion = finalize_completion_reports(session=session, run=run, wf=wf)
    if completion == "cancel":
        print(f"Report: {run.run_dir / _REPORT_MARKDOWN_FILENAME}")
        return run, 130
    if can_transition(wf.stage, "complete"):
        transition_stage(wf, "complete")
        save_workflow_state(run.run_dir, wf)
    print(f"Build output: {run.run_dir / '05-build-output.md'}")
    print(f"Review summary: {run.run_dir / '08-build-review-summary.md'}")
    print(f"Report: {run.run_dir / _REPORT_MARKDOWN_FILENAME}")
    return run, 0


def finalize_completion_reports(*, session: InteractiveSession, run: RunState, wf: WorkflowState) -> str | None:
    """Return ``cancel`` when user aborts completion reporting."""
    choice = session.ask_completion_report()
    if choice.type == "cancel":
        wf.lifecycle_status = "canceled"
        save_workflow_state(run.run_dir, wf)
        return "cancel"
    if choice.type == "skip":
        wf.lifecycle_status = "completed"
        wf.report_markdown_path = _REPORT_MARKDOWN_FILENAME
        save_workflow_state(run.run_dir, wf)
        return None

    markdown_report = render_report(run)

    if choice.type in {"terminal", "both"}:
        print(_format_terminal_report(markdown_report))
        wf.terminal_report_printed = True
    if choice.type in {"html", "both"}:
        html_path = deterministic_html_report_path(run.run_dir)
        try:
            written = write_html_report(run)
            wf.report_html_path = written.name
            print(f"HTML report written to: {written}")
        except OSError as exc:
            print(f"Failed to write HTML report to {html_path}: {exc}")

    wf.report_markdown_path = _REPORT_MARKDOWN_FILENAME
    wf.lifecycle_status = "completed"
    save_workflow_state(run.run_dir, wf)
    return None


def _ask_cli_question(question: str) -> str:
    print(question)
    return input("> ").strip()


def _resolve_build_review_selection(
    decision: BuildReviewDecision,
    recommendations: list[BuildRecommendation],
    session: InteractiveSession,
) -> list[str]:
    if not recommendations:
        return []
    if decision.type == "applyAll":
        return [rec.id for rec in recommendations]
    if decision.type != "select":
        return []
    indices = session.read_recommendation_selection(len(recommendations))
    mapped = map_numeric_selection_to_ids(indices, recommendations)
    return list(mapped)


def _write_apply_follow_up_notes(
    run: RunState,
    recommendations: list[BuildRecommendation],
    expanded: list[str],
) -> str:
    by_id = {rec.id: rec for rec in recommendations}
    instructions = "\n".join(
        f"- {by_id[rec_id].title}: {by_id[rec_id].description}" for rec_id in expanded if rec_id in by_id
    )
    note = "# Follow-up application instructions\n\n" + instructions + "\n"
    write_text(run, "10-build-follow-up.md", note)
    write_text(run, _REPORT_MARKDOWN_FILENAME, render_report(run))
    write_state(run)
    return instructions


def _checkpoint_review_fixes(
    workspace: Path, run: RunState, wf: WorkflowState, *, runs_root: Path | None = None
) -> None:
    git_root = find_git_root(workspace)
    if not git_root:
        return
    excluded = [runs_root] if runs_root else None
    paths = git_changed_pathspecs(git_root, excluded_roots=excluded)
    if not paths:
        return
    ctx = ContextStore(workspace)
    ctx.init()
    commit_checkpoint(
        git_root,
        "review-fixes-complete",
        "applied build review recommendations",
        paths,
        record_context=ctx,
        task_id=wf.task_id or run.run_id,
        excluded_roots=excluded,
    )


def _checkpoint_build_milestones(
    workspace: Path,
    git_root: Path,
    run: RunState,
    paths: list[str],
    *,
    runs_root: Path | None = None,
) -> None:
    ctx = ContextStore(workspace)
    ctx.init()
    has_tests = any(_looks_like_test_path(path) for path in paths)
    commit_checkpoint(
        git_root,
        "implementation-complete",
        "post-build workspace changes",
        paths,
        record_context=ctx,
        task_id=run.run_id,
        excluded_roots=[runs_root] if runs_root else None,
    )
    if has_tests:
        ctx.add_context_item(
            task_id=run.run_id,
            kind="checkpoint",
            path=None,
            content="tests-added-or-updated\n",
            metadata={"label": "tests-added-or-updated", "paths": [p for p in paths if _looks_like_test_path(p)]},
        )


def _looks_like_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return normalized.startswith("tests/") or "/tests/" in normalized or normalized.startswith("test_")


def _emit_local_progress(progress: ProgressHook, run: RunState, message: str) -> None:
    run._last_local_progress = message  # type: ignore[attr-defined]
    progress("local", "workflow", run, None, None, None)


def _format_terminal_report(markdown: str) -> str:
    lines = ["", "=" * 72, " WORKFLOW REPORT", "=" * 72, ""]
    lines.extend(markdown.rstrip().splitlines())
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)
