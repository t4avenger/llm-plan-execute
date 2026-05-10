"""Central interactive workflow orchestration for plan/run/build commands.

Command inventory
---------------

- ``plan``: shared clarification path, planning, and typed plan review loop; stops after acceptance,
  cancellation, or non-interactive auto-accept. Excluded from automatically chaining into build execution.

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
from .html_report import deterministic_html_report_path, write_html_report
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
    complete_planning,
    record_build_recommendation_application,
    render_clarification,
    rerun_build_review,
    revise_proposed_plan,
    run_build,
    run_clarification,
    run_planning,
)
from .workflow_state import WorkflowState, save_workflow_state

ProgressHook = Callable[..., None]


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
        wf.stage = "plan_review"
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
            print(f"Report: {run.run_dir / 'report.md'}")
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
    wf.stage = "plan_review"
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
        wf.stage = "plan_review"
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
) -> str | None:
    """Return ``cancel`` when canceled; otherwise ``None``."""
    summary_path = run.run_dir / "08-build-review-summary.md"
    if not summary_path.exists():
        return None

    while True:
        recommendations = parse_recommendations_from_summary(summary_path.read_text(encoding="utf-8"))
        decision = session.ask_build_review()
        wf.build_review_selected_action = decision.type
        save_workflow_state(run.run_dir, wf)

        if decision.type == "cancel":
            wf.lifecycle_status = "canceled"
            wf.build_review_selected_ids = []
            save_workflow_state(run.run_dir, wf)
            return "cancel"
        if decision.type == "continueWithoutApplying":
            wf.build_review_selected_ids = []
            save_workflow_state(run.run_dir, wf)
            return None
        if decision.type == "feedback":
            feedback = session.read_build_feedback()
            if session.non_interactive and not feedback:
                continue
            wf.build_review_feedback_history.append(feedback)
            save_workflow_state(run.run_dir, wf)
            run = rerun_build_review(
                run,
                router,
                execution=execution,
                permission_mode=permission_mode,
                progress=progress,
                feedback_history=wf.build_review_feedback_history,
            )
            save_workflow_state(run.run_dir, wf)
            continue

        selected_ids = _resolve_build_review_selection(decision, recommendations, session)
        wf.build_review_selected_ids = list(selected_ids)
        save_workflow_state(run.run_dir, wf)
        expanded = expand_with_dependencies(selected_ids, recommendations)
        missing = selection_requires_missing_dependency(expanded, recommendations)
        if missing:
            print(missing)
            continue
        wf.build_review_applied_ids = list(expanded)
        save_workflow_state(run.run_dir, wf)
        record_build_recommendation_application(run, recommendations, expanded)
        _write_apply_follow_up_notes(run, recommendations, expanded)
        return None


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
) -> tuple[RunState, int]:
    """Run build, build-review loop, and completion reporting; return final run state and exit code."""
    build_permission = permission_mode_cli or resolve_build_permission(execution.build_mode, session)
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
        run = exc.run
        wf.lifecycle_status = "failed"
        save_workflow_state(run.run_dir, wf)
        print(f"Build failed: {run.run_dir / '05-build-output.md'}")
        print(f"Report: {run.run_dir / 'report.md'}")
        return run, 1

    wf.stage = "build_review"
    save_workflow_state(run.run_dir, wf)
    review_outcome = interactive_build_review_loop(
        session=session,
        run=run,
        router=router,
        execution=execution,
        permission_mode=build_permission,
        progress=progress,
        wf=wf,
    )
    if review_outcome == "cancel":
        print(f"Report: {run.run_dir / 'report.md'}")
        return run, 130

    completion = finalize_completion_reports(session=session, run=run, wf=wf)
    if completion == "cancel":
        print(f"Report: {run.run_dir / 'report.md'}")
        return run, 130

    print(f"Build output: {run.run_dir / '05-build-output.md'}")
    print(f"Review summary: {run.run_dir / '08-build-review-summary.md'}")
    print(f"Report: {run.run_dir / 'report.md'}")
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
        wf.report_markdown_path = "report.md"
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

    wf.report_markdown_path = "report.md"
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
) -> None:
    by_id = {rec.id: rec for rec in recommendations}
    instructions = "\n".join(
        f"- {by_id[rec_id].title}: {by_id[rec_id].description}" for rec_id in expanded if rec_id in by_id
    )
    note = "# Follow-up application instructions\n\n" + instructions + "\n"
    write_text(run, "10-build-follow-up.md", note)
    write_text(run, "report.md", render_report(run))
    write_state(run)


def _format_terminal_report(markdown: str) -> str:
    lines = ["", "=" * 72, " WORKFLOW REPORT", "=" * 72, ""]
    lines.extend(markdown.rstrip().splitlines())
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)
