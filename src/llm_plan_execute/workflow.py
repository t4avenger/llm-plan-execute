from __future__ import annotations

import hashlib
import shutil
import subprocess
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from .artifacts import write_state, write_text
from .build_review_schema import BuildRecommendation
from .config import ExecutionConfig
from .prompts import (
    build_arbiter_prompt,
    build_prompt,
    build_review_feedback_suffix,
    build_review_prompt,
    clarification_prompt,
    clarified_planner_prompt,
    plan_arbiter_prompt,
    plan_review_prompt,
    plan_revision_prompt,
    planner_prompt,
)
from .providers import ProviderRouter
from .reporting import render_report
from .selection import assign_models
from .types import ROLES, Clarification, ExecutionPolicy, ModelInfo, ProviderActivity, ProviderResult, RunState, Usage

ProgressCallback = Callable[[str, str, RunState, ModelInfo | None, ProviderResult | None, Path | None], None]
_PROPOSED_PLAN_ARTIFACT = "04-proposed-plan.md"


class BuildFailedError(ValueError):
    def __init__(self, message: str, run: RunState) -> None:
        super().__init__(message)
        self.run = run


def run_planning(
    prompt: str,
    runs_dir: Path,
    router: ProviderRouter,
    *,
    auto_accept: bool,
    execution: ExecutionConfig | None = None,
    permission_mode: str | None = None,
    progress: ProgressCallback | None = None,
) -> RunState:
    models = router.available_models()
    assignments, warnings = assign_models(models)
    run = RunState.create(prompt, runs_dir)
    run.assignments = assignments
    run.warnings.extend(warnings)
    return complete_planning(
        run,
        router,
        auto_accept=auto_accept,
        execution=execution,
        permission_mode=permission_mode,
        progress=progress,
    )


def run_clarification(
    prompt: str,
    runs_dir: Path,
    router: ProviderRouter,
    *,
    execution: ExecutionConfig | None = None,
    permission_mode: str | None = None,
    progress: ProgressCallback | None = None,
) -> RunState:
    models = router.available_models()
    assignments, warnings = assign_models(models)
    run = RunState.create(prompt, runs_dir)
    run.assignments = assignments
    run.warnings.extend(warnings)

    result = _run_provider(
        run,
        router,
        "clarifier",
        assignments["planner"].model,
        clarification_prompt(prompt),
        _execution_policy(execution, "clarifier", permission_mode),
        progress,
    )
    run.results.append(result)
    run.clarification = parse_clarification(result.output)
    write_text(run, "00-clarification.md", render_clarification(run.clarification))
    write_state(run)
    write_text(run, "report.md", render_report(run))
    return run


def complete_planning(
    run: RunState,
    router: ProviderRouter,
    *,
    auto_accept: bool,
    execution: ExecutionConfig | None = None,
    permission_mode: str | None = None,
    progress: ProgressCallback | None = None,
    planner_prompt_override: str | None = None,
) -> RunState:
    planning_prompt = planner_prompt_override if planner_prompt_override is not None else _planning_prompt(run)
    planner = _run_provider(
        run,
        router,
        "planner",
        run.assignments["planner"].model,
        planning_prompt,
        _execution_policy(execution, "planner", permission_mode),
        progress,
    )
    run.results.append(planner)
    write_text(run, "01-draft-plan.md", planner.output)

    review_a = _run_provider(
        run,
        router,
        "plan_reviewer_a",
        run.assignments["plan_reviewer_a"].model,
        plan_review_prompt(planner.output, "reviewer A"),
        _execution_policy(execution, "plan_reviewer_a", permission_mode),
        progress,
    )
    review_b = _run_provider(
        run,
        router,
        "plan_reviewer_b",
        run.assignments["plan_reviewer_b"].model,
        plan_review_prompt(planner.output, "reviewer B"),
        _execution_policy(execution, "plan_reviewer_b", permission_mode),
        progress,
    )
    run.results.extend([review_a, review_b])
    write_text(run, "02-plan-review-a.md", review_a.output)
    write_text(run, "03-plan-review-b.md", review_b.output)

    arbiter = _run_provider(
        run,
        router,
        "plan_arbiter",
        run.assignments["plan_arbiter"].model,
        plan_arbiter_prompt(planner.output, review_a.output, review_b.output),
        _execution_policy(execution, "plan_arbiter", permission_mode),
        progress,
    )
    run.results.append(arbiter)
    if auto_accept:
        run.accepted_plan = arbiter.output
        write_text(run, "04-accepted-plan.md", run.accepted_plan)
    else:
        write_text(run, _PROPOSED_PLAN_ARTIFACT, arbiter.output)

    write_state(run)
    report = render_report(run)
    write_text(run, "report.md", report)
    return run


def revise_proposed_plan(
    run: RunState,
    router: ProviderRouter,
    *,
    execution: ExecutionConfig | None = None,
    permission_mode: str | None = None,
    progress: ProgressCallback | None = None,
    feedback_history: list[str],
) -> RunState:
    proposed_path = run.run_dir / _PROPOSED_PLAN_ARTIFACT
    prior = proposed_path.read_text(encoding="utf-8") if proposed_path.exists() else ""
    history = list(feedback_history)
    planner_prompt_override = plan_revision_prompt(run.prompt, prior, history)
    return complete_planning(
        run,
        router,
        auto_accept=False,
        execution=execution,
        permission_mode=permission_mode,
        progress=progress,
        planner_prompt_override=planner_prompt_override,
    )


def rerun_build_review(
    existing: RunState,
    router: ProviderRouter,
    *,
    execution: ExecutionConfig | None = None,
    permission_mode: str | None = None,
    progress: ProgressCallback | None = None,
    feedback_history: list[str],
) -> RunState:
    if not existing.accepted_plan or not existing.build_output:
        raise ValueError("Run must include accepted plan and build output before rerunning build review.")

    feedback_suffix = build_review_feedback_suffix(feedback_history)
    review_a = _run_provider(
        existing,
        router,
        "build_reviewer_a",
        existing.assignments["build_reviewer_a"].model,
        build_review_prompt(existing.accepted_plan, existing.build_output, "implementation reviewer A")
        + feedback_suffix,
        _execution_policy(execution, "build_reviewer_a", permission_mode),
        progress,
    )
    review_b = _run_provider(
        existing,
        router,
        "build_reviewer_b",
        existing.assignments["build_reviewer_b"].model,
        build_review_prompt(existing.accepted_plan, existing.build_output, "implementation reviewer B")
        + feedback_suffix,
        _execution_policy(execution, "build_reviewer_b", permission_mode),
        progress,
    )
    existing.results.extend([review_a, review_b])
    write_text(existing, "06-build-review-a.md", review_a.output)
    write_text(existing, "07-build-review-b.md", review_b.output)

    arbiter = _run_provider(
        existing,
        router,
        "build_arbiter",
        existing.assignments["build_arbiter"].model,
        build_arbiter_prompt(review_a.output, review_b.output),
        _execution_policy(execution, "build_arbiter", permission_mode),
        progress,
    )
    existing.results.append(arbiter)
    write_text(existing, "08-build-review-summary.md", arbiter.output)

    existing.next_options = [
        "fix findings with the builder model",
        "accept the build as-is",
        "return to planning with review feedback",
    ]
    write_state(existing)
    write_text(existing, "report.md", render_report(existing))
    return existing


def record_build_recommendation_application(
    run: RunState,
    recommendations: list[BuildRecommendation],
    selected_ids: list[str],
) -> None:
    by_id = {rec.id: rec for rec in recommendations}
    lines = ["# Applied Build Review Recommendations", ""]
    for rec_id in selected_ids:
        rec = by_id.get(rec_id)
        if rec is None:
            continue
        lines.append(f"## {rec.title} (`{rec.id}`)")
        lines.append("")
        lines.append(rec.description)
        lines.append("")
    write_text(run, "09-build-review-applied.md", "\n".join(lines).rstrip())
    write_text(run, "report.md", render_report(run))
    write_state(run)


def apply_build_recommendation_fixes(
    existing: RunState,
    router: ProviderRouter,
    recommendations_text: str,
    *,
    execution: ExecutionConfig | None = None,
    permission_mode: str | None = None,
    progress: ProgressCallback | None = None,
    runs_root: Path | None = None,
) -> RunState:
    if not existing.accepted_plan or not existing.build_output:
        raise ValueError("Run must include accepted plan and build output before applying review fixes.")

    prompt = "\n\n".join(
        [
            "Apply the selected build-review recommendations to the workspace.",
            "Accepted plan:",
            existing.accepted_plan,
            "Current build output:",
            existing.build_output,
            "Selected recommendations:",
            recommendations_text,
            "Make the code changes, update or add tests when needed, and summarize what changed.",
        ]
    )
    before_changes = _workspace_changes(router.workspace, exclude_runs_under=runs_root)
    build = _run_provider(
        existing,
        router,
        "builder",
        existing.assignments["builder"].model,
        prompt,
        _execution_policy(execution, "builder", permission_mode),
        progress,
    )
    existing.results.append(build)
    existing.build_output = build.output
    write_text(existing, "11-build-review-fix-output.md", build.output)
    after_changes = _workspace_changes(router.workspace, exclude_runs_under=runs_root)
    failure = _build_failure(existing, build, before_changes, after_changes, router.dry_run)
    if failure:
        existing.build_status = "failed"
        existing.build_failure = failure
        existing.warnings.append(f"Build review fix failed: {failure}")
        write_state(existing)
        write_text(existing, "report.md", render_report(existing))
        raise BuildFailedError(failure, existing)
    existing.build_status = "succeeded"
    write_state(existing)
    write_text(existing, "report.md", render_report(existing))
    return rerun_build_review(
        existing,
        router,
        execution=execution,
        permission_mode=permission_mode,
        progress=progress,
        feedback_history=["Verify the selected build-review recommendations were applied."],
    )


def accept_plan(existing: RunState) -> RunState:
    if existing.build_output:
        raise ValueError("Run already has build output and cannot accept a different plan.")
    proposed_path = existing.run_dir / _PROPOSED_PLAN_ARTIFACT
    if not proposed_path.exists():
        raise ValueError("Run has no proposed plan to accept.")

    existing.accepted_plan = proposed_path.read_text(encoding="utf-8").rstrip()
    write_text(existing, "04-accepted-plan.md", existing.accepted_plan)
    write_state(existing)
    write_text(existing, "report.md", render_report(existing))
    return existing


def run_build(
    existing: RunState,
    router: ProviderRouter,
    *,
    execution: ExecutionConfig | None = None,
    permission_mode: str | None = None,
    progress: ProgressCallback | None = None,
    runs_root: Path | None = None,
) -> RunState:
    if not existing.accepted_plan:
        raise ValueError("Run has no accepted plan. Review the proposed plan, then run the accept command.")
    _ensure_assignments(existing, router)

    before_changes = _workspace_changes(router.workspace, exclude_runs_under=runs_root)
    build = _run_provider(
        existing,
        router,
        "builder",
        existing.assignments["builder"].model,
        build_prompt(existing.accepted_plan),
        _execution_policy(execution, "builder", permission_mode),
        progress,
    )
    existing.results.append(build)
    existing.build_output = build.output
    write_text(existing, "05-build-output.md", build.output)
    after_changes = _workspace_changes(router.workspace, exclude_runs_under=runs_root)
    failure = _build_failure(existing, build, before_changes, after_changes, router.dry_run)
    if failure:
        existing.build_status = "failed"
        existing.build_failure = failure
        existing.warnings.append(f"Build failed: {failure}")
        existing.next_options = [
            "inspect 05-build-output.md",
            "fix provider availability or builder errors",
            "rerun build after the failure is resolved",
        ]
        write_text(existing, "05-build-output.md", build.output + "\n\n## Build Failure\n\n" + failure)
        write_state(existing)
        write_text(existing, "report.md", render_report(existing))
        raise BuildFailedError(failure, existing)

    existing.build_status = "succeeded"

    review_a = _run_provider(
        existing,
        router,
        "build_reviewer_a",
        existing.assignments["build_reviewer_a"].model,
        build_review_prompt(existing.accepted_plan, build.output, "implementation reviewer A"),
        _execution_policy(execution, "build_reviewer_a", permission_mode),
        progress,
    )
    review_b = _run_provider(
        existing,
        router,
        "build_reviewer_b",
        existing.assignments["build_reviewer_b"].model,
        build_review_prompt(existing.accepted_plan, build.output, "implementation reviewer B"),
        _execution_policy(execution, "build_reviewer_b", permission_mode),
        progress,
    )
    existing.results.extend([review_a, review_b])
    write_text(existing, "06-build-review-a.md", review_a.output)
    write_text(existing, "07-build-review-b.md", review_b.output)

    arbiter = _run_provider(
        existing,
        router,
        "build_arbiter",
        existing.assignments["build_arbiter"].model,
        build_arbiter_prompt(review_a.output, review_b.output),
        _execution_policy(execution, "build_arbiter", permission_mode),
        progress,
    )
    existing.results.append(arbiter)
    write_text(existing, "08-build-review-summary.md", arbiter.output)

    existing.next_options = [
        "fix findings with the builder model",
        "accept the build as-is",
        "return to planning with review feedback",
    ]
    write_state(existing)
    write_text(existing, "report.md", render_report(existing))
    return existing


_HEARTBEAT_INTERVAL_SECONDS = 30


@contextmanager
def _provider_heartbeat(progress: ProgressCallback | None, role: str, run: RunState, model: ModelInfo):  # type: ignore[misc]
    """Emit heartbeat events on a background thread while a provider call runs."""
    if not progress:
        yield
        return
    stop_event = threading.Event()

    def _loop() -> None:
        while not stop_event.wait(timeout=_HEARTBEAT_INTERVAL_SECONDS):
            progress("heartbeat", role, run, model, None, None)  # type: ignore[misc]

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)


def _run_provider(
    run: RunState,
    router: ProviderRouter,
    role: str,
    model: ModelInfo,
    prompt: str,
    execution_policy: ExecutionPolicy,
    progress: ProgressCallback | None,
) -> ProviderResult:
    run.execution_policies[role] = execution_policy
    if progress:
        progress("start", role, run, model, None, None)
    with _provider_heartbeat(progress, role, run, model):
        try:
            activity = None
            if progress:

                def activity(item: ProviderActivity) -> None:
                    _emit_provider_activity(progress, run, item)

            result = router.run(role, model, prompt, execution_policy, activity)
        except Exception as exc:
            if progress:
                progress("finish", role, run, model, _failed_provider_result(role, model, prompt, exc), None)
            raise
        if result.warning and result.warning not in run.warnings:
            run.warnings.append(result.warning)
        if progress:
            progress("finish", role, run, model, result, None)
        return result


def _emit_provider_activity(progress: ProgressCallback, run: RunState, activity: ProviderActivity) -> None:
    # Keep the normalized event object available to richer progress reporters without
    # changing the public callback signature used by older tests.
    run._last_provider_activity = activity  # type: ignore[attr-defined]
    progress("activity", activity.role, run, activity.model, None, None)


def _failed_provider_result(role: str, model: ModelInfo, prompt: str, exc: Exception) -> ProviderResult:
    error = str(exc)
    return ProviderResult(role, model, prompt, error, Usage(), 0.0, error)


def _execution_policy(
    execution: ExecutionConfig | None,
    role: str,
    permission_mode: str | None,
) -> ExecutionPolicy:
    return (execution or ExecutionConfig()).policy_for_role(role, mode_override=permission_mode)


def _git_exclude_pathspecs(workspace: Path, exclude_runs_under: Path | None) -> list[str]:
    if exclude_runs_under is None:
        return []
    try:
        rel = exclude_runs_under.resolve().relative_to(workspace.resolve())
    except ValueError:
        return []
    if rel == Path("."):
        return []
    return [f":(exclude){rel.as_posix()}"]


def _workspace_changes(workspace: Path, *, exclude_runs_under: Path | None = None) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    exclude = _git_exclude_pathspecs(workspace, exclude_runs_under)
    status = _git_output(git, workspace, ["status", "--porcelain=v1", "-z"], pathspec_excludes=exclude)
    staged_diff = _git_output(git, workspace, ["diff", "--cached", "--binary", "HEAD", "--"], pathspec_excludes=exclude)
    unstaged_diff = _git_output(git, workspace, ["diff", "--binary", "--"], pathspec_excludes=exclude)
    untracked = _git_output(
        git, workspace, ["ls-files", "--others", "--exclude-standard", "-z"], pathspec_excludes=exclude
    )
    if status is None or staged_diff is None or unstaged_diff is None or untracked is None:
        return None
    return "\0".join((status, staged_diff, unstaged_diff, _untracked_file_hashes(workspace, untracked)))


def _git_output(
    git: str,
    workspace: Path,
    args: list[str],
    *,
    pathspec_excludes: list[str] | None = None,
) -> str | None:
    excludes = pathspec_excludes or []
    if not excludes:
        cmd = [git, *args]
    elif args and args[-1] == "--":
        cmd = [git, *args[:-1], "--", ".", *excludes]
    else:
        cmd = [git, *args, "--", ".", *excludes]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed git executable with static arguments.
            cmd,
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _untracked_file_hashes(workspace: Path, output: str) -> str:
    entries: list[str] = []
    for name in output.split("\0"):
        if not name:
            continue
        path = workspace / name
        if not path.is_file():
            entries.append(f"{name}:<not-file>")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{name}:{digest}")
    return "\0".join(entries)


def _build_failure(
    run: RunState,
    build: ProviderResult,
    before_changes: str | None,
    after_changes: str | None,
    dry_run: bool,
) -> str | None:
    if build.error:
        return build.error
    if dry_run or not _looks_code_changing(run.accepted_plan or ""):
        return None
    if before_changes is not None and after_changes is not None and before_changes == after_changes:
        return "Builder completed without changing the workspace for a code-changing plan."
    return None


def _looks_code_changing(plan: str) -> bool:
    text = plan.lower()
    indicators = (
        "implement",
        "code",
        "file",
        "test",
        "refactor",
        "fix",
        "build",
        "feature",
        "change",
    )
    return any(indicator in text for indicator in indicators)


def _ensure_assignments(existing: RunState, router: ProviderRouter) -> None:
    missing_roles = [role for role in ROLES if role not in existing.assignments]
    if not missing_roles:
        return
    assignments, warnings = assign_models(router.available_models())
    for role in missing_roles:
        existing.assignments[role] = assignments[role]
    existing.warnings.extend(warnings)
    existing.warnings.append(f"Filled missing model assignments for roles: {', '.join(missing_roles)}.")


def parse_clarification(output: str) -> Clarification:
    status = "clear"
    questions: list[str] = []
    assumptions: list[str] = []
    section: str | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if _is_status_line(lower):
            raw_status = line.partition(":")[2].strip().lower()
            status = "needs_questions" if "question" in raw_status else "clear"
            continue
        if _is_questions_header(lower):
            section = "questions"
            continue
        if _is_assumptions_header(lower):
            section = "assumptions"
            continue
        if not line.startswith("-"):
            continue

        _append_clarification_item(section, line, questions, assumptions)

    if questions:
        status = "needs_questions"
    return Clarification(status=status, questions=questions, assumptions=assumptions, raw_output=output)


def render_clarification(clarification: Clarification) -> str:
    lines = ["# Clarification", "", f"- Status: {clarification.status}", "", "## Questions"]
    lines.extend(f"- {question}" for question in clarification.questions)
    if not clarification.questions:
        lines.append("- none")
    lines.extend(["", "## Answers"])
    lines.extend(f"- {answer}" for answer in clarification.answers)
    if not clarification.answers:
        lines.append("- none")
    lines.extend(["", "## Assumptions"])
    lines.extend(f"- {assumption}" for assumption in clarification.assumptions)
    if not clarification.assumptions:
        lines.append("- none")
    lines.extend(["", "## Raw Output", clarification.raw_output.rstrip()])
    return "\n".join(lines)


def _planning_prompt(run: RunState) -> str:
    clarification = run.clarification
    if clarification and clarification.questions and clarification.answers:
        return clarified_planner_prompt(run.prompt, clarification.questions, clarification.answers)
    return planner_prompt(run.prompt)


def _is_status_line(line: str) -> bool:
    return line.startswith("status:")


def _is_questions_header(line: str) -> bool:
    return line.startswith("questions:")


def _is_assumptions_header(line: str) -> bool:
    return line.startswith("assumptions:")


def _append_clarification_item(
    section: str | None,
    line: str,
    questions: list[str],
    assumptions: list[str],
) -> None:
    item = line[1:].strip()
    if not item or item.lower() == "none":
        return
    if section == "questions":
        questions.append(item)
    elif section == "assumptions":
        assumptions.append(item)
