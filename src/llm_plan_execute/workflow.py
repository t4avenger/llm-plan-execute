from __future__ import annotations

from pathlib import Path

from .artifacts import write_state, write_text
from .prompts import (
    build_arbiter_prompt,
    build_prompt,
    build_review_prompt,
    clarification_prompt,
    clarified_planner_prompt,
    plan_arbiter_prompt,
    plan_review_prompt,
    planner_prompt,
)
from .providers import ProviderRouter
from .reporting import render_report
from .selection import assign_models
from .types import ROLES, Clarification, RunState


def run_planning(prompt: str, runs_dir: Path, router: ProviderRouter, *, auto_accept: bool) -> RunState:
    models = router.available_models()
    assignments, warnings = assign_models(models)
    run = RunState.create(prompt, runs_dir)
    run.assignments = assignments
    run.warnings.extend(warnings)
    return complete_planning(run, router, auto_accept=auto_accept)


def run_clarification(prompt: str, runs_dir: Path, router: ProviderRouter) -> RunState:
    models = router.available_models()
    assignments, warnings = assign_models(models)
    run = RunState.create(prompt, runs_dir)
    run.assignments = assignments
    run.warnings.extend(warnings)

    result = router.run("clarifier", assignments["planner"].model, clarification_prompt(prompt))
    run.results.append(result)
    run.clarification = parse_clarification(result.output)
    write_text(run, "00-clarification.md", render_clarification(run.clarification))
    write_state(run)
    write_text(run, "report.md", render_report(run))
    return run


def complete_planning(run: RunState, router: ProviderRouter, *, auto_accept: bool) -> RunState:
    planning_prompt = _planning_prompt(run)
    planner = router.run("planner", run.assignments["planner"].model, planning_prompt)
    run.results.append(planner)
    write_text(run, "01-draft-plan.md", planner.output)

    review_a = router.run(
        "plan_reviewer_a",
        run.assignments["plan_reviewer_a"].model,
        plan_review_prompt(planner.output, "reviewer A"),
    )
    review_b = router.run(
        "plan_reviewer_b",
        run.assignments["plan_reviewer_b"].model,
        plan_review_prompt(planner.output, "reviewer B"),
    )
    run.results.extend([review_a, review_b])
    write_text(run, "02-plan-review-a.md", review_a.output)
    write_text(run, "03-plan-review-b.md", review_b.output)

    arbiter = router.run(
        "plan_arbiter",
        run.assignments["plan_arbiter"].model,
        plan_arbiter_prompt(planner.output, review_a.output, review_b.output),
    )
    run.results.append(arbiter)
    if auto_accept:
        run.accepted_plan = arbiter.output
        write_text(run, "04-accepted-plan.md", run.accepted_plan)
    else:
        write_text(run, "04-proposed-plan.md", arbiter.output)

    write_state(run)
    report = render_report(run)
    write_text(run, "report.md", report)
    return run


def accept_plan(existing: RunState) -> RunState:
    if existing.build_output:
        raise ValueError("Run already has build output and cannot accept a different plan.")
    proposed_path = existing.run_dir / "04-proposed-plan.md"
    if not proposed_path.exists():
        raise ValueError("Run has no proposed plan to accept.")

    existing.accepted_plan = proposed_path.read_text(encoding="utf-8").rstrip()
    write_text(existing, "04-accepted-plan.md", existing.accepted_plan)
    write_state(existing)
    write_text(existing, "report.md", render_report(existing))
    return existing


def run_build(existing: RunState, router: ProviderRouter) -> RunState:
    if not existing.accepted_plan:
        raise ValueError("Run has no accepted plan. Review the proposed plan, then run the accept command.")
    _ensure_assignments(existing, router)

    build = router.run("builder", existing.assignments["builder"].model, build_prompt(existing.accepted_plan))
    existing.results.append(build)
    existing.build_output = build.output
    write_text(existing, "05-build-output.md", build.output)

    review_a = router.run(
        "build_reviewer_a",
        existing.assignments["build_reviewer_a"].model,
        build_review_prompt(existing.accepted_plan, build.output, "implementation reviewer A"),
    )
    review_b = router.run(
        "build_reviewer_b",
        existing.assignments["build_reviewer_b"].model,
        build_review_prompt(existing.accepted_plan, build.output, "implementation reviewer B"),
    )
    existing.results.extend([review_a, review_b])
    write_text(existing, "06-build-review-a.md", review_a.output)
    write_text(existing, "07-build-review-b.md", review_b.output)

    arbiter = router.run(
        "build_arbiter",
        existing.assignments["build_arbiter"].model,
        build_arbiter_prompt(review_a.output, review_b.output),
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
