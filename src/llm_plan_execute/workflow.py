from __future__ import annotations

from pathlib import Path

from .artifacts import write_state, write_text
from .prompts import (
    build_arbiter_prompt,
    build_prompt,
    build_review_prompt,
    plan_arbiter_prompt,
    plan_review_prompt,
    planner_prompt,
)
from .providers import ProviderRouter
from .reporting import render_report
from .selection import assign_models
from .types import ROLES, RunState


def run_planning(prompt: str, runs_dir: Path, router: ProviderRouter, *, auto_accept: bool) -> RunState:
    models = router.available_models()
    assignments, warnings = assign_models(models)
    run = RunState.create(prompt, runs_dir)
    run.assignments = assignments
    run.warnings.extend(warnings)

    planner = router.run("planner", assignments["planner"].model, planner_prompt(prompt))
    run.results.append(planner)
    write_text(run, "01-draft-plan.md", planner.output)

    review_a = router.run(
        "plan_reviewer_a",
        assignments["plan_reviewer_a"].model,
        plan_review_prompt(planner.output, "reviewer A"),
    )
    review_b = router.run(
        "plan_reviewer_b",
        assignments["plan_reviewer_b"].model,
        plan_review_prompt(planner.output, "reviewer B"),
    )
    run.results.extend([review_a, review_b])
    write_text(run, "02-plan-review-a.md", review_a.output)
    write_text(run, "03-plan-review-b.md", review_b.output)

    arbiter = router.run(
        "plan_arbiter",
        assignments["plan_arbiter"].model,
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


def run_build(existing: RunState, router: ProviderRouter) -> RunState:
    if not existing.accepted_plan:
        raise ValueError("Run has no accepted plan. Run the plan command with --yes before building.")
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
