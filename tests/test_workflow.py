import pytest

from llm_plan_execute.providers import DryRunProvider, ProviderRouter
from llm_plan_execute.types import ModelAssignment, ModelInfo, RunState
from llm_plan_execute.workflow import accept_plan, complete_planning, run_build, run_clarification, run_planning


def test_dry_run_plan_build_report(tmp_path):
    router = ProviderRouter([DryRunProvider()])

    run = run_planning("Add a small feature", tmp_path, router, auto_accept=True)
    run = run_build(run, router)

    assert (run.run_dir / "04-accepted-plan.md").exists()
    assert (run.run_dir / "08-build-review-summary.md").exists()
    report = (run.run_dir / "report.md").read_text(encoding="utf-8")
    assert "Prompt Improvement Advice" in report
    assert "Estimated cost" in report


def test_planning_without_auto_accept_writes_proposed_plan_only(tmp_path):
    router = ProviderRouter([DryRunProvider()])

    run = run_planning("Add a small feature", tmp_path, router, auto_accept=False)

    assert run.accepted_plan is None
    assert (run.run_dir / "04-proposed-plan.md").exists()
    assert not (run.run_dir / "04-accepted-plan.md").exists()


def test_build_requires_accepted_plan(tmp_path):
    router = ProviderRouter([DryRunProvider()])
    run = run_planning("Add a small feature", tmp_path, router, auto_accept=False)

    with pytest.raises(ValueError, match="accept command"):
        run_build(run, router)


def test_accept_plan_promotes_proposed_plan(tmp_path):
    router = ProviderRouter([DryRunProvider()])
    run = run_planning("Add a small feature", tmp_path, router, auto_accept=False)
    proposed = (run.run_dir / "04-proposed-plan.md").read_text(encoding="utf-8").rstrip()

    run = accept_plan(run)

    assert run.accepted_plan == proposed
    assert (run.run_dir / "04-accepted-plan.md").read_text(encoding="utf-8").rstrip() == proposed


def test_build_succeeds_after_accept(tmp_path):
    router = ProviderRouter([DryRunProvider()])
    run = run_planning("Add a small feature", tmp_path, router, auto_accept=False)
    run = accept_plan(run)

    run = run_build(run, router)

    assert (run.run_dir / "08-build-review-summary.md").exists()


def test_accept_requires_proposed_plan(tmp_path):
    run = RunState.create("prompt", tmp_path)
    run.run_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="no proposed plan"):
        accept_plan(run)


def test_accept_refuses_after_build_output(tmp_path):
    router = ProviderRouter([DryRunProvider()])
    run = run_planning("Add a small feature", tmp_path, router, auto_accept=False)
    run.build_output = "already built"

    with pytest.raises(ValueError, match="already has build output"):
        accept_plan(run)


def test_clarification_clear_prompt_can_complete(tmp_path):
    router = ProviderRouter([DryRunProvider()])

    run = run_clarification("Add a small feature", tmp_path, router)
    run = complete_planning(run, router, auto_accept=False)

    assert run.clarification is not None
    assert run.clarification.status == "clear"
    assert (run.run_dir / "00-clarification.md").exists()
    assert (run.run_dir / "04-proposed-plan.md").exists()


def test_clarification_records_answers(tmp_path):
    router = ProviderRouter([DryRunProvider()])

    run = run_clarification("Do an ambiguous thing", tmp_path, router)
    assert run.clarification is not None
    run.clarification.answers = ["Add the smallest useful behavior."]
    run = complete_planning(run, router, auto_accept=False)

    assert run.clarification.answers == ["Add the smallest useful behavior."]
    assert (run.run_dir / "04-proposed-plan.md").exists()


def test_build_fills_missing_assignments(tmp_path):
    router = ProviderRouter([DryRunProvider()])
    run = RunState.create("prompt", tmp_path)
    run.accepted_plan = "accepted"
    run.assignments["builder"] = ModelAssignment("builder", ModelInfo("dry-cursor", "fast-builder"))

    run_build(run, router)

    assert "build_reviewer_a" in run.assignments
    assert any("Filled missing model assignments" in warning for warning in run.warnings)
