import pytest

from llm_plan_execute.providers import DryRunProvider, ProviderRouter
from llm_plan_execute.types import ModelAssignment, ModelInfo, RunState
from llm_plan_execute.workflow import run_build, run_planning


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

    with pytest.raises(ValueError, match="--yes"):
        run_build(run, router)


def test_build_fills_missing_assignments(tmp_path):
    router = ProviderRouter([DryRunProvider()])
    run = RunState.create("prompt", tmp_path)
    run.accepted_plan = "accepted"
    run.assignments["builder"] = ModelAssignment("builder", ModelInfo("dry-cursor", "fast-builder"))

    run_build(run, router)

    assert "build_reviewer_a" in run.assignments
    assert any("Filled missing model assignments" in warning for warning in run.warnings)
