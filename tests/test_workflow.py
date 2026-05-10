from llm_plan_execute.providers import DryRunProvider, ProviderRouter
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
