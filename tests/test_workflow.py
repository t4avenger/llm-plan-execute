import pytest

from llm_plan_execute.config import ExecutionConfig
from llm_plan_execute.providers import DryRunProvider, Provider, ProviderRouter
from llm_plan_execute.types import ExecutionPolicy, ModelAssignment, ModelInfo, ProviderResult, RunState, Usage
from llm_plan_execute.workflow import (
    BuildFailedError,
    _run_provider,
    accept_plan,
    complete_planning,
    run_build,
    run_clarification,
    run_planning,
)


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


def test_builder_error_fails_build_and_skips_reviewers(tmp_path):
    provider = RecordingBuildProvider(builder_error="provider failed")
    router = ProviderRouter([provider], workspace=tmp_path)
    run = _accepted_build_run(tmp_path)

    with pytest.raises(BuildFailedError, match="provider failed") as exc:
        run_build(run, router)

    assert provider.calls == ["builder"]
    assert exc.value.run.build_status == "failed"
    assert exc.value.run.build_failure == "provider failed"
    assert "Build Failure" in (run.run_dir / "05-build-output.md").read_text(encoding="utf-8")
    assert not (run.run_dir / "06-build-review-a.md").exists()
    assert "Build status: failed" in (run.run_dir / "report.md").read_text(encoding="utf-8")


def test_noop_code_build_fails_and_skips_reviewers(tmp_path, monkeypatch):
    provider = RecordingBuildProvider()
    router = ProviderRouter([provider], workspace=tmp_path)
    run = _accepted_build_run(tmp_path)
    monkeypatch.setattr("llm_plan_execute.workflow._workspace_changes", lambda _workspace: "unchanged-diff")

    with pytest.raises(BuildFailedError, match="without changing the workspace"):
        run_build(run, router)

    assert provider.calls == ["builder"]
    assert run.build_status == "failed"
    assert not (run.run_dir / "08-build-review-summary.md").exists()


def test_dirty_workspace_with_changed_diff_does_not_fail_as_noop(tmp_path, monkeypatch):
    provider = RecordingBuildProvider()
    router = ProviderRouter([provider], workspace=tmp_path)
    run = _accepted_build_run(tmp_path)
    snapshots = iter(["dirty-before", "dirty-after"])
    monkeypatch.setattr("llm_plan_execute.workflow._workspace_changes", lambda _workspace: next(snapshots))

    run_build(run, router)

    assert provider.calls == ["builder", "build_reviewer_a", "build_reviewer_b", "build_arbiter"]
    assert run.build_status == "succeeded"


def test_workflow_records_phase_execution_policies(tmp_path, monkeypatch):
    provider = RecordingBuildProvider()
    router = ProviderRouter([provider], workspace=tmp_path)
    run = _accepted_build_run(tmp_path)
    snapshots = iter(["dirty-before", "dirty-after"])
    monkeypatch.setattr("llm_plan_execute.workflow._workspace_changes", lambda _workspace: next(snapshots))

    run_build(run, router, execution=ExecutionConfig())

    assert provider.policies["builder"].mode == "workspace-write"
    assert provider.policies["build_reviewer_a"].mode == "read-only"
    assert run.execution_policies["builder"].mode == "workspace-write"
    assert run.execution_policies["build_arbiter"].mode == "read-only"


def test_permission_override_applies_to_provider_calls(tmp_path, monkeypatch):
    provider = RecordingBuildProvider()
    router = ProviderRouter([provider], workspace=tmp_path)
    run = _accepted_build_run(tmp_path)
    snapshots = iter(["dirty-before", "dirty-after"])
    monkeypatch.setattr("llm_plan_execute.workflow._workspace_changes", lambda _workspace: next(snapshots))

    run_build(run, router, execution=ExecutionConfig(), permission_mode="full-access")

    assert provider.policies["builder"].mode == "full-access"
    assert provider.policies["build_reviewer_a"].mode == "full-access"


def test_run_provider_emits_finish_progress_when_router_raises(tmp_path):
    run = RunState.create("prompt", tmp_path)
    model = ModelInfo("missing", "model")
    events = []

    def record(*event):
        events.append(event)

    with pytest.raises(ValueError, match="No provider can run"):
        _run_provider(run, ProviderRouter([]), "builder", model, "prompt", ExecutionPolicy(), record)

    assert [event[0] for event in events] == ["start", "finish"]
    assert events[1][4] is not None
    assert events[1][4].error is not None


def test_provider_warning_is_recorded_on_run(tmp_path):
    model = ModelInfo("cursor", "auto")
    provider = WarningProvider(model)
    router = ProviderRouter([provider], workspace=tmp_path)
    run = RunState.create("prompt", tmp_path)

    result = _run_provider(run, router, "builder", model, "prompt", ExecutionPolicy(), None)

    assert result.warning == "warning text"
    assert run.warnings == ["warning text"]


def _accepted_build_run(tmp_path):
    run = RunState.create("prompt", tmp_path / "runs")
    run.accepted_plan = "Implement a code feature and add tests."
    for role in (
        "builder",
        "build_reviewer_a",
        "build_reviewer_b",
        "build_arbiter",
    ):
        run.assignments[role] = ModelAssignment(role, ModelInfo("local", role, (role,)))
    return run


class RecordingBuildProvider(Provider):
    def __init__(self, *, builder_error: str | None = None) -> None:
        self.builder_error = builder_error
        self.calls: list[str] = []
        self.policies: dict[str, ExecutionPolicy] = {}

    def available_models(self) -> list[ModelInfo]:
        return [
            ModelInfo("local", "builder", ("builder",)),
            ModelInfo("local", "build_reviewer_a", ("build_reviewer_a",)),
            ModelInfo("local", "build_reviewer_b", ("build_reviewer_b",)),
            ModelInfo("local", "build_arbiter", ("build_arbiter",)),
        ]

    def run(
        self,
        role: str,
        model: ModelInfo,
        prompt: str,
        execution_policy: ExecutionPolicy | None = None,
    ) -> ProviderResult:
        self.calls.append(role)
        if execution_policy:
            self.policies[role] = execution_policy
        error = self.builder_error if role == "builder" else None
        output = error or f"{role} output"
        return ProviderResult(role, model, prompt, output, Usage(), 0.0, error)


class WarningProvider(Provider):
    def __init__(self, model: ModelInfo) -> None:
        self.model = model

    def available_models(self) -> list[ModelInfo]:
        return [self.model]

    def run(
        self,
        role: str,
        model: ModelInfo,
        prompt: str,
        _execution_policy: ExecutionPolicy | None = None,
    ) -> ProviderResult:
        return ProviderResult(role, model, prompt, "ok", Usage(), 0.0, warning="warning text")
