import subprocess

from llm_plan_execute.config import ProviderConfig
from llm_plan_execute.providers import CLIProvider, Provider, ProviderRouter
from llm_plan_execute.types import ModelInfo, ProviderResult, Usage


def test_cli_provider_captures_nonzero_exit_code(monkeypatch):
    model = ModelInfo("test", "model")
    provider = CLIProvider(ProviderConfig("test", "test-provider", True, (model,)))

    def fake_run(*args, **_kwargs):
        return subprocess.CompletedProcess(args[0], 42, stdout="", stderr="bad input")

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda command: f"/bin/{command}")
    monkeypatch.setattr("llm_plan_execute.providers.subprocess.run", fake_run)

    result = provider.run("planner", model, "prompt")

    assert result.error == "Provider exited with code 42: bad input"
    assert result.output == "Provider returned no stdout. stderr:\nbad input"


def test_cli_provider_captures_nonzero_exit_code_without_stderr(monkeypatch):
    model = ModelInfo("test", "model")
    provider = CLIProvider(ProviderConfig("test", "test-provider", True, (model,)))

    def fake_run(*args, **_kwargs):
        return subprocess.CompletedProcess(args[0], 2, stdout="", stderr="")

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda command: f"/bin/{command}")
    monkeypatch.setattr("llm_plan_execute.providers.subprocess.run", fake_run)

    result = provider.run("planner", model, "prompt")

    assert result.error == "Provider exited with code 2"
    assert result.output == "Provider returned no stdout. Provider exited with code 2"


def test_provider_router_uses_provider_that_owns_model():
    model_a = ModelInfo("a", "model")
    model_b = ModelInfo("b", "model")
    provider_a = RecordingProvider(model_a)
    provider_b = RecordingProvider(model_b)
    router = ProviderRouter([provider_a, provider_b])

    result = router.run("builder", model_b, "prompt")

    assert result.model == model_b
    assert provider_a.calls == []
    assert provider_b.calls == ["builder"]


class RecordingProvider(Provider):
    def __init__(self, model: ModelInfo) -> None:
        self.model = model
        self.calls: list[str] = []

    def available_models(self) -> list[ModelInfo]:
        return [self.model]

    def run(self, role: str, model: ModelInfo, prompt: str) -> ProviderResult:
        self.calls.append(role)
        return ProviderResult(role, model, prompt, "ok", Usage(), 0.0)
