import subprocess
from pathlib import Path

from llm_plan_execute.config import ProviderConfig
from llm_plan_execute.providers import ClaudeAdapter, CLIProvider, CodexAdapter, CursorAdapter, Provider, ProviderRouter
from llm_plan_execute.types import ModelInfo, ProviderResult, Usage


def test_cli_provider_captures_nonzero_exit_code(monkeypatch):
    model = ModelInfo("codex", "model")
    provider = CLIProvider(ProviderConfig("codex", "codex", True, (model,)))

    def fake_run(*args, **_kwargs):
        return subprocess.CompletedProcess(args[0], 42, stdout="", stderr="bad input")

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda command: f"/bin/{command}")
    monkeypatch.setattr("llm_plan_execute.providers.subprocess.run", fake_run)

    result = provider.run("planner", model, "prompt")

    assert result.error == "Provider exited with code 42: bad input"
    assert result.output == "Provider returned no stdout. stderr:\nbad input"


def test_cli_provider_captures_nonzero_exit_code_without_stderr(monkeypatch):
    model = ModelInfo("codex", "model")
    provider = CLIProvider(ProviderConfig("codex", "codex", True, (model,)))

    def fake_run(*args, **_kwargs):
        return subprocess.CompletedProcess(args[0], 2, stdout="", stderr="")

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda command: f"/bin/{command}")
    monkeypatch.setattr("llm_plan_execute.providers.subprocess.run", fake_run)

    result = provider.run("planner", model, "prompt")

    assert result.error == "Provider exited with code 2"
    assert result.output == "Provider returned no stdout. Provider exited with code 2"


def test_codex_adapter_builds_noninteractive_command():
    model = ModelInfo("codex", "gpt-5.4")
    config = ProviderConfig("codex", "codex", True, (model,))
    workspace = Path(".")

    command = CodexAdapter().build_command(config, model, "write tests", workspace)

    assert command.args == [
        "codex",
        "exec",
        "--model",
        "gpt-5.4",
        "--cd",
        str(workspace.resolve()),
        "write tests",
    ]
    assert command.cwd == workspace.resolve()


def test_cursor_adapter_builds_headless_command():
    model = ModelInfo("cursor", "auto")
    config = ProviderConfig("cursor", "cursor-agent", True, (model,))
    workspace = Path(".")

    command = CursorAdapter().build_command(config, model, "build feature", workspace)

    assert command.args == [
        "cursor-agent",
        "--print",
        "--output-format",
        "text",
        "--model",
        "auto",
        "--workspace",
        str(workspace.resolve()),
        "--trust",
        "build feature",
    ]
    assert command.cwd == workspace.resolve()


def test_claude_adapter_builds_documented_extension_command():
    model = ModelInfo("claude", "sonnet")
    config = ProviderConfig("claude", "claude", True, (model,))
    workspace = Path(".")

    command = ClaudeAdapter().build_command(config, model, "review plan", workspace)

    assert command.args == ["claude", "--model", "sonnet", "review plan"]
    assert command.cwd == workspace.resolve()


def test_cli_provider_reports_missing_command(monkeypatch):
    model = ModelInfo("codex", "gpt-5.4")
    provider = CLIProvider(ProviderConfig("codex", "missing-codex", True, (model,)))

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda _command: None)

    result = provider.run("planner", model, "prompt")

    assert result.error == "Provider command 'missing-codex' is not available on PATH."
    assert result.output == result.error


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
