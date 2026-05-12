import subprocess
from pathlib import Path

import pytest

from llm_plan_execute.config import ProviderConfig
from llm_plan_execute.providers import (
    CURSOR_SANDBOX_RETRY_WARNING,
    PROVIDER_RUN_TIMEOUT_SEC,
    ClaudeAdapter,
    CLIProvider,
    CodexAdapter,
    CursorAdapter,
    Provider,
    ProviderCommand,
    ProviderRouter,
    _activity_from_stream_line,
    _feed_provider_stdin,
    _run_provider_command_streaming,
    _streaming_args,
)
from llm_plan_execute.types import ExecutionPolicy, ModelInfo, ProviderResult, Usage


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

    command = CodexAdapter().build_command(config, model, "write tests", workspace, ExecutionPolicy())

    assert command.args == [
        "codex",
        "exec",
        "--model",
        "gpt-5.4",
        "--sandbox",
        "workspace-write",
        "--cd",
        str(workspace.resolve()),
        "-",
    ]
    assert command.cwd == workspace.resolve()
    assert command.stdin == "write tests"


def test_codex_adapter_builds_full_access_command():
    model = ModelInfo("codex", "gpt-5.4")
    config = ProviderConfig("codex", "codex", True, (model,))
    workspace = Path(".")

    command = CodexAdapter().build_command(config, model, "write tests", workspace, ExecutionPolicy("full-access"))

    assert "--dangerously-bypass-approvals-and-sandbox" in command.args
    assert "--sandbox" not in command.args


def test_codex_adapter_adds_writable_dirs():
    model = ModelInfo("codex", "gpt-5.4")
    config = ProviderConfig("codex", "codex", True, (model,))
    workspace = Path(".")
    extra_dir = Path("tmp-extra")

    command = CodexAdapter().build_command(
        config,
        model,
        "write tests",
        workspace,
        ExecutionPolicy("workspace-write", (extra_dir,)),
    )

    assert command.args[command.args.index("--add-dir") + 1] == str(extra_dir.resolve())


def test_cursor_adapter_builds_headless_command():
    model = ModelInfo("cursor", "auto")
    config = ProviderConfig("cursor", "cursor-agent", True, (model,))
    workspace = Path(".")

    command = CursorAdapter().build_command(config, model, "build feature", workspace, ExecutionPolicy())

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


def test_cursor_adapter_builds_full_access_command():
    model = ModelInfo("cursor", "auto")
    config = ProviderConfig("cursor", "cursor-agent", True, (model,))
    workspace = Path(".")

    command = CursorAdapter().build_command(config, model, "build feature", workspace, ExecutionPolicy("full-access"))

    assert "--force" in command.args
    assert "--sandbox" in command.args
    assert "disabled" in command.args


def test_cursor_adapter_builds_read_only_command():
    model = ModelInfo("cursor", "auto")
    config = ProviderConfig("cursor", "cursor-agent", True, (model,))
    workspace = Path(".")

    command = CursorAdapter().build_command(config, model, "review plan", workspace, ExecutionPolicy("read-only"))

    assert "--mode" in command.args
    assert "plan" in command.args
    assert "--sandbox" not in command.args


def test_cursor_provider_retries_without_sandbox_when_system_sandbox_unavailable(monkeypatch):
    model = ModelInfo("cursor", "auto")
    provider = CLIProvider(ProviderConfig("cursor", "cursor-agent", True, (model,)))
    calls: list[list[str]] = []

    def fake_run(*args, **_kwargs):
        command = args[0]
        calls.append(command)
        if "--sandbox" in command:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="Error: Sandbox mode is enabled but not available on this system.",
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda command: f"/bin/{command}")
    monkeypatch.setattr("llm_plan_execute.providers.subprocess.run", fake_run)

    result = provider.run("builder", model, "build feature", ExecutionPolicy("full-access"))

    assert result.output == "ok"
    assert result.error is None
    assert result.warning == CURSOR_SANDBOX_RETRY_WARNING
    expected_call_count = 2
    assert len(calls) == expected_call_count
    assert "--sandbox" in calls[0]
    assert "--sandbox" not in calls[1]


def test_claude_adapter_builds_documented_extension_command():
    model = ModelInfo("claude", "sonnet")
    config = ProviderConfig("claude", "claude", True, (model,))
    workspace = Path(".")

    command = ClaudeAdapter().build_command(config, model, "review plan", workspace, ExecutionPolicy())

    assert command.args == ["claude", "--model", "sonnet", "--print", "--input-format", "text"]
    assert command.cwd == workspace.resolve()
    assert command.stdin == "review plan"


def test_claude_provider_invokes_subprocess_with_documented_contract(monkeypatch, tmp_path):
    """Spawning uses subprocess.run with cwd, capture_output, timeout (owned error boundary)."""
    model = ModelInfo("claude", "opus")
    provider = CLIProvider(
        ProviderConfig("claude", "claude", True, (model,)),
        workspace=tmp_path,
    )
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda command: f"/fake/{command}")
    monkeypatch.setattr("llm_plan_execute.providers.subprocess.run", fake_run)

    result = provider.run("plan_reviewer_b", model, "review this plan", ExecutionPolicy())

    assert result.error is None
    assert captured["cmd"] == ["claude", "--model", "opus", "--print", "--input-format", "text"]
    kwargs = captured["kwargs"]
    assert kwargs["cwd"] == tmp_path.resolve()
    assert kwargs["timeout"] == PROVIDER_RUN_TIMEOUT_SEC
    assert kwargs["text"] is True
    assert kwargs["input"] == "review this plan"
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False


def test_codex_provider_sends_prompt_via_stdin(monkeypatch, tmp_path):
    model = ModelInfo("codex", "gpt-5.4")
    provider = CLIProvider(
        ProviderConfig("codex", "codex", True, (model,)),
        workspace=tmp_path,
    )
    prompt = "large prompt " * 1000
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda command: f"/fake/{command}")
    monkeypatch.setattr("llm_plan_execute.providers.subprocess.run", fake_run)

    result = provider.run("builder", model, prompt, ExecutionPolicy())

    assert result.error is None
    assert captured["cmd"][-1] == "-"
    assert prompt not in captured["cmd"]
    assert captured["kwargs"]["input"] == prompt


def test_feed_provider_stdin_writes_and_closes_pipe():
    class FakeStdin:
        def __init__(self) -> None:
            self.value = ""
            self.closed_called = False

        def write(self, text: str) -> None:
            self.value += text

        def close(self) -> None:
            self.closed_called = True

    class FakeProcess:
        stdin = FakeStdin()

    proc = FakeProcess()
    thread = _feed_provider_stdin(proc, "streamed prompt")

    assert thread is not None
    thread.join(timeout=1)
    assert proc.stdin.value == "streamed prompt"
    assert proc.stdin.closed_called is True


def test_feed_provider_stdin_requires_pipe():
    class FakeProcess:
        stdin = None

    with pytest.raises(subprocess.SubprocessError, match="stdin pipe"):
        _feed_provider_stdin(FakeProcess(), "prompt")


def test_feed_provider_stdin_no_prompt_returns_none():
    class FakeProcess:
        stdin = None

    assert _feed_provider_stdin(FakeProcess(), None) is None


def test_feed_provider_stdin_ignores_broken_pipe():
    class BrokenStdin:
        def write(self, _text: str) -> None:
            raise BrokenPipeError

        def close(self) -> None:
            raise AssertionError("close should not run after a broken write")

    class FakeProcess:
        stdin = BrokenStdin()

    thread = _feed_provider_stdin(FakeProcess(), "prompt")

    assert thread is not None
    thread.join(timeout=1)


def test_streaming_provider_command_uses_stdin_pipe(monkeypatch, tmp_path):
    model = ModelInfo("codex", "gpt")
    captured: dict[str, object] = {}

    class FakeStdin:
        def __init__(self) -> None:
            self.value = ""
            self.closed = False

        def write(self, text: str) -> None:
            self.value += text

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self, args, **kwargs) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.stdin = FakeStdin()
            self.returncode = 0

    def fake_collect(proc, **_kwargs):
        captured["stdin"] = proc.stdin
        return "out", "err"

    monkeypatch.setattr("llm_plan_execute.providers.subprocess.Popen", FakeProcess)
    monkeypatch.setattr("llm_plan_execute.providers._collect_streaming_output", fake_collect)

    completed = _run_provider_command_streaming(
        ProviderCommand(["codex", "exec", "--model", "gpt", "-"], tmp_path, "streamed prompt"),
        provider_name="codex",
        role="builder",
        model=model,
        activity=lambda _item: None,
        start=0,
    )

    assert completed.stdout == "out"
    assert completed.stderr == "err"
    assert captured["args"] == ["codex", "exec", "--model", "gpt", "--json", "-"]
    assert captured["kwargs"]["stdin"] is subprocess.PIPE
    stdin = captured["stdin"]
    assert stdin.value == "streamed prompt"
    assert stdin.closed is True


def test_claude_provider_maps_subprocess_timeout(monkeypatch):
    model = ModelInfo("claude", "sonnet")
    provider = CLIProvider(ProviderConfig("claude", "claude", True, (model,)))

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        cmd = args[0] if args else []
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=int(kwargs.get("timeout", 0)))

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda command: f"/bin/{command}")
    monkeypatch.setattr("llm_plan_execute.providers.subprocess.run", fake_run)

    result = provider.run("builder", model, "prompt")

    assert "timed out" in result.error


def test_claude_provider_captures_nonzero_exit_code_and_stderr(monkeypatch):
    model = ModelInfo("claude", "sonnet")
    provider = CLIProvider(ProviderConfig("claude", "claude", True, (model,)))

    def fake_run(*args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 7, stdout="", stderr="auth required")

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda command: f"/bin/{command}")
    monkeypatch.setattr("llm_plan_execute.providers.subprocess.run", fake_run)

    result = provider.run("builder", model, "prompt")

    assert result.error == "Provider exited with code 7: auth required"
    assert result.output == "Provider returned no stdout. stderr:\nauth required"


def test_claude_provider_reports_unresolved_command(monkeypatch):
    model = ModelInfo("claude", "sonnet")
    provider = CLIProvider(ProviderConfig("claude", "claude", True, (model,)))

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda _command: None)

    result = provider.run("builder", model, "prompt")

    assert result.error == (
        "Provider command 'claude' could not be resolved via PATH lookup or as a configured executable path."
    )
    assert result.output == result.error


def test_streaming_args_enable_provider_json_modes():
    assert "--json" in _streaming_args("codex", ["codex", "exec", "--model", "m", "prompt"])
    cursor = _streaming_args("cursor", ["cursor-agent", "--print", "--output-format", "text", "prompt"])
    assert "stream-json" in cursor
    assert "--stream-partial-output" in cursor
    claude = _streaming_args("claude", ["claude", "--model", "sonnet", "prompt"])
    assert "--print" in claude
    assert "--verbose" in claude
    assert "stream-json" in claude


def test_activity_from_stream_line_extracts_file_activity(tmp_path):
    model = ModelInfo("codex", "gpt")
    activity = _activity_from_stream_line(
        "codex",
        '{"type": "tool_call", "tool_name": "read", "path": "src/app.py"}',
        role="builder",
        model=model,
        elapsed=12.5,
        workspace=tmp_path,
    )

    assert activity is not None
    assert activity.workspace_path == "src/app.py"
    assert activity.message == "reading src/app.py"


def test_cli_provider_reports_missing_command(monkeypatch):
    model = ModelInfo("codex", "gpt-5.4")
    provider = CLIProvider(ProviderConfig("codex", "missing-codex", True, (model,)))

    monkeypatch.setattr("llm_plan_execute.providers.shutil.which", lambda _command: None)

    result = provider.run("planner", model, "prompt")

    assert result.error == (
        "Provider command 'missing-codex' could not be resolved via PATH lookup or as a configured executable path."
    )
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

    def run(
        self,
        role: str,
        model: ModelInfo,
        prompt: str,
        _execution_policy: ExecutionPolicy | None = None,
    ) -> ProviderResult:
        self.calls.append(role)
        return ProviderResult(role, model, prompt, "ok", Usage(), 0.0)
