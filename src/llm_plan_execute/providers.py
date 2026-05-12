from __future__ import annotations

import json
import selectors
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig, ProviderConfig
from .types import ExecutionPolicy, ModelInfo, ProviderActivity, ProviderResult, Usage

ActivityCallback = Callable[[ProviderActivity], None]


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


def estimate_cost(model: ModelInfo, usage: Usage) -> float:
    # Conservative placeholder: lower cost score means cheaper model.
    dollars_per_million = {1: 1.0, 2: 3.0, 3: 8.0, 4: 15.0, 5: 30.0}.get(model.cost, 8.0)
    return round((usage.total_tokens / 1_000_000) * dollars_per_million, 6)


class Provider:
    def available_models(self) -> list[ModelInfo]:
        raise NotImplementedError

    def run(
        self,
        role: str,
        model: ModelInfo,
        prompt: str,
        _execution_policy: ExecutionPolicy | None = None,
        _activity: ActivityCallback | None = None,
    ) -> ProviderResult:
        raise NotImplementedError


class DryRunProvider(Provider):
    def __init__(self) -> None:
        self.models = [
            ModelInfo("dry-codex", "frontier-planner", ("planner", "plan_arbiter"), 5, 3, 4, 5),
            ModelInfo("dry-claude", "deep-reviewer", ("plan_reviewer_b", "build_reviewer_b"), 5, 3, 4, 5),
            ModelInfo("dry-cursor", "fast-builder", ("builder",), 3, 5, 2, 4),
            ModelInfo(
                "dry-codex",
                "balanced-reviewer",
                ("plan_reviewer_a", "build_reviewer_a", "build_arbiter"),
                4,
                4,
                3,
                4,
            ),
        ]

    def available_models(self) -> list[ModelInfo]:
        return list(self.models)

    def run(
        self,
        role: str,
        model: ModelInfo,
        prompt: str,
        _execution_policy: ExecutionPolicy | None = None,
        _activity: ActivityCallback | None = None,
    ) -> ProviderResult:
        start = time.monotonic()
        output = dry_response(role, prompt)
        usage = Usage(
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(output),
            exact=False,
            confidence="estimated",
        )
        usage.cost_usd = estimate_cost(model, usage)
        return ProviderResult(role, model, prompt, output, usage, time.monotonic() - start)


@dataclass(frozen=True)
class ProviderCommand:
    args: list[str]
    cwd: Path
    stdin: str | None = None


class ProviderAdapter:
    def build_command(
        self,
        config: ProviderConfig,
        model: ModelInfo,
        prompt: str,
        workspace: Path,
        _execution_policy: ExecutionPolicy,
    ) -> ProviderCommand:
        raise NotImplementedError


CURSOR_SANDBOX_UNAVAILABLE = "Sandbox mode is enabled but not available on this system"
CURSOR_SANDBOX_RETRY_WARNING = (
    "Cursor sandbox was unavailable on this system; retried without Cursor sandbox flags. "
    "Cursor may run in allowlist or approval mode instead."
)
PROVIDER_RUN_TIMEOUT_SEC = 1800


class CodexAdapter(ProviderAdapter):
    def build_command(
        self,
        config: ProviderConfig,
        model: ModelInfo,
        prompt: str,
        workspace: Path,
        execution_policy: ExecutionPolicy,
    ) -> ProviderCommand:
        resolved_workspace = workspace.resolve()
        permission_args = _codex_permission_args(execution_policy)
        return ProviderCommand(
            [
                config.command,
                "exec",
                "--model",
                model.name,
                *permission_args,
                "--cd",
                str(resolved_workspace),
                "-",
            ],
            resolved_workspace,
            prompt,
        )


class CursorAdapter(ProviderAdapter):
    def build_command(
        self,
        config: ProviderConfig,
        model: ModelInfo,
        prompt: str,
        workspace: Path,
        execution_policy: ExecutionPolicy,
    ) -> ProviderCommand:
        resolved_workspace = workspace.resolve()
        permission_args = _cursor_permission_args(execution_policy)
        return ProviderCommand(
            [
                config.command,
                "--print",
                "--output-format",
                "text",
                "--model",
                model.name,
                "--workspace",
                str(resolved_workspace),
                "--trust",
                *permission_args,
                prompt,
            ],
            resolved_workspace,
        )


class ClaudeAdapter(ProviderAdapter):
    def build_command(
        self,
        config: ProviderConfig,
        model: ModelInfo,
        prompt: str,
        workspace: Path,
        _execution_policy: ExecutionPolicy,
    ) -> ProviderCommand:
        resolved_workspace = workspace.resolve()
        return ProviderCommand(
            [config.command, "--model", model.name, "--print", "--input-format", "text"],
            resolved_workspace,
            prompt,
        )


ADAPTERS: dict[str, ProviderAdapter] = {
    "codex": CodexAdapter(),
    "cursor": CursorAdapter(),
    "claude": ClaudeAdapter(),
}


@dataclass
class CLIProvider(Provider):
    config: ProviderConfig
    workspace: Path = Path(".")

    @property
    def adapter(self) -> ProviderAdapter | None:
        return ADAPTERS.get(self.config.name)

    def available_models(self) -> list[ModelInfo]:
        if not self.config.enabled or self.adapter is None or shutil.which(self.config.command) is None:
            return []
        return list(self.config.models)

    def run(
        self,
        role: str,
        model: ModelInfo,
        prompt: str,
        execution_policy: ExecutionPolicy | None = None,
        activity: ActivityCallback | None = None,
    ) -> ProviderResult:
        start = time.monotonic()
        usage = Usage(input_tokens=estimate_tokens(prompt), exact=False, confidence="estimated")
        adapter = self.adapter
        if adapter is None:
            output = f"No adapter is registered for provider {self.config.name!r}."
            usage.output_tokens = estimate_tokens(output)
            usage.cost_usd = estimate_cost(model, usage)
            return ProviderResult(role, model, prompt, output, usage, time.monotonic() - start, output)
        if shutil.which(self.config.command) is None:
            error = (
                f"Provider command {self.config.command!r} could not be resolved via PATH lookup "
                "or as a configured executable path."
            )
            usage.output_tokens = estimate_tokens(error)
            usage.cost_usd = estimate_cost(model, usage)
            return ProviderResult(role, model, prompt, error, usage, time.monotonic() - start, error)

        policy = execution_policy or ExecutionPolicy()
        provider_command = adapter.build_command(self.config, model, prompt, self.workspace, policy)
        try:
            completed, warning = self._run_command_with_cursor_sandbox_fallback(
                provider_command,
                role=role,
                model=model,
                activity=activity,
                start=start,
            )
            output, error = _provider_output_and_error(completed)
            usage.output_tokens = estimate_tokens(output)
            usage.cost_usd = estimate_cost(model, usage)
            return ProviderResult(role, model, prompt, output, usage, time.monotonic() - start, error, warning)
        except (OSError, subprocess.SubprocessError) as exc:
            output = f"Provider execution failed for {model.id}: {exc}"
            usage.output_tokens = estimate_tokens(output)
            usage.cost_usd = estimate_cost(model, usage)
            return ProviderResult(role, model, prompt, output, usage, time.monotonic() - start, str(exc))

    def _run_command_with_cursor_sandbox_fallback(
        self,
        provider_command: ProviderCommand,
        *,
        role: str,
        model: ModelInfo,
        activity: ActivityCallback | None,
        start: float,
    ) -> tuple[subprocess.CompletedProcess[str], str | None]:
        completed = _run_provider_command(
            provider_command,
            provider_name=self.config.name,
            role=role,
            model=model,
            activity=activity,
            start=start,
        )
        if not _should_retry_cursor_without_sandbox(self.config.name, provider_command.args, completed):
            return completed, None

        retry_command = ProviderCommand(
            _without_cursor_sandbox_args(provider_command.args),
            provider_command.cwd,
            provider_command.stdin,
        )
        retry = _run_provider_command(
            retry_command,
            provider_name=self.config.name,
            role=role,
            model=model,
            activity=activity,
            start=start,
        )
        if retry.returncode == 0 or retry.stdout.strip():
            return retry, CURSOR_SANDBOX_RETRY_WARNING
        return completed, None


class ProviderRouter:
    def __init__(self, providers: list[Provider], *, dry_run: bool = False, workspace: Path = Path(".")) -> None:
        self.providers = providers
        self.dry_run = dry_run or any(isinstance(provider, DryRunProvider) for provider in providers)
        self.workspace = workspace

    @classmethod
    def from_config(cls, config: AppConfig) -> ProviderRouter:
        if config.dry_run:
            return cls([DryRunProvider()], dry_run=True, workspace=config.workspace)
        providers: list[Provider] = []
        providers.extend(CLIProvider(provider, workspace=config.workspace) for provider in config.providers)
        return cls(providers, dry_run=False, workspace=config.workspace)

    def available_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        for provider in self.providers:
            models.extend(provider.available_models())
        return models

    def run(
        self,
        role: str,
        model: ModelInfo,
        prompt: str,
        execution_policy: ExecutionPolicy | None = None,
        activity: ActivityCallback | None = None,
    ) -> ProviderResult:
        for provider in self.providers:
            if any(candidate.id == model.id for candidate in provider.available_models()):
                if activity is None:
                    return provider.run(role, model, prompt, execution_policy)
                try:
                    return provider.run(role, model, prompt, execution_policy, activity)
                except TypeError:
                    return provider.run(role, model, prompt, execution_policy)
        raise ValueError(f"No provider can run selected model {model.id}")


def _codex_permission_args(policy: ExecutionPolicy) -> list[str]:
    if policy.mode == "full-access":
        return ["--dangerously-bypass-approvals-and-sandbox"]
    args = ["--sandbox", policy.mode]
    for directory in policy.writable_dirs:
        args.extend(["--add-dir", str(directory.resolve())])
    return args


def _cursor_permission_args(policy: ExecutionPolicy) -> list[str]:
    if policy.mode == "full-access":
        return ["--force", "--sandbox", "disabled"]
    if policy.mode == "read-only":
        return ["--mode", "plan"]
    return []


def _run_provider_command(
    provider_command: ProviderCommand,
    *,
    provider_name: str = "",
    role: str | None = None,
    model: ModelInfo | None = None,
    activity: ActivityCallback | None = None,
    start: float | None = None,
) -> subprocess.CompletedProcess[str]:
    if activity is not None and role is not None and model is not None:
        return _run_provider_command_streaming(
            provider_command,
            provider_name=provider_name,
            role=role,
            model=model,
            activity=activity,
            start=start or time.monotonic(),
        )
    # Provider command is explicit user config, never a shell.
    return subprocess.run(  # noqa: S603
        provider_command.args,
        cwd=provider_command.cwd,
        text=True,
        input=provider_command.stdin,
        capture_output=True,
        check=False,
        timeout=PROVIDER_RUN_TIMEOUT_SEC,
    )


def _run_provider_command_streaming(
    provider_command: ProviderCommand,
    *,
    provider_name: str,
    role: str,
    model: ModelInfo,
    activity: ActivityCallback,
    start: float,
) -> subprocess.CompletedProcess[str]:
    args = _streaming_args(provider_name, provider_command.args)
    proc = subprocess.Popen(  # noqa: S603
        args,
        cwd=provider_command.cwd,
        text=True,
        stdin=subprocess.PIPE if provider_command.stdin is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdin_thread = _feed_provider_stdin(proc, provider_command.stdin)
    stdout, stderr = _collect_streaming_output(
        proc,
        args=args,
        provider_name=provider_name,
        role=role,
        model=model,
        activity=activity,
        start=start,
        workspace=provider_command.cwd,
    )
    if stdin_thread is not None:
        stdin_thread.join(timeout=1)
    return subprocess.CompletedProcess(
        args,
        proc.returncode if proc.returncode is not None else 0,
        stdout=stdout,
        stderr=stderr,
    )


def _feed_provider_stdin(proc: subprocess.Popen[str], stdin: str | None) -> threading.Thread | None:
    if stdin is None:
        return None
    if proc.stdin is None:
        raise subprocess.SubprocessError("provider stdin pipe was not available")

    def feed() -> None:
        try:
            proc.stdin.write(stdin)
            proc.stdin.close()
        except OSError:
            return

    thread = threading.Thread(target=feed, daemon=True)
    thread.start()
    return thread


def _collect_streaming_output(
    proc: subprocess.Popen[str],
    *,
    args: list[str],
    provider_name: str,
    role: str,
    model: ModelInfo,
    activity: ActivityCallback,
    start: float,
    workspace: Path,
) -> tuple[str, str]:
    timeout_at = start + PROVIDER_RUN_TIMEOUT_SEC
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    if proc.stdout is None or proc.stderr is None:
        raise subprocess.SubprocessError("provider stdout/stderr pipes were not available")
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
    try:
        while selector.get_map():
            if time.monotonic() > timeout_at:
                proc.kill()
                proc.communicate()
                raise subprocess.TimeoutExpired(args, PROVIDER_RUN_TIMEOUT_SEC)
            for key, _mask in selector.select(timeout=0.2):
                _handle_stream_line(
                    key,
                    selector=selector,
                    stdout_lines=stdout_lines,
                    stderr_lines=stderr_lines,
                    provider_name=provider_name,
                    role=role,
                    model=model,
                    activity=activity,
                    start=start,
                    workspace=workspace,
                )
        proc.wait(timeout=0)
    finally:
        selector.close()
    return "".join(stdout_lines), "".join(stderr_lines)


def _handle_stream_line(
    key: selectors.SelectorKey,
    *,
    selector: selectors.BaseSelector,
    stdout_lines: list[str],
    stderr_lines: list[str],
    provider_name: str,
    role: str,
    model: ModelInfo,
    activity: ActivityCallback,
    start: float,
    workspace: Path,
) -> None:
    line = key.fileobj.readline()
    if line == "":
        selector.unregister(key.fileobj)
        return
    if key.data == "stderr":
        stderr_lines.append(line)
        return
    stdout_lines.append(line)
    parsed = _activity_from_stream_line(
        provider_name,
        line,
        role=role,
        model=model,
        elapsed=time.monotonic() - start,
        workspace=workspace,
    )
    if parsed is not None:
        activity(parsed)


def _streaming_args(provider_name: str, args: list[str]) -> list[str]:
    if provider_name == "codex" and "--json" not in args:
        return [*args[:-1], "--json", args[-1]] if args else args
    if provider_name == "cursor":
        return _cursor_streaming_args(args)
    if provider_name == "claude":
        return _claude_streaming_args(args)
    return args


def _cursor_streaming_args(args: list[str]) -> list[str]:
    updated = list(args)
    if "--output-format" in updated:
        index = updated.index("--output-format")
        if index + 1 < len(updated):
            updated[index + 1] = "stream-json"
    else:
        updated[1:1] = ["--output-format", "stream-json"]
    if "--stream-partial-output" not in updated:
        prompt = updated.pop() if updated else ""
        updated.extend(["--stream-partial-output", prompt])
    return updated


def _claude_streaming_args(args: list[str]) -> list[str]:
    updated = list(args)
    if "--print" not in updated and "-p" not in updated:
        updated.insert(1, "--print")
    if "--output-format" not in updated:
        updated[1:1] = ["--output-format", "stream-json"]
    if "stream-json" in updated and "--verbose" not in updated:
        updated.insert(1, "--verbose")
    for flag in ("--include-partial-messages", "--include-hook-events"):
        if flag not in updated:
            updated.insert(1, flag)
    return updated


def _activity_from_stream_line(
    provider_name: str,
    line: str,
    *,
    role: str,
    model: ModelInfo,
    elapsed: float,
    workspace: Path,
) -> ProviderActivity | None:
    text = line.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ProviderActivity(role, model, "message", _compact_text(text), elapsed_seconds=elapsed)
    return _activity_from_json_payload(
        provider_name,
        payload,
        role=role,
        model=model,
        elapsed=elapsed,
        workspace=workspace,
    )


def _activity_from_json_payload(
    provider_name: str,
    payload: object,
    *,
    role: str,
    model: ModelInfo,
    elapsed: float,
    workspace: Path,
) -> ProviderActivity | None:
    if not isinstance(payload, dict):
        return None
    flat = json.dumps(payload, sort_keys=True)
    tool_name = _first_string(payload, ("tool_name", "tool", "name"))
    command = _first_string(payload, ("command", "cmd"))
    path = _extract_workspace_path(payload, workspace)
    message = _first_string(payload, ("message", "text", "delta", "content", "summary"))
    kind = str(payload.get("type") or payload.get("event") or payload.get("kind") or "activity")
    if path:
        action = _file_action_from_payload(flat)
        message = f"{action} {path}"
    elif command:
        message = f"running {command}"
    elif tool_name:
        message = f"using {tool_name}"
    elif message:
        message = _compact_text(message)
    else:
        provider_label = provider_name or model.provider
        message = f"{provider_label} reported {kind}"
    return ProviderActivity(
        role,
        model,
        kind,
        message,
        tool_name=tool_name,
        workspace_path=path,
        command=command,
        elapsed_seconds=elapsed,
    )


def _first_string(payload: dict[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in payload.values():
        if isinstance(value, dict):
            found = _first_string(value, keys)
            if found:
                return found
    return None


def _extract_workspace_path(payload: dict[str, object], workspace: Path) -> str | None:
    raw = _first_string(payload, ("path", "file", "file_path", "filepath", "uri"))
    if not raw:
        return None
    raw = raw.removeprefix("file://")
    candidate = Path(raw)
    try:
        resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
        return resolved.relative_to(workspace.resolve()).as_posix()
    except (OSError, ValueError):
        return raw


def _file_action_from_payload(flat_payload: str) -> str:
    lower = flat_payload.lower()
    if any(token in lower for token in ("edit", "write", "patch", "update", "create")):
        return "editing"
    if any(token in lower for token in ("read", "open", "view")):
        return "reading"
    return "touching"


def _compact_text(text: str, *, max_len: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "..."


def _provider_output_and_error(completed: subprocess.CompletedProcess[str]) -> tuple[str, str | None]:
    output = completed.stdout.strip()
    stderr = completed.stderr.strip() or None
    error = _process_error(completed.returncode, stderr)
    if not output and stderr:
        output = f"Provider returned no stdout. stderr:\n{stderr}"
    elif not output and error:
        output = f"Provider returned no stdout. {error}"
    return output, error


def _should_retry_cursor_without_sandbox(
    provider_name: str,
    args: list[str],
    completed: subprocess.CompletedProcess[str],
) -> bool:
    return (
        provider_name == "cursor"
        and "--sandbox" in args
        and completed.returncode != 0
        and CURSOR_SANDBOX_UNAVAILABLE in completed.stderr
    )


def _without_cursor_sandbox_args(args: list[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "--sandbox":
            index += 2
            continue
        stripped.append(args[index])
        index += 1
    return stripped


def _process_error(returncode: int, stderr: str | None) -> str | None:
    if returncode == 0:
        return None
    if stderr:
        return f"Provider exited with code {returncode}: {stderr}"
    return f"Provider exited with code {returncode}"


def _dry_planner_output(_prompt: str) -> str:
    return (
        "# Draft Plan\n\n"
        "1. Clarify the goal and acceptance criteria.\n"
        "2. Inspect the repository and identify the smallest implementation boundary.\n"
        "3. Implement the feature behind clear interfaces.\n"
        "4. Add focused tests and run verification.\n"
        "5. Produce a concise report with risks and follow-up options.\n"
    )


def _dry_plan_arbiter_output(_prompt: str) -> str:
    return (
        "# Arbiter Decision\n\n"
        "Incorporate reviewer suggestions about fallback handling, artifact formats, and tests. "
        "Do not add provider-specific secrets or hard-coded credentials.\n"
    )


def _dry_build_arbiter_output(_prompt: str) -> str:
    payload = json.dumps(
        [
            {
                "id": "rec-1",
                "title": "Harden provider fallbacks",
                "description": "Add explicit fallback handling when provider CLIs are unavailable.",
                "status": "applicable",
                "depends_on": [],
            },
            {
                "id": "rec-2",
                "title": "Expand workflow tests",
                "description": "Add coverage for reporting, selection parsing, and interactive defaults.",
                "status": "applicable",
                "depends_on": ["rec-1"],
            },
        ],
        indent=2,
    )
    return (
        "# Build Arbiter Decision\n\n"
        f"<!-- llm-plan-execute:recommendations\n{payload}\n-->\n\n"
        "- Highest priority: keep artifacts deterministic and machine-readable.\n"
        "- Recommendation: fix findings unless risk is explicitly accepted.\n"
    )


def _dry_builder_output(_prompt: str) -> str:
    return (
        "# Build Result\n\n"
        "The implementation was completed according to the accepted plan. "
        "Run artifacts and report outputs were generated for review.\n"
    )


def _dry_reviewer_output(role: str, _prompt: str) -> str:
    return (
        f"# {role} Review\n\n"
        "- Add explicit fallback handling for unavailable models.\n"
        "- Ensure run artifacts are written in machine-readable and human-readable formats.\n"
        "- Include tests for selection, reporting, and the dry-run provider.\n"
    )


def dry_response(role: str, prompt: str) -> str:
    if role == "clarifier":
        text = _dry_clarifier_response(prompt)
    elif role == "planner":
        text = _dry_planner_output(prompt)
    elif "reviewer" in role:
        text = _dry_reviewer_output(role, prompt)
    elif role == "plan_arbiter":
        text = _dry_plan_arbiter_output(prompt)
    elif role == "build_arbiter":
        text = _dry_build_arbiter_output(prompt)
    elif role.endswith("arbiter"):
        text = "# Arbiter Decision\n\nConsolidate reviewer guidance and keep changes narrowly scoped.\n"
    elif role == "builder":
        text = _dry_builder_output(prompt)
    else:
        text = f"# {role}\n\nProcessed prompt of {len(prompt)} characters."
    return text


def _dry_clarifier_response(prompt: str) -> str:
    if "ambiguous" in prompt.lower():
        return (
            "STATUS: needs_questions\n"
            "QUESTIONS:\n"
            "- What behavior should the change implement?\n"
            "ASSUMPTIONS:\n"
            "- The implementation should stay scoped to the current repository.\n"
        )
    return (
        "STATUS: clear\n"
        "QUESTIONS:\n"
        "- none\n"
        "ASSUMPTIONS:\n"
        "- Use the existing project patterns and verification commands.\n"
    )
