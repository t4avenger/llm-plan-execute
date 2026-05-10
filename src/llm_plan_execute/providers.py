from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig, ProviderConfig
from .types import ExecutionPolicy, ModelInfo, ProviderResult, Usage


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
                prompt,
            ],
            resolved_workspace,
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
        return ProviderCommand([config.command, "--model", model.name, prompt], resolved_workspace)


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
            error = f"Provider command {self.config.command!r} is not available on PATH."
            usage.output_tokens = estimate_tokens(error)
            usage.cost_usd = estimate_cost(model, usage)
            return ProviderResult(role, model, prompt, error, usage, time.monotonic() - start, error)

        policy = execution_policy or ExecutionPolicy()
        provider_command = adapter.build_command(self.config, model, prompt, self.workspace, policy)
        try:
            completed, warning = self._run_command_with_cursor_sandbox_fallback(provider_command)
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
    ) -> tuple[subprocess.CompletedProcess[str], str | None]:
        completed = _run_provider_command(provider_command)
        if not _should_retry_cursor_without_sandbox(self.config.name, provider_command.args, completed):
            return completed, None

        retry_command = ProviderCommand(_without_cursor_sandbox_args(provider_command.args), provider_command.cwd)
        retry = _run_provider_command(retry_command)
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
    ) -> ProviderResult:
        for provider in self.providers:
            if any(candidate.id == model.id for candidate in provider.available_models()):
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


def _run_provider_command(provider_command: ProviderCommand) -> subprocess.CompletedProcess[str]:
    # Provider command is explicit user config, never a shell.
    return subprocess.run(  # noqa: S603
        provider_command.args,
        cwd=provider_command.cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=1800,
    )


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
