from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig, ProviderConfig
from .types import ModelInfo, ProviderResult, Usage


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


def estimate_cost(model: ModelInfo, usage: Usage) -> float:
    # Conservative placeholder: lower cost score means cheaper model.
    dollars_per_million = {1: 1.0, 2: 3.0, 3: 8.0, 4: 15.0, 5: 30.0}.get(model.cost, 8.0)
    return round((usage.total_tokens / 1_000_000) * dollars_per_million, 6)


class Provider:
    def available_models(self) -> list[ModelInfo]:
        raise NotImplementedError

    def run(self, role: str, model: ModelInfo, prompt: str) -> ProviderResult:
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

    def run(self, role: str, model: ModelInfo, prompt: str) -> ProviderResult:
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
    def build_command(self, config: ProviderConfig, model: ModelInfo, prompt: str, workspace: Path) -> ProviderCommand:
        raise NotImplementedError


class CodexAdapter(ProviderAdapter):
    def build_command(self, config: ProviderConfig, model: ModelInfo, prompt: str, workspace: Path) -> ProviderCommand:
        resolved_workspace = workspace.resolve()
        return ProviderCommand(
            [
                config.command,
                "exec",
                "--model",
                model.name,
                "--cd",
                str(resolved_workspace),
                prompt,
            ],
            resolved_workspace,
        )


class CursorAdapter(ProviderAdapter):
    def build_command(self, config: ProviderConfig, model: ModelInfo, prompt: str, workspace: Path) -> ProviderCommand:
        resolved_workspace = workspace.resolve()
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
                prompt,
            ],
            resolved_workspace,
        )


class ClaudeAdapter(ProviderAdapter):
    def build_command(self, config: ProviderConfig, model: ModelInfo, prompt: str, workspace: Path) -> ProviderCommand:
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

    def run(self, role: str, model: ModelInfo, prompt: str) -> ProviderResult:
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

        provider_command = adapter.build_command(self.config, model, prompt, self.workspace)
        try:
            completed = subprocess.run(  # noqa: S603 - provider command is explicit user config, never a shell.
                provider_command.args,
                cwd=provider_command.cwd,
                text=True,
                capture_output=True,
                check=False,
                timeout=1800,
            )
            output = completed.stdout.strip()
            stderr = completed.stderr.strip() or None
            error = _process_error(completed.returncode, stderr)
            if not output and stderr:
                output = f"Provider returned no stdout. stderr:\n{stderr}"
            elif not output and error:
                output = f"Provider returned no stdout. {error}"
            usage.output_tokens = estimate_tokens(output)
            usage.cost_usd = estimate_cost(model, usage)
            return ProviderResult(role, model, prompt, output, usage, time.monotonic() - start, error)
        except (OSError, subprocess.SubprocessError) as exc:
            output = f"Provider execution failed for {model.id}: {exc}"
            usage.output_tokens = estimate_tokens(output)
            usage.cost_usd = estimate_cost(model, usage)
            return ProviderResult(role, model, prompt, output, usage, time.monotonic() - start, str(exc))


class ProviderRouter:
    def __init__(self, providers: list[Provider]) -> None:
        self.providers = providers

    @classmethod
    def from_config(cls, config: AppConfig) -> ProviderRouter:
        if config.dry_run:
            return cls([DryRunProvider()])
        providers: list[Provider] = []
        providers.extend(CLIProvider(provider, workspace=config.workspace) for provider in config.providers)
        return cls(providers)

    def available_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        for provider in self.providers:
            models.extend(provider.available_models())
        return models

    def run(self, role: str, model: ModelInfo, prompt: str) -> ProviderResult:
        for provider in self.providers:
            if any(candidate.id == model.id for candidate in provider.available_models()):
                return provider.run(role, model, prompt)
        raise ValueError(f"No provider can run selected model {model.id}")


def _process_error(returncode: int, stderr: str | None) -> str | None:
    if returncode == 0:
        return None
    if stderr:
        return f"Provider exited with code {returncode}: {stderr}"
    return f"Provider exited with code {returncode}"


def dry_response(role: str, prompt: str) -> str:
    if role == "clarifier":
        return _dry_clarifier_response(prompt)
    if role == "planner":
        return (
            "# Draft Plan\n\n"
            "1. Clarify the goal and acceptance criteria.\n"
            "2. Inspect the repository and identify the smallest implementation boundary.\n"
            "3. Implement the feature behind clear interfaces.\n"
            "4. Add focused tests and run verification.\n"
            "5. Produce a concise report with risks and follow-up options.\n"
        )
    if "reviewer" in role:
        return (
            f"# {role} Review\n\n"
            "- Add explicit fallback handling for unavailable models.\n"
            "- Ensure run artifacts are written in machine-readable and human-readable formats.\n"
            "- Include tests for selection, reporting, and the dry-run provider.\n"
        )
    if role.endswith("arbiter"):
        return (
            "# Arbiter Decision\n\n"
            "Incorporate reviewer suggestions about fallback handling, artifact formats, and tests. "
            "Do not add provider-specific secrets or hard-coded credentials.\n"
        )
    if role == "builder":
        return (
            "# Build Result\n\n"
            "The implementation was completed according to the accepted plan. "
            "Run artifacts and report outputs were generated for review.\n"
        )
    return f"# {role}\n\nProcessed prompt of {len(prompt)} characters."


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
