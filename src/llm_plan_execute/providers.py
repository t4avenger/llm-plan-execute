from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass

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


@dataclass
class CLIProvider(Provider):
    config: ProviderConfig

    def available_models(self) -> list[ModelInfo]:
        if not self.config.enabled or shutil.which(self.config.command) is None:
            return []
        return list(self.config.models)

    def run(self, role: str, model: ModelInfo, prompt: str) -> ProviderResult:
        start = time.monotonic()
        usage = Usage(input_tokens=estimate_tokens(prompt), exact=False, confidence="estimated")
        try:
            completed = subprocess.run(  # noqa: S603 - provider command is explicit user config, never a shell.
                [self.config.command, "--model", model.name],
                input=prompt,
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
        providers.extend(CLIProvider(provider) for provider in config.providers)
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
