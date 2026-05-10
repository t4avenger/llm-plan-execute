from __future__ import annotations

from .types import RunState


def render_report(run: RunState) -> str:
    total_tokens = sum(result.usage.total_tokens for result in run.results)
    exact_tokens = sum(result.usage.total_tokens for result in run.results if result.usage.exact)
    estimated_tokens = total_tokens - exact_tokens
    total_cost = sum(result.usage.cost_usd or 0 for result in run.results)

    lines = _summary_lines(run, total_tokens, exact_tokens, estimated_tokens, total_cost)
    for role, assignment in run.assignments.items():
        reused = " reused" if assignment.reused else ""
        lines.append(f"- `{role}`: `{assignment.model.id}` ({assignment.reason}{reused})")

    _append_warnings(lines, run)
    _append_clarification(lines, run)
    _append_usage(lines, run)
    _append_prompt_advice(lines)
    _append_next_options(lines, run)

    return "\n".join(lines) + "\n"


def _summary_lines(
    run: RunState,
    total_tokens: int,
    exact_tokens: int,
    estimated_tokens: int,
    total_cost: float,
) -> list[str]:
    lines = [
        "# LLM Plan Execute Report",
        "",
        f"- Run: `{run.run_id}`",
        f"- Results: {len(run.results)} model calls",
        f"- Tokens: {total_tokens} total ({exact_tokens} exact, {estimated_tokens} estimated)",
        f"- Estimated cost: ${total_cost:.6f}",
    ]
    if run.build_status:
        lines.append(f"- Build status: {run.build_status}")
    if run.build_failure:
        lines.append(f"- Build failure: {run.build_failure}")
    lines.extend(["", "## Model Assignments"])
    return lines


def _append_warnings(lines: list[str], run: RunState) -> None:
    if run.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in run.warnings)


def _append_clarification(lines: list[str], run: RunState) -> None:
    if not run.clarification:
        return
    lines.extend(["", "## Clarification", f"- Status: {run.clarification.status}"])
    if run.clarification.questions:
        lines.append(f"- Questions: {len(run.clarification.questions)}")
    if run.clarification.answers:
        lines.append(f"- Answers: {len(run.clarification.answers)}")


def _append_usage(lines: list[str], run: RunState) -> None:
    lines.extend(["", "## Usage"])
    for result in run.results:
        exact = "exact" if result.usage.exact else result.usage.confidence
        lines.append(
            f"- `{result.role}` via `{result.model.id}`: "
            f"{result.usage.total_tokens} tokens, ${result.usage.cost_usd or 0:.6f}, {exact}"
        )


def _append_prompt_advice(lines: list[str]) -> None:
    lines.extend(
        [
            "",
            "## Prompt Improvement Advice",
            "- State acceptance criteria explicitly before planning.",
            "- Name constraints such as provider preferences, budget, speed, and files that must not change.",
            "- Include expected verification commands when they matter.",
            "- Separate product intent from implementation hints so reviewers can challenge the right things.",
        ]
    )


def _append_next_options(lines: list[str], run: RunState) -> None:
    if run.next_options:
        lines.extend(["", "## Next Options"])
        lines.extend(f"- {option}" for option in run.next_options)
