from __future__ import annotations

from .types import RunState


def render_report(run: RunState) -> str:
    total_tokens = sum(result.usage.total_tokens for result in run.results)
    exact_tokens = sum(result.usage.total_tokens for result in run.results if result.usage.exact)
    estimated_tokens = total_tokens - exact_tokens
    total_cost = sum(result.usage.cost_usd or 0 for result in run.results)

    lines = [
        "# LLM Plan Execute Report",
        "",
        f"- Run: `{run.run_id}`",
        f"- Results: {len(run.results)} model calls",
        f"- Tokens: {total_tokens} total ({exact_tokens} exact, {estimated_tokens} estimated)",
        f"- Estimated cost: ${total_cost:.6f}",
        "",
        "## Model Assignments",
    ]
    for role, assignment in run.assignments.items():
        reused = " reused" if assignment.reused else ""
        lines.append(f"- `{role}`: `{assignment.model.id}` ({assignment.reason}{reused})")

    if run.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in run.warnings)

    if run.clarification:
        lines.extend(["", "## Clarification", f"- Status: {run.clarification.status}"])
        if run.clarification.questions:
            lines.append(f"- Questions: {len(run.clarification.questions)}")
        if run.clarification.answers:
            lines.append(f"- Answers: {len(run.clarification.answers)}")

    lines.extend(["", "## Usage"])
    for result in run.results:
        exact = "exact" if result.usage.exact else result.usage.confidence
        lines.append(
            f"- `{result.role}` via `{result.model.id}`: "
            f"{result.usage.total_tokens} tokens, ${result.usage.cost_usd or 0:.6f}, {exact}"
        )

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

    if run.next_options:
        lines.extend(["", "## Next Options"])
        lines.extend(f"- {option}" for option in run.next_options)

    return "\n".join(lines) + "\n"
