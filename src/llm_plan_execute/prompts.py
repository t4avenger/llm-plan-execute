from __future__ import annotations


def clarification_prompt(user_prompt: str) -> str:
    return f"""Decide whether this implementation request has enough information to plan safely.

Request:
{user_prompt}

Return exactly this plain-text format:
STATUS: clear or needs_questions
QUESTIONS:
- question text, or none
ASSUMPTIONS:
- assumption text, or none

Ask questions only for decisions that materially change scope, interfaces, data flow, tests, or acceptance criteria."""


def planner_prompt(user_prompt: str) -> str:
    return f"""Create a decision-complete implementation plan for this request.

Request:
{user_prompt}

Return concise Markdown with summary, implementation changes, tests, assumptions, and any open questions."""


def clarified_planner_prompt(user_prompt: str, questions: list[str], answers: list[str]) -> str:
    answer_lines = "\n".join(
        f"- Q: {question}\n  A: {answer}" for question, answer in zip(questions, answers, strict=False)
    )
    return planner_prompt(f"{user_prompt}\n\nClarifications:\n{answer_lines}")


def plan_review_prompt(plan: str, reviewer_name: str) -> str:
    return f"""Review this implementation plan as {reviewer_name}.

Focus on missing decisions, risky assumptions, unclear interfaces, and tests. Return actionable suggestions only.

Plan:
{plan}"""


def plan_arbiter_prompt(plan: str, review_a: str, review_b: str) -> str:
    return f"""Decide which review suggestions should be incorporated into the plan.

Original plan:
{plan}

Review A:
{review_a}

Review B:
{review_b}

Return a concise decision list and then the revised accepted plan."""


def build_prompt(accepted_plan: str) -> str:
    return f"""Implement the following accepted plan in the current repository.

Follow the plan closely. Keep changes scoped. Run appropriate verification and summarize the result.

Accepted plan:
{accepted_plan}"""


def build_review_prompt(accepted_plan: str, build_output: str, reviewer_name: str) -> str:
    return f"""Review the completed build as {reviewer_name}.

Find bugs, missed acceptance criteria, risky behavior, and missing tests.

Accepted plan:
{accepted_plan}

Build output:
{build_output}"""


def build_arbiter_prompt(review_a: str, review_b: str) -> str:
    return f"""Consolidate these build review findings.

Return:
- highest-priority findings
- whether the user should fix findings, accept as-is, or return to planning
- concise rationale

Review A:
{review_a}

Review B:
{review_b}"""
