from llm_plan_execute.prompts import (
    build_review_feedback_suffix,
    clarified_planner_prompt,
    plan_revision_prompt,
)


def test_clarified_planner_prompt_includes_question_answers():
    prompt = clarified_planner_prompt("Build it", ["Scope?"], ["Smallest useful change."])

    assert "Clarifications:" in prompt
    assert "Q: Scope?" in prompt
    assert "A: Smallest useful change." in prompt


def test_plan_revision_prompt_handles_empty_and_populated_feedback():
    empty = plan_revision_prompt("Build it", "Old plan", [])
    populated = plan_revision_prompt("Build it", "Old plan", ["Add tests", "Clarify rollback"])

    assert "- none" in empty
    assert "- Add tests" in populated
    assert "- Clarify rollback" in populated


def test_build_review_feedback_suffix_is_empty_or_cumulative():
    assert build_review_feedback_suffix([]) == ""

    suffix = build_review_feedback_suffix(["Check edge cases", "Retest CLI"])

    assert "Operator feedback" in suffix
    assert "- Check edge cases" in suffix
    assert "- Retest CLI" in suffix
