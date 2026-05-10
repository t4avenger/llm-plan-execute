import pytest

from llm_plan_execute.interactive import InteractiveCanceledError, session_with_mock_stdin


def test_prompt_confirm_empty_input_respects_default_no():
    session, _stdout, _stderr = session_with_mock_stdin([""])
    assert session.prompt_confirm("Proceed?", default_yes=False) is False


def test_prompt_confirm_empty_input_respects_default_yes():
    session, _stdout, _stderr = session_with_mock_stdin([""])
    assert session.prompt_confirm("Proceed?", default_yes=True) is True


def test_prompt_choice_retries_until_valid_choice():
    session, _stdout, stderr = session_with_mock_stdin(["", "x", "2"])

    decision = session.ask_stage_transition()

    assert decision.type == "pause"
    assert any("Please enter a choice." in line for line in stderr.lines)
    assert any("Invalid choice" in line for line in stderr.lines)


def test_prompt_choice_raises_after_retries():
    session, _stdout, _stderr = session_with_mock_stdin(["bad", "still-bad"], max_retries=2)

    with pytest.raises(InteractiveCanceledError, match="Exceeded maximum retries"):
        session.ask_plan_review()


def test_prompt_free_text_required_retries_then_returns_text():
    session, _stdout, stderr = session_with_mock_stdin(["", "Use a clearer plan."])

    assert session.read_plan_feedback() == "Use a clearer plan."
    assert any("Input cannot be empty." in line for line in stderr.lines)


def test_prompt_free_text_non_interactive_skips():
    session, _stdout, stderr = session_with_mock_stdin([], non_interactive=True)

    assert session.read_build_feedback() == ""
    assert any("skipping free-text" in line for line in stderr.lines)


def test_step_through_sections_next_stop_cancel_and_non_interactive():
    session, stdout, _stderr = session_with_mock_stdin(["n", "s"])
    sections = [("One", "Body one"), ("Two", "Body two")]

    assert session.step_through_sections("Review", sections).type == "stop"
    assert any("## One" in line for line in stdout.lines)
    assert any("## Two" in line for line in stdout.lines)

    cancel_session, _stdout, _stderr = session_with_mock_stdin(["c"])
    assert cancel_session.step_through_sections("Review", sections).type == "cancel"

    non_interactive, _stdout, stderr = session_with_mock_stdin([], non_interactive=True)
    assert non_interactive.step_through_sections("Review", sections).type == "stop"
    assert any("skipping step-through" in line for line in stderr.lines)


def test_menu_helpers_return_typed_decisions_and_non_interactive_defaults():
    session, _stdout, _stderr = session_with_mock_stdin(["3", "2", "5", "3", "2, 1"])

    assert session.ask_plan_review().type == "stepThrough"
    assert session.ask_stage_transition().type == "pause"
    assert session.ask_build_review().type == "cancel"
    assert session.ask_completion_report().type == "both"
    assert session.read_recommendation_selection(3) == ("2", "1")

    non_interactive, _stdout, _stderr = session_with_mock_stdin([], non_interactive=True)
    assert non_interactive.ask_build_review().type == "continueWithoutApplying"
    assert non_interactive.ask_completion_report().type == "skip"
    assert non_interactive.read_recommendation_selection(3) == ()
