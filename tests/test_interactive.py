import pytest

from llm_plan_execute.interactive import (
    ChoiceOption,
    InteractiveCanceledError,
    InteractiveSession,
    ListBuffer,
    session_with_mock_stdin,
)


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
    with pytest.raises(InteractiveCanceledError, match="non-interactive mode"):
        session.read_build_feedback()
    assert stderr.lines == []


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
    with pytest.raises(InteractiveCanceledError, match="non-interactive mode"):
        non_interactive.ask_build_review()
    assert non_interactive.ask_completion_report().type == "skip"
    with pytest.raises(InteractiveCanceledError, match="non-interactive mode"):
        non_interactive.read_recommendation_selection(3)


def test_slash_commands_and_escaped_input():
    session, _stdout, _stderr = session_with_mock_stdin(["/help", "/continue"])
    assert session.ask_stage_transition().type == "proceed"

    unknown, _stdout2, err2 = session_with_mock_stdin(["/wat", "1"])
    assert unknown.ask_plan_review().type == "accept"
    assert any("Unknown slash command" in line for line in err2.lines)

    escaped, _stdout3, err3 = session_with_mock_stdin(["//status", "1"])
    assert escaped.ask_plan_review().type == "accept"
    assert any("Escaped slash input treated as data" in line for line in err3.lines)


def test_ctrl_c_first_press_warns_second_cancels():
    """First Ctrl-C warns; second Ctrl-C raises InteractiveCanceledError."""

    class _CtrlCThenValidStdin:
        def __init__(self, responses):
            self._iter = iter(responses)

        def readline(self) -> str:
            value = next(self._iter)
            if value is KeyboardInterrupt:
                raise KeyboardInterrupt
            return value + "\n"

    # First Ctrl-C → warning printed, then valid input "1" → proceed
    stdin = _CtrlCThenValidStdin([KeyboardInterrupt, "1"])
    stdout = ListBuffer()
    stderr = ListBuffer()
    session = InteractiveSession(stdin=stdin, stdout=stdout, stderr=stderr)
    result = session.ask_stage_transition()
    assert result.type == "proceed"
    assert any("Ctrl+C again" in line for line in stderr.lines)

    # Two consecutive Ctrl-Cs → InteractiveCanceledError
    stdin2 = _CtrlCThenValidStdin([KeyboardInterrupt, KeyboardInterrupt])
    session2 = InteractiveSession(stdin=stdin2, stdout=ListBuffer(), stderr=ListBuffer())
    with pytest.raises(InteractiveCanceledError, match="Interrupted"):
        session2.ask_stage_transition()


def test_status_slash_command_continues_then_resolves():
    session, stdout, _stderr = session_with_mock_stdin(["/status", "1"])
    result = session.ask_stage_transition()
    assert result.type == "proceed"
    assert any("Session is idle" in line for line in stdout.lines)


def test_verbose_slash_command_on_and_off():
    calls: list[bool] = []
    session, stdout, _stderr = session_with_mock_stdin(["/verbose on", "1"])
    session._on_verbose_change = calls.append  # type: ignore[assignment]
    result = session.ask_stage_transition()
    assert result.type == "proceed"
    assert any("Verbose mode enabled" in line for line in stdout.lines)
    assert calls == [True]

    calls2: list[bool] = []
    session2, stdout2, _stderr2 = session_with_mock_stdin(["/verbose off", "1"])
    session2._on_verbose_change = calls2.append  # type: ignore[assignment]
    result2 = session2.ask_stage_transition()
    assert result2.type == "proceed"
    assert any("Verbose mode disabled" in line for line in stdout2.lines)
    assert calls2 == [False]


def test_verbose_slash_command_without_callback():
    session, stdout, _stderr = session_with_mock_stdin(["/verbose on", "1"])
    result = session.ask_stage_transition()
    assert result.type == "proceed"
    assert any("Verbose mode enabled" in line for line in stdout.lines)


def test_verbose_slash_typo_is_rejected():
    session, _stdout, stderr = session_with_mock_stdin(["/verbose onn", "1"])
    result = session.ask_stage_transition()
    assert result.type == "proceed"
    assert any("Usage: /verbose on" in line for line in stderr.lines)


def test_prompt_confirm_slash_cancel_raises():
    session, _stdout, _stderr = session_with_mock_stdin(["/cancel"])
    with pytest.raises(InteractiveCanceledError, match="Canceled from slash command"):
        session.prompt_confirm("Are you sure?")


def test_retry_slash_with_single_option_falls_back_to_first():
    session, _stdout, _stderr = session_with_mock_stdin(["/retry"])
    result = session.prompt_choice("Pick one?", [ChoiceOption("1", "Only option", "sole")], include_retry=True)
    assert result == "sole"


def test_prompt_confirm_valid_yes_no_tokens():
    for token, expected in (("y", True), ("yes", True), ("n", False), ("no", False)):
        session, _stdout, _stderr = session_with_mock_stdin([token])
        assert session.prompt_confirm("Ok?") is expected


def test_prompt_confirm_invalid_token_retries():
    session, _stdout, stderr = session_with_mock_stdin(["maybe", "y"])
    assert session.prompt_confirm("Ok?") is True
    assert any("Please answer y or n." in line for line in stderr.lines)


def test_cancel_slash_in_choice_menu_raises():
    session, _stdout, _stderr = session_with_mock_stdin(["/cancel"])
    with pytest.raises(InteractiveCanceledError, match="Workflow cancel"):
        session.ask_stage_transition()


def test_stdin_exhaustion_raises_eof_during_choice():
    session, _stdout, _stderr = session_with_mock_stdin(["bad"], max_retries=5)
    with pytest.raises(InteractiveCanceledError, match="EOF"):
        session.ask_stage_transition()


def test_non_interactive_prompt_choice_raises():
    session, _stdout, _stderr = session_with_mock_stdin([], non_interactive=True)
    with pytest.raises(InteractiveCanceledError, match="non-interactive mode"):
        session.ask_stage_transition()


def test_prompt_confirm_non_cancel_slash_continues_to_next_input():
    session, _stdout, _stderr = session_with_mock_stdin(["/help", "y"])
    assert session.prompt_confirm("Proceed?") is True


def test_step_through_sections_completes_all_sections():
    session, stdout, _stderr = session_with_mock_stdin(["n", "n"])
    sections = [("A", "Body A"), ("B", "Body B")]
    result = session.step_through_sections("Title", sections)
    assert result.type == "stop"
    assert any("## A" in line for line in stdout.lines)
    assert any("## B" in line for line in stdout.lines)


def test_step_through_sections_invalid_token_then_stop():
    session, _stdout, stderr = session_with_mock_stdin(["x", "s"])
    result = session.step_through_sections("Title", [("Sec", "Body")])
    assert result.type == "stop"
    assert any("Invalid section control" in line for line in stderr.lines)


def test_non_interactive_prompt_confirm_raises():
    session, _stdout, _stderr = session_with_mock_stdin([], non_interactive=True)
    with pytest.raises(InteractiveCanceledError, match="non-interactive mode"):
        session.prompt_confirm("Are you sure?")


def test_prompt_confirm_eof_raises():
    session, _stdout, _stderr = session_with_mock_stdin([])
    with pytest.raises(InteractiveCanceledError, match="EOF"):
        session.prompt_confirm("Are you sure?")
