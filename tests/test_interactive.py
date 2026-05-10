from llm_plan_execute.interactive import session_with_mock_stdin


def test_prompt_confirm_empty_input_respects_default_no():
    session, _stdout, _stderr = session_with_mock_stdin([""])
    assert session.prompt_confirm("Proceed?", default_yes=False) is False


def test_prompt_confirm_empty_input_respects_default_yes():
    session, _stdout, _stderr = session_with_mock_stdin([""])
    assert session.prompt_confirm("Proceed?", default_yes=True) is True
