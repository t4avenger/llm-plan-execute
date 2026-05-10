from pathlib import Path

from llm_plan_execute.cli import _state_from_json, main


def test_state_from_json_ignores_malformed_string_lists(tmp_path):
    run = _state_from_json(
        {
            "run_id": "run-1",
            "prompt": "prompt",
            "warnings": "warning",
            "next_options": None,
            "assignments": {},
            "results": [],
        },
        tmp_path,
    )

    assert run.warnings == []
    assert run.next_options == []


def test_main_reports_user_facing_errors(capsys):
    exit_code = main(["build", "--run-dir", str(Path("missing-run"))])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error:" in captured.err
