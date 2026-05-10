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


def test_config_validate_reports_invalid_provider(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(
        """
{
  "providers": [
    {
      "name": "unknown",
      "command": "unknown",
      "models": [
        {
          "name": "model",
          "roles": ["planner"],
          "reasoning": 3,
          "speed": 3,
          "cost": 3,
          "context": 3
        }
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config), "config", "validate"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "unsupported provider" in captured.err


def test_config_validate_reports_missing_enabled_command(tmp_path, capsys, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(
        """
{
  "providers": [
    {
      "name": "codex",
      "command": "missing-codex",
      "enabled": true,
      "models": [
        {
          "name": "gpt-5.4",
          "roles": ["planner"],
          "reasoning": 3,
          "speed": 3,
          "cost": 3,
          "context": 3
        }
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("llm_plan_execute.cli.shutil.which", lambda _command: None)

    exit_code = main(["--config", str(config), "config", "validate"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "was not found on PATH" in captured.err


def test_config_validate_skips_missing_commands_in_dry_run(tmp_path, capsys, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(
        """
{
  "dry_run": true,
  "providers": [
    {
      "name": "codex",
      "command": "missing-codex",
      "enabled": true,
      "models": [
        {
          "name": "gpt-5.4",
          "roles": ["planner"],
          "reasoning": 3,
          "speed": 3,
          "cost": 3,
          "context": 3
        }
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("llm_plan_execute.cli.shutil.which", lambda _command: None)

    exit_code = main(["--config", str(config), "config", "validate"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "was not found on PATH" not in captured.err


def test_dry_run_allows_empty_provider_list(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(
        """
{
  "dry_run": true,
  "providers": []
}
""",
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config), "models"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "dry-codex:frontier-planner" in captured.out
