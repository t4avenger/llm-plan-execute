from pathlib import Path

from llm_plan_execute.cli import _state_from_json, main

CLARIFICATION_NEEDED_EXIT = 2


def _write_dry_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.json"
    runs_dir = tmp_path / "runs"
    config.write_text(
        f"""
{{
  "dry_run": true,
  "runs_dir": "{runs_dir}",
  "providers": []
}}
""",
        encoding="utf-8",
    )
    return config


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


def test_plan_without_clarify_prints_accept_command(tmp_path, capsys):
    config = _write_dry_config(tmp_path)

    exit_code = main(["--config", str(config), "plan", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Proposed plan:" in captured.out
    assert "llm-plan-execute accept --run-dir" in captured.out


def test_accept_command_promotes_plan(tmp_path, capsys):
    config = _write_dry_config(tmp_path)
    main(["--config", str(config), "plan", "--prompt", "Add a small feature", "--no-clarify"])
    captured = capsys.readouterr()
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()

    exit_code = main(["--config", str(config), "accept", "--run-dir", str(run_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Accepted plan:" in captured.out
    assert (run_dir / "04-accepted-plan.md").exists()


def test_noninteractive_clarification_exits_before_plan(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    exit_code = main(["--config", str(config), "plan", "--prompt", "Do an ambiguous thing"])

    captured = capsys.readouterr()
    assert exit_code == CLARIFICATION_NEEDED_EXIT
    assert "Clarification needed:" in captured.out
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()
    assert (run_dir / "00-clarification.md").exists()
    assert not (run_dir / "01-draft-plan.md").exists()


def test_build_unaccepted_run_reports_accept_guidance(tmp_path, capsys):
    config = _write_dry_config(tmp_path)
    main(["--config", str(config), "plan", "--prompt", "Add a small feature", "--no-clarify"])
    captured = capsys.readouterr()
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()

    exit_code = main(["--config", str(config), "build", "--run-dir", str(run_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "accept command" in captured.err
