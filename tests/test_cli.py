import json
from pathlib import Path

from llm_plan_execute.cli import _state_from_json, main
from llm_plan_execute.config import sample_config

CLARIFICATION_NEEDED_EXIT = 2
CANCELED_EXIT = 130


def _write_dry_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.json"
    runs_dir = tmp_path / "runs"
    config.write_text(
        json.dumps(
            {
                "dry_run": True,
                "runs_dir": str(runs_dir),
                "providers": [],
            },
            indent=2,
        )
        + "\n",
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


def test_missing_config_requires_init_config_or_dry_run(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["models"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Config file .llm-plan-execute/config.json was not found" in captured.err
    assert "use --dry-run explicitly" in captured.err


def test_config_validate_reports_missing_default_config(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["config", "validate"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "config file was not found" in captured.err


def test_dry_run_without_config_still_works(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--dry-run", "models"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "dry-codex:frontier-planner" in captured.out


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


def test_init_config_enables_installed_commands(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "llm_plan_execute.config.shutil.which",
        lambda command: f"/bin/{command}" if command in {"codex", "cursor-agent"} else None,
    )
    config_path = tmp_path / "config.json"

    exit_code = main(["init-config", "--path", str(config_path)])

    captured = capsys.readouterr()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    enabled = {provider["name"]: provider["enabled"] for provider in raw["providers"]}
    assert exit_code == 0
    assert f"Wrote {config_path}" in captured.out
    assert enabled == {"codex": True, "claude": False, "cursor": True}


def test_cursor_default_model_is_builder_only(monkeypatch):
    monkeypatch.setattr("llm_plan_execute.config.shutil.which", lambda _command: "/bin/provider")

    raw = sample_config()
    cursor = next(provider for provider in raw["providers"] if provider["name"] == "cursor")

    assert cursor["models"][0]["roles"] == ["builder"]


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


def test_plan_prints_clear_clarification_status(tmp_path, capsys):
    config = _write_dry_config(tmp_path)

    exit_code = main(["--config", str(config), "plan", "--prompt", "Add a small feature"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Clarification: no questions required" in captured.out
    assert "Proposed plan:" in captured.out


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


def test_interactive_clarification_marks_answered_questions_clear(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "Implement the requested behavior.")

    exit_code = main(["--config", str(config), "plan", "--prompt", "Do an ambiguous thing"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Clarification: answered 1 question(s)." in captured.out
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()
    clarification = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["clarification"]
    assert clarification["status"] == "clear"
    assert clarification["answers"] == ["Implement the requested behavior."]
    assert "- Status: clear" in (run_dir / "00-clarification.md").read_text(encoding="utf-8")


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


def test_run_save_only_stops_before_build(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "save-only")

    exit_code = main(["--config", str(config), "run", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Saved proposed plan." in captured.out
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()
    assert (run_dir / "04-proposed-plan.md").exists()
    assert not (run_dir / "05-build-output.md").exists()


def test_run_approve_builds_after_inline_plan_review(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "approve")

    exit_code = main(["--config", str(config), "run", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "# Arbiter Decision" in captured.out
    assert "Build output:" in captured.out
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()
    assert (run_dir / "08-build-review-summary.md").exists()


def test_run_cancel_stops_before_build(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "cancel")

    exit_code = main(["--config", str(config), "run", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == CANCELED_EXIT
    assert "Canceled before build." in captured.out


def test_quiet_suppresses_progress(tmp_path, capsys):
    config = _write_dry_config(tmp_path)

    exit_code = main(["--config", str(config), "--quiet", "plan", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Proposed plan:" in captured.out
    assert captured.err == ""


def test_progress_uses_stderr_and_leaves_stdout_clean(tmp_path, capsys):
    config = _write_dry_config(tmp_path)

    exit_code = main(["--config", str(config), "plan", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Proposed plan:" in captured.out
    assert "planner: starting" in captured.err
    assert "planner: starting" not in captured.out
