import io
import json
from pathlib import Path

from llm_plan_execute.cli import _state_from_json, main
from llm_plan_execute.config import sample_config
from llm_plan_execute.interactive import session_with_mock_stdin

CLARIFICATION_NEEDED_EXIT = 2
CANCELED_EXIT = 130
PAUSED_EXIT = 3


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


def _config_args(tmp_path: Path, config: Path) -> list[str]:
    return ["--repo", str(tmp_path), "--config", str(config)]


def _scripted_tty_stdin(monkeypatch, *lines: str) -> None:
    """Patch sys.stdin for InteractiveSession (readline, isatty); lines need no trailing newline."""

    queue = [line if line.endswith("\n") else line + "\n" for line in lines]

    class _Stdin:
        def isatty(self) -> bool:
            return True

        def readline(self) -> str:
            return queue.pop(0) if queue else ""

    monkeypatch.setattr("sys.stdin", _Stdin())


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
    assert "was not found for workspace" in captured.err
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


def test_config_validate_reports_invalid_execution_mode(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(
        """
{
  "dry_run": true,
  "execution": {
    "default_mode": "root"
  },
  "providers": []
}
""",
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config), "config", "validate"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "execution.default_mode" in captured.err


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


def test_plan_without_clarify_prints_accept_command(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    _scripted_tty_stdin(monkeypatch, "1")

    exit_code = main([*_config_args(tmp_path, config), "plan", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Proposed plan:" in captured.out
    assert "llm-plan-execute accept --run-dir" in captured.out


def test_plan_prints_clear_clarification_status(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    _scripted_tty_stdin(monkeypatch, "1")

    exit_code = main([*_config_args(tmp_path, config), "plan", "--prompt", "Add a small feature"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Clarification: no questions required" in captured.out
    assert "Proposed plan:" in captured.out


def test_accept_command_promotes_plan(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    _scripted_tty_stdin(monkeypatch, "4")

    exit_plan = main([*_config_args(tmp_path, config), "plan", "--prompt", "Add a small feature", "--no-clarify"])
    assert exit_plan == CANCELED_EXIT
    captured = capsys.readouterr()
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()

    exit_code = main([*_config_args(tmp_path, config), "accept", "--run-dir", str(run_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Accepted plan:" in captured.out
    assert (run_dir / "04-accepted-plan.md").exists()


def test_noninteractive_clarification_exits_before_plan(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO())

    exit_code = main([*_config_args(tmp_path, config), "plan", "--prompt", "Do an ambiguous thing"])

    captured = capsys.readouterr()
    assert exit_code == CLARIFICATION_NEEDED_EXIT
    assert "Clarification needed:" in captured.out
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()
    assert (run_dir / "00-clarification.md").exists()
    assert not (run_dir / "01-draft-plan.md").exists()


def test_interactive_clarification_marks_answered_questions_clear(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "Implement the requested behavior.")
    _scripted_tty_stdin(monkeypatch, "1")

    exit_code = main([*_config_args(tmp_path, config), "plan", "--prompt", "Do an ambiguous thing"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Clarification: answered 1 question(s)." in captured.out
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()
    clarification = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["clarification"]
    assert clarification["status"] == "clear"
    assert clarification["answers"] == ["Implement the requested behavior."]
    assert "- Status: clear" in (run_dir / "00-clarification.md").read_text(encoding="utf-8")


def test_build_unaccepted_run_reports_accept_guidance(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    _scripted_tty_stdin(monkeypatch, "4")

    exit_plan = main([*_config_args(tmp_path, config), "plan", "--prompt", "Add a small feature", "--no-clarify"])
    assert exit_plan == CANCELED_EXIT
    captured = capsys.readouterr()
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()

    monkeypatch.setattr("sys.stdin", io.StringIO())
    exit_code = main([*_config_args(tmp_path, config), "build", "--run-dir", str(run_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "accept command" in captured.err


def test_run_pause_before_build_preserves_accepted_plan(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    _scripted_tty_stdin(monkeypatch, "1", "2")

    exit_code = main([*_config_args(tmp_path, config), "run", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == PAUSED_EXIT
    assert "Workflow paused" in captured.out
    assert "Continue with: llm-plan-execute build --run-dir" in captured.out
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()
    assert (run_dir / "04-accepted-plan.md").exists()
    assert (run_dir / "workflow-state.json").exists()
    assert not (run_dir / "05-build-output.md").exists()


def test_run_approve_builds_after_inline_plan_review(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    _scripted_tty_stdin(monkeypatch, "1", "1", "2", "4", "4")

    exit_code = main([*_config_args(tmp_path, config), "run", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "# Arbiter Decision" in captured.out
    assert "Build output:" in captured.out
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()
    assert (run_dir / "08-build-review-summary.md").exists()


def test_build_permission_mode_is_recorded(tmp_path, capsys):
    config = _write_dry_config(tmp_path)
    main([*_config_args(tmp_path, config), "plan", "--prompt", "Add a small feature", "--no-clarify", "--yes"])
    captured = capsys.readouterr()
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()

    exit_code = main(
        [*_config_args(tmp_path, config), "build", "--run-dir", str(run_dir), "--permission-mode", "full-access"]
    )

    capsys.readouterr()
    assert exit_code == 0
    policies = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["execution_policies"]
    assert policies["builder"]["mode"] == "full-access"


def test_run_cancel_stops_before_build(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    _scripted_tty_stdin(monkeypatch, "4")

    exit_code = main([*_config_args(tmp_path, config), "run", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == CANCELED_EXIT
    assert "Report:" in captured.out


def test_quiet_suppresses_progress(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    _scripted_tty_stdin(monkeypatch, "1")

    exit_code = main(
        [*_config_args(tmp_path, config), "--quiet", "plan", "--prompt", "Add a small feature", "--no-clarify"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Proposed plan:" in captured.out
    assert captured.err == ""


def test_progress_uses_stderr_and_leaves_stdout_clean(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    _scripted_tty_stdin(monkeypatch, "1")

    exit_code = main([*_config_args(tmp_path, config), "plan", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Proposed plan:" in captured.out
    assert "planner: starting" in captured.err
    assert "planner: starting" not in captured.out


def test_plan_non_tty_auto_accepts_without_flag(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO())

    exit_code = main([*_config_args(tmp_path, config), "plan", "--prompt", "Add a small feature", "--no-clarify"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Accepted plan:" in captured.out
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()
    assert (run_dir / "04-accepted-plan.md").exists()


def test_pause_then_build_non_interactive_completes(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    _scripted_tty_stdin(monkeypatch, "1", "2")

    exit_run = main([*_config_args(tmp_path, config), "run", "--prompt", "Add a small feature", "--no-clarify"])
    assert exit_run == PAUSED_EXIT
    captured = capsys.readouterr()
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()

    exit_build = main([*_config_args(tmp_path, config), "--non-interactive", "build", "--run-dir", str(run_dir)])
    capsys.readouterr()
    assert exit_build == 0
    assert (run_dir / "05-build-output.md").exists()
    wf = json.loads((run_dir / "workflow-state.json").read_text(encoding="utf-8"))
    assert wf.get("lifecycle_status") == "completed"
    assert wf.get("stage") == "complete"


def test_non_interactive_run_flag_completes(tmp_path, capsys):
    config = _write_dry_config(tmp_path)

    exit_code = main(
        [
            *_config_args(tmp_path, config),
            "--non-interactive",
            "run",
            "--prompt",
            "Add a small feature",
            "--no-clarify",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Build output:" in captured.out
    assert "Review summary:" in captured.out


def test_ci_alias_behaves_like_non_interactive(tmp_path, capsys):
    config = _write_dry_config(tmp_path)
    exit_code = main([*_config_args(tmp_path, config), "--ci", "run", "--prompt", "Add feature", "--no-clarify"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Build output:" in captured.out


def test_plan_save_workflow_state_failure_returns_one(tmp_path, capsys, monkeypatch):
    config = _write_dry_config(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO())

    def _boom(_run_dir, _wf):
        raise OSError("simulated write failure")

    monkeypatch.setattr("llm_plan_execute.cli.save_workflow_state", _boom)

    exit_code = main([*_config_args(tmp_path, config), "plan", "--prompt", "Add a small feature", "--no-clarify"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "simulated write failure" in captured.err


def test_plan_review_menu_lists_core_actions():
    session, stdout, _stderr = session_with_mock_stdin(["4"])
    session.ask_plan_review()
    transcript = "".join(stdout.lines)
    assert "Accept plan" in transcript
    assert "Modify plan" in transcript
    assert "Step through plan" in transcript


def test_complete_stage_is_reached_after_full_non_interactive_run(tmp_path, capsys):
    config = _write_dry_config(tmp_path)
    exit_code = main(
        [
            *_config_args(tmp_path, config),
            "--non-interactive",
            "run",
            "--prompt",
            "Add a small feature",
            "--no-clarify",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()
    wf = json.loads((run_dir / "workflow-state.json").read_text(encoding="utf-8"))
    assert wf.get("stage") == "complete"
    assert wf.get("lifecycle_status") == "completed"


def test_force_session_overrides_stale_lock(tmp_path, capsys):
    """--force-session lets build proceed when a stale lock file is present."""
    from llm_plan_execute.workflow_state import workflow_lock_path

    config = _write_dry_config(tmp_path)
    main([*_config_args(tmp_path, config), "plan", "--prompt", "Add feature", "--no-clarify", "--yes"])
    captured = capsys.readouterr()
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()

    # Inject a stale lock (PID 0 is always dead)
    lock_path = workflow_lock_path(run_dir)
    lock_path.write_text("0\n", encoding="utf-8")

    exit_build = main(
        [
            *_config_args(tmp_path, config),
            "--non-interactive",
            "build",
            "--run-dir",
            str(run_dir),
            "--force-session",
        ]
    )
    capsys.readouterr()
    assert exit_build == 0


def test_build_without_force_session_fails_on_live_lock(tmp_path, capsys):
    """Without --force-session, a live-PID lock raises an error."""
    import os

    from llm_plan_execute.workflow_state import workflow_lock_path

    config = _write_dry_config(tmp_path)
    main([*_config_args(tmp_path, config), "plan", "--prompt", "Add feature", "--no-clarify", "--yes"])
    captured = capsys.readouterr()
    run_line = next(line for line in captured.out.splitlines() if line.startswith("Run:"))
    run_dir = tmp_path / "runs" / run_line.partition(":")[2].strip()

    # Write current process PID — definitely alive
    workflow_lock_path(run_dir).write_text(f"{os.getpid()}\n", encoding="utf-8")

    exit_build = main(
        [
            *_config_args(tmp_path, config),
            "--non-interactive",
            "build",
            "--run-dir",
            str(run_dir),
        ]
    )
    captured = capsys.readouterr()
    assert exit_build == 1
    assert "--force-session" in captured.err
