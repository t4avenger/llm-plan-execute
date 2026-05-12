"""Tests for the interactive wizard launched by bare ``llm-plan-execute``."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from llm_plan_execute import wizard
from llm_plan_execute.cli import main
from llm_plan_execute.config import DEFAULT_ROOT

CANCELED_EXIT = 130


def _scripted_tty_stdin(monkeypatch: pytest.MonkeyPatch, *lines: str) -> None:
    """Patch ``sys.stdin`` with a TTY-like, line-buffered scripted source."""

    queue = [line if line.endswith("\n") else line + "\n" for line in lines]

    class _Stdin:
        def isatty(self) -> bool:
            return True

        def readline(self) -> str:
            return queue.pop(0) if queue else ""

    monkeypatch.setattr("sys.stdin", _Stdin())


def _write_dry_config(workspace: Path) -> Path:
    config = workspace / ".llm-plan-execute" / "config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "dry_run": True,
                "runs_dir": ".llm-plan-execute/runs",
                "providers": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def _paused_run(workspace: Path, name: str = "20260101-000000-deadbeef") -> Path:
    runs_dir = workspace / DEFAULT_ROOT / "runs" / name
    runs_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 3,
        "stage": "plan_review",
        "lifecycle_status": "paused",
        "accepted_plan_artifact": "04-accepted-plan.md",
        "accepted_plan_version": 1,
    }
    (runs_dir / "workflow-state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return runs_dir


def test_is_tty_handles_streams_without_isatty() -> None:
    assert wizard.is_tty(io.StringIO()) is False

    class _Yes:
        def isatty(self) -> bool:
            return True

    assert wizard.is_tty(_Yes()) is True


def test_bare_command_non_tty_prints_help_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO())

    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "usage:" in captured.err.lower()
    assert "No subcommand provided" in captured.err


def test_bare_command_with_non_interactive_flag_prints_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _scripted_tty_stdin(monkeypatch)  # TTY, but flag forces non-interactive

    exit_code = main(["--non-interactive"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No subcommand provided" in captured.err


def test_wizard_decline_workspace_exits_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _scripted_tty_stdin(monkeypatch, "n")

    exit_code = main(["--repo", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Workspace declined" in captured.out


def test_wizard_validate_existing_config_prints_ok(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _write_dry_config(tmp_path)
    _scripted_tty_stdin(monkeypatch, "y", "3")  # confirm workspace; choose validate

    exit_code = main(["--repo", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Config is valid." in captured.out


def test_wizard_creates_config_when_missing_and_dispatches_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("llm_plan_execute.config.shutil.which", lambda _command: None)

    captured_argv: list[list[str]] = []

    def _fake_main(argv: list[str]) -> int:
        captured_argv.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_main)
    _scripted_tty_stdin(
        monkeypatch,
        "y",  # confirm workspace
        "2",  # continue with --dry-run
        "1",  # inline prompt entry
        "Add a small feature",  # prompt text
    )

    exit_code = main(["--repo", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0, f"stdout={captured.out!r}\nstderr={captured.err!r}"
    assert "llm-plan-execute wizard" in captured.out
    assert "(simulated providers)" in captured.out
    assert captured_argv, "wizard did not dispatch the run command"
    last_argv = captured_argv[-1]
    assert "--dry-run" in last_argv
    assert "Add a small feature" in last_argv


def test_wizard_dry_run_skips_config_setup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured_argv: list[list[str]] = []

    def _fake_main(argv: list[str]) -> int:
        captured_argv.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_main)
    _scripted_tty_stdin(
        monkeypatch,
        "y",  # confirm workspace
        "1",  # inline prompt entry
        "Add a small feature",  # prompt text
    )

    exit_code = main(["--repo", str(tmp_path), "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0, f"stdout={captured.out!r}\nstderr={captured.err!r}"
    assert "(dry-run)" in captured.out
    assert captured_argv, "wizard did not dispatch the run command"
    assert "--dry-run" in captured_argv[-1]


def test_wizard_creates_default_config_when_user_chooses_create(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("llm_plan_execute.config.shutil.which", lambda _command: None)
    _scripted_tty_stdin(
        monkeypatch,
        "y",  # confirm workspace
        "1",  # create default config
        "1",  # save provider setup using auto-detected defaults
        "3",  # cancel at prompt entry to keep test focused
    )

    exit_code = main(["--repo", str(tmp_path)])

    captured = capsys.readouterr()
    config_path = tmp_path / ".llm-plan-execute" / "config.json"
    assert config_path.exists()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert {p["name"] for p in raw["providers"]} == {"codex", "claude", "cursor"}
    assert exit_code == CANCELED_EXIT
    assert "Workflow canceled." in captured.err


def test_wizard_reconfigure_flag_overwrites_after_confirm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = _write_dry_config(tmp_path)
    original = config_path.read_text(encoding="utf-8")
    monkeypatch.setattr("llm_plan_execute.config.shutil.which", lambda _command: None)
    _scripted_tty_stdin(
        monkeypatch,
        "y",  # confirm workspace
        "y",  # confirm overwrite
        "1",  # save provider setup using auto-detected defaults
        "3",  # cancel at prompt entry
    )

    exit_code = main(["--repo", str(tmp_path), "--reconfigure"])

    assert exit_code == CANCELED_EXIT
    rewritten = config_path.read_text(encoding="utf-8")
    assert rewritten != original
    assert "providers" in rewritten


def test_wizard_reconfigure_overwrite_refusal_keeps_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config_path = _write_dry_config(tmp_path)
    original = config_path.read_text(encoding="utf-8")
    _scripted_tty_stdin(
        monkeypatch,
        "y",  # confirm workspace
        "n",  # refuse overwrite (no second provider-setup menu)
        "3",  # cancel prompt entry
    )

    exit_code = main(["--repo", str(tmp_path), "--reconfigure"])

    captured = capsys.readouterr()
    assert exit_code == CANCELED_EXIT
    assert "Keeping existing config." in captured.out
    assert config_path.read_text(encoding="utf-8") == original


def test_wizard_detects_paused_run_and_dispatches_build(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _write_dry_config(tmp_path)
    run_dir = _paused_run(tmp_path)
    captured_argv: list[list[str]] = []

    def _fake_main(argv: list[str]) -> int:
        captured_argv.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_main)
    _scripted_tty_stdin(
        monkeypatch,
        "y",  # confirm workspace
        "1",  # use existing config
        "1",  # resume existing paused run
    )

    exit_code = main(["--repo", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0, f"stdout={captured.out!r}\nstderr={captured.err!r}"
    assert "Detected 1 paused run" in captured.out
    assert captured_argv, "wizard did not dispatch the build command"
    last_argv = captured_argv[-1]
    assert "build" in last_argv
    assert "--run-dir" in last_argv
    assert str(run_dir) in last_argv


def test_wizard_new_run_path_ignores_paused_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _write_dry_config(tmp_path)
    _paused_run(tmp_path)

    captured_argv: list[list[str]] = []

    def _fake_main(argv: list[str]) -> int:
        captured_argv.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_main)
    _scripted_tty_stdin(
        monkeypatch,
        "y",  # confirm workspace
        "1",  # use existing config
        "2",  # start new run
        "1",  # inline prompt entry
        "Add a small feature",  # prompt text
    )

    exit_code = main(["--repo", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0, f"stdout={captured.out!r}\nstderr={captured.err!r}"
    assert "Detected 1 paused run" in captured.out
    assert captured_argv, "wizard did not dispatch the run command"
    last_argv = captured_argv[-1]
    assert "run" in last_argv
    assert "--prompt" in last_argv
    assert "Add a small feature" in last_argv


def test_wizard_editor_unset_falls_back_to_inline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("EDITOR", raising=False)
    _write_dry_config(tmp_path)

    captured_argv: list[list[str]] = []

    def _fake_main(argv: list[str]) -> int:
        captured_argv.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_main)
    _scripted_tty_stdin(
        monkeypatch,
        "y",  # confirm workspace
        "1",  # use existing config
        "2",  # editor (unset)
        "1",  # fall back to inline
        "Add a small feature",  # prompt text
    )

    exit_code = main(["--repo", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "$EDITOR is not set" in captured.err
    assert captured_argv, "wizard did not dispatch the run command"


def test_wizard_editor_nonzero_falls_back_to_inline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EDITOR", "fake-editor")

    class _Completed:
        returncode = 1

    def _fake_run(argv: list[str], **_kwargs: Any) -> _Completed:
        assert argv[0] == "fake-editor"
        return _Completed()

    monkeypatch.setattr("llm_plan_execute.wizard.subprocess.run", _fake_run)
    _write_dry_config(tmp_path)

    captured_argv: list[list[str]] = []

    def _fake_main(argv: list[str]) -> int:
        captured_argv.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_main)
    _scripted_tty_stdin(
        monkeypatch,
        "y",  # confirm workspace
        "1",  # use existing config
        "2",  # editor (fails)
        "1",  # fall back to inline
        "Add a small feature",  # prompt text
    )

    exit_code = main(["--repo", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "$EDITOR exited with code 1" in captured.err
    assert captured_argv, "wizard did not dispatch the run command"
    assert "Add a small feature" in captured_argv[-1]


def test_wizard_editor_empty_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EDITOR", "fake-editor")

    class _Completed:
        returncode = 0

    def _fake_run(argv: list[str], **_kwargs: Any) -> _Completed:
        Path(argv[-1]).write_text("# only comments\n\n", encoding="utf-8")
        return _Completed()

    monkeypatch.setattr("llm_plan_execute.wizard.subprocess.run", _fake_run)
    _write_dry_config(tmp_path)

    captured_argv: list[list[str]] = []

    def _fake_main(argv: list[str]) -> int:
        captured_argv.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_main)
    _scripted_tty_stdin(
        monkeypatch,
        "y",  # confirm workspace
        "1",  # use existing config
        "2",  # editor (empty)
        "1",  # inline fallback
        "Add a small feature",  # prompt text
    )

    exit_code = main(["--repo", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "empty prompt" in captured.err
    assert captured_argv, "wizard did not dispatch the run command"


def test_wizard_editor_success_uses_typed_prompt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EDITOR", "fake-editor")

    class _Completed:
        returncode = 0

    def _fake_run(argv: list[str], **_kwargs: Any) -> _Completed:
        Path(argv[-1]).write_text("Add a small feature from the editor\n", encoding="utf-8")
        return _Completed()

    monkeypatch.setattr("llm_plan_execute.wizard.subprocess.run", _fake_run)
    _write_dry_config(tmp_path)

    captured_argv: list[list[str]] = []

    def _fake_main(argv: list[str]) -> int:
        captured_argv.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_main)
    _scripted_tty_stdin(
        monkeypatch,
        "y",  # confirm workspace
        "1",  # use existing config
        "2",  # editor
        "y",  # confirm prompt
    )

    exit_code = main(["--repo", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0, f"stdout={captured.out!r}\nstderr={captured.err!r}"
    assert captured_argv, "wizard did not dispatch the run command"
    last_argv = captured_argv[-1]
    assert "Add a small feature from the editor" in last_argv


def test_wizard_create_config_defaults_then_dispatches_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("llm_plan_execute.config.shutil.which", lambda _command: None)
    captured_argv: list[list[str]] = []

    def _fake_dispatch(argv: list[str]) -> int:
        captured_argv.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_dispatch)
    _scripted_tty_stdin(
        monkeypatch,
        "y",
        "1",  # create default config
        "1",  # auto-detected provider defaults
        "1",  # inline prompt
        "Ship the feature",
    )

    exit_code = main(["--repo", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0, f"stdout={captured.out!r}\nstderr={captured.err!r}"
    assert (tmp_path / ".llm-plan-execute" / "config.json").exists()
    assert captured_argv
    assert "run" in captured_argv[-1]
    assert "Ship the feature" in captured_argv[-1]


def test_wizard_dispatched_exit_code_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_dry_config(tmp_path)
    dispatch_exit = 42

    def _fake_dispatch(_argv: list[str]) -> int:
        return dispatch_exit

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_dispatch)
    _scripted_tty_stdin(monkeypatch, "y", "1", "1", "task")

    exit_code = main(["--repo", str(tmp_path)])

    assert exit_code == dispatch_exit


def test_wizard_forwards_ui_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_dry_config(tmp_path)
    captured: list[list[str]] = []

    def _fake_dispatch(argv: list[str]) -> int:
        captured.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_dispatch)
    _scripted_tty_stdin(monkeypatch, "y", "1", "1", "go")

    exit_code = main(["--repo", str(tmp_path), "--ui", "jsonl"])

    assert exit_code == 0
    assert "--ui" in captured[-1]
    assert "jsonl" in captured[-1]


def test_wizard_reconfigure_without_existing_config_still_offers_create_menu(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """``--reconfigure`` is ignored until a config path exists; wizard behaves like a fresh workspace."""

    captured_dispatch: list[list[str]] = []

    def _fake_dispatch(argv: list[str]) -> int:
        captured_dispatch.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_dispatch)
    _scripted_tty_stdin(monkeypatch, "y", "2", "1", "dry-run path")

    exit_code = main(["--repo", str(tmp_path), "--reconfigure"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No config found" in captured.out
    assert captured_dispatch
    assert "dry-run path" in captured_dispatch[-1]


def test_wizard_paused_scan_warns_on_newer_schema_but_lists_readable_paused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _write_dry_config(tmp_path)
    good = _paused_run(tmp_path, "good-paused")
    bad = tmp_path / DEFAULT_ROOT / "runs" / "future-schema"
    bad.mkdir(parents=True)
    (bad / "workflow-state.json").write_text(
        json.dumps(
            {
                "schema_version": 99,
                "stage": "plan_review",
                "lifecycle_status": "paused",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    captured_argv: list[list[str]] = []

    def _fake_dispatch(argv: list[str]) -> int:
        captured_argv.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_dispatch)
    _scripted_tty_stdin(monkeypatch, "y", "1", "1")

    exit_code = main(["--repo", str(tmp_path)])

    err = capsys.readouterr().err
    assert "newer schema" in err.lower() or "newer" in err.lower()
    assert exit_code == 0
    assert captured_argv
    assert "build" in captured_argv[-1]
    assert str(good) in captured_argv[-1]


def test_wizard_editor_decline_prompt_loops_back_to_entry_menu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EDITOR", "fake-editor")

    class _Completed:
        returncode = 0

    def _fake_run(argv: list[str], **_kwargs: Any) -> _Completed:
        Path(argv[-1]).write_text("first draft from editor\n", encoding="utf-8")
        return _Completed()

    monkeypatch.setattr("llm_plan_execute.wizard.subprocess.run", _fake_run)
    _write_dry_config(tmp_path)
    captured_argv: list[list[str]] = []

    def _fake_dispatch(argv: list[str]) -> int:
        captured_argv.append(argv)
        return 0

    monkeypatch.setattr("llm_plan_execute.cli.dispatch_argv", _fake_dispatch)
    _scripted_tty_stdin(
        monkeypatch,
        "y",
        "1",
        "2",  # editor
        "n",  # decline "Use this prompt?"
        "1",  # inline fallback
        "final prompt text",
    )

    exit_code = main(["--repo", str(tmp_path)])

    assert exit_code == 0
    assert captured_argv
    assert "final prompt text" in captured_argv[-1]


def test_wizard_customize_shows_role_help_and_writes_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("llm_plan_execute.config.shutil.which", lambda _command: None)
    # Three providers: empty answers keep ``enabled`` false (default_no) and skip model prompts.
    _scripted_tty_stdin(
        monkeypatch,
        "y",
        "1",  # create default config
        "2",  # customize
        "",  # codex: decline enable (default false)
        "",  # claude
        "",  # cursor
        "3",  # cancel at task prompt
    )

    exit_code = main(["--repo", str(tmp_path)])

    out = capsys.readouterr().out
    assert "Roles assigned by selection" in out
    assert exit_code == CANCELED_EXIT
    cfg = tmp_path / ".llm-plan-execute" / "config.json"
    assert cfg.exists()
    raw = json.loads(cfg.read_text(encoding="utf-8"))
    assert all(not p.get("enabled") for p in raw["providers"])


def test_existing_subcommand_still_works_with_optional_parser(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Subcommands continue to dispatch after the wizard made command optional."""
    config = _write_dry_config(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO())

    exit_code = main(
        [
            "--repo",
            str(tmp_path),
            "--config",
            str(config),
            "plan",
            "--prompt",
            "Add a small feature",
            "--no-clarify",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Accepted plan:" in captured.out
