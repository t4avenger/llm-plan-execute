"""Workspace resolution, config paths, and CLI --repo behavior."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from llm_plan_execute.cli import main
from llm_plan_execute.config import (
    load_config,
    normalize_writable_dirs,
    resolve_config_path,
    resolve_repo,
    resolve_workspace_relative_path,
)
from llm_plan_execute.workflow import _git_exclude_pathspecs, _workspace_changes


def test_resolve_repo_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = resolve_repo(None)
    assert resolved == tmp_path.resolve()


def test_resolve_repo_rejects_missing(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(ValueError, match="does not exist"):
        resolve_repo(missing)


def test_resolve_repo_rejects_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        resolve_repo(f)


def test_normalize_writable_dirs_blocks_escape(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    inner = ws / "inside"
    inner.mkdir()
    escape = Path("../outside")
    with pytest.raises(ValueError, match="outside workspace"):
        normalize_writable_dirs(ws, (escape,), field="execution.writable_dirs")


def test_resolve_config_path_explicit_absolute(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = tmp_path / "elsewhere.json"
    cfg.touch()
    assert resolve_config_path(cfg, ws) == cfg.resolve()


def test_resolve_config_path_explicit_relative_to_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = ws / "custom.json"
    cfg.write_text("{}", encoding="utf-8")
    assert resolve_config_path(Path("custom.json"), ws) == cfg.resolve()


def test_resolve_config_path_expands_user(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    cfg = home / "cfg.json"
    cfg.touch()
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("HOME", str(home))

    assert resolve_config_path(Path("~/cfg.json"), ws) == cfg.resolve()


def test_resolve_config_path_default_under_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    expected = ws / ".llm-plan-execute" / "config.json"
    assert resolve_config_path(None, ws) == expected.resolve()


def test_workspace_relative_path_expands_user(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    prompt = home / "prompt.txt"
    prompt.touch()
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("HOME", str(home))

    assert resolve_workspace_relative_path(ws, Path("~/prompt.txt")) == prompt.resolve()


def test_load_config_normalizes_paths(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = ws / ".llm-plan-execute" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps(
            {
                "dry_run": True,
                "runs_dir": ".llm-plan-execute/runs",
                "providers": [],
                "execution": {"writable_dirs": ["subdir"]},
            }
        ),
        encoding="utf-8",
    )
    (ws / "subdir").mkdir()
    app = load_config(None, workspace=ws, dry_run=False)
    assert app.workspace == ws.resolve()
    assert app.runs_dir == (ws / ".llm-plan-execute/runs").resolve()
    assert app.execution.writable_dirs == ((ws / "subdir").resolve(),)


def test_normalize_writable_dirs_expands_user(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    allowed = home / "allowed"
    allowed.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("HOME", str(home))

    assert normalize_writable_dirs(ws, (Path("~/allowed"),), field="execution.writable_dirs") == (allowed.resolve(),)


def test_load_config_rejects_runs_dir_outside_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = ws / ".llm-plan-execute" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps({"dry_run": True, "runs_dir": str(tmp_path / "outside-runs"), "providers": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"runs_dir: path .* outside workspace"):
        load_config(None, workspace=ws, dry_run=False)


def test_main_init_config_writes_under_repo(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "proj"
    repo.mkdir()
    exit_code = main(["--repo", str(repo), "init-config"])
    captured = capsys.readouterr()
    assert exit_code == 0
    out = (repo / ".llm-plan-execute" / "config.json").resolve()
    assert str(out) in captured.out or captured.out.strip().endswith(str(out))


def test_main_plan_with_repo_writes_runs_under_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "fixture"
    repo.mkdir()
    cfg = repo / "cfg.json"
    runs = repo / "runs-here"
    cfg.write_text(
        json.dumps({"dry_run": True, "runs_dir": str(runs), "providers": []}),
        encoding="utf-8",
    )
    exit_code = main(
        [
            "--repo",
            str(repo),
            "--config",
            "cfg.json",
            "plan",
            "--prompt",
            "test",
            "--no-clarify",
        ]
    )
    assert exit_code == 0
    assert runs.exists()
    assert any(r.name.startswith("20") for r in runs.iterdir())


def test_relative_run_dir_resolves_against_repo(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "r"
    repo.mkdir()
    cfg = repo / "c.json"
    runs = repo / "custom-runs"
    cfg.write_text(
        json.dumps({"dry_run": True, "runs_dir": str(runs), "providers": []}),
        encoding="utf-8",
    )
    main(["--repo", str(repo), "--config", "c.json", "plan", "--prompt", "x", "--no-clarify"])
    capsys.readouterr()
    run_ids = [p.name for p in runs.iterdir()]
    run_dir_name = run_ids[0]

    exit_code = main(
        [
            "--repo",
            str(repo),
            "--config",
            "c.json",
            "accept",
            "--run-dir",
            f"custom-runs/{run_dir_name}",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Accepted plan" in captured.out


def test_cli_writable_dir_resolves_under_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "w"
    repo.mkdir()
    sub = repo / "extra-writable"
    sub.mkdir()
    cfg = repo / "cfg.json"
    cfg.write_text(json.dumps({"dry_run": True, "runs_dir": str(repo / "runs"), "providers": []}), encoding="utf-8")
    exit_code = main(
        [
            "--repo",
            str(repo),
            "--config",
            "cfg.json",
            "plan",
            "--prompt",
            "implement code and tests",
            "--no-clarify",
            "--writable-dir",
            "extra-writable",
        ]
    )
    assert exit_code == 0


def test_explicit_config_missing_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "missing.json"
    exit_code = main(["--config", str(missing), "models"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "was not found" in captured.err


def test_repo_path_with_spaces(tmp_path, capsys, monkeypatch):
    repo = tmp_path / "my workspace"
    repo.mkdir()
    cfg = repo / ".llm-plan-execute" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"dry_run": True, "providers": []}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    exit_code = main(["--repo", str(repo), "models"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "dry-codex" in captured.out


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only chmod read-only parent")
def test_init_config_unwritable_workspace(tmp_path, capsys):
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o755)
    ro = parent / "locked"
    ro.mkdir(mode=0o755)
    os.chmod(ro, 0o555)  # noqa: S103 - test intentionally makes workspace unwritable.
    try:
        exit_code = main(["--repo", str(ro), "init-config"])
    finally:
        os.chmod(ro, 0o755)  # noqa: S103 - restore permissions after read-only test.
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error:" in captured.err


def test_workspace_changes_ignores_run_artifacts(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available")

    subprocess.run([git, "init"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    subprocess.run([git, "config", "user.email", "t@t"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    subprocess.run([git, "config", "user.name", "t"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    (tmp_path / "tracked.txt").write_text("a\n", encoding="utf-8")
    subprocess.run([git, "add", "tracked.txt"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    subprocess.run([git, "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603

    runs_root = tmp_path / ".llm-plan-execute" / "runs"
    clean = _workspace_changes(tmp_path, exclude_runs_under=runs_root)
    assert clean is not None
    runs_root.mkdir(parents=True)
    (runs_root / "noise.txt").write_text("run artifact\n", encoding="utf-8")

    before = _workspace_changes(tmp_path, exclude_runs_under=runs_root)
    assert before == clean
    (tmp_path / "edit.txt").write_text("user edit\n", encoding="utf-8")
    after = _workspace_changes(tmp_path, exclude_runs_under=runs_root)
    assert after is not None
    assert before != after


def test_workspace_changes_does_not_exclude_workspace_root(tmp_path):
    assert _git_exclude_pathspecs(tmp_path, tmp_path) == []
