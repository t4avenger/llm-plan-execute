"""Tests for git-flow helpers (requires git executable)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from llm_plan_execute.git_flow import (
    GitFlowError,
    commit_checkpoint,
    filter_checkpoint_pathspecs,
    find_git_root,
    format_task_branch_name,
    gh_available_and_authenticated,
    guard_build_start,
    maybe_offer_github_pr,
    resolve_base_branch,
    slugify_task_branch_component,
    try_github_pr_create,
)
from llm_plan_execute.workflow_state import WorkflowState

_GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(_GIT is None, reason="git executable is required for git-flow tests")


def _git(repo: Path, *args: str) -> None:
    subprocess.run([str(_GIT), *args], cwd=repo, check=True, capture_output=True, text=True)  # noqa: S603


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    subprocess.run([str(_GIT), "init"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "test")
    _git(tmp_path, "checkout", "-b", "main")
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-m", "init")
    return tmp_path


def test_find_git_root(git_repo: Path) -> None:
    assert find_git_root(git_repo) == git_repo.resolve()


def test_resolve_base_branch(git_repo: Path) -> None:
    assert resolve_base_branch(git_repo, None) == "main"


def test_resolve_base_branch_override(git_repo: Path) -> None:
    assert resolve_base_branch(git_repo, "main") == "main"


def test_slugify_and_branch_name() -> None:
    slug_max = 50
    assert slugify_task_branch_component("My Feature!!!") == "my-feature"
    long_slug = "a" * 80
    assert len(slugify_task_branch_component(long_slug)) <= slug_max
    name = format_task_branch_name("run-abc123xyz")
    assert name.startswith("feat/")
    assert "abc123xy" in name or "run-abc123" in name


def test_guard_skips_without_git(tmp_path: Path) -> None:
    wf = WorkflowState()
    guard_build_start(tmp_path, wf, base_branch_override=None, task_id="t1")
    assert wf.branch is None


def test_guard_dirty_tree(git_repo: Path) -> None:
    (git_repo / "dirty.txt").write_text("x", encoding="utf-8")
    wf = WorkflowState()
    with pytest.raises(GitFlowError, match="uncommitted"):
        guard_build_start(git_repo, wf, base_branch_override=None, task_id="task-x")


def test_guard_creates_branch(git_repo: Path) -> None:
    wf = WorkflowState()
    guard_build_start(git_repo, wf, base_branch_override=None, task_id="my-task-12345")
    assert wf.branch == format_task_branch_name("my-task-12345")
    assert wf.base_branch == "main"


def test_guard_wrong_branch(git_repo: Path) -> None:
    _git(git_repo, "checkout", "-b", "other")
    wf = WorkflowState()
    with pytest.raises(GitFlowError, match="checkout base branch"):
        guard_build_start(git_repo, wf, base_branch_override=None, task_id="z")


def test_filter_checkpoint_pathspecs(git_repo: Path) -> None:
    specs = ["README.md", ".llm-plan-execute/runs/x/foo.md", "custom-runs/run.json", "src/a.py"]
    kept = filter_checkpoint_pathspecs(git_repo, specs, excluded_roots=[git_repo / "custom-runs"])
    assert "README.md" in kept
    assert "src/a.py" in kept
    assert "custom-runs/run.json" not in kept
    assert all(".llm-plan-execute/runs" not in k for k in kept)


def test_commit_checkpoint_respects_pathspecs(git_repo: Path) -> None:
    (git_repo / "a.txt").write_text("a", encoding="utf-8")
    (git_repo / "b.txt").write_text("b", encoding="utf-8")
    res = commit_checkpoint(git_repo, "checkpoint", "test", ["a.txt"])
    assert res.committed is True
    proc = subprocess.run(  # noqa: S603
        [str(_GIT), "status", "--porcelain"], cwd=git_repo, capture_output=True, text=True, check=False
    )
    assert "b.txt" in proc.stdout


def test_commit_checkpoint_hook_failure(git_repo: Path) -> None:
    hook_dir = git_repo / ".git" / "hooks"
    hook_dir.mkdir(parents=True, exist_ok=True)
    (hook_dir / "pre-commit").write_text("#!/bin/sh\necho hook says no >&2\nexit 1\n", encoding="utf-8")
    (hook_dir / "pre-commit").chmod(0o755)
    (git_repo / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(GitFlowError, match=r"(hook says no|failed)"):
        commit_checkpoint(git_repo, "checkpoint", "x", ["x.txt"])


def test_gh_auth_probe() -> None:
    ok, _reason = gh_available_and_authenticated()
    assert isinstance(ok, bool)


def test_try_github_pr_create_reports_missing_gh(monkeypatch, git_repo: Path) -> None:
    monkeypatch.setattr("llm_plan_execute.git_flow.shutil.which", lambda _command: None)

    created, message = try_github_pr_create(git_repo, "title", "body")

    assert created is False
    assert "GitHub CLI" in message


def test_maybe_offer_github_pr_non_git_writes_body(tmp_path: Path, capsys) -> None:
    wf = WorkflowState(branch="feat/x", base_branch="main")

    maybe_offer_github_pr(
        workspace=tmp_path,
        wf=wf,
        task_id="tid",
        title_hint="Title",
        interactive_tty=False,
        create_pr_without_prompt=False,
    )

    output = capsys.readouterr().out
    assert "PR creation skipped" in output
    body = (tmp_path / ".llm-plan-execute" / "pr-body.md").read_text(encoding="utf-8")
    assert "Branch: `feat/x`" in body


def test_maybe_offer_github_pr_declined_writes_body(git_repo: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda: "n")

    maybe_offer_github_pr(
        workspace=git_repo,
        wf=WorkflowState(),
        task_id="tid",
        title_hint="Title",
        interactive_tty=True,
        create_pr_without_prompt=False,
    )

    assert "Skipped PR creation" in capsys.readouterr().out
    assert (git_repo / ".llm-plan-execute" / "pr-body.md").exists()
