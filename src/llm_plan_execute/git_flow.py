"""Git guardrails and checkpoint commits for task workspaces (not the CLI install location)."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .workflow_state import WorkflowState

RUNS_DIR_SEGMENT = ".llm-plan-execute/runs"
_PR_BODY_FILL_IN_LINE = "- (fill in)"


class GitFlowError(RuntimeError):
    """User-actionable git guard or checkpoint failure."""


@dataclass(frozen=True)
class CommitResult:
    committed: bool
    commit_sha: str | None
    message: str


def _git(git: str, repo: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [git, *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=check,
    )


def _git_capture(git: str, repo: Path, args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run([git, *args], cwd=repo, text=True, capture_output=True, check=False)  # noqa: S603
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def find_git_root(start: Path) -> Path | None:
    """Return absolute git root containing ``start``, or ``None`` if not in a worktree."""
    git = shutil.which("git")
    if git is None:
        return None
    code, out, _err = _git_capture(git, start, ["rev-parse", "--show-toplevel"])
    if code != 0 or not out:
        return None
    return Path(out).resolve()


def _is_dirty(git: str, repo: Path) -> bool:
    proc = subprocess.run([git, "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise GitFlowError(proc.stderr.strip() or "git status failed.")
    return bool(proc.stdout.strip())


def _current_branch(git: str, repo: Path) -> str | None:
    code, out, _err = _git_capture(git, repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    if code != 0:
        return None
    if out == "HEAD":
        return None
    return out


def _branch_exists_local(git: str, repo: Path, branch: str) -> bool:
    proc = subprocess.run(  # noqa: S603
        [git, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _branch_exists_remote(git: str, repo: Path, branch: str, remote: str = "origin") -> bool:
    proc = subprocess.run(  # noqa: S603
        [git, "ls-remote", "--heads", remote, branch],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())


def resolve_base_branch(git_root: Path, override: str | None) -> str:
    """Resolve the integration branch name (e.g. ``main``), not a full ref path."""
    git = shutil.which("git")
    if git is None:
        raise GitFlowError("git executable not found on PATH; cannot resolve base branch.")

    if override:
        return override.strip()

    sym_code, sym_out, _ = _git_capture(git, git_root, ["symbolic-ref", "refs/remotes/origin/HEAD"])
    if sym_code == 0 and sym_out.startswith("refs/remotes/origin/"):
        name = sym_out.removeprefix("refs/remotes/origin/")
        if name and _ref_exists(git, git_root, f"refs/heads/{name}"):
            return name

    for candidate in ("main", "master"):
        if _ref_exists(git, git_root, f"refs/heads/{candidate}"):
            return candidate

    raise GitFlowError(
        "Could not resolve a base branch: set build.base_branch in config, ensure "
        "refs/remotes/origin/HEAD exists, or create local branch 'main' or 'master'."
    )


def _ref_exists(git: str, repo: Path, ref: str) -> bool:
    proc = subprocess.run([git, "show-ref", "--verify", ref], cwd=repo, capture_output=True, text=True, check=False)  # noqa: S603
    return proc.returncode == 0


def slugify_task_branch_component(raw: str, *, max_len: int = 50) -> str:
    """Lowercase slug: ``[a-z0-9-]``, collapsed separators, max length."""
    s = raw.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = "task"
    return s[:max_len].strip("-")


def short_run_id(task_id: str, *, max_len: int = 8) -> str:
    """Short suffix from task/run id for branch names."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", task_id)
    if len(cleaned) >= max_len:
        return cleaned[-max_len:].lower()
    return cleaned.lower() or "run"


def format_task_branch_name(task_id: str) -> str:
    slug = slugify_task_branch_component(task_id)
    short = short_run_id(task_id)
    return f"feat/{slug}-{short}"


def filter_checkpoint_pathspecs(
    repo_root: Path,
    pathspecs: list[str],
    *,
    excluded_roots: list[Path] | None = None,
) -> list[str]:
    """Drop pathspecs under ``.llm-plan-execute/runs/`` (relative to repo root)."""
    excluded = _checkpoint_excluded_roots(repo_root, excluded_roots)
    kept: list[str] = []
    for ps in pathspecs:
        if not ps or ps.strip() == "":
            continue
        norm = ps.replace("\\", "/").strip()
        if RUNS_DIR_SEGMENT in norm or norm.startswith(RUNS_DIR_SEGMENT):
            continue
        target = (repo_root / norm).resolve()
        if _is_under_any(target, excluded):
            continue
        kept.append(ps)
    return kept


def _checkpoint_excluded_roots(repo_root: Path, extra: list[Path] | None) -> list[Path]:
    roots = [(repo_root / ".llm-plan-execute" / "runs").resolve()]
    for path in extra or []:
        roots.append(path.resolve() if path.is_absolute() else (repo_root / path).resolve())
    return roots


def _is_under_any(target: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            target.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def guard_build_start(
    workspace: Path,
    wf: WorkflowState,
    *,
    base_branch_override: str | None,
    task_id: str,
) -> None:
    """
    Enforce a clean base branch and create ``feat/<slug>-<id>`` before build work.

    Skips when ``workspace`` is not inside a git repository. Updates ``wf`` with
    ``task_id``, ``base_branch``, and ``branch`` when a branch is created or reused.
    """
    wf.task_id = task_id
    git_root = find_git_root(workspace.resolve())
    if git_root is None:
        return

    git = shutil.which("git")
    if git is None:
        raise GitFlowError("git executable not found on PATH.")

    _ensure_clean_worktree(git, git_root)
    cur = _current_branch(git, git_root)
    if cur is None:
        raise GitFlowError("Cannot start build: HEAD is detached. Checkout your base branch first.")

    if wf.branch:
        _validate_existing_workflow_branch(cur, wf, git_root, base_branch_override)
        return

    base_branch = resolve_base_branch(git_root, base_branch_override)
    wf.base_branch = base_branch

    branch_name = format_task_branch_name(task_id)
    _validate_new_task_branch(git, git_root, current=cur, base_branch=base_branch, branch_name=branch_name)
    _checkout_new_branch(git, git_root, branch_name)
    wf.branch = branch_name


def _ensure_clean_worktree(git: str, git_root: Path) -> None:
    try:
        if _is_dirty(git, git_root):
            raise GitFlowError(
                "Cannot start build: git worktree has uncommitted changes. "
                "Commit or discard changes before branching (auto-stash is not enabled)."
            )
    except GitFlowError:
        raise
    except OSError as exc:
        raise GitFlowError(f"Cannot inspect git status: {exc}") from exc


def _validate_existing_workflow_branch(
    current: str,
    wf: WorkflowState,
    git_root: Path,
    base_branch_override: str | None,
) -> None:
    if current != wf.branch:
        raise GitFlowError(
            f"Cannot start build: expected branch {wf.branch!r} but HEAD is {current!r}. "
            "Checkout the task branch or clear workflow state."
        )
    if wf.base_branch is None:
        wf.base_branch = resolve_base_branch(git_root, base_branch_override)


def _validate_new_task_branch(
    git: str,
    git_root: Path,
    *,
    current: str,
    base_branch: str,
    branch_name: str,
) -> None:
    if current != base_branch:
        raise GitFlowError(
            f"Cannot start build: checkout base branch {base_branch!r} first (currently on {current!r})."
        )
    if _branch_exists_local(git, git_root, branch_name):
        raise GitFlowError(f"Branch {branch_name!r} already exists locally; remove or rename it first.")
    if _branch_exists_remote(git, git_root, branch_name):
        raise GitFlowError(
            f"Branch {branch_name!r} already exists on the remote; delete the remote branch or pick a new task id."
        )


def _checkout_new_branch(git: str, git_root: Path, branch_name: str) -> None:
    proc = subprocess.run(  # noqa: S603
        [git, "checkout", "-b", branch_name],
        cwd=git_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GitFlowError(proc.stderr.strip() or proc.stdout.strip() or "git checkout -b failed.")


def commit_checkpoint(
    repo_root: Path,
    label: str,
    summary: str,
    pathspecs: list[str],
    *,
    record_context: Any | None = None,
    task_id: str | None = None,
    kind: str = "commit",
    excluded_roots: list[Path] | None = None,
) -> CommitResult:
    """
    Stage explicit pathspecs and commit when there are staged changes.

    Hooks and GPG signing behave like normal ``git commit``. Never passes ``--no-verify``.
    """
    git = shutil.which("git")
    if git is None:
        raise GitFlowError("git executable not found on PATH.")

    filtered = filter_checkpoint_pathspecs(repo_root.resolve(), pathspecs, excluded_roots=excluded_roots)
    if not filtered:
        return CommitResult(False, None, f"no pathspecs after excluding {RUNS_DIR_SEGMENT}/")

    repo_root = repo_root.resolve()
    add_proc = subprocess.run([git, "add", "--", *filtered], cwd=repo_root, capture_output=True, text=True, check=False)  # noqa: S603
    if add_proc.returncode != 0:
        raise GitFlowError(add_proc.stderr.strip() or "git add failed.")

    diff_proc = subprocess.run(  # noqa: S603
        [git, "diff", "--cached", "--quiet"], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if diff_proc.returncode == 0:
        return CommitResult(False, None, "no staged changes for checkpoint")

    msg = f"{label}: {summary}".strip()
    commit_proc = subprocess.run(  # noqa: S603
        [git, "commit", "-m", msg],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_proc.returncode != 0:
        err = commit_proc.stderr.strip() or commit_proc.stdout.strip() or "git commit failed"
        raise GitFlowError(err)

    rev_proc = subprocess.run([git, "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True)  # noqa: S603
    sha = rev_proc.stdout.strip()

    if record_context is not None and task_id:
        metadata = {"label": label, "summary": summary, "sha": sha, "pathspecs": filtered}
        record_context.add_context_item(
            task_id=task_id,
            kind=kind,
            path=None,
            content=f"{msg}\n",
            metadata=metadata,
        )

    return CommitResult(True, sha, msg)


def list_tracked_pathspecs(repo_root: Path, *, exclude_runs: bool = True) -> list[str]:
    """Paths changed vs HEAD for checkpoint staging (explicit pathspec list)."""
    git = shutil.which("git")
    if git is None:
        return []
    proc = subprocess.run(  # noqa: S603
        [git, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    paths = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not exclude_runs:
        return paths
    return [p for p in paths if RUNS_DIR_SEGMENT not in p.replace("\\", "/") and not p.startswith(RUNS_DIR_SEGMENT)]


def git_changed_pathspecs(repo_root: Path, *, excluded_roots: list[Path] | None = None) -> list[str]:
    """Uncommitted + unstaged changes compared to HEAD, suitable for ``git add``."""
    git = shutil.which("git")
    if git is None:
        return []
    names: set[str] = set()
    for args in (
        ["diff", "--name-only", "HEAD"],
        ["diff", "--cached", "--name-only", "HEAD"],
        ["ls-files", "--others", "--exclude-standard"],
    ):
        proc = subprocess.run([git, *args], cwd=repo_root, capture_output=True, text=True, check=False)  # noqa: S603
        if proc.returncode == 0:
            names.update(line.strip() for line in proc.stdout.splitlines() if line.strip())
    excluded = _checkpoint_excluded_roots(repo_root.resolve(), excluded_roots)
    filtered = []
    for p in sorted(names):
        if RUNS_DIR_SEGMENT in p.replace("\\", "/"):
            continue
        if _is_under_any((repo_root / p).resolve(), excluded):
            continue
        filtered.append(p)
    return filtered


def gh_available_and_authenticated() -> tuple[bool, str]:
    """Return (ok, reason)."""
    gh = shutil.which("gh")
    if gh is None:
        return False, "GitHub CLI (gh) is not installed."
    proc = subprocess.run([gh, "auth", "status"], capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        return False, proc.stderr.strip() or proc.stdout.strip() or "gh auth status failed."
    return True, ""


def try_github_pr_create(repo_root: Path, title: str, body: str) -> tuple[bool, str]:
    """Return (created, message)."""
    ok, reason = gh_available_and_authenticated()
    if not ok:
        return False, reason
    gh = shutil.which("gh")
    if gh is None:
        return False, "GitHub CLI (gh) not found."
    proc = subprocess.run(  # noqa: S603
        [gh, "pr", "create", "--title", title, "--body", body],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, proc.stderr.strip() or proc.stdout.strip() or "gh pr create failed."
    return True, proc.stdout.strip() or "Pull request created."


def build_pr_body(task_id: str) -> str:
    """Markdown PR body scaffold including sections required by the workflow."""
    fill = _PR_BODY_FILL_IN_LINE
    lines = [
        "## Summary",
        "",
        fill,
        "",
        "## Implementation notes",
        "",
        fill,
        "",
        "## Tests run",
        "",
        fill,
        "",
        "## Commit list",
        "",
        "- (fill in — use `git log` on this branch)",
        "",
        "## Agent handoff / context notes",
        "",
        f"- Task id: `{task_id}`",
        "- See workspace context store for captured notes.",
        "",
        "## Known limitations",
        "",
        fill,
        "",
    ]
    return "\n".join(lines)


def _interactive_user_accepts_pr_prompt() -> bool:
    print("Build review is complete. Create a PR? [y/N]", flush=True)
    try:
        answer = input().strip().lower()
    except EOFError:
        answer = ""
    return answer in {"y", "yes"}


def _try_github_pr_or_write_fallback(workspace: Path, repo: Path, pr_title: str, body: str) -> None:
    """Create PR via gh when possible; otherwise write pr-body.md with diagnostics."""
    out_path = workspace / ".llm-plan-execute" / "pr-body.md"
    ok_auth, auth_reason = gh_available_and_authenticated()
    if not ok_auth:
        write_text_report(out_path, body)
        print(f"GitHub CLI unavailable or not authenticated ({auth_reason}). PR creation skipped.")
        print(f"Wrote proposed PR body to {out_path}")
        return
    created, msg = try_github_pr_create(repo, pr_title, body)
    if created:
        print(msg)
        return
    write_text_report(out_path, body)
    print(f"Could not create PR: {msg}")
    print(f"Wrote proposed PR body to {out_path}")


def maybe_offer_github_pr(
    *,
    workspace: Path,
    wf: WorkflowState,
    task_id: str,
    title_hint: str,
    interactive_tty: bool,
    create_pr_without_prompt: bool,
) -> None:
    """After build review, optionally create a GitHub PR via ``gh`` or print the body."""
    repo = find_git_root(workspace.resolve())
    body = build_pr_body(task_id)
    if wf.branch or wf.base_branch:
        body += "\n## Workflow\n\n"
        if wf.branch:
            body += f"- Branch: `{wf.branch}`\n"
        if wf.base_branch:
            body += f"- Base branch: `{wf.base_branch}`\n"

    pr_title = title_hint.strip().splitlines()[0][:120] if title_hint.strip() else f"Task {task_id}"

    out_path = workspace / ".llm-plan-execute" / "pr-body.md"

    if repo is None:
        write_text_report(out_path, body)
        print("Not in a git repository; PR creation skipped.")
        print(f"Wrote proposed PR body to {out_path}")
        return

    attempt_pr = False
    if create_pr_without_prompt:
        attempt_pr = True
    elif interactive_tty:
        attempt_pr = _interactive_user_accepts_pr_prompt()
    else:
        write_text_report(out_path, body)
        print("Non-interactive session: PR creation skipped (set build.create_pr to true to enable).")
        print(f"Wrote proposed PR body to {out_path}")
        return

    if not attempt_pr:
        write_text_report(out_path, body)
        print(f"Skipped PR creation. Draft body written to {out_path}")
        return

    _try_github_pr_or_write_fallback(workspace, repo, pr_title, body)


def write_text_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
