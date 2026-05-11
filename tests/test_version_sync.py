"""Keep canonical package version (pyproject.toml) aligned with mirrors."""

from __future__ import annotations

import tomllib
from pathlib import Path

from llm_plan_execute import __version__

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    raw = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    data = tomllib.loads(raw)
    return data["project"]["version"]


def test_package_version_matches_pyproject() -> None:
    assert __version__ == _pyproject_version()


def test_uv_lock_pins_same_version_as_pyproject() -> None:
    py_ver = _pyproject_version()
    lock_data = tomllib.loads((_REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = lock_data.get("package", [])
    ours = [p for p in packages if p.get("name") == "llm-plan-execute"]
    assert len(ours) == 1
    assert ours[0]["version"] == py_ver
