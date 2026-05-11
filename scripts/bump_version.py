from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_INIT = ROOT / "src" / "llm_plan_execute" / "__init__.py"
UV_LOCK = ROOT / "uv.lock"
CHANGELOG = ROOT / "CHANGELOG.md"


def next_version(current: str, part: str, prerelease: str | None = None) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-.].*)?", current)
    if not match:
        raise ValueError(f"Unsupported version {current!r}; expected MAJOR.MINOR.PATCH.")
    major, minor, patch = (int(group) for group in match.groups())
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError("part must be one of: major, minor, patch")
    version = f"{major}.{minor}.{patch}"
    return f"{version}-{prerelease}" if prerelease else version


def read_pyproject_version() -> str:
    raw = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', raw, flags=re.MULTILINE)
    if not match:
        raise ValueError("Could not find project.version in pyproject.toml")
    return match.group(1)


def replace_once(path: Path, pattern: str, replacement: str) -> None:
    raw = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, raw, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one replacement in {path}")
    path.write_text(updated, encoding="utf-8")


def bump_files(new_version: str) -> None:
    replace_once(PYPROJECT, r'^version = "[^"]+"$', f'version = "{new_version}"')
    replace_once(PACKAGE_INIT, r'^__version__ = "[^"]+"$', f'__version__ = "{new_version}"')
    replace_once(
        UV_LOCK,
        r'(\[\[package\]\]\nname = "llm-plan-execute"\nversion = )"[^"]+"',
        rf'\1"{new_version}"',
    )
    raw = CHANGELOG.read_text(encoding="utf-8")
    marker = "# Changelog\n\n"
    if f"## {new_version}\n" not in raw:
        raw = raw.replace(marker, f"{marker}## {new_version}\n\n- Release notes pending.\n\n", 1)
        CHANGELOG.write_text(raw, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=("patch", "minor", "major"))
    parser.add_argument("--prerelease", default=None)
    args = parser.parse_args()
    current = read_pyproject_version()
    new_version = next_version(current, args.part, args.prerelease)
    bump_files(new_version)
    print(new_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
