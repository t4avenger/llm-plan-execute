import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bump_version.py"
_SPEC = importlib.util.spec_from_file_location("bump_version", _SCRIPT)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
next_version = _MODULE.next_version


def test_next_version_patch_minor_major() -> None:
    assert next_version("1.2.3", "patch") == "1.2.4"
    assert next_version("1.2.3", "minor") == "1.3.0"
    assert next_version("1.2.3", "major") == "2.0.0"


def test_next_version_prerelease() -> None:
    assert next_version("1.2.3", "patch", "rc.1") == "1.2.4-rc.1"
