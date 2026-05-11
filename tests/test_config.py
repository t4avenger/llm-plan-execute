"""Tests for config parsing and validation (Sonar new-code coverage)."""

from __future__ import annotations

from pathlib import Path

from llm_plan_execute.config import (
    ConfigIssue,
    ConfigValidation,
    ExecutionConfig,
    parse_config,
    validate_config_data,
    validate_config_file,
)


def test_execution_config_uses_default_mode_for_other_roles() -> None:
    ex = ExecutionConfig(default_mode="full-access", planning_mode="read-only")
    assert ex.mode_for_role("unknown_future_role") == "full-access"


def test_config_validation_ok_false_when_errors_present() -> None:
    bad = ConfigValidation(errors=(ConfigIssue("error", "p", "m"),), warnings=())
    assert bad.ok is False


def test_parse_config_coerces_non_object_execution_and_build(tmp_path: Path) -> None:
    cfg = parse_config(
        {
            "providers": [],
            "dry_run": True,
            "execution": "bad",
            "build": ["not", "a", "dict"],
        },
        workspace=tmp_path,
    )
    assert isinstance(cfg.execution, ExecutionConfig)
    assert cfg.build.base_branch is None


def test_parse_config_execution_phases_and_dirs_fallback(tmp_path: Path) -> None:
    cfg = parse_config(
        {
            "providers": [],
            "dry_run": True,
            "execution": {
                "phases": "not-a-dict",
                "writable_dirs": "not-a-list",
            },
        },
        workspace=tmp_path,
    )
    assert cfg.execution.planning_mode == "read-only"
    assert cfg.execution.writable_dirs == ()


def test_validate_config_data_root_must_be_object() -> None:
    v = validate_config_data([], require_providers=False)
    assert not v.ok
    assert any(i.path == "$" for i in v.errors)


def test_validate_config_data_providers_must_be_list() -> None:
    v = validate_config_data({"providers": "codex", "dry_run": True})
    assert not v.ok
    assert any(i.path == "providers" for i in v.errors)


def test_validate_config_data_reports_execution_build_field_errors() -> None:
    v = validate_config_data(
        {
            "providers": [{"name": "codex", "command": "codex", "enabled": True, "models": []}],
            "execution": "not-an-object",
            "build": 42,
        }
    )
    paths = {i.path for i in v.errors}
    assert "execution" in paths
    assert "build" in paths


def test_validate_config_file_missing_with_dry_run_succeeds(tmp_path: Path) -> None:
    # No config file on disk — dry_run uses bundled sample config path in validation flow
    v = validate_config_file(None, workspace=tmp_path, dry_run=True)
    assert v.ok


def test_validate_config_file_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / ".llm-plan-execute" / "config.json"
    p.parent.mkdir(parents=True)
    p.write_text("{ not json", encoding="utf-8")
    v = validate_config_file(p, workspace=tmp_path)
    assert not v.ok
    assert any("invalid JSON" in i.message for i in v.errors)
