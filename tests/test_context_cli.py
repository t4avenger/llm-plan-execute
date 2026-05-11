from __future__ import annotations

import argparse
import io
import json
import types
from pathlib import Path

from llm_plan_execute.context_cli import run_context_command


def _args(command: str, **kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(context_command=command, **kwargs)


def test_context_cli_init_add_search_summary_and_prune(tmp_path: Path, capsys, monkeypatch) -> None:
    assert run_context_command(_args("init"), tmp_path) == 0
    assert (tmp_path / ".llm-plan-execute" / "context.sqlite").exists()
    capsys.readouterr()

    monkeypatch.setattr("sys.stdin", io.StringIO("alpha beta gamma"))
    add = _args("add", task="t1", kind="note", path="notes.md", stdin=True, file=None)
    assert run_context_command(add, tmp_path) == 0
    item_id = capsys.readouterr().out.strip()
    assert item_id

    search = _args("search", task="t1", query="beta", kind=None, top_k=5, min_score=None)
    assert run_context_command(search, tmp_path) == 0
    results = json.loads(capsys.readouterr().out)
    assert results[0]["id"] == item_id

    assert run_context_command(_args("summarize", task="t1"), tmp_path) == 0
    assert "t1" in capsys.readouterr().out
    assert run_context_command(_args("prune"), tmp_path) == 0
    assert "future" in capsys.readouterr().out


def test_context_cli_add_from_file_and_handoff(tmp_path: Path, capsys) -> None:
    source = tmp_path / "context.txt"
    source.write_text("file context\n", encoding="utf-8")

    add = _args("add", task="t2", kind="file", path=None, stdin=False, file=source)
    assert run_context_command(add, tmp_path) == 0
    assert capsys.readouterr().out.strip()

    handoff = _args("handoff", task="t2", to_agent="builder", from_agent="planner", goal="ship", branch="feat/x")
    assert run_context_command(handoff, tmp_path) == 0
    assert capsys.readouterr().out.strip()


def test_context_cli_reports_errors(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.txt"
    add = _args("add", task="t3", kind="file", path=None, stdin=False, file=missing)
    assert run_context_command(add, tmp_path) == 1
    assert "Error:" in capsys.readouterr().err


def test_context_cli_unknown_command_returns_one(tmp_path: Path) -> None:
    assert run_context_command(_args("unknown"), tmp_path) == 1


def test_context_cli_reindex_embeds_items(monkeypatch, tmp_path: Path, capsys) -> None:
    """Covers context ``reindex`` command (fastembed path mocked)."""

    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts):
            for _text in texts:
                yield [0.1, 0.2]

    fake_module = types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
    monkeypatch.setattr("llm_plan_execute.context_store.importlib.import_module", lambda _name: fake_module)

    assert run_context_command(_args("init"), tmp_path) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO("embed me"))
    assert run_context_command(_args("add", task="t1", kind="note", path=None, stdin=True, file=None), tmp_path) == 0
    capsys.readouterr()

    assert run_context_command(_args("reindex", task=None), tmp_path) == 0
    out = capsys.readouterr().out
    assert "Reindexed" in out
    assert "item" in out.lower()
