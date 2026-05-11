"""Tests for SQLite context store."""

from __future__ import annotations

import builtins
import sqlite3
import threading
import time
import types
from pathlib import Path

import pytest

from llm_plan_execute.context_store import (
    DEFAULT_MAX_CONTENT_BYTES,
    ContextStore,
    ContextStoreError,
    ContextStoreLockedError,
    redact_secrets,
    row_to_embedding_record,
    truncate_content,
)
from llm_plan_execute.git_flow import build_pr_body

EXPECTED_EMBEDDING_COUNT = 2
EXPECTED_EMBEDDING_DIM = 2


def test_redact_and_truncate() -> None:
    assert "sk-abc" not in redact_secrets("token=sk-abc123456789012345678901234567890")
    long_text = "x" * (DEFAULT_MAX_CONTENT_BYTES + 50)
    cut = truncate_content(long_text)
    assert len(cut.encode("utf-8")) <= DEFAULT_MAX_CONTENT_BYTES + 200
    assert "truncated" in cut.lower()


def test_context_init_search_fts(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.init()
    store.upsert_task("t1", title="hello")
    store.add_context_item(task_id="t1", kind="plan", path=None, content="alpha beta gamma", metadata={})
    rows = store.search_context("beta", task_id="t1")
    assert rows
    assert rows[0].provider == "fts5"


def test_schema_version_row(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.init()
    conn = store._connect()
    try:
        v = conn.execute("SELECT version FROM schema_version").fetchone()
        assert int(v[0]) == 1
    finally:
        conn.close()


def test_concurrent_writer_lock(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.init()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def first() -> None:
        lock = store._acquire_writer_lock(blocking=True)
        try:
            barrier.wait()
            time.sleep(0.3)
        finally:
            lock.release()

    def second() -> None:
        barrier.wait()
        try:
            store._acquire_writer_lock(blocking=False)
        except ContextStoreLockedError as exc:
            errors.append(exc)

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors


def test_embedding_import_error_when_fastembed_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ContextStore(tmp_path)
    store.init()
    store.add_context_item(task_id="t1", kind="x", path=None, content="hello world", metadata={})
    row = store._connect().execute("SELECT id FROM context_items LIMIT 1").fetchone()
    assert row is not None

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "fastembed":
            raise ImportError("no fastembed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ContextStoreError, match="fastembed"):
        store.reindex_embeddings()


def test_search_requires_reindex_on_model_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ContextStore(tmp_path)
    store.init()
    store.add_context_item(task_id="t1", kind="note", path=None, content="hello", metadata={})

    class FakeEmb:
        name = "fastembed"
        model = "m1"

        def available(self) -> bool:
            return True

        def embed(self, _text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    monkeypatch.setattr("llm_plan_execute.context_store._load_fastembed_provider", lambda _m: FakeEmb())

    row = store._connect().execute("SELECT id FROM context_items LIMIT 1").fetchone()
    store.embed_item(str(row[0]))
    with pytest.raises(ContextStoreError, match="reindex"):
        store.search_context("hello", embedding_model="some-other-model")


def test_reindex_embeddings_and_vector_search_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ContextStore(tmp_path)
    store.init()
    store.add_context_item(task_id="t1", kind="note", path=None, content="alpha", metadata={})
    store.add_context_item(task_id="t1", kind="note", path=None, content="beta", metadata={})

    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts):
            for text in texts:
                yield [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    fake_module = types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
    monkeypatch.setattr("llm_plan_execute.context_store.importlib.import_module", lambda _name: fake_module)

    assert store.reindex_embeddings(task_id="t1") == EXPECTED_EMBEDDING_COUNT
    results = store.search_context("alpha", task_id="t1", use_embeddings=True)
    assert results[0].item.content == "alpha"
    assert results[0].provider == "fastembed"


def test_row_to_item_handles_bad_metadata_and_embedding_record(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.init()
    conn = store._connect()
    try:
        conn.execute(
            """
            INSERT INTO tasks(id, title, status, created_at, updated_at)
            VALUES ('t', 't', 'active', 'now', 'now')
            """
        )
        conn.execute(
            """
            INSERT INTO context_items(id, task_id, kind, path, content, metadata_json, created_at)
            VALUES ('c', 't', 'note', NULL, 'body', '{bad json', 'now')
            """
        )
        item_row = conn.execute("SELECT * FROM context_items WHERE id = 'c'").fetchone()
        assert store._row_to_item(item_row).metadata == {}

        conn.execute(
            """
            INSERT INTO embeddings(context_item_id, provider, model, dim, vector_blob, created_at)
            VALUES ('c', 'p', 'm', 2, ?, 'now')
            """,
            (sqlite3.Binary(b"12345678"),),
        )
        emb_row = conn.execute("SELECT * FROM embeddings WHERE context_item_id = 'c'").fetchone()
        record = row_to_embedding_record(emb_row)
        assert record.provider == "p"
        assert record.dim == EXPECTED_EMBEDDING_DIM
    finally:
        conn.close()


def test_context_sqlite_migration_guard(tmp_path: Path) -> None:
    db = tmp_path / ".llm-plan-execute" / "context.sqlite"
    db.parent.mkdir(parents=True)
    conn = __import__("sqlite3").connect(db)
    conn.execute("CREATE TABLE schema_version (version INTEGER)")
    conn.execute("INSERT INTO schema_version VALUES (99)")
    conn.commit()
    conn.close()
    store = ContextStore(tmp_path)
    with pytest.raises(ContextStoreError, match="Unsupported"):
        store.init()


def test_build_pr_body_sections() -> None:
    body = build_pr_body("tid")
    for heading in ("## Summary", "## Implementation notes", "## Tests run", "## Commit list", "## Known limitations"):
        assert heading in body
