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
    TRUNCATION_MARKER,
    ContextStore,
    ContextStoreError,
    ContextStoreLockedError,
    cosine_similarity,
    ensure_context_gitignore,
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


def test_truncate_content_marker_only_when_budget_equals_marker_length() -> None:
    marker_bytes = len(TRUNCATION_MARKER.encode("utf-8"))
    text = "z" * (marker_bytes + 1)
    assert truncate_content(text, max_bytes=marker_bytes) == TRUNCATION_MARKER


def test_cosine_similarity_zero_norm() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([3.0, 4.0], [0.0, 0.0]) == 0.0


def test_ensure_context_gitignore_appends_without_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / ".gitignore"
    path.write_text("keep\nthis", encoding="utf-8")
    written = ensure_context_gitignore(tmp_path)
    assert written == path
    body = path.read_text(encoding="utf-8")
    assert ".llm-plan-execute/" in body.splitlines()
    assert body.endswith(".llm-plan-execute/\n")


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


def test_embed_item_unknown_context_item_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ContextStore(tmp_path)
    store.init()

    def boom(_m: str) -> object:
        raise AssertionError("_load_fastembed_provider should not run without a row")

    monkeypatch.setattr("llm_plan_execute.context_store._load_fastembed_provider", boom)
    with pytest.raises(ContextStoreError, match="Unknown context item"):
        store.embed_item("does-not-exist")


def test_embed_item_raises_when_existing_embeddings_differ_from_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContextStore(tmp_path)
    store.init()
    store.add_context_item(task_id="t1", kind="note", path=None, content="hello", metadata={})
    conn = store._connect()
    try:
        cid = str(conn.execute("SELECT id FROM context_items LIMIT 1").fetchone()[0])
        conn.execute(
            """
            INSERT INTO embeddings(context_item_id, provider, model, dim, vector_blob, created_at)
            VALUES (?, 'fastembed', 'stored-model', 2, ?, 'now')
            """,
            (cid, sqlite3.Binary(b"\x00\x00\x00\x00")),
        )
        conn.commit()
    finally:
        conn.close()

    class Fake:
        name = "fastembed"

        def embed(self, _t: str) -> list[float]:
            return [1.0, 0.0]

    monkeypatch.setattr("llm_plan_execute.context_store._load_fastembed_provider", lambda _m: Fake())
    with pytest.raises(ContextStoreError, match="different provider/model"):
        store.embed_item(cid, model="other-model")


def test_search_context_raises_when_multiple_embedding_pairs(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.init()
    store.add_context_item(task_id="t1", kind="note", path=None, content="hello", metadata={})
    conn = store._connect()
    try:
        cid = str(conn.execute("SELECT id FROM context_items LIMIT 1").fetchone()[0])
        conn.execute(
            """
            INSERT INTO embeddings(context_item_id, provider, model, dim, vector_blob, created_at)
            VALUES (?, 'p1', 'm1', 2, ?, 'now')
            """,
            (cid, sqlite3.Binary(b"\x00\x00\x00\x00")),
        )
        conn.execute(
            """
            INSERT INTO embeddings(context_item_id, provider, model, dim, vector_blob, created_at)
            VALUES (?, 'p2', 'm2', 2, ?, 'now')
            """,
            (cid, sqlite3.Binary(b"\x00\x00\x00\x00")),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ContextStoreError, match="Multiple embedding"):
        store.search_context("hello", task_id="t1")


def test_search_context_whitespace_only_query_returns_empty(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.init()
    assert store.search_context("  \t ") == []


def test_search_context_min_score_excludes_fts_hits(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.init()
    store.add_context_item(task_id="t1", kind="note", path=None, content="hello world", metadata={})
    assert store.search_context("hello", task_id="t1", min_score=1.01) == []


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
