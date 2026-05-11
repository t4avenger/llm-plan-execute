"""Workspace-scoped SQLite store for task context, search, and embeddings."""

from __future__ import annotations

import fcntl
import importlib
import json
import math
import re
import sqlite3
import struct
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .context_types import ContextItem, EmbeddingRecord, HandoffPayload, SearchResult

CONTEXT_SQLITE_INTERNAL_VERSION = 1
DEFAULT_MAX_CONTENT_BYTES = 64 * 1024
TRUNCATION_MARKER = "\n\n[truncated by llm-plan-execute]\n"

DEFAULT_EMBEDDING_PROVIDER = "fastembed"
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

_RE_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password|bearer)\s*[:=]\s*\S+"),
    re.compile(r"sk-[a-zA-Z0-9]{10,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
    re.compile(r"xox[baprs]-[a-zA-Z0-9-]+"),
)


def redact_secrets(text: str, *, patterns: tuple[re.Pattern[str], ...] = _RE_SECRET_PATTERNS) -> str:
    """Best-effort secret redaction before persistence."""
    redacted = text
    for pat in patterns:
        redacted = pat.sub(lambda _m: "[REDACTED]", redacted)
    return redacted


def truncate_content(text: str, max_bytes: int = DEFAULT_MAX_CONTENT_BYTES) -> str:
    """Truncate UTF-8 by bytes with a marker (best-effort)."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    cut = raw[: max_bytes - len(TRUNCATION_MARKER.encode("utf-8"))]
    while cut:
        try:
            return cut.decode("utf-8") + TRUNCATION_MARKER
        except UnicodeDecodeError:
            cut = cut[:-1]
    return TRUNCATION_MARKER


def context_db_path(workspace: Path) -> Path:
    return (workspace / ".llm-plan-execute" / "context.sqlite").resolve()


def ensure_context_gitignore(workspace: Path) -> Path | None:
    """Append ``.llm-plan-execute/`` to workspace ``.gitignore`` when missing."""
    path = workspace / ".gitignore"
    line = ".llm-plan-execute/"
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if line in text.splitlines() or any(line.rstrip("/") in ln for ln in text.splitlines()):
            return None
        append = "" if text.endswith("\n") or not text else "\n"
        path.write_text(text + append + line + "\n", encoding="utf-8")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line + "\n", encoding="utf-8")
    return path


class ContextStoreLockedError(RuntimeError):
    """Another process holds the writer lock."""


class ContextStoreError(RuntimeError):
    """Logical error from the context store."""


def _iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _vec_to_blob(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _blob_to_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _load_fastembed_provider(model: str) -> Any:
    """Import optional dependency only when embeddings are requested."""
    try:
        fastembed = importlib.import_module("fastembed")
    except ImportError as exc:
        raise ContextStoreError(
            "fastembed is not installed. Install the optional extra (e.g. pip install "
            "'llm-plan-execute[fastembed]') or use lexical search only."
        ) from exc
    text_embedding = fastembed.TextEmbedding

    class _FastEmbedWrapper:
        name = "fastembed"

        def __init__(self, model_name: str) -> None:
            self.model = model_name
            self._impl = text_embedding(model_name=model_name)

        def available(self) -> bool:
            return True

        def embed(self, text: str) -> list[float]:
            vectors = list(self._impl.embed(text))
            if not vectors:
                raise ContextStoreError("Embedding provider returned no vectors.")
            first = vectors[0]
            if hasattr(first, "tolist"):
                return [float(x) for x in first.tolist()]
            return [float(x) for x in list(first)]

    return _FastEmbedWrapper(model)


class ContextStore:
    """SQLite-backed task context with FTS5 fallback and optional embeddings."""

    def __init__(
        self,
        workspace: Path,
        *,
        max_content_bytes: int = DEFAULT_MAX_CONTENT_BYTES,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.db_path = context_db_path(self.workspace)
        self.max_content_bytes = max_content_bytes
        self.redact = redact or redact_secrets

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=60.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _writer_lock_path(self) -> Path:
        return self.db_path.parent / ".context-write.lock"

    def _acquire_writer_lock(self, *, blocking: bool = True) -> Any:
        lock_path = self._writer_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115 - manual lifecycle
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(fh.fileno(), flags)
        except BlockingIOError as exc:
            fh.close()
            raise ContextStoreLockedError("Context store is locked by another writer.") from exc
        return fh

    def init(self) -> None:
        """Create database file and schema."""
        fh = self._acquire_writer_lock()
        try:
            conn = self._connect()
            try:
                self._init_schema(conn)
            finally:
                conn.close()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
              version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              branch TEXT,
              base_branch TEXT,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agents (
              id TEXT PRIMARY KEY,
              role TEXT NOT NULL,
              status TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS handoffs (
              id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL REFERENCES tasks(id),
              from_agent TEXT,
              to_agent TEXT,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS context_items (
              id TEXT PRIMARY KEY,
              task_id TEXT REFERENCES tasks(id),
              kind TEXT NOT NULL,
              path TEXT,
              content TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS embeddings (
              context_item_id TEXT NOT NULL REFERENCES context_items(id),
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              dim INTEGER NOT NULL,
              vector_blob BLOB NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY(context_item_id, provider, model)
            );

            CREATE INDEX IF NOT EXISTS idx_context_items_task ON context_items(task_id);
            CREATE INDEX IF NOT EXISTS idx_context_items_kind ON context_items(kind);
            CREATE INDEX IF NOT EXISTS idx_embeddings_provider_model ON embeddings(provider, model);

            CREATE VIRTUAL TABLE IF NOT EXISTS context_items_fts USING fts5(
              item_id UNINDEXED,
              content,
              tokenize = 'porter unicode61'
            );
            """
        )
        rows = conn.execute("SELECT rowid, version FROM schema_version").fetchall()
        if not rows:
            conn.execute("INSERT INTO schema_version(version) VALUES (?)", (CONTEXT_SQLITE_INTERNAL_VERSION,))
            return
        if any(int(row["version"]) != CONTEXT_SQLITE_INTERNAL_VERSION for row in rows):
            raise ContextStoreError("Unsupported context.sqlite schema; remove or migrate the database.")
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", (CONTEXT_SQLITE_INTERNAL_VERSION,))

    def upsert_task(
        self,
        task_id: str,
        *,
        title: str,
        status: str = "active",
        branch: str | None = None,
        base_branch: str | None = None,
    ) -> None:
        now = _iso_now()
        fh = self._acquire_writer_lock()
        try:
            conn = self._connect()
            try:
                self._init_schema(conn)
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
                if row:
                    conn.execute(
                        """
                        UPDATE tasks SET title = ?, branch = ?, base_branch = ?, status = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (title, branch, base_branch, status, now, task_id),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO tasks(id, title, branch, base_branch, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (task_id, title, branch, base_branch, status, now, now),
                    )
                conn.execute("COMMIT")
            finally:
                conn.close()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    def add_context_item(
        self,
        *,
        task_id: str | None,
        kind: str,
        path: str | None,
        content: str,
        metadata: dict[str, Any],
        item_id: str | None = None,
    ) -> str:
        rid = item_id or str(uuid.uuid4())
        safe = self.redact(content)
        safe = truncate_content(safe, self.max_content_bytes)
        meta = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        now = _iso_now()
        fh = self._acquire_writer_lock()
        try:
            conn = self._connect()
            try:
                self._init_schema(conn)
                conn.execute("BEGIN IMMEDIATE")
                if task_id:
                    conn.execute(
                        """
                        INSERT INTO tasks(id, title, branch, base_branch, status, created_at, updated_at)
                        VALUES (?, ?, NULL, NULL, 'active', ?, ?)
                        ON CONFLICT(id) DO NOTHING
                        """,
                        (task_id, task_id, now, now),
                    )
                conn.execute(
                    """
                    INSERT INTO context_items(id, task_id, kind, path, content, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (rid, task_id, kind, path, safe, meta, now),
                )
                conn.execute(
                    "INSERT INTO context_items_fts(item_id, content) VALUES (?, ?)",
                    (rid, safe),
                )
                conn.execute("COMMIT")
            finally:
                conn.close()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
        return rid

    def add_handoff(
        self,
        *,
        task_id: str,
        payload: HandoffPayload,
        from_agent: str | None,
        to_agent: str | None,
        handoff_id: str | None = None,
    ) -> str:
        hid = handoff_id or str(uuid.uuid4())
        body = json.dumps(asdict(payload), separators=(",", ":"), sort_keys=True)
        now = _iso_now()
        fh = self._acquire_writer_lock()
        try:
            conn = self._connect()
            try:
                self._init_schema(conn)
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO tasks(id, title, branch, base_branch, status, created_at, updated_at)
                    VALUES (?, ?, NULL, NULL, 'active', ?, ?)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (task_id, task_id, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO handoffs(id, task_id, from_agent, to_agent, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (hid, task_id, from_agent, to_agent, body, now),
                )
                conn.execute("COMMIT")
            finally:
                conn.close()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
        return hid

    def _embedding_pairs(self, conn: sqlite3.Connection) -> list[tuple[str, str]]:
        rows = conn.execute("SELECT DISTINCT provider, model FROM embeddings ORDER BY provider, model").fetchall()
        return [(str(r["provider"]), str(r["model"])) for r in rows]

    def embed_item(
        self,
        item_id: str,
        *,
        provider_name: str = DEFAULT_EMBEDDING_PROVIDER,
        model: str | None = None,
    ) -> None:
        """Create or replace embedding row for one context item."""
        model = model or DEFAULT_FASTEMBED_MODEL
        fh = self._acquire_writer_lock()
        try:
            conn = self._connect()
            try:
                self._init_schema(conn)
                row = conn.execute("SELECT content FROM context_items WHERE id = ?", (item_id,)).fetchone()
                if row is None:
                    raise ContextStoreError(f"Unknown context item {item_id!r}.")
                text = str(row["content"])
                pairs = self._embedding_pairs(conn)
                if pairs and any((p, m) != (provider_name, model) for p, m in pairs):
                    raise ContextStoreError(
                        "Embeddings use a different provider/model than requested. Run `context reindex` first."
                    )
                provider = _load_fastembed_provider(model)
                vec = provider.embed(text)
                blob = _vec_to_blob(vec)
                dim = len(vec)
                now = _iso_now()
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO embeddings(context_item_id, provider, model, dim, vector_blob, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(context_item_id, provider, model) DO UPDATE SET
                      dim = excluded.dim,
                      vector_blob = excluded.vector_blob,
                      created_at = excluded.created_at
                    """,
                    (item_id, provider_name, model, dim, blob, now),
                )
                conn.execute("COMMIT")
            finally:
                conn.close()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    def reindex_embeddings(self, *, task_id: str | None = None, model: str | None = None) -> int:
        """Recompute embeddings for items using the configured fastembed model."""
        model = model or DEFAULT_FASTEMBED_MODEL
        provider_name = DEFAULT_EMBEDDING_PROVIDER
        provider = _load_fastembed_provider(model)
        fh = self._acquire_writer_lock()
        count = 0
        try:
            conn = self._connect()
            try:
                self._init_schema(conn)
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM embeddings")
                sql = "SELECT id, content FROM context_items"
                params: tuple[Any, ...] = ()
                if task_id:
                    sql += " WHERE task_id = ?"
                    params = (task_id,)
                rows = conn.execute(sql, params).fetchall()
                now = _iso_now()
                for row in rows:
                    cid = str(row["id"])
                    vec = provider.embed(str(row["content"]))
                    conn.execute(
                        """
                        INSERT INTO embeddings(context_item_id, provider, model, dim, vector_blob, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (cid, provider_name, model, len(vec), _vec_to_blob(vec), now),
                    )
                    count += 1
                conn.execute("COMMIT")
            finally:
                conn.close()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
        return count

    def search_context(
        self,
        query: str,
        *,
        task_id: str | None = None,
        kind: str | None = None,
        top_k: int = 10,
        min_score: float | None = None,
        use_embeddings: bool = True,
        embedding_model: str | None = None,
    ) -> list[SearchResult]:
        embedding_model = embedding_model or DEFAULT_FASTEMBED_MODEL
        fh = self._acquire_writer_lock(blocking=True)
        try:
            conn = self._connect()
            try:
                self._init_schema(conn)
                pairs = self._embedding_pairs(conn)
                if use_embeddings and pairs:
                    if len(pairs) > 1:
                        raise ContextStoreError(
                            "Multiple embedding provider/model pairs exist. Run `context reindex` to rebuild."
                        )
                    epair = pairs[0]
                    if epair != (DEFAULT_EMBEDDING_PROVIDER, embedding_model):
                        raise ContextStoreError(
                            f"Stored embeddings are for {epair[0]}/{epair[1]} but search expects "
                            f"{DEFAULT_EMBEDDING_PROVIDER}/{embedding_model}. Run `context reindex`."
                        )
                    provider = _load_fastembed_provider(epair[1])
                    qvec = provider.embed(query)
                    return self._vector_search(conn, qvec, task_id=task_id, kind=kind, top_k=top_k, min_score=min_score)
                return self._fts_search(conn, query, task_id=task_id, kind=kind, top_k=top_k, min_score=min_score)
            finally:
                conn.close()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    def _row_to_item(self, row: sqlite3.Row) -> ContextItem:
        meta_raw = row["metadata_json"]
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
        except json.JSONDecodeError:
            meta = {}
        return ContextItem(
            id=str(row["id"]),
            task_id=row["task_id"],
            kind=str(row["kind"]),
            path=row["path"],
            content=str(row["content"]),
            metadata=meta if isinstance(meta, dict) else {},
            created_at=str(row["created_at"]),
        )

    def _fts_search(
        self,
        conn: sqlite3.Connection,
        query: str,
        *,
        task_id: str | None,
        kind: str | None,
        top_k: int,
        min_score: float | None,
    ) -> list[SearchResult]:
        if not query.strip():
            return []
        match_expr = " ".join(f'"{part}"' for part in query.replace('"', "").split() if part)
        if not match_expr:
            match_expr = query
        sql = """
          SELECT c.*, bm25(context_items_fts) AS rank
          FROM context_items_fts
          JOIN context_items c ON c.id = context_items_fts.item_id
          WHERE context_items_fts MATCH ?
        """
        params: list[Any] = [match_expr]
        if task_id:
            sql += " AND c.task_id = ?"
            params.append(task_id)
        if kind:
            sql += " AND c.kind = ?"
            params.append(kind)
        sql += " ORDER BY rank LIMIT ?"
        params.append(top_k)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        results: list[SearchResult] = []
        for row in rows:
            rank = float(row["rank"])
            score = 1.0 / (1.0 + max(rank, 0.0))
            if min_score is not None and score < min_score:
                continue
            item = self._row_to_item(row)
            results.append(SearchResult(item=item, score=score, provider="fts5"))
        return results

    def _vector_search(
        self,
        conn: sqlite3.Connection,
        query_vec: list[float],
        *,
        task_id: str | None,
        kind: str | None,
        top_k: int,
        min_score: float | None,
    ) -> list[SearchResult]:
        sql = """
          SELECT c.*, e.vector_blob, e.provider
          FROM embeddings e
          JOIN context_items c ON c.id = e.context_item_id
          WHERE 1=1
        """
        params: list[Any] = []
        if task_id:
            sql += " AND c.task_id = ?"
            params.append(task_id)
        if kind:
            sql += " AND c.kind = ?"
            params.append(kind)
        rows = conn.execute(sql, params).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            blob = row["vector_blob"]
            if not isinstance(blob, bytes):
                continue
            vec = _blob_to_vec(blob)
            sim = cosine_similarity(query_vec, vec)
            scored.append((sim, row))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[SearchResult] = []
        for sim, row in scored[:top_k]:
            if min_score is not None and sim < min_score:
                continue
            item = self._row_to_item(row)
            out.append(SearchResult(item=item, score=sim, provider=str(row["provider"])))
        return out

    def summarize_task(self, task_id: str, *, limit: int = 40) -> str:
        fh = self._acquire_writer_lock()
        try:
            conn = self._connect()
            try:
                self._init_schema(conn)
                rows = conn.execute(
                    """
                    SELECT kind, content, created_at FROM context_items
                    WHERE task_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (task_id, limit),
                ).fetchall()
            finally:
                conn.close()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
        lines = [f"# Task {task_id} summary", ""]
        for row in reversed(rows):
            lines.append(f"## {row['kind']} @ {row['created_at']}")
            lines.append("")
            lines.append(str(row["content"])[:2000])
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def prune_placeholder(self) -> str:
        """Reserved for future retention policy."""
        return "context prune is not implemented in v1 (reserved for future retention policy)."


def row_to_embedding_record(row: sqlite3.Row) -> EmbeddingRecord:
    return EmbeddingRecord(
        context_item_id=str(row["context_item_id"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        dim=int(row["dim"]),
        vector_blob=bytes(row["vector_blob"]),
        created_at=str(row["created_at"]),
    )
