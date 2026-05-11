"""CLI handlers for ``context`` subcommands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .context_store import ContextStore, ContextStoreError, ContextStoreLockedError, ensure_context_gitignore
from .context_types import HandoffPayload

HANDOFF_VERSION = 1


def register_context_parser(sub: Any) -> None:
    ctx = sub.add_parser("context", help="Workspace task context store (SQLite).")
    ctx_sub = ctx.add_subparsers(dest="context_command", required=True)

    ctx_sub.add_parser("init", help="Create context.sqlite and ensure .gitignore entry.")

    add = ctx_sub.add_parser("add", help="Append a context item.")
    add.add_argument("--task", required=True, metavar="ID")
    add.add_argument("--kind", required=True)
    add.add_argument("--path", default=None)
    src = add.add_mutually_exclusive_group(required=True)
    src.add_argument("--stdin", action="store_true", help="Read content from stdin.")
    src.add_argument("--file", type=Path, default=None)

    search = ctx_sub.add_parser("search", help="Search stored context.")
    search.add_argument("--task", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("-k", "--top-k", type=int, default=10)
    search.add_argument("--kind", default=None)
    search.add_argument("--min-score", type=float, default=None)

    handoff = ctx_sub.add_parser("handoff", help="Record a structured handoff payload.")
    handoff.add_argument("--task", required=True)
    handoff.add_argument("--to-agent", required=True, metavar="ROLE")
    handoff.add_argument("--from-agent", default=None)
    handoff.add_argument("--goal", default="")
    handoff.add_argument("--branch", default=None)

    summarize = ctx_sub.add_parser("summarize", help="Print a simple roll-up for a task.")
    summarize.add_argument("--task", required=True)

    reindex = ctx_sub.add_parser("reindex", help="Rebuild embeddings (requires optional fastembed extra).")
    reindex.add_argument("--task", default=None)

    ctx_sub.add_parser("prune", help="Placeholder for future retention controls.")


def run_context_command(args: argparse.Namespace, workspace: Path) -> int:
    store = ContextStore(workspace.resolve())
    handlers = {
        "init": _cmd_init,
        "add": _cmd_add,
        "search": _cmd_search,
        "handoff": _cmd_handoff,
        "summarize": _cmd_summarize,
        "reindex": _cmd_reindex,
        "prune": _cmd_prune,
    }
    handler = handlers.get(args.context_command)
    if handler is None:
        return 1
    try:
        return handler(args, workspace, store)
    except (ContextStoreError, ContextStoreLockedError, OSError, UnicodeDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_init(_args: argparse.Namespace, workspace: Path, store: ContextStore) -> int:
    store.init()
    store.db_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_context_gitignore(workspace.resolve())
    print(f"Initialized context database at {store.db_path}")
    return 0


def _cmd_add(args: argparse.Namespace, _workspace: Path, store: ContextStore) -> int:
    content = sys.stdin.read() if args.stdin else Path(args.file).read_text(encoding="utf-8")
    store.init()
    store.upsert_task(args.task, title=args.task)
    cid = store.add_context_item(
        task_id=args.task,
        kind=args.kind,
        path=args.path,
        content=content,
        metadata={"source": "cli"},
    )
    print(cid)
    return 0


def _cmd_search(args: argparse.Namespace, _workspace: Path, store: ContextStore) -> int:
    store.init()
    results = store.search_context(
        args.query,
        task_id=args.task,
        kind=args.kind,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    payload = [
        {
            "score": r.score,
            "provider": r.provider,
            "kind": r.item.kind,
            "id": r.item.id,
            "content": r.item.content,
        }
        for r in results
    ]
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_handoff(args: argparse.Namespace, _workspace: Path, store: ContextStore) -> int:
    store.init()
    payload = HandoffPayload(
        v=HANDOFF_VERSION,
        current_goal=args.goal,
        current_branch=args.branch,
        important_files=[],
        decisions_made=[],
        work_completed=[],
        tests_run=[],
        remaining_work=[],
        risks_blockers=[],
        suggested_next_agent_role=args.to_agent,
    )
    hid = store.add_handoff(
        task_id=args.task,
        payload=payload,
        from_agent=args.from_agent,
        to_agent=args.to_agent,
    )
    print(hid)
    return 0


def _cmd_summarize(args: argparse.Namespace, _workspace: Path, store: ContextStore) -> int:
    store.init()
    print(store.summarize_task(args.task), end="")
    return 0


def _cmd_reindex(args: argparse.Namespace, _workspace: Path, store: ContextStore) -> int:
    store.init()
    n = store.reindex_embeddings(task_id=args.task)
    print(f"Reindexed {n} item(s).")
    return 0


def _cmd_prune(_args: argparse.Namespace, _workspace: Path, store: ContextStore) -> int:
    store.init()
    print(store.prune_placeholder())
    return 0
