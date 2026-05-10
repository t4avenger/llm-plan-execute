"""Minimal HTML report generation without third-party dependencies."""

from __future__ import annotations

import html
from pathlib import Path

from .reporting import render_report
from .types import RunState

REPORT_HTML_NAME = "report.html"


def deterministic_html_report_path(run_dir: Path) -> Path:
    """Always ``{run_dir}/report.html`` (overwrite allowed)."""
    return run_dir / REPORT_HTML_NAME


def write_html_report(run: RunState) -> Path:
    """Write UTF-8 HTML alongside Markdown artifacts."""
    path = deterministic_html_report_path(run.run_dir)
    markdown = render_report(run)
    escaped = html.escape(markdown)
    title = html.escape(f"Run {run.run_id}")
    document = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        f'<meta charset="utf-8"><title>{title}</title>\n'
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; base-uri 'none'; style-src 'unsafe-inline'\">\n"
        "<style>body{font-family:system-ui,Segoe UI,sans-serif;margin:1.5rem;line-height:1.5}"
        "pre{white-space:pre-wrap;word-break:break-word}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{title}</h1>\n"
        f"<pre>{escaped}</pre>\n"
        "</body>\n"
        "</html>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
