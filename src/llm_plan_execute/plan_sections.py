"""Split Markdown plans into ordered sections for step-through review."""

from __future__ import annotations


def split_plan_sections(markdown: str) -> list[tuple[str, str]]:
    """Return ``(heading, body)`` pairs using ``##`` headings as boundaries."""
    lines = markdown.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = "Overview"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_title, body))
        buffer = []

    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            flush()
            current_title = line[3:].strip() or "Section"
            buffer = []
            continue
        buffer.append(line)

    flush()

    if not sections:
        stripped = markdown.strip()
        if stripped:
            sections.append(("Plan", stripped))

    return sections
