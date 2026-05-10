"""Structured build-review recommendations parsed from arbiter Markdown."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

_MARKER_OPEN = "<!-- llm-plan-execute:recommendations"
_MARKER_CLOSE = "-->"
_MIN_HEADING_BLOCKS = 2


@dataclass(frozen=True)
class BuildRecommendation:
    """Selectable unit produced after build review."""

    id: str
    title: str
    description: str
    status: str = "applicable"
    depends_on: tuple[str, ...] = ()


def parse_recommendations_from_summary(markdown: str) -> list[BuildRecommendation]:
    """Prefer embedded JSON; fall back to headings / bullets."""
    embedded = _parse_embedded_json(markdown)
    if embedded:
        return embedded
    heading_recs = _parse_heading_blocks(markdown)
    if heading_recs:
        return heading_recs
    return _parse_bullet_findings(markdown)


def expand_with_dependencies(selected: Sequence[str], recommendations: Sequence[BuildRecommendation]) -> list[str]:
    """Auto-include missing dependencies; preserves stable ordering."""
    by_id = {rec.id: rec for rec in recommendations}
    ordered: list[str] = []
    seen: set[str] = set()
    visiting: set[str] = set()

    def visit(rec_id: str) -> None:
        if rec_id in seen:
            return
        rec = by_id.get(rec_id)
        if rec is None:
            return
        if rec_id in visiting:
            return
        visiting.add(rec_id)
        for dependency in rec.depends_on:
            visit(dependency)
        visiting.discard(rec_id)
        seen.add(rec_id)
        ordered.append(rec_id)

    for rec_id in selected:
        visit(rec_id)
    return ordered


def selection_requires_missing_dependency(
    selected: Sequence[str], recommendations: Sequence[BuildRecommendation]
) -> str | None:
    """Return error message if a dependency is missing and could not be auto-resolved."""
    by_id = {rec.id: rec for rec in recommendations}
    selected_set = set(selected)
    for rec_id in selected:
        rec = by_id.get(rec_id)
        if rec is None:
            continue
        for dependency in rec.depends_on:
            if dependency not in selected_set:
                return f"Recommendation {rec_id!r} depends on {dependency!r}, which was not included in the selection."
    return None


def map_numeric_selection_to_ids(
    indices: Sequence[str], recommendations: Sequence[BuildRecommendation]
) -> tuple[str, ...]:
    """Map outputs of :func:`selection_parser.parse_index_selection` to recommendation IDs."""
    by_index = {str(i): rec.id for i, rec in enumerate(recommendations, start=1)}
    resolved: list[str] = []
    seen: set[str] = set()
    for key in indices:
        rec_id = by_index.get(key)
        if rec_id is None or rec_id in seen:
            continue
        seen.add(rec_id)
        resolved.append(rec_id)
    return tuple(resolved)


def _parse_embedded_json(markdown: str) -> list[BuildRecommendation]:
    if _MARKER_OPEN not in markdown:
        return []
    start = markdown.index(_MARKER_OPEN) + len(_MARKER_OPEN)
    end = markdown.find(_MARKER_CLOSE, start)
    if end == -1:
        return []
    payload = markdown[start:end].strip()
    try:
        raw_items = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw_items, list):
        return []
    recommendations: list[BuildRecommendation] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        rec_id = item.get("id")
        title = item.get("title")
        description = item.get("description", "")
        if not isinstance(rec_id, str) or not isinstance(title, str):
            continue
        status = item.get("status", "applicable")
        depends = item.get("depends_on", [])
        depends_tuple = tuple(str(dep) for dep in depends) if isinstance(depends, list) else ()
        recommendations.append(
            BuildRecommendation(
                id=rec_id,
                title=title,
                description=str(description),
                status=str(status),
                depends_on=depends_tuple,
            )
        )
    return recommendations


_HEADING_PATTERN = re.compile(r"^###\s+(.+)$", re.MULTILINE)


def _parse_heading_blocks(markdown: str) -> list[BuildRecommendation]:
    matches = list(_HEADING_PATTERN.finditer(markdown))
    if len(matches) < _MIN_HEADING_BLOCKS:
        return []
    recommendations: list[BuildRecommendation] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        rec_id = f"heading-{index + 1}"
        recommendations.append(BuildRecommendation(id=rec_id, title=title, description=body))
    return recommendations


def _parse_bullet_findings(markdown: str) -> list[BuildRecommendation]:
    lines = markdown.splitlines()
    bullets: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    if not bullets:
        snippet = markdown.strip()
        if snippet:
            return [BuildRecommendation(id="finding-1", title="Build review summary", description=snippet)]
        return []
    return [
        BuildRecommendation(id=f"finding-{index}", title=text[:80], description=text)
        for index, text in enumerate(bullets, start=1)
    ]
