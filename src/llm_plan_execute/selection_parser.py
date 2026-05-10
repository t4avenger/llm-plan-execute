"""Parse multi-select recommendation input with explicit whitespace/duplicate/out-of-range rules."""

from __future__ import annotations


def parse_index_selection(raw: str, max_index: int) -> tuple[str, ...]:
    """Parse 1-based indices into numeric string tokens.

    - Whitespace around commas is ignored.
    - Duplicates are removed while preserving first-seen order.
    - Ranges are not supported; tokens containing ``-`` are skipped.
    - Non-integer tokens are skipped.
    - Out-of-range integers are skipped.
    - Empty submission returns an empty tuple.
    """
    text = raw.strip()
    if not text:
        return ()

    seen: set[int] = set()
    ordered: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token or "-" in token:
            continue
        if not token.isdigit():
            continue
        value = int(token)
        if value < 1 or value > max_index:
            continue
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)

    return tuple(str(index) for index in ordered)
