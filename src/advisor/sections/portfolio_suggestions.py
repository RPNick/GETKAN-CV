from __future__ import annotations

from typing import Any

from src.advisor.common import normalize_text


def _bullet_lines(items: list[str], heading: str, placeholder: str) -> list[str]:
    lines = [heading, ""]
    if not items:
        lines.append(f"- {placeholder}")
        return lines
    for item in items:
        text = normalize_text(item)
        if text:
            lines.append(f"- {text}")
    return lines


def render_portfolio_suggestions_section(*, items: list[str]) -> dict[str, Any]:
    return {"heading": "## Portfolio or Project Suggestions", "lines": _bullet_lines(items, "## Portfolio or Project Suggestions", "No portfolio or project suggestions yet."), "items": items}
