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


def render_interview_prep_section(*, items: list[str]) -> dict[str, Any]:
    return {"heading": "## Interview Prep", "lines": _bullet_lines(items, "## Interview Prep", "Review the packets, role fit, and recent project stories."), "items": items}
