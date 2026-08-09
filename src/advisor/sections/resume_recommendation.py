from __future__ import annotations

from typing import Any

from src.advisor.common import normalize_text


def _recommendation_bullet_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Resume Recommendation", ""]
    if not rows:
        lines.append("- No resume recommendations available.")
        return lines

    for row in rows:
        summary = f"{row['recommendation']}. {row['reason']}".strip()
        lines.append(f"- {row['area']}: {summary} (P{row['priority']})")
    return lines


def render_resume_recommendation_section(*, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"heading": "## Resume Recommendation", "lines": _recommendation_bullet_lines(rows), "rows": rows}
