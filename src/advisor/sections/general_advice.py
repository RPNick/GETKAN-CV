from __future__ import annotations

from typing import Any

from src.advisor.common import normalize_text


def _section_paragraph(summary: str, advice: str) -> str:
    parts: list[str] = []
    summary_text = normalize_text(summary)
    if summary_text:
        parts.append(f"Summary: {summary_text}.")

    advice_text = normalize_text(advice)
    if advice_text:
        parts.append(f"Advice: {advice_text}.")

    if not parts:
        parts.append("Summary: Add job packets to generate a tailored summary and advice.")

    return " ".join(parts)


def render_general_advice_section(*, summary: str, general_advice: str) -> dict[str, Any]:
    paragraph = _section_paragraph(summary, general_advice)
    return {
        "heading": "## General Advice and Summary",
        "lines": ["## General Advice and Summary", "", paragraph],
        "summary": summary,
        "general_advice": general_advice,
    }