from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from textwrap import shorten

from src.utils.advisor_common import build_resume_corpus, compact_job_packets, load_prompt_config, normalize_text, post_openrouter_json

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "advisor" / "prompts.json"

DEFAULT_PROMPTS: dict[str, str] = {
    "advisor_resume_recommendation_system_prompt": "You are a senior career advisor. Return only valid JSON with a resume_recommendations array of concise resume improvement suggestions, no markdown.",
    "advisor_resume_recommendation_user_prompt_template": (
        "Based on the current resume corpus, job packets, and skills summary, recommend resume changes that improve hireability without inventing more work experience. Focus on sections, bullets, wording, and skills presentation.\n\n"
        "Resume corpus:\n{resume_corpus}\n\nJob packets summary:\n{job_packets}"
    ),
}


def _load_prompts() -> dict[str, str]:
    return load_prompt_config(PROMPT_FILE, DEFAULT_PROMPTS)


def _recommendation_bullet_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Resume Recommendation", ""]
    if not rows:
        lines.append("- No resume recommendations available.")
        return lines

    for row in rows:
        summary = shorten(f"{row['recommendation']}. {row['reason']}", width=100, placeholder="...")
        lines.append(f"- {row['area']}: {summary} (P{row['priority']})")
    return lines


def build_resume_recommendation_section(*, resume_modules_dir: Path, packets: list[dict[str, Any]], model_name: str | None) -> dict[str, Any]:
    if not packets:
        lines = _recommendation_bullet_lines([])
        return {"heading": "## Resume Recommendation", "lines": lines, "rows": []}

    model = model_name or "openai/gpt-4o-mini"
    prompts = _load_prompts()
    resume_corpus = build_resume_corpus(resume_modules_dir)
    compact_packets = compact_job_packets(packets)
    payload = post_openrouter_json(
        model=model,
        system_prompt=prompts["advisor_resume_recommendation_system_prompt"],
        user_prompt=prompts["advisor_resume_recommendation_user_prompt_template"].format(
            resume_corpus=resume_corpus,
            job_packets=json.dumps(compact_packets, ensure_ascii=False),
        ),
        schema_name="advisor_resume_recommendations",
        schema={
            "type": "object",
            "properties": {
                "resume_recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "area": {"type": "string"},
                            "recommendation": {"type": "string"},
                            "reason": {"type": "string"},
                            "priority": {"type": "integer"},
                        },
                        "required": ["area", "recommendation", "reason", "priority"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": 6,
                },
            },
            "required": ["resume_recommendations"],
            "additionalProperties": False,
        },
    )

    rows: list[dict[str, Any]] = []
    for item in payload.get("resume_recommendations", [])[:6]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "area": normalize_text(str(item.get("area") or "")),
                "recommendation": normalize_text(str(item.get("recommendation") or "")),
                "reason": normalize_text(str(item.get("reason") or "")),
                "priority": int(item.get("priority") or 0),
            }
        )
    rows.sort(key=lambda row: (-row["priority"], row["area"].lower()))
    lines = _recommendation_bullet_lines(rows)
    return {"heading": "## Resume Recommendation", "lines": lines, "rows": rows}
