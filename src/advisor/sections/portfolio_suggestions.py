from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from textwrap import shorten

from src.utils.advisor_common import build_resume_corpus, compact_job_packets, load_prompt_config, normalize_text, post_openrouter_json

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "advisor" / "prompts.json"

DEFAULT_PROMPTS: dict[str, str] = {
    "advisor_portfolio_suggestions_system_prompt": "You are a senior career advisor. Return only valid JSON with a portfolio_suggestions array of short portfolio or project suggestion bullets, no markdown.",
    "advisor_portfolio_suggestions_user_prompt_template": (
        "Using the resume corpus and job packets, suggest 3-5 concise portfolio or project ideas that would improve hireability. Keep each bullet brief and practical.\n\n"
        "Resume corpus:\n{resume_corpus}\n\nJob packets summary:\n{job_packets}"
    ),
}


def _load_prompts() -> dict[str, str]:
    return load_prompt_config(PROMPT_FILE, DEFAULT_PROMPTS)


def _bullet_lines(items: list[str], heading: str, placeholder: str) -> list[str]:
    lines = [heading, ""]
    if not items:
        lines.append(f"- {placeholder}")
        return lines
    for item in items:
        text = shorten(normalize_text(item), width=120, placeholder="...")
        if text:
            lines.append(f"- {text}")
    return lines


def build_portfolio_suggestions_section(*, resume_modules_dir: Path, packets: list[dict[str, Any]], model_name: str | None) -> dict[str, Any]:
    if not packets:
        lines = _bullet_lines([], "## Portfolio or Project Suggestions", "Add job packets to generate project ideas.")
        return {"heading": "## Portfolio or Project Suggestions", "lines": lines, "items": []}

    model = model_name or "openai/gpt-4o-mini"
    prompts = _load_prompts()
    resume_corpus = build_resume_corpus(resume_modules_dir)
    compact_packets = compact_job_packets(packets)
    payload = post_openrouter_json(
        model=model,
        system_prompt=prompts["advisor_portfolio_suggestions_system_prompt"],
        user_prompt=prompts["advisor_portfolio_suggestions_user_prompt_template"].format(
            resume_corpus=resume_corpus,
            job_packets=json.dumps(compact_packets, ensure_ascii=False),
        ),
        schema_name="advisor_portfolio_suggestions",
        schema={
            "type": "object",
            "properties": {
                "portfolio_suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 6,
                },
            },
            "required": ["portfolio_suggestions"],
            "additionalProperties": False,
        },
    )

    items = [shorten(normalize_text(str(item)), width=120, placeholder="...") for item in payload.get("portfolio_suggestions", [])]
    items = [item for item in items if item]
    lines = _bullet_lines(items, "## Portfolio or Project Suggestions", "No portfolio or project suggestions yet.")
    return {"heading": "## Portfolio or Project Suggestions", "lines": lines, "items": items}
