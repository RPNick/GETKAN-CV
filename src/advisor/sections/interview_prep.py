from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from textwrap import shorten

from src.utils.advisor_common import build_resume_corpus, compact_job_packets, load_prompt_config, normalize_text, post_openrouter_json

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "advisor" / "prompts.json"

DEFAULT_PROMPTS: dict[str, str] = {
    "advisor_interview_prep_system_prompt": "You are a senior career advisor. Return only valid JSON with an interview_prep array of short interview prep bullets, no markdown.",
    "advisor_interview_prep_user_prompt_template": (
        "Using the resume corpus and job packets, produce 4-6 short interview prep bullets that cover likely topics, questions, and evidence to review. Keep each bullet brief.\n\n"
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


def build_interview_prep_section(*, resume_modules_dir: Path, packets: list[dict[str, Any]], model_name: str | None) -> dict[str, Any]:
    if not packets:
        lines = _bullet_lines([], "## Interview Prep", "Add job packets to generate targeted interview prep.")
        return {"heading": "## Interview Prep", "lines": lines, "items": []}

    model = model_name or "openai/gpt-4o-mini"
    prompts = _load_prompts()
    resume_corpus = build_resume_corpus(resume_modules_dir)
    compact_packets = compact_job_packets(packets)
    payload = post_openrouter_json(
        model=model,
        system_prompt=prompts["advisor_interview_prep_system_prompt"],
        user_prompt=prompts["advisor_interview_prep_user_prompt_template"].format(
            resume_corpus=resume_corpus,
            job_packets=json.dumps(compact_packets, ensure_ascii=False),
        ),
        schema_name="advisor_interview_prep",
        schema={
            "type": "object",
            "properties": {
                "interview_prep": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 8,
                },
            },
            "required": ["interview_prep"],
            "additionalProperties": False,
        },
    )

    items = [shorten(normalize_text(str(item)), width=120, placeholder="...") for item in payload.get("interview_prep", [])]
    items = [item for item in items if item]
    lines = _bullet_lines(items, "## Interview Prep", "Review the packets, role fit, and recent project stories.")
    return {"heading": "## Interview Prep", "lines": lines, "items": items}
