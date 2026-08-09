from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from textwrap import shorten

from src.utils.advisor_common import build_resume_corpus, compact_job_packets, load_prompt_config, normalize_text, post_openrouter_json

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "advisor" / "prompts.json"

DEFAULT_PROMPTS: dict[str, str] = {
    "advisor_general_advice_system_prompt": "You are a senior career advisor. Return only valid JSON with a summary string and a general_advice array of short bullets, no markdown.",
    "advisor_general_advice_user_prompt_template": (
        "Using the resume corpus and job packets, write one concise summary of the current job fit and 3-5 short general advice bullets. Keep everything practical and brief.\n\n"
        "Resume corpus:\n{resume_corpus}\n\nJob packets summary:\n{job_packets}"
    ),
}


def _load_prompts() -> dict[str, str]:
    return load_prompt_config(PROMPT_FILE, DEFAULT_PROMPTS)


def _section_lines(summary: str, advice: list[str]) -> list[str]:
    lines = ["## General Advice and Summary", ""]
    if summary:
        lines.append(f"- Summary: {shorten(normalize_text(summary), width=120, placeholder='...')}")
    for item in advice:
        text = shorten(normalize_text(item), width=120, placeholder="...")
        if text:
            lines.append(f"- Advice: {text}")
    if len(lines) == 2:
        lines.append("- Summary: Add job packets to generate a tailored summary and advice.")
    return lines


def build_general_advice_section(*, resume_modules_dir: Path, packets: list[dict[str, Any]], model_name: str | None) -> dict[str, Any]:
    prompts = _load_prompts()
    resume_corpus = build_resume_corpus(resume_modules_dir)

    if not packets:
        fallback_summary = "The current resume shows experience worth tailoring, but job packets are needed for targeted guidance."
        fallback_advice = [
            "Lead with the roles you want and the impact you have shipped.",
            "Mirror the language from target jobs in your summary and bullets.",
            "Keep the strongest evidence near the top of the resume.",
        ]
        lines = _section_lines(fallback_summary, fallback_advice)
        return {"heading": "## General Advice and Summary", "lines": lines, "summary": fallback_summary, "general_advice": fallback_advice}

    model = model_name or "openai/gpt-4o-mini"
    compact_packets = compact_job_packets(packets)
    payload = post_openrouter_json(
        model=model,
        system_prompt=prompts["advisor_general_advice_system_prompt"],
        user_prompt=prompts["advisor_general_advice_user_prompt_template"].format(
            resume_corpus=resume_corpus,
            job_packets=json.dumps(compact_packets, ensure_ascii=False),
        ),
        schema_name="advisor_general_advice",
        schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "general_advice": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 6,
                },
            },
            "required": ["summary", "general_advice"],
            "additionalProperties": False,
        },
    )

    summary = normalize_text(str(payload.get("summary") or ""))
    general_advice = [normalize_text(str(item)) for item in payload.get("general_advice", [])]
    general_advice = [item for item in general_advice if item]
    lines = _section_lines(summary, general_advice)
    return {"heading": "## General Advice and Summary", "lines": lines, "summary": summary, "general_advice": general_advice}