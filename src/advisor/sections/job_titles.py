from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from textwrap import shorten

from src.utils.advisor_common import build_resume_corpus, compact_job_packets, load_prompt_config, normalize_text, post_openrouter_json
from src.compatibility_score import calculate_compatibility_score

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "advisor" / "prompts.json"

DEFAULT_PROMPTS: dict[str, str] = {
    "advisor_job_titles_system_prompt": "You are a senior career advisor. Return only valid JSON with a recommended_job_titles array of concise role titles and short descriptions, no markdown.",
    "advisor_job_titles_user_prompt_template": (
        "Based on the resume corpus, the job packets, and the compatibility scores below, recommend job titles that best match the current resume. Include title, description, and a short rationale for each.\n\n"
        "Resume corpus:\n{resume_corpus}\n\nJob packets summary:\n{job_packets}"
    ),
}


def _load_prompts() -> dict[str, str]:
    return load_prompt_config(PROMPT_FILE, DEFAULT_PROMPTS)


def _compact_packets_with_scores(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_packets = compact_job_packets(packets)
    scored_packets: list[dict[str, Any]] = []
    for payload, compact in zip(packets, compact_packets):
        scored_packets.append({**compact, "compatibility_score": calculate_compatibility_score(payload)})
    scored_packets.sort(key=lambda item: (-int(item.get("compatibility_score") or 0), str(item.get("title") or "").lower()))
    return scored_packets


def _job_titles_bullet_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Recommended Job Titles", ""]
    if not rows:
        lines.append("- No job packets provided.")
        return lines

    for row in rows:
        summary = shorten(f"{row['description']}. {row['rationale']}", width=100, placeholder="...")
        lines.append(f"- {row['job_title']} (score {row['compatibility_score']}): {summary}")
    return lines


def build_recommended_job_titles_section(*, resume_modules_dir: Path, packets: list[dict[str, Any]], model_name: str | None) -> dict[str, Any]:
    if not packets:
        lines = _job_titles_bullet_lines([])
        return {"heading": "## Recommended Job Titles", "lines": lines, "rows": []}

    model = model_name or "openai/gpt-4o-mini"
    prompts = _load_prompts()
    resume_corpus = build_resume_corpus(resume_modules_dir)
    scored_packets = _compact_packets_with_scores(packets)
    payload = post_openrouter_json(
        model=model,
        system_prompt=prompts["advisor_job_titles_system_prompt"],
        user_prompt=prompts["advisor_job_titles_user_prompt_template"].format(
            resume_corpus=resume_corpus,
            job_packets=json.dumps(scored_packets, ensure_ascii=False),
        ),
        schema_name="advisor_job_titles",
        schema={
            "type": "object",
            "properties": {
                "recommended_job_titles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "job_title": {"type": "string"},
                            "description": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["job_title", "description", "rationale"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": 5,
                },
            },
            "required": ["recommended_job_titles"],
            "additionalProperties": False,
        },
    )

    model_rows = payload.get("recommended_job_titles", [])
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(model_rows[:5]):
        if not isinstance(item, dict):
            continue
        source = scored_packets[index] if index < len(scored_packets) else {}
        rows.append(
            {
                "job_title": normalize_text(str(item.get("job_title") or "")),
                "description": normalize_text(str(item.get("description") or "")),
                "rationale": normalize_text(str(item.get("rationale") or "")),
                "compatibility_score": int(source.get("compatibility_score") or 0),
            }
        )
    rows.sort(key=lambda row: (-row["compatibility_score"], row["job_title"].lower()))
    lines = _job_titles_bullet_lines(rows)
    return {"heading": "## Recommended Job Titles", "lines": lines, "rows": rows}
