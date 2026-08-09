from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.advisor_common import compact_job_packets, load_prompt_config, normalize_text, post_openrouter_json

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "advisor" / "prompts.json"

DEFAULT_PROMPTS: dict[str, str] = {
    "advisor_skills_system_prompt": "You are a senior career advisor. Return only valid JSON with a skills array of concise skill names extracted from the job packets, no markdown.",
    "advisor_skills_user_prompt_template": (
        "Extract 5-15 concise skills from the job packets only. Focus on skills that recur across the job packets or appear as important requirements. Return skill names only, with no explanations.\n\n"
        "Job packets summary:\n{job_packets}"
    ),
}


def _load_prompts() -> dict[str, str]:
    return load_prompt_config(PROMPT_FILE, DEFAULT_PROMPTS)


def _unique_skills(skills: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for skill in skills:
        cleaned = normalize_text(skill)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _count_skill_mentions(skill: str, packets: list[dict[str, Any]]) -> dict[str, int]:
    normalized_skill = normalize_text(skill).lower()
    must_have_count = 0
    good_to_have_count = 0

    for payload in packets:
        job = payload.get("job", {}) if isinstance(payload, dict) else {}
        must_have_items = {normalize_text(str(item)).lower() for item in (job.get("must_have") or []) if normalize_text(str(item))}
        nice_to_have_items = {normalize_text(str(item)).lower() for item in (job.get("nice_to_have") or []) if normalize_text(str(item))}

        if normalized_skill in must_have_items:
            must_have_count += 1
        if normalized_skill in nice_to_have_items:
            good_to_have_count += 1

    return {
        "skill": normalize_text(skill),
        "must_haves": must_have_count,
        "good_to_haves": good_to_have_count,
        "total": must_have_count + good_to_have_count,
    }


def _skill_table_lines(skill_rows: list[dict[str, int]]) -> list[str]:
    lines = ["## Recommend Skills", "", "| Skill | Must Haves | Good To Haves | Total |", "| --- | ---: | ---: | ---: |"]
    if not skill_rows:
        lines.append("| None identified | 0 | 0 | 0 |")
        return lines

    for row in skill_rows:
        lines.append(f"| {row['skill']} | {row['must_haves']} | {row['good_to_haves']} | {row['total']} |")
    return lines


def parse_skill_rows_from_table_lines(lines: list[str]) -> list[dict[str, int]]:
    skill_rows: list[dict[str, int]] = []
    for line in lines:
        if not line.startswith("| ") or line.startswith("| Skill") or line.startswith("| ---") or line.startswith("| None identified"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 4:
            continue
        skill_rows.append(
            {
                "skill": parts[0],
                "must_haves": int(parts[1]),
                "good_to_haves": int(parts[2]),
                "total": int(parts[3]),
            }
        )
    return skill_rows


def build_recommend_skills_section(*, packets: list[dict[str, Any]], model_name: str | None) -> dict[str, Any]:
    if not packets:
        lines = _skill_table_lines([])
        return {"heading": "## Recommend Skills", "lines": lines, "skills": [], "skill_rows": []}

    model = model_name or "openai/gpt-4o-mini"
    prompts = _load_prompts()
    compact_packets = compact_job_packets(packets)
    payload = post_openrouter_json(
        model=model,
        system_prompt=prompts["advisor_skills_system_prompt"],
        user_prompt=prompts["advisor_skills_user_prompt_template"].format(job_packets=json.dumps(compact_packets, ensure_ascii=False)),
        schema_name="advisor_skills",
        schema={
            "type": "object",
            "properties": {
                "skills": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20},
            },
            "required": ["skills"],
            "additionalProperties": False,
        },
    )

    skills = _unique_skills([str(item) for item in payload.get("skills", [])])
    skill_rows = [_count_skill_mentions(skill, packets) for skill in skills]
    skill_rows = [row for row in skill_rows if row["total"] > 0]
    skill_rows.sort(key=lambda row: (-row["total"], -row["must_haves"], -row["good_to_haves"], row["skill"].lower()))
    lines = _skill_table_lines(skill_rows)
    return {"heading": "## Recommend Skills", "lines": lines, "skills": [row["skill"] for row in skill_rows], "skill_rows": skill_rows}
