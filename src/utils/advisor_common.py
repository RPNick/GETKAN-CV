from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.config.resume_modules import RESUME_MODULE_NAMES

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


def load_prompt_config(prompt_path: Path, defaults: dict[str, str]) -> dict[str, str]:
    prompts = dict(defaults)
    if not prompt_path.exists():
        return prompts

    try:
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return prompts

    if not isinstance(payload, dict):
        return prompts

    for key, default_value in defaults.items():
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            prompts[key] = value
        else:
            prompts[key] = default_value
    return prompts


def load_dotenv(env_path: Path | None = None) -> None:
    if env_path is None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_text(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip()
    cleaned = re.sub(r"^[\-\*\d\.)\(\s]+", "", cleaned)
    cleaned = cleaned.strip(" ,.;:")
    return cleaned


def extract_json_payload(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:markdown)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def compact_job_packets(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_packets: list[dict[str, Any]] = []
    for payload in packets[:40]:
        job = payload.get("job", {}) if isinstance(payload, dict) else {}
        compact_packets.append(
            {
                "title": str(job.get("title") or ""),
                "company": str(job.get("company") or ""),
                "domain": str(job.get("domain") or ""),
                "description": str(job.get("description") or "")[:1200],
                "must_have": [str(item) for item in (job.get("must_have") or [])][:20],
                "nice_to_have": [str(item) for item in (job.get("nice_to_have") or [])][:20],
                "responsibilities": [str(item) for item in (job.get("responsibilities") or [])][:20],
            }
        )
    return compact_packets


def build_resume_corpus(resume_modules_dir: Path) -> str:
    chunks: list[str] = []
    for name in RESUME_MODULE_NAMES:
        path = resume_modules_dir / name
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def post_openrouter_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required for advisor recommendations")

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/GETKAN-CV",
            "X-Title": "GETKAN-CV Job Hunt Advisor",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": 0.2,
        },
        timeout=60,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]["content"]
    payload = extract_json_payload(message)
    if not payload:
        raise RuntimeError("OpenRouter did not return valid advisor JSON")
    return payload
