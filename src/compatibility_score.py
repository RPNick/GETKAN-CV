from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config.resume_modules import RESUME_MODULE_NAMES


def _contains_keyword(text: str, keyword: str) -> bool:
    escaped = re.escape(keyword)
    if re.search(r"[^A-Za-z0-9]", keyword):
        return escaped.lower() in text.lower()
    return re.search(rf"\b{escaped}\b", text, flags=re.I) is not None


def _resume_match_corpus() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    modules_dir = repo_root / "resume" / "modules"
    parts: list[str] = []
    for name in RESUME_MODULE_NAMES:
        path = modules_dir / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))

    skills_path = modules_dir / "skills.json"
    if skills_path.exists():
        try:
            payload = json.loads(skills_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for values in payload.values():
                    if isinstance(values, list):
                        parts.append(" ".join(str(item) for item in values))
        except (json.JSONDecodeError, OSError):
            pass
    return "\n".join(parts).lower()


def _known_skill_terms() -> list[str]:
    repo_root = Path(__file__).resolve().parent.parent
    skills_path = repo_root / "resume" / "modules" / "skills.json"
    terms: list[str] = []
    if skills_path.exists():
        try:
            payload = json.loads(skills_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for values in payload.values():
                    if isinstance(values, list):
                        for item in values:
                            cleaned = str(item).strip().lower()
                            if cleaned:
                                terms.append(cleaned)
        except (json.JSONDecodeError, OSError):
            pass

    # Add common role terms that frequently appear inside long requirement sentences.
    terms.extend(
        [
            "api",
            "json:api",
            "backend",
            "frontend",
            "distributed systems",
            "microservices",
            "mysql",
            "postgresql",
            "redis",
            "observability",
            "opentelemetry",
            "ci/cd",
            "websockets",
            "kubernetes",
            "docker",
            "go",
            "python",
            "php",
            "laravel",
            "react",
            "vue",
            "typescript",
            "javascript",
        ]
    )
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def _extract_requirement_terms(requirement: str, known_terms: list[str]) -> list[str]:
    text = (requirement or "").strip().lower()
    if not text:
        return []

    candidates: list[str] = []

    # Skill-aware extraction first so multiword technologies are preserved.
    for term in known_terms:
        if _contains_keyword(text, term):
            candidates.append(term)

    # Split long requirements into smaller chunks.
    chunked = re.split(r"[,;/]|\band\b|\bor\b|\bwith\b|\bsuch as\b|\bincluding\b|\blike\b", text)
    for chunk in chunked:
        cleaned = re.sub(r"\s+", " ", chunk).strip(" .:-")
        if len(cleaned) < 3:
            continue
        if len(cleaned.split()) > 8:
            continue
        candidates.append(cleaned)

    # Single-token fallback for common terms/acronyms.
    for token in re.findall(r"[a-z0-9+#\.:-]{2,}", text):
        if token in {"years", "experience", "strong", "skills", "ability", "understanding"}:
            continue
        if token.isdigit():
            continue
        candidates.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _requirement_match_score(corpus: str, requirement: str, known_terms: list[str]) -> float:
    req = (requirement or "").strip().lower()
    if not req:
        return 0.0

    # Full phrase match gets full credit.
    if _contains_keyword(corpus, req):
        return 1.0

    terms = _extract_requirement_terms(req, known_terms)
    if not terms:
        return 0.0

    matches = sum(1 for term in terms if _contains_keyword(corpus, term))
    ratio = matches / len(terms)
    return min(0.95, ratio)


def calculate_compatibility_score(job_packet: dict[str, Any]) -> int:
    job = job_packet.get("job", {}) if isinstance(job_packet, dict) else {}
    must_have = [str(item).strip() for item in (job.get("must_have") or []) if str(item).strip()]
    nice_to_have = [str(item).strip() for item in (job.get("nice_to_have") or []) if str(item).strip()]
    title = str(job.get("title") or "")
    domain = str(job.get("domain") or "")

    corpus = _resume_match_corpus()
    known_terms = _known_skill_terms()

    must_ratio = 0.0
    if must_have:
        must_scores = [_requirement_match_score(corpus, item, known_terms) for item in must_have]
        must_ratio = sum(must_scores) / len(must_scores)

    nice_ratio = 0.0
    if nice_to_have:
        nice_scores = [_requirement_match_score(corpus, item, known_terms) for item in nice_to_have]
        nice_ratio = sum(nice_scores) / len(nice_scores)

    title_tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z+#]{2,}", f"{title} {domain}")
        if token.lower() not in {"senior", "software", "engineer"}
    ]
    title_ratio = 0.0
    if title_tokens:
        title_matches = sum(1 for token in title_tokens if _contains_keyword(corpus, token.lower()))
        title_ratio = title_matches / len(title_tokens)

    weighted_ratio = (must_ratio * 0.7) + (nice_ratio * 0.2) + (title_ratio * 0.1)
    if not must_have and not nice_to_have and not title_tokens:
        weighted_ratio = 0.45

    score = int(round(1 + (weighted_ratio * 9)))
    return max(1, min(10, score))
