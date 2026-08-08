from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SKILL_STOPWORDS = {
    "experience",
    "knowledge",
    "skills",
    "ability",
    "communication",
    "team",
    "teams",
    "preferred",
    "required",
    "plus",
}

THEME_KEYWORDS: dict[str, list[str]] = {
    "testing_and_quality": ["testing", "unit test", "integration test", "qa", "quality", "jest", "cypress", "playwright"],
    "devops_and_delivery": ["docker", "kubernetes", "ci", "cd", "github actions", "deploy", "release", "pipeline"],
    "backend_and_api": ["backend", "api", "microservice", "service", "database", "sql", "reliability"],
    "frontend_product": ["frontend", "react", "vue", "typescript", "javascript", "ui", "ux"],
    "cloud_and_platform": ["aws", "gcp", "azure", "cloud", "infrastructure", "terraform"],
}


def _normalize_skill(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip()
    cleaned = re.sub(r"^[\-\*\d\.)\(\s]+", "", cleaned)
    cleaned = cleaned.strip(" ,.;:")
    return cleaned


def _is_skill_candidate(value: str) -> bool:
    lowered = value.lower()
    if not lowered:
        return False
    if lowered in SKILL_STOPWORDS:
        return False
    if len(lowered) < 2 or len(lowered) > 48:
        return False
    return True


def _contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term)
    if re.search(r"[^A-Za-z0-9]", term):
        return escaped.lower() in text.lower()
    return re.search(rf"\b{escaped}\b", text, flags=re.I) is not None


def _load_skills_catalog(resume_modules_dir: Path) -> dict[str, list[str]]:
    catalog_path = resume_modules_dir / "skills.json"
    if not catalog_path.exists():
        return {}

    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(payload, dict):
        return {}

    cleaned: dict[str, list[str]] = {}
    for category, values in payload.items():
        if not isinstance(category, str) or not isinstance(values, list):
            continue
        items = []
        for raw in values:
            item = _normalize_skill(str(raw))
            if item:
                items.append(item)
        if items:
            cleaned[category] = items
    return cleaned


def _build_resume_corpus(resume_modules_dir: Path) -> str:
    module_names = ["summary.tex", "experience.tex", "personalprojects.tex", "aboutme.tex"]
    chunks: list[str] = []
    for name in module_names:
        path = resume_modules_dir / name
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks).lower()


def _collect_job_packets(output_root: Path) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for packet_path in sorted(output_root.rglob("job_packet.json")):
        try:
            payload = json.loads(packet_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("job"), dict):
            packets.append(payload)
    return packets


def _collect_job_packets_from_files(packet_files: list[str] | None) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for raw_path in packet_files or []:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("job"), dict):
            packets.append(payload)
    return packets


def _build_specific_tech_recommendations(resume_corpus: str, skills_catalog: dict[str, list[str]]) -> list[str]:
    known_terms = {term.lower() for values in skills_catalog.values() for term in values}

    def is_known(tech: str) -> bool:
        lower = tech.lower()
        return lower in known_terms or _contains_term(resume_corpus, lower)

    # Ordered by common demand for senior software-engineering roles.
    tech_targets: list[tuple[str, str]] = [
        ("Go", "strong backend demand and practical systems/concurrency signal"),
        ("TypeScript", "high-value signal for maintainable frontend and full-stack codebases"),
        ("Python", "portable across backend, automation, and AI workflows"),
        ("PostgreSQL", "core data-layer competency expected by many product teams"),
        ("Docker", "baseline expectation for reproducible environments and delivery"),
        ("Kubernetes", "frequently requested for production orchestration"),
        ("GitHub Actions", "clear CI/CD ownership signal"),
        ("AWS", "broad cloud relevance across engineering teams"),
        ("Terraform", "infrastructure-as-code credibility for platform maturity"),
        ("OpenTelemetry", "modern observability signal for reliability-focused roles"),
        ("Playwright", "strong end-to-end testing signal for quality-focused teams"),
        ("Redis", "commonly needed for caching and performance-sensitive systems"),
    ]

    recommendations: list[str] = []
    for tech, reason in tech_targets:
        if is_known(tech):
            continue
        recommendations.append(f"- {tech}: {reason}.")
        if len(recommendations) >= 6:
            break

    if recommendations:
        return recommendations

    # If all targets already appear known, suggest deeper proof areas.
    return [
        "- TypeScript: deepen advanced typing and API-contract patterns to show senior-level code quality.",
        "- Kubernetes: highlight production deployment ownership and rollout strategy.",
        "- Playwright: demonstrate robust end-to-end testing and regression prevention.",
    ]


def _build_general_advice_lines(resume_corpus: str, skills_catalog: dict[str, list[str]]) -> list[str]:
    lines: list[str] = []
    lines.append("## General Recommendations (No Job Packets Provided)")
    lines.append("- Keep a primary role target (for example: Senior Backend Engineer) and tune summary + top bullets for that role only.")
    lines.append("- Add quantified outcomes to your latest role bullets: latency, uptime, release frequency, defect rate, or cost impact.")
    lines.append("- Show end-to-end ownership: design, implementation, testing, deployment, and production support.")
    lines.append("- Keep technology mentions concentrated around your strongest stack and remove low-signal tool sprawl.")
    lines.append("- Add one portfolio proof per priority area: API/backend, testing quality, and delivery automation.")
    lines.append("")
    lines.append("## Skills Hiring Managers Commonly Expect")
    lines.append("- Backend/API: API design, service reliability, database performance, observability.")
    lines.append("- Testing/Quality: unit + integration tests, coverage strategy, regression prevention.")
    lines.append("- Delivery: CI/CD ownership, release process, incident response and postmortems.")
    lines.append("- Leadership: mentoring, architecture ownership, cross-team collaboration.")
    lines.append("")
    lines.append("## Suggested Technologies And Languages To Work On")
    lines.extend(_build_specific_tech_recommendations(resume_corpus, skills_catalog))
    lines.append("")
    lines.append("## Practical Next Steps")
    lines.append("- Save 5-10 target job packets, then rerun advisor mode for market-specific gap analysis.")
    lines.append("- Keep a running achievements bank with measurable outcomes to speed up tailoring.")
    lines.append("- Refresh one resume section per week so examples stay recent and concrete.")
    lines.append("")
    return lines


def _theme_signal(text: str, keywords: list[str]) -> int:
    count = 0
    for keyword in keywords:
        if _contains_term(text, keyword.lower()):
            count += 1
    return count


def generate_job_hunt_recommendations(
    output_root: str | Path,
    resume_modules_dir: str | Path | None = None,
    recommendations_filename: str = "job_hunt_recommendations.md",
    job_packet_files: list[str] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    modules_dir = Path(resume_modules_dir) if resume_modules_dir else (repo_root / "resume" / "modules")

    resume_corpus = _build_resume_corpus(modules_dir)
    skills_catalog = _load_skills_catalog(modules_dir)
    catalog_terms = [term.lower() for values in skills_catalog.values() for term in values]

    explicit_packets = _collect_job_packets_from_files(job_packet_files)
    packets = explicit_packets or _collect_job_packets(output_path)
    if not packets:
        recommendations_path = output_path / recommendations_filename
        lines = [
            "# Job Hunt Recommendations",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Advisor model: {model_name or 'default'}",
            "Job packets analyzed: 0",
            "",
        ]
        lines.extend(_build_general_advice_lines(resume_corpus, skills_catalog))
        recommendations_path.write_text("\n".join(lines), encoding="utf-8")
        return {
            "recommendations_path": str(recommendations_path),
            "packet_count": 0,
            "missing_skills": [],
            "matched_skills": [],
        }

    must_have_counter: Counter[str] = Counter()
    nice_to_have_counter: Counter[str] = Counter()
    title_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    skill_notes: dict[str, dict[str, int]] = defaultdict(lambda: {"must": 0, "nice": 0})

    market_theme_counts: Counter[str] = Counter()
    resume_theme_counts: Counter[str] = Counter()

    for payload in packets:
        job = payload.get("job", {})
        title = _normalize_skill(str(job.get("title") or ""))
        domain = _normalize_skill(str(job.get("domain") or ""))

        if title and title.lower() != "unknown":
            title_counter[title] += 1
        if domain and domain.lower() != "unknown":
            domain_counter[domain] += 1

        must_have_values = job.get("must_have") or []
        nice_to_have_values = job.get("nice_to_have") or []
        responsibilities = job.get("responsibilities") or []
        description = str(job.get("description") or "")

        for raw in must_have_values:
            skill = _normalize_skill(str(raw))
            if _is_skill_candidate(skill):
                must_have_counter[skill] += 1
                skill_notes[skill]["must"] += 1

        for raw in nice_to_have_values:
            skill = _normalize_skill(str(raw))
            if _is_skill_candidate(skill):
                nice_to_have_counter[skill] += 1
                skill_notes[skill]["nice"] += 1

        market_text = " ".join(
            [
                title,
                description,
                " ".join(str(item) for item in must_have_values),
                " ".join(str(item) for item in nice_to_have_values),
                " ".join(str(item) for item in responsibilities),
            ]
        ).lower()

        for theme, keywords in THEME_KEYWORDS.items():
            market_theme_counts[theme] += _theme_signal(market_text, keywords)

    for theme, keywords in THEME_KEYWORDS.items():
        resume_theme_counts[theme] = _theme_signal(resume_corpus, keywords)

    weighted_skills: list[tuple[str, int]] = []
    for skill in set(must_have_counter.keys()) | set(nice_to_have_counter.keys()):
        weighted_score = (must_have_counter[skill] * 3) + nice_to_have_counter[skill]
        weighted_skills.append((skill, weighted_score))
    weighted_skills.sort(key=lambda row: (-row[1], row[0].lower()))

    missing_skills: list[dict[str, Any]] = []
    matched_skills: list[dict[str, Any]] = []

    for skill, score in weighted_skills:
        lowered = skill.lower()
        on_resume = _contains_term(resume_corpus, lowered) or any(_contains_term(lowered, term) or _contains_term(term, lowered) for term in catalog_terms)
        record = {
            "skill": skill,
            "weighted_score": score,
            "must_have_mentions": skill_notes[skill]["must"],
            "nice_to_have_mentions": skill_notes[skill]["nice"],
        }
        if on_resume:
            matched_skills.append(record)
        else:
            missing_skills.append(record)

    top_missing = missing_skills[:12]
    top_matched = matched_skills[:10]

    theme_gaps: list[dict[str, Any]] = []
    for theme in THEME_KEYWORDS:
        market_signal = market_theme_counts.get(theme, 0)
        resume_signal = resume_theme_counts.get(theme, 0)
        if market_signal <= 0:
            continue
        if market_signal >= max(2, resume_signal + 2):
            theme_gaps.append(
                {
                    "theme": theme,
                    "market_signal": market_signal,
                    "resume_signal": resume_signal,
                }
            )
    theme_gaps.sort(key=lambda row: (-row["market_signal"], row["theme"]))

    recommendations_path = output_path / recommendations_filename
    lines: list[str] = []
    lines.append("# Job Hunt Recommendations")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Advisor model: {model_name or 'default'}")
    lines.append(f"Job packets analyzed: {len(packets)}")
    lines.append("")

    lines.append("## Targeting Snapshot")
    if title_counter:
        lines.append("Top role titles from saved job packets:")
        for title, count in title_counter.most_common(6):
            lines.append(f"- {title}: {count}")
    else:
        lines.append("- No consistent job title signal found yet.")

    if domain_counter:
        lines.append("Top domains:")
        for domain, count in domain_counter.most_common(5):
            lines.append(f"- {domain}: {count}")
    lines.append("")

    lines.append("## Highest-Impact Skills To Build Or Show")
    if top_missing:
        for item in top_missing:
            lines.append(
                f"- {item['skill']} (mentions: must-have {item['must_have_mentions']}, nice-to-have {item['nice_to_have_mentions']})"
            )
    else:
        lines.append("- Your saved packets mostly align with your current visible skill inventory.")
    lines.append("")

    lines.append("## Skills Already Aligned")
    if top_matched:
        for item in top_matched:
            lines.append(
                f"- {item['skill']} (mentions: must-have {item['must_have_mentions']}, nice-to-have {item['nice_to_have_mentions']})"
            )
    else:
        lines.append("- No strong overlaps detected yet.")
    lines.append("")

    lines.append("## Thematic Gaps Hiring Managers Will Notice")
    if theme_gaps:
        for gap in theme_gaps[:6]:
            lines.append(
                f"- {gap['theme'].replace('_', ' ')}: market signal {gap['market_signal']} vs resume signal {gap['resume_signal']}"
            )
    else:
        lines.append("- No major theme gap detected from current packet sample.")
    lines.append("")

    lines.append("## Concrete Actions")
    lines.append("- Add one quantified impact bullet per core theme in your most recent role (performance, reliability, delivery speed, quality).")
    lines.append("- Add one portfolio proof per top missing skill where possible (small repo, write-up, or demo with clear scope and outcomes).")
    lines.append("- Mirror exact high-frequency phrasing from packets in your resume bullets (for ATS match) while keeping claims factual.")
    lines.append("- Keep a short wins section for leadership outcomes: mentoring, architecture ownership, and cross-team delivery.")
    lines.append("")

    recommendations_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "recommendations_path": str(recommendations_path),
        "packet_count": len(packets),
        "missing_skills": top_missing,
        "matched_skills": top_matched,
        "theme_gaps": theme_gaps[:6],
    }
