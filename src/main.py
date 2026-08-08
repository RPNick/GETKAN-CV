from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.job_parser_agent import JobParserState, extract_facts, fetch_or_load_listing, handoff_to_tailor, normalize_packet, validate_packet
from src.agents.job_hunt_advisor import generate_job_hunt_recommendations
from src.agents.resume_tailor_agent import build_tailored_payload, recompile_existing_output, render_env_placeholders


ROLE_DEFAULT_MODELS: dict[str, str] = {
    "PARSER": "openai/gpt-4o-mini",
    "TAILOR": "anthropic/claude-3.7-sonnet",
    "ADVISOR": "openai/gpt-4.1-mini",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tailor-resume")
    parser.add_argument("job_name", nargs="?", help="Short name for the tailored resume run")
    parser.add_argument("-f", "--file", dest="file_path", help="Path to a job packet or listing file")
    parser.add_argument("-u", "--url", dest="job_url", help="Job listing URL")
    parser.add_argument("-l", "--url-list-file", dest="url_list_file", help="Path to a text file containing job URLs (one per line) for batch processing")
    parser.add_argument("-o", "--output", dest="output_dir", help="Output directory for generated artifacts")
    parser.add_argument("--model", dest="model_name", help="Optional OpenRouter model override")
    parser.add_argument(
        "--build-basic",
        dest="build_basic",
        action="store_true",
        help="Build the base resume from resume/resume.tex without tailoring",
    )
    parser.add_argument(
        "--recompile",
        dest="recompile_existing",
        action="store_true",
        help="Recompile an existing generated output (uses <output>/resume/resume.tex after manual .tex edits)",
    )
    parser.add_argument(
        "--job-hunt-advice",
        dest="job_hunt_advice",
        action="store_true",
        help="Analyze saved output job_packet.json files and resume modules, then write job_hunt_recommendations.md",
    )
    parser.add_argument(
        "--job-packets",
        dest="job_packet_files",
        nargs="*",
        help="Optional explicit job_packet.json file paths for advisor mode (falls back to scanning output/**/job_packet.json)",
    )
    return parser


def load_listing_from_file(file_path: Optional[str]) -> str:
    if not file_path:
        return ""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    return path.read_text(encoding="utf-8")


def load_urls_from_file(list_file_path: str) -> list[str]:
    path = Path(list_file_path)
    if not path.exists():
        raise FileNotFoundError(f"URL list file not found: {list_file_path}")
    urls: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    if not urls:
        raise ValueError("URL list file is empty")
    return urls


def _load_dotenv() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve_model_for_role(role: str, cli_override: Optional[str]) -> Optional[str]:
    if cli_override:
        return cli_override

    _load_dotenv()
    role_key = f"OPENROUTER_MODEL_{role.upper()}"
    return os.getenv(role_key) or os.getenv("OPENROUTER_MODEL") or ROLE_DEFAULT_MODELS.get(role.upper())


def _contains_keyword(text: str, keyword: str) -> bool:
    escaped = re.escape(keyword)
    if re.search(r"[^A-Za-z0-9]", keyword):
        return escaped.lower() in text.lower()
    return re.search(rf"\b{escaped}\b", text, flags=re.I) is not None


def _resume_match_corpus() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    modules_dir = repo_root / "resume" / "modules"
    module_names = ["summary.tex", "experience.tex", "personalprojects.tex", "aboutme.tex"]
    parts: list[str] = []
    for name in module_names:
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

    title_tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z+#]{2,}", f"{title} {domain}") if token.lower() not in {"senior", "software", "engineer"}]
    title_ratio = 0.0
    if title_tokens:
        title_matches = sum(1 for token in title_tokens if _contains_keyword(corpus, token.lower()))
        title_ratio = title_matches / len(title_tokens)

    weighted_ratio = (must_ratio * 0.7) + (nice_ratio * 0.2) + (title_ratio * 0.1)
    if not must_have and not nice_to_have and not title_tokens:
        weighted_ratio = 0.45

    score = int(round(1 + (weighted_ratio * 9)))
    return max(1, min(10, score))


def append_source_log(job_name: str, file_path: Optional[str], job_url: Optional[str], compatibility_score: int, model_name: Optional[str] = None) -> str:
    log_dir = Path.cwd() / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "source_history.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_name": job_name,
        "url": job_url or "",
        "file": str(Path(file_path).resolve()) if file_path else "",
        "compatibility_score": compatibility_score,
        "model_name": model_name or "",
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return str(log_path)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", (value or "").strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned or "job"


def _auto_job_name(job_packet: dict[str, Any], job_url: str, used_names: set[str]) -> str:
    job = job_packet.get("job", {}) if isinstance(job_packet, dict) else {}
    company = str(job.get("company") or "").strip()
    title = str(job.get("title") or "").strip()

    parts: list[str] = []
    if company and company.lower() != "unknown":
        parts.append(company)
    if title and title.lower() != "unknown":
        parts.append(title)

    if not parts:
        from urllib.parse import urlparse

        parsed = urlparse(job_url)
        tail = parsed.path.rstrip("/").split("/")[-1] if parsed.path else ""
        if tail:
            parts.append(tail)
        elif parsed.netloc:
            parts.append(parsed.netloc)
        else:
            parts.append("job")

    base = _slugify("-".join(parts))
    candidate = base
    index = 2
    while candidate in used_names:
        candidate = f"{base}-{index}"
        index += 1
    used_names.add(candidate)
    return candidate


def _run_single_tailor(
    job_name: str,
    file_path: Optional[str],
    job_url: Optional[str],
    output_root: Path,
    model_name: Optional[str],
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_tailor_model = _resolve_model_for_role("TAILOR", model_name)

    listing_text = load_listing_from_file(file_path)
    state: JobParserState = {
        "source": {"job_url": job_url, "listing_text": listing_text},
        "raw_listing_text": listing_text,
        "extracted_facts": {},
        "normalized_packet": {},
        "confidence": 0.0,
    }

    fetch_or_load_listing(state)
    extract_facts(state)
    normalize_packet(state)
    validate_packet(state)

    compatibility_score = calculate_compatibility_score(state.get("normalized_packet", {}))
    source_log_path = append_source_log(job_name, file_path, job_url, compatibility_score, model_name=resolved_tailor_model)

    handoff_result = handoff_to_tailor(state, output_dir=output_root)
    payload = build_tailored_payload(state["normalized_packet"], job_name=job_name, output_dir=str(output_root), model_name=resolved_tailor_model)
    payload["compatibility_score"] = compatibility_score

    summary_path = output_root / "tailored_resume.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "job_name": job_name,
        "output_dir": str(output_root),
        "job_packet": handoff_result["output_path"],
        "summary": str(summary_path),
        "pdf": payload.get("compile", {}).get("pdf_path", ""),
        "compatibility_score": compatibility_score,
        "source_log": source_log_path,
    }


def build_basic_resume(output_dir: Optional[str]) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parent.parent
    resume_dir = repo_root / "resume"
    destination = Path(output_dir) if output_dir else Path.cwd() / "output" / "general"
    destination.mkdir(parents=True, exist_ok=True)
    resume_output_root = destination / "resume"
    resume_output_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(repo_root / "getkan-cv.cls", resume_output_root / "getkan-cv.cls")
    shutil.copytree(resume_dir / "modules", resume_output_root / "modules", dirs_exist_ok=True)
    fonts_dir = resume_dir / "fonts"
    if fonts_dir.exists():
        shutil.copytree(fonts_dir, resume_output_root / "fonts", dirs_exist_ok=True)

    resume_text = (resume_dir / "resume.tex").read_text(encoding="utf-8")
    resume_text = resume_text.replace("\\documentclass[11pt, letterpaper]{../getkan-cv}", "\\documentclass[11pt, letterpaper]{getkan-cv}")
    resume_text = resume_text.replace("\\fontdir[../fonts/]", "\\fontdir[fonts/]")
    resume_text = render_env_placeholders(resume_text)
    (resume_output_root / "resume.tex").write_text(resume_text, encoding="utf-8")

    xelatex = shutil.which("xelatex")
    if not xelatex:
        raise RuntimeError("xelatex not available on PATH")

    logs: list[str] = []
    for pass_index in range(2):
        result = subprocess.run(
            [xelatex, "-interaction=nonstopmode", "-halt-on-error", "resume.tex"],
            cwd=resume_output_root,
            capture_output=True,
            text=True,
        )
        logs.append(f"pass {pass_index + 1} exit={result.returncode}")
        if result.stdout:
            logs.append(result.stdout[-1200:])
        if result.stderr:
            logs.append(result.stderr[-1200:])
        if result.returncode != 0:
            raise RuntimeError("Basic resume compile failed")

    compiled_pdf = resume_output_root / "resume.pdf"
    pdf_path = destination / "resume.pdf"
    if compiled_pdf.exists():
        shutil.copy2(compiled_pdf, pdf_path)
    return {
        "output_dir": str(destination),
        "pdf": str(pdf_path if pdf_path.exists() else ""),
        "compile_log": "\n".join(logs),
    }


def run(
    job_name: Optional[str],
    file_path: Optional[str],
    job_url: Optional[str],
    output_dir: Optional[str],
    model_name: Optional[str],
    build_basic: bool = False,
    recompile_existing: bool = False,
    job_hunt_advice: bool = False,
    job_packet_files: list[str] | None = None,
    url_list_file: str | None = None,
) -> int:
    if job_hunt_advice:
        advisor_model = _resolve_model_for_role("ADVISOR", model_name)
        advice_output_root = Path(output_dir) if output_dir else Path.cwd() / "output"
        recommendations = generate_job_hunt_recommendations(advice_output_root, job_packet_files=job_packet_files, model_name=advisor_model)
        print(
            json.dumps(
                {
                    "output_dir": str(advice_output_root),
                    "recommendations": recommendations.get("recommendations_path", ""),
                    "job_packet_count": recommendations.get("packet_count", 0),
                    "model_name": advisor_model or "",
                    "mode": "job-hunt-advice",
                },
                indent=2,
            )
        )
        return 0

    if build_basic:
        basic_result = build_basic_resume(output_dir)
        print(
            json.dumps(
                {
                    "output_dir": basic_result["output_dir"],
                    "pdf": basic_result["pdf"],
                    "mode": "build-basic",
                },
                indent=2,
            )
        )
        return 0

    if url_list_file:
        if job_name:
            raise ValueError("Do not provide job_name when using -l/--url-list-file")
        if file_path:
            raise ValueError("-f/--file is not supported with -l/--url-list-file")
        if job_url:
            raise ValueError("-u/--url cannot be combined with -l/--url-list-file")

        job_urls = load_urls_from_file(url_list_file)

        output_base = Path(output_dir) if output_dir else Path.cwd() / "output"
        output_base.mkdir(parents=True, exist_ok=True)

        used_names: set[str] = set()
        batch_results: list[dict[str, Any]] = []
        resolved_tailor_model = _resolve_model_for_role("TAILOR", model_name)
        for url in job_urls:
            staging_state: JobParserState = {
                "source": {"job_url": url, "listing_text": ""},
                "raw_listing_text": "",
                "extracted_facts": {},
                "normalized_packet": {},
                "confidence": 0.0,
            }
            fetch_or_load_listing(staging_state)
            extract_facts(staging_state)
            normalize_packet(staging_state)
            validate_packet(staging_state)

            auto_name = _auto_job_name(staging_state.get("normalized_packet", {}), url, used_names)
            output_root = output_base / auto_name
            output_root.mkdir(parents=True, exist_ok=True)

            # Reuse fully processed packet to avoid duplicate fetch/parse calls.
            compatibility_score = calculate_compatibility_score(staging_state.get("normalized_packet", {}))
            source_log_path = append_source_log(auto_name, None, url, compatibility_score, model_name=resolved_tailor_model)
            handoff_result = handoff_to_tailor(staging_state, output_dir=output_root)
            payload = build_tailored_payload(staging_state["normalized_packet"], job_name=auto_name, output_dir=str(output_root), model_name=resolved_tailor_model)
            payload["compatibility_score"] = compatibility_score
            summary_path = output_root / "tailored_resume.json"
            summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            batch_results.append(
                {
                    "job_name": auto_name,
                    "url": url,
                    "output_dir": str(output_root),
                    "job_packet": handoff_result["output_path"],
                    "summary": str(summary_path),
                    "pdf": payload.get("compile", {}).get("pdf_path", ""),
                    "compatibility_score": compatibility_score,
                    "source_log": source_log_path,
                }
            )

        print(
            json.dumps(
                {
                    "mode": "batch-urls",
                    "url_list_file": str(Path(url_list_file).resolve()),
                    "count": len(batch_results),
                    "runs": batch_results,
                },
                indent=2,
            )
        )
        return 0

    if not job_name:
        raise ValueError("Provide job_name unless using --build-basic")

    default_output_root = Path.cwd() / "output" / job_name
    output_root = Path(output_dir or str(default_output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    if recompile_existing:
        recompile_result = recompile_existing_output(output_root)
        compile_payload = recompile_result.get("compile", {})
        print(
            json.dumps(
                {
                    "job_name": job_name,
                    "output_dir": str(output_root),
                    "summary": recompile_result.get("summary", ""),
                    "pdf": compile_payload.get("pdf_path", ""),
                    "page_count": compile_payload.get("page_count"),
                    "mode": "recompile",
                },
                indent=2,
            )
        )
        return 0

    if not file_path and not job_url:
        raise ValueError("Provide either -f/--file or -u/--url")

    result = _run_single_tailor(job_name, file_path, job_url, output_root, model_name)
    print(
        json.dumps(result, indent=2)
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(
            args.job_name,
            args.file_path,
            args.job_url,
            args.output_dir,
            args.model_name,
            args.build_basic,
            args.recompile_existing,
            args.job_hunt_advice,
            args.job_packet_files,
            args.url_list_file,
        )
    except Exception as exc:  # pragma: no cover - CLI error surface
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
