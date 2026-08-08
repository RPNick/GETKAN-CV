# GETKAN-CV

GETKAN-CV is a Python + LaTeX resume tailoring tool.

It takes a job listing (URL or file), extracts structured job requirements, tailors resume modules with truth-preserving edits, and compiles a PDF resume that is constrained to one page when possible.

## What This Project Does

- Parses a job listing into a normalized job packet.
- Tailors selected resume modules (`summary.tex`, `experience.tex`, `personalprojects.tex`, `aboutme.tex`).
- Writes generated artifacts to a dedicated output folder.
- Compiles a LaTeX PDF using `xelatex`.
- Supports recompiling an existing generated output after manual `.tex` edits.

## Project Structure

- `src/main.py`: CLI entrypoint and workflow orchestration.
- `src/agents/job_parser_agent.py`: Job extraction, fallback parsing, normalization, validation.
- `src/agents/job_hunt_advisor.py`: Cross-job analysis of saved job packets with resume-based recommendation output.
- `src/agents/resume_tailor_agent.py`: Module tailoring, one-page fit profiles, artifact writing, compile logic.
- `src/prompts/tailor_prompts.json`: Editable prompt templates for parser and tailor behavior.
- `resume/modules/skills.json`: Editable technical skills catalog used by tailoring allowlist prioritization.

Skills categories in `resume/modules/skills.json` also influence bullet prioritization strength during tailoring (for example testing-focused roles prioritize testing-heavy bullets).
- `resume/`: Source resume and base modules.
- `output/`: Generated job-specific artifacts.
- `tailor-resume`: Executable wrapper script.

## LaTeX File Structure

The resume is assembled from a root TeX file plus section modules.

```text
getkan-cv.cls
resume/
  fonts/
  resume.tex
  modules/
    summary.tex
    experience.tex
    personalprojects.tex
    aboutme.tex
    education.tex
```

### Root and Class Files

- `getkan-cv.cls`
  - Custom document class for layout, typography, spacing, and CV macros (`\cvsection`, `\cventry`, `\cvitems`, etc.).
- `resume/resume.tex`
  - Main resume entrypoint.
  - Sets page geometry, color theme, fonts, and header/footer identity fields.
  - Imports active content modules using `\input{modules/...}`.
- `resume/fonts/`
  - Local font files consumed by `\fontdir[fonts/]` in `resume/resume.tex`.
  - Provides Roboto variants and FontAwesome used by the custom class.

### Section Module Files (`resume/modules/*.tex`)

- `resume/modules/summary.tex`
  - Summary section (`\cvsection{Summary}`) with a single paragraph describing your profile.
- `resume/modules/experience.tex`
  - Work Experience section with role blocks and bullet groups.
  - Uses `\cventry`, `\cvitems`, and `\cvsubitems` for job history and achievements.
- `resume/modules/personalprojects.tex`
  - Personal Projects section.
  - Includes optional intro text and project bullets used by tailoring/project-priority logic.
- `resume/modules/aboutme.tex`
  - About Me section.
  - Mixes education anchor content plus personal interest/context bullets.
- `resume/modules/education.tex`
  - Standalone Education section content.
  - Currently present as a source module but not imported by default in `resume/resume.tex`.

### Generated TeX (Per Tailoring Run)

For each tailored run, TeX files are copied/generated into:

- `output/<job_name>/resume/resume.tex`
  - Compilable run-specific root file.
- `output/<job_name>/resume/modules/*.tex`
  - Tailored versions of section modules used for that job target.
- `output/<job_name>/<job_name>.pdf`
  - Published final PDF at the output root.

## Requirements

- Python 3.10+ (recommended)
- `xelatex` available on `PATH`
- Optional but recommended: `pdfinfo` (for page count metadata)
- OpenRouter API key for model-assisted parsing:
  - `OPENROUTER_API_KEY`
  - Optional global fallback: `OPENROUTER_MODEL`
  - Optional parser model override: `OPENROUTER_MODEL_PARSER`
  - Optional tailor model override: `OPENROUTER_MODEL_TAILOR`
  - Optional advisor model override: `OPENROUTER_MODEL_ADVISOR`

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Minimal Python packages used:

- `langchain-core`
- `langchain-openai`
- `requests`

## Environment Setup

Create a `.env` file in the repository root (optional if env vars are already exported):

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_MODEL_PARSER=openai/gpt-4o-mini
OPENROUTER_MODEL_TAILOR=anthropic/claude-3.7-sonnet
OPENROUTER_MODEL_ADVISOR=openai/gpt-4.1-mini
RESUME_ADDRESS=123 Main St, Austin, TX 78701
RESUME_MOBILE=(+1) 555-555-5555
RESUME_EMAIL=your.email@example.com
```

Model resolution order is:

- Parser: `OPENROUTER_MODEL_PARSER` -> `OPENROUTER_MODEL` -> built-in default
- Tailor: `--model` CLI override -> `OPENROUTER_MODEL_TAILOR` -> `OPENROUTER_MODEL` -> `anthropic/claude-3.7-sonnet`
- Advisor: `--model` CLI override -> `OPENROUTER_MODEL_ADVISOR` -> `OPENROUTER_MODEL` -> `openai/gpt-4.1-mini`

Resume template identity placeholders resolve from env vars:

- `RESUME_ADDRESS`
- `RESUME_MOBILE`
- `RESUME_EMAIL`

Section-level tailoring controls live in `src/prompts/tailor_prompts.json`.
Common keys:

- `summary_section_prompt`
- `experience_section_prompt`
- `personalprojects_section_prompt`
- `aboutme_section_prompt`
- `aboutme_required_items` (delimiter: `||`)

To customize personal project order, include it directly inside `personalprojects_section_prompt`:

`Priority order: getkan-cv||linux enthusiast||mystic type-writer||notesboard plus plus`

## Command Reference

### 1) Tailor from a job URL

```bash
./tailor-resume <job_name> -u <job_url>
```

Example:

```bash
./tailor-resume github-careers -u "https://www.github.careers/careers-home/jobs/5682?lang=en-us"
```

### 2) Tailor from a local listing file

```bash
./tailor-resume <job_name> -f <path_to_listing_text_or_html>
```

### 2b) Batch process multiple URLs with auto job names

```bash
./tailor-resume -l <path_to_url_list_file>
```

Optional custom output root for batch runs:

```bash
./tailor-resume -l <path_to_url_list_file> -o <output_dir>
```

The URL list file should contain one URL per line (blank lines and lines starting with `#` are ignored).
This mode auto-generates unique job names from parsed company/title and writes each run to its own output folder.

### 3) Set custom output directory

```bash
./tailor-resume <job_name> -u <job_url> -o <output_dir>
```

If `-o` is omitted, default output is:

```text
output/<job_name>
```

### 4) Override model (optional)

```bash
./tailor-resume <job_name> -u <job_url> --model <model_id>
```

### 5) Recompile existing output after manual `.tex` edits

Use this after you edit files in `output/<job_name>/resume` (for example `modules/experience.tex`):

```bash
./tailor-resume <job_name> --recompile
```

Or with an explicit output directory:

```bash
./tailor-resume <job_name> --recompile -o <output_dir>
```

This mode:

- Skips parser/tailoring.
- Rebuilds from `resume/resume.tex` and republishes `<job_name>.pdf` at output root.
- Updates compile metadata in `tailored_resume.json` when available.

### 6) Build the base resume (no tailoring)

```bash
./tailor-resume --build-basic
```

Optional custom output directory:

```bash
./tailor-resume --build-basic -o <output_dir>
```

Default output when `-o` is omitted:

```text
output/general
```

### 7) Show CLI help

```bash
./tailor-resume -h
```

### 8) Generate job hunt recommendations from saved packets

```bash
./tailor-resume --job-hunt-advice
```

Optional custom output root to scan and write recommendations:

```bash
./tailor-resume --job-hunt-advice -o <output_dir>
```

Optional explicit packet files (instead of discovery scan):

```bash
./tailor-resume --job-hunt-advice --job-packets output/role-a/job_packet.json output/role-b/job_packet.json
```

This mode:

- Scans `output/**/job_packet.json`.
- Or uses explicit files from `--job-packets` when provided.
- Compares market demand from saved packets against your current resume modules and `skills.json`.
- Writes `job_hunt_recommendations.md` with skills and positioning recommendations.
- If no packets are available, it writes general job-hunt recommendations instead of a blank/no-data message.

## Test Commands

Run the current unit test suite:

```bash
python -m unittest -q tests.test_job_parser_agent
```

## Generated Output Layout

For a run like `./tailor-resume github-careers ...`:

- `output/github-careers/job_packet.json`: Parsed and normalized job data.
- `output/github-careers/tailored_resume.json`: Tailoring payload + compile metadata.
- `output/github-careers/resume/resume.tex`: Compilable resume root.
- `output/github-careers/resume/modules/*.tex`: Tailored module files.
- `output/github-careers/github-careers.pdf`: Final PDF output.
- `log/source_history.jsonl`: Append-only history of previously used URL/file inputs with timestamps.

For each tailored run, a `compatibility_score` (1-10) is computed and:

- printed in CLI JSON output,
- stored in `tailored_resume.json`,
- appended to `log/source_history.jsonl`.

## Typical Workflow

1. Run tailoring from URL or file.
2. Inspect generated modules in `output/<job_name>/resume/modules`.
3. Make manual edits if needed.
4. Recompile with `--recompile`.

For a non-tailored base resume build, use `--build-basic`.

## Notes

- Tailoring is constrained to use existing resume facts only.
- Project selection/prioritization logic for personalprojects content is deterministic and independent of job description.
- One-page fitting uses progressive compactness profiles when generating tailored output.
