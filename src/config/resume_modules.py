from __future__ import annotations

# Centralized list of resume module filenames used across analysis/build flows.
# Edit this tuple to add/remove modules consumed by helpers that build a resume corpus.
RESUME_MODULE_NAMES: tuple[str, ...] = (
    "summary.tex",
    "experience.tex",
    "personalprojects.tex",
    "aboutme.tex",
)
