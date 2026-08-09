from __future__ import annotations

import re


def normalize_text(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip()
    cleaned = re.sub(r"^[\-\*\d\.)\(\s]+", "", cleaned)
    cleaned = cleaned.strip(" ,.;:")
    return cleaned