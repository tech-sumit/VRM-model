"""Multiple-choice verifier. Extracts a single uppercase letter A-J from a free-form response."""

from __future__ import annotations

import re
from typing import Any

# Specific patterns first (longer, more constrained), then a generic single-letter fallback.
_PATTERNS = [
    re.compile(r"\b(?:THE\s+ANSWER\s+IS|FINAL\s+ANSWER)\s*[:\s]*\(?([A-J])\)?(?!\w)"),
    re.compile(r"\bANSWER\s*[:\s]+\(?([A-J])\)?(?!\w)"),
    re.compile(r"(?<!\w)\(([A-J])\)(?!\w)"),
]
_FALLBACK = re.compile(r"\b([A-J])\b")


def _extract(text: str) -> str | None:
    if not text:
        return None
    text = text.strip()
    text = text.lstrip("(").rstrip(").")
    if len(text) == 1 and text.upper() in "ABCDEFGHIJ":
        return text.upper()
    upper = text.upper()
    for pat in _PATTERNS:
        m = pat.search(upper)
        if m:
            return m.group(1)
    m = _FALLBACK.search(upper)
    return m.group(1) if m else None


def normalize_choice(prediction: str, gold: dict[str, Any]) -> float:
    p = _extract(prediction)
    g = (gold.get("answer") or "").strip().upper()
    if not p or not g:
        return 0.0
    return 1.0 if p == g else 0.0
