"""Numeric verifier with permissive normalization.

Handles dollar signs, percent signs, commas, latex fractions, plain fractions,
exponential notation, leading/trailing whitespace.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

_LATEX_FRAC = re.compile(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}")
_PLAIN_FRAC = re.compile(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$")


def _coerce(s: str) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = s.strip("$ ").rstrip("%").replace(",", "")
    m = _LATEX_FRAC.search(s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    m = _PLAIN_FRAC.match(s)
    if m:
        try:
            return float(Fraction(m.group(1)) / Fraction(m.group(2)))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def exact_numeric(prediction: str, gold: dict[str, Any]) -> float:
    """Returns 1.0 if numeric prediction matches gold within tolerance, else 0.0."""
    p = _coerce(prediction)
    g = _coerce(str(gold.get("answer", "")))
    if p is None or g is None:
        return 0.0
    tol = float(gold.get("tolerance", 0.0))
    return 1.0 if abs(p - g) <= tol else 0.0
