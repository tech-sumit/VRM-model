"""LaTeX-math verifier using sympy.

`math_equal(pred, gold)` returns 1.0 iff `simplify(pred - gold) == 0`
after parsing both via sympy's latex parser or `parse_expr` with implicit-multiplication
and caret-as-power transformations enabled.
"""

from __future__ import annotations

from typing import Any

from sympy import simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMATIONS = (*standard_transformations, implicit_multiplication_application, convert_xor)
_LATEX_MARKERS = ("\\frac", "\\sqrt", "\\sum", "\\int", "\\pi", "\\theta", "^{", "_{")


def _looks_latex(s: str) -> bool:
    return any(m in s for m in _LATEX_MARKERS)


def _to_sympy(s: str):
    s = s.strip().strip("$ ")
    if not s:
        return None
    try:
        if _looks_latex(s):
            return parse_latex(s)
        return parse_expr(s, transformations=_TRANSFORMATIONS)
    except Exception:
        return None


def math_equal(prediction: str, gold: dict[str, Any]) -> float:
    p = _to_sympy(prediction)
    g = _to_sympy(str(gold.get("answer", "")))
    if p is None or g is None:
        return 0.0
    try:
        diff = simplify(p - g)
        return 1.0 if diff == 0 else 0.0
    except Exception:
        return 0.0
