"""Verifier registry. Use `score(record, prediction)` to compute composite reward."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vrm.data.verifiers.format import extract_answer, has_valid_format
from vrm.data.verifiers.latex_math import math_equal
from vrm.data.verifiers.multiple_choice import normalize_choice
from vrm.data.verifiers.numeric import exact_numeric
from vrm.data.verifiers.span import span_match

VerifierFn = Callable[[str, dict[str, Any]], float]

REGISTRY: dict[str, VerifierFn] = {
    "exact_numeric": exact_numeric,
    "normalize_choice": normalize_choice,
    "math_equal": math_equal,
    "span_match": span_match,
}


def score(
    record_or_gold: dict[str, Any],
    prediction: str,
    *,
    format_weight: float = 0.1,
) -> dict[str, float]:
    """Composite reward per spec §3.3.

        total = format_weight * format + (1 - format_weight) * accuracy

    `record_or_gold` is a dict with at least keys 'verifier', 'answer', and
    optional 'tolerance'.
    """
    fmt = 1.0 if has_valid_format(prediction) else 0.0
    if fmt == 0.0:
        return {"format": 0.0, "accuracy": 0.0, "total": 0.0}
    pred_answer = extract_answer(prediction)
    name = record_or_gold.get("verifier")
    fn = REGISTRY.get(name)  # type: ignore[arg-type]
    if fn is None:
        return {"format": fmt, "accuracy": 0.0, "total": format_weight * fmt}
    acc = fn(pred_answer, record_or_gold)
    total = format_weight * fmt + (1.0 - format_weight) * acc
    return {"format": fmt, "accuracy": acc, "total": total}
