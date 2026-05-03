"""Span verifier -- case-insensitive substring containment.

For ChartQA / span-style answers where the model emits a sentence and the
gold is a short noun-phrase. Returns 1.0 iff `gold.lower() in pred.lower()`.
"""

from __future__ import annotations

from typing import Any


def span_match(prediction: str, gold: dict[str, Any]) -> float:
    g = (gold.get("answer") or "").strip().lower()
    p = (prediction or "").strip().lower()
    if not g or not p:
        return 0.0
    return 1.0 if g in p else 0.0
