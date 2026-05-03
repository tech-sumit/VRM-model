"""Common types for dataset normalizers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vrm.data.schema import Record

NormalizeFn = Callable[[dict], "Record | None"]


@dataclass(frozen=True)
class NormalizeSpec:
    hf_id: str
    split: str
    normalize: NormalizeFn
    image_column: str | None = None
    config: str | None = None
    default_verifier: str = "exact_numeric"


SYSTEM_PROMPT = (
    "You are a careful visual reasoner. Solve step-by-step. "
    "Put your reasoning in <think>...</think> and your final answer in "
    "<answer>...</answer>."
)


def _verifier_for(answer_type: str) -> str:
    return {
        "numeric": "exact_numeric",
        "multiple_choice": "normalize_choice",
        "latex_math": "math_equal",
        "span": "span_match",
    }.get(answer_type, "exact_numeric")
