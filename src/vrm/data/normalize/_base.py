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
    # Optional data_files hint forwarded to datasets.load_dataset, used when
    # the HF repo ships multiple JSONs with incompatible schemas (e.g.
    # Osilly/Vision-R1-cold has two unrelated JSONs at the repo root).
    data_files: str | None = None
    default_verifier: str = "exact_numeric"


SYSTEM_PROMPT = (
    "You are a careful visual reasoner. Solve step-by-step. "
    "Put your reasoning in <think>...</think> and your final answer in "
    "<answer>...</answer>."
)


def _verifier_for(answer_type: str) -> str:
    if answer_type == "multiple_choice":
        return "normalize_choice"
    if answer_type == "latex_math":
        return "math_equal"
    if answer_type == "span":
        return "span_match"
    return "exact_numeric"
