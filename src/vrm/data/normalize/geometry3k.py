"""Geometry3K normalizer.

Schema (hiyouga/geometry3k, split=train, verified 2026-05-03):
  images: list[PIL.Image]  (usually length 1)
  problem: str  (includes '<image>' placeholder)
  answer: str  (single letter A/B/C/D or number)

The driver converts PIL elements in list columns to str paths before this
runs, so raw["images"] is list[str].
"""

from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec, _verifier_for
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    images = raw.get("images") or ([raw["image"]] if raw.get("image") else [])
    images = [str(i) for i in images if i]
    problem = (raw.get("problem") or raw.get("question") or "").strip()
    answer = (raw.get("answer") or "").strip()
    if not images or not problem or not answer:
        return None
    # Problem string in this dataset already contains "<image>" tokens;
    # preserve them so the VLM knows where the image slots in.
    answer_type = "multiple_choice" if len(answer) == 1 and answer.upper() in "ABCDEFGH" else "numeric"
    verifier = _verifier_for(answer_type)
    return Record(
        id=str(raw.get("id") or hash(problem)),
        images=images,
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=problem),
        ],
        answer=answer,
        answer_type=answer_type,  # type: ignore[arg-type]
        verifier=verifier,  # type: ignore[arg-type]
        tolerance=0.001 if answer_type == "numeric" else 0.0,
        source="geometry3k",
    )


SPEC = NormalizeSpec(
    hf_id="hiyouga/geometry3k",
    split="train",
    normalize=normalize,
    default_verifier="exact_numeric",
)
