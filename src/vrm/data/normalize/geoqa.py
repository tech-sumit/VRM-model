"""GeoQA-R1V normalizer.

Schema (leonardPKU/GEOQA_R1V_Train_8K, split=train, verified 2026-05-03):
  image: PIL.Image  (driver persists to path)
  problem: str
  solution: str  (already formatted like "<answer> 145° </answer>" --
                  repack's ground-truth field, NOT a CoT)
"""

from __future__ import annotations

import re

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec, _verifier_for
from vrm.data.schema import Message, Record

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def _extract_answer(solution: str) -> str:
    m = _ANSWER_RE.search(solution or "")
    if m:
        return m.group(1).strip()
    return (solution or "").strip()


def normalize(raw: dict) -> Record | None:
    image = raw.get("image")
    problem = (raw.get("problem") or raw.get("question") or "").strip()
    raw_answer = raw.get("answer") or raw.get("label")
    answer = str(raw_answer).strip() if raw_answer else _extract_answer(raw.get("solution") or "")
    if not image or not problem or not answer:
        return None
    # Strip trailing unit symbols like "°" for numeric equality if pure number.
    stripped = answer.rstrip("°").strip()
    answer_type = "numeric" if stripped.replace(".", "", 1).replace("-", "", 1).isdigit() else "span"
    verifier = _verifier_for(answer_type)
    return Record(
        id=str(raw.get("id") or hash(problem)),
        images=[str(image)],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{problem}"),
        ],
        answer=stripped if answer_type == "numeric" else answer,
        answer_type=answer_type,  # type: ignore[arg-type]
        verifier=verifier,  # type: ignore[arg-type]
        tolerance=0.1 if answer_type == "numeric" else 0.0,
        source="geoqa",
    )


SPEC = NormalizeSpec(
    hf_id="leonardPKU/GEOQA_R1V_Train_8K",
    split="train",
    normalize=normalize,
    default_verifier="exact_numeric",
)
