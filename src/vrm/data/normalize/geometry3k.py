"""Geometry3K (multiple-choice) normalizer."""

from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    image = raw.get("image")
    answer = (raw.get("answer") or "").strip()
    if not image or not answer:
        return None
    question = raw.get("question") or raw.get("problem_text") or ""
    choices = raw.get("choices") or raw.get("options")
    if isinstance(choices, list):
        question = (
            question + "\nChoices:\n" + "\n".join(f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(choices))
        )
    return Record(
        id=str(raw.get("id") or hash(question)),
        images=[str(image)],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{question}"),
        ],
        answer=answer,
        answer_type="multiple_choice",
        verifier="normalize_choice",
        tolerance=0.0,
        source="geometry3k",
    )


SPEC = NormalizeSpec(
    hf_id="InfiMM/Geometry3K",
    split="train",
    normalize=normalize,
    default_verifier="normalize_choice",
)
