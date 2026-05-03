"""Geo170K (geometry problems) normalizer."""

from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    image = raw.get("image")
    answer = (raw.get("answer") or "").strip()
    if not image or not answer:
        return None
    question = raw.get("question") or raw.get("problem") or ""
    return Record(
        id=str(raw.get("id") or hash(question)),
        images=[str(image)],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{question}"),
        ],
        answer=answer,
        answer_type="numeric",
        verifier="exact_numeric",
        tolerance=0.001,
        source="geo170k",
    )


SPEC = NormalizeSpec(
    hf_id="Luckyjhg/Geo170K",
    split="train",
    normalize=normalize,
)
