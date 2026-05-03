"""TabMWP (numeric tabular math word problems) normalizer."""

from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    image = raw.get("image") or raw.get("table_image")
    answer = (raw.get("answer") or "").strip()
    if not image or not answer:
        return None
    question = raw.get("question") or ""
    return Record(
        id=str(raw.get("id") or raw.get("pid") or hash(question)),
        images=[str(image)],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{question}"),
        ],
        answer=answer,
        answer_type="numeric",
        verifier="exact_numeric",
        tolerance=0.001,
        source="tabmwp",
    )


SPEC = NormalizeSpec(
    hf_id="TabMWP/TabMWP",
    split="train",
    normalize=normalize,
)
