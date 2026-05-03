"""ChartQA normalizer."""

from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    image = raw.get("image") or raw.get("imgname")
    answer = (raw.get("label") or raw.get("answer") or "").strip()
    if not image or not answer:
        return None
    question = raw.get("query") or raw.get("question") or ""
    is_numeric = answer.replace(".", "", 1).replace("-", "", 1).isdigit()
    return Record(
        id=str(raw.get("id") or raw.get("imgname") or hash(question)),
        images=[str(image)],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{question}"),
        ],
        answer=answer,
        answer_type="numeric" if is_numeric else "span",
        verifier="exact_numeric" if is_numeric else "span_match",
        tolerance=0.01 if is_numeric else 0.0,
        source="chartqa",
    )


SPEC = NormalizeSpec(
    hf_id="HuggingFaceM4/ChartQA",
    split="train",
    normalize=normalize,
    default_verifier="span_match",
)
