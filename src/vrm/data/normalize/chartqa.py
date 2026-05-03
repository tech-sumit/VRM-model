"""ChartQA normalizer.

Schema (HuggingFaceM4/ChartQA, split=train, verified 2026-05-03):
  image: PIL.Image (driver persists to disk before this sees it)
  query: str  (the question)
  label: list[str]  (answers, usually length 1)
  human_or_machine: int (0/1 flag, unused)
"""

from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec, _verifier_for
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    image = raw.get("image")
    query = (raw.get("query") or raw.get("question") or "").strip()
    label = raw.get("label") or raw.get("answer")
    if isinstance(label, list):
        label = label[0] if label else ""
    label = (label or "").strip()
    if not image or not query or not label:
        return None
    answer_type = "numeric" if label.replace(".", "", 1).replace("-", "", 1).isdigit() else "span"
    verifier = _verifier_for(answer_type)
    return Record(
        id=str(raw.get("id") or raw.get("imgname") or hash(query)),
        images=[str(image)],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{query}"),
        ],
        answer=label,
        answer_type=answer_type,  # type: ignore[arg-type]
        verifier=verifier,  # type: ignore[arg-type]
        tolerance=0.001 if answer_type == "numeric" else 0.0,
        source="chartqa",
    )


SPEC = NormalizeSpec(
    hf_id="HuggingFaceM4/ChartQA",
    split="train",
    normalize=normalize,
    default_verifier="span_match",
)
