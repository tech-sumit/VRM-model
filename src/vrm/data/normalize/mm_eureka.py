"""MM-Eureka K12 normalizer."""

from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec, _verifier_for
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    image = raw.get("image") or raw.get("image_path")
    answer = (raw.get("answer") or raw.get("ground_truth") or "").strip()
    if not image or not answer:
        return None
    answer_type = raw.get("answer_type") or (
        "multiple_choice" if len(answer) == 1 and answer.upper() in "ABCDEFGH" else "numeric"
    )
    question = raw.get("question") or raw.get("problem") or ""
    return Record(
        id=str(raw.get("id") or hash(question)),
        images=[str(image)],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{question}"),
        ],
        answer=answer,
        answer_type=answer_type,  # type: ignore[arg-type]
        verifier=_verifier_for(answer_type),  # type: ignore[arg-type]
        tolerance=0.001 if answer_type == "numeric" else 0.0,
        source="mm_eureka",
    )


SPEC = NormalizeSpec(
    hf_id="FanqingM/MM-Eureka-Dataset",
    split="train",
    normalize=normalize,
)
