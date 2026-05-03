"""Vision-R1-cold (distilled long-CoT) normalizer.

This dataset already contains assistant CoT in `<think>...</think><answer>...</answer>`
format -- we preserve it as the assistant turn, so it's directly SFT-ready.
"""

from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec, _verifier_for
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    image = raw.get("image")
    answer = (raw.get("answer") or "").strip()
    if not image or not answer:
        return None
    answer_type = raw.get("answer_type") or "numeric"
    question = raw.get("question") or raw.get("instruction") or ""
    distilled = raw.get("response") or raw.get("solution") or ""
    messages: list[Message] = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=f"<image>\n{question}"),
    ]
    if distilled:
        messages.append(Message(role="assistant", content=distilled))
    return Record(
        id=str(raw.get("id") or hash(question)),
        images=[str(image)],
        messages=messages,
        answer=answer,
        answer_type=answer_type,  # type: ignore[arg-type]
        verifier=_verifier_for(answer_type),  # type: ignore[arg-type]
        tolerance=0.001 if answer_type == "numeric" else 0.0,
        source="vision_r1_cold",
    )


SPEC = NormalizeSpec(
    hf_id="Osilly/Vision-R1-cold-distill",
    split="train",
    normalize=normalize,
)
