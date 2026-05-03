"""MathVista normalizer."""

from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec, _verifier_for
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    image = raw.get("image") or raw.get("decoded_image")
    answer = (raw.get("answer") or "").strip()
    if not image or not answer:
        return None
    answer_type_raw = (raw.get("question_type") or "").lower()
    answer_type = "multiple_choice" if "multi_choice" in answer_type_raw or raw.get("choices") else "numeric"
    question = raw.get("question") or ""
    if raw.get("choices"):
        choices = raw["choices"]
        question = (
            question + "\nChoices:\n" + "\n".join(f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(choices))
        )
    return Record(
        id=str(raw.get("pid") or raw.get("id") or hash(question)),
        images=[str(image)],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{question}"),
        ],
        answer=answer,
        answer_type=answer_type,  # type: ignore[arg-type]
        verifier=_verifier_for(answer_type),  # type: ignore[arg-type]
        tolerance=0.001 if answer_type == "numeric" else 0.0,
        source="mathvista",
    )


SPEC = NormalizeSpec(
    hf_id="AI4Math/MathVista",
    split="testmini",
    normalize=normalize,
)
