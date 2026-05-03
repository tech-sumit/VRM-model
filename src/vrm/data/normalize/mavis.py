"""MAVIS-Instruct normalizer.

Expected raw fields: id, image (path or PIL), question, answer, answer_type.
HF dataset id is approximate -- verify against the actual dataset card before
running data-build.
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
    verifier = _verifier_for(answer_type)
    question = raw.get("question") or ""
    return Record(
        id=str(raw.get("id") or raw.get("uid") or hash(question)),
        images=[str(image)],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{question}"),
        ],
        answer=answer,
        answer_type=answer_type,  # type: ignore[arg-type]
        verifier=verifier,  # type: ignore[arg-type]
        tolerance=0.001 if answer_type == "numeric" else 0.0,
        source="mavis",
    )


SPEC = NormalizeSpec(
    hf_id="PKU-Alignment/MAVIS-Instruct",
    split="train",
    normalize=normalize,
    default_verifier="exact_numeric",
)
