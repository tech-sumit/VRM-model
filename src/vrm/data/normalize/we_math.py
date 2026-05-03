"""We-Math normalizer."""

from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec, _verifier_for
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    image = raw.get("image_path") or raw.get("image")
    answer = (raw.get("answer") or "").strip()
    if not image or not answer:
        return None
    answer_type = raw.get("answer_type") or ("multiple_choice" if raw.get("options") else "numeric")
    question = raw.get("question") or ""
    if raw.get("options"):
        opts = raw["options"]
        if isinstance(opts, dict):
            question = question + "\n" + "\n".join(f"({k}) {v}" for k, v in opts.items())
        elif isinstance(opts, list):
            question = question + "\n" + "\n".join(f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(opts))
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
        source="we_math",
    )


SPEC = NormalizeSpec(
    hf_id="We-Math/We-Math",
    split="train",
    normalize=normalize,
)
