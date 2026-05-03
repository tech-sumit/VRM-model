"""MAVIS-Geometry normalizer.

The full 834K MAVIS-Instruct corpus was never open-sourced; only
MAVIS-Geometry and MAVIS-Function were released. We use MAVIS-Geometry
(the larger math-reasoning subset).

Schema (2026-05-03, CaraJ/MAVIS-Geometry, split=visualization):
  image_text_lite      : image with minimal rendered text (preferred)
  answer_index         : int (index into choices)
  choices              : list[str]
  text_en              : dict with keys including text_lite_question,
                         vision_dominant_question, CoT_reasoning, ...

The driver pre-converts PIL image columns to relative paths before this
runs, so raw["image_text_lite"] is a string path here.
"""

from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec
from vrm.data.schema import Message, Record

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def normalize(raw: dict) -> Record | None:
    image = raw.get("image_text_lite") or raw.get("image_vision_dominant")
    text_en = raw.get("text_en") or {}
    if isinstance(text_en, str):
        question = text_en.strip()
    elif isinstance(text_en, dict):
        question = (
            text_en.get("text_lite_question")
            or text_en.get("vision_dominant_question")
            or text_en.get("text_dominant_question")
            or ""
        ).strip()
    else:
        question = ""
    choices = raw.get("choices") or []
    answer_index = raw.get("answer_index")
    if image is None or not question or not choices or answer_index is None:
        return None
    try:
        answer_index = int(answer_index)
    except (TypeError, ValueError):
        return None
    if not 0 <= answer_index < len(choices):
        return None
    answer_letter = _LETTERS[answer_index]
    choices_fmt = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return Record(
        id=str(raw.get("id") or hash(question)),
        images=[str(image)],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{question}\n\n{choices_fmt}"),
        ],
        answer=answer_letter,
        answer_type="multiple_choice",
        verifier="normalize_choice",
        tolerance=0.0,
        source="mavis",
    )


SPEC = NormalizeSpec(
    hf_id="CaraJ/MAVIS-Geometry",
    split="visualization",
    normalize=normalize,
    default_verifier="normalize_choice",
)
