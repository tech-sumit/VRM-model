"""Shared helper for LLaVA-style conversations-format datasets.

Many VLM SFT datasets (MathV360K, Vision-R1-cold, Geo170K, LLaVA-OneVision,
etc.) ship as a JSONL of records shaped like:

    {"id": ..., "image": "<relative-path>", "conversations": [
        {"from": "human", "value": "<image>\nQuestion..."},
        {"from": "gpt",   "value": "Answer..."},
    ]}

The image string is a RELATIVE path inside a companion zip archive that
these repos distribute separately. We accept such rows even when the image
file cannot be resolved locally -- Record.images stores the relative path
and the training pipeline is responsible for resolving it at load time
(or skipping records with missing images).
"""

from __future__ import annotations

import re

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec, _verifier_for
from vrm.data.schema import Message, Record

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def _extract_qa(conversations: list) -> tuple[str, str]:
    q = a = ""
    for turn in conversations or []:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("from") or "").lower()
        val = (turn.get("value") or "").strip()
        if role in ("human", "user") and not q:
            q = val
        elif role in ("gpt", "assistant") and not a:
            a = val
    return q, a


def _short_answer(full: str) -> str:
    """Extract final answer from <answer>X</answer> tags or fall back to
    the raw string (truncated). Keeps full CoT as the assistant message."""
    m = _ANSWER_TAG_RE.search(full)
    if m:
        return m.group(1).strip()
    # Heuristic: last short line often holds the final answer.
    lines = [line.strip() for line in full.splitlines() if line.strip()]
    if lines and len(lines[-1]) <= 80:
        return lines[-1]
    return full[:200]


def make_normalizer(source: str):
    def normalize(raw: dict) -> Record | None:
        image = raw.get("image") or raw.get("images")
        if isinstance(image, list):
            image = image[0] if image else None
        if not image:
            return None
        question, assistant = _extract_qa(raw.get("conversations") or [])
        if not question or not assistant:
            return None
        final_answer = _short_answer(assistant)
        if not final_answer:
            return None
        answer_type = "numeric" if final_answer.replace(".", "", 1).replace("-", "", 1).isdigit() else "span"
        verifier = _verifier_for(answer_type)
        return Record(
            id=str(raw.get("id") or hash(question)),
            images=[str(image)],
            messages=[
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=question),
                Message(role="assistant", content=assistant),
            ],
            answer=final_answer,
            answer_type=answer_type,  # type: ignore[arg-type]
            verifier=verifier,  # type: ignore[arg-type]
            tolerance=0.001 if answer_type == "numeric" else 0.0,
            source=source,
        )

    return normalize


def make_spec(source: str, hf_id: str, split: str = "train", **kw) -> NormalizeSpec:
    return NormalizeSpec(
        hf_id=hf_id,
        split=split,
        normalize=make_normalizer(source),
        default_verifier="span_match",
        **kw,
    )
