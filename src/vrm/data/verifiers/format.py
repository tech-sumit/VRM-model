"""Format verifier: per spec §3.3, response must be `<think>...</think><answer>...</answer>`.

Used both as part of the composite reward and to extract the model's predicted answer.
"""

from __future__ import annotations

import re

_FMT = re.compile(
    r"<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>",
    re.DOTALL,
)
MIN_THINK_TOKENS = 50  # rough whitespace-token count to discourage reward hacking


def has_valid_format(response: str) -> bool:
    m = _FMT.search(response or "")
    if not m:
        return False
    think_tokens = len(m.group("think").split())
    return think_tokens >= MIN_THINK_TOKENS


def extract_answer(response: str) -> str:
    m = _FMT.search(response or "")
    return m.group("answer").strip() if m else ""
