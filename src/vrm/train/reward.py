"""TRL `GRPOTrainer` reward function adapter.

TRL invokes a reward function with batched completions and the matching
records' columns spread out as kwargs (e.g., `verifier=[...]`, `answer=[...]`).
We wrap our deterministic verifiers + format check.
"""

from __future__ import annotations

from collections.abc import Sequence

from vrm.data.verifiers import REGISTRY
from vrm.data.verifiers.format import extract_answer, has_valid_format

RewardFn = "callable"


def _format_reward(
    completions: Sequence[str],
    **_: object,
) -> list[float]:
    return [1.0 if has_valid_format(c) else 0.0 for c in completions]


def _accuracy_reward(
    completions: Sequence[str],
    *,
    verifier: Sequence[str],
    answer: Sequence[str],
    tolerance: Sequence[float] | None = None,
    **_: object,
) -> list[float]:
    out: list[float] = []
    tolerance = tolerance or [0.0] * len(completions)
    for c, v, a, t in zip(completions, verifier, answer, tolerance, strict=False):
        if not has_valid_format(c):
            out.append(0.0)
            continue
        fn = REGISTRY.get(v)
        if fn is None:
            out.append(0.0)
            continue
        out.append(fn(extract_answer(c), {"answer": a, "tolerance": t}))
    return out


def make_reward_funcs() -> list:
    """Returns a list of reward functions in the order TRL expects.

    Their weights (configured in stage2_grpo.yaml) sum to 1.0:
      0.1 * format + 0.9 * accuracy.
    """
    return [_format_reward, _accuracy_reward]
