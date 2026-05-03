"""vLLM batch inference helper used by the difficulty filter and Stage 3 sampling.

vLLM and torch are import-guarded -- non-GPU environments (CI, dev laptops)
won't crash on import.
"""

from __future__ import annotations

from collections.abc import Sequence

from vrm.data.schema import Record


def _to_chat_template(rec: Record) -> str:
    parts = []
    for m in rec.messages:
        if m.role == "assistant":
            continue
        parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _load_images(paths: list[str]) -> list:
    from PIL import Image

    return [Image.open(p).convert("RGB") for p in paths]


def generate_responses(
    records: Sequence[Record],
    *,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    n_per_prompt: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 8192,
) -> list[list[str]]:
    """Returns one inner list of n_per_prompt strings per record."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 4},
    )
    sp = SamplingParams(n=n_per_prompt, temperature=temperature, top_p=1.0, max_tokens=max_tokens)
    prompts = [
        {
            "prompt": _to_chat_template(r),
            "multi_modal_data": {"image": _load_images(r.images)},
        }
        for r in records
    ]
    outputs = llm.generate(prompts, sp)
    return [[o.text for o in out.outputs] for out in outputs]
