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


def _patch_vllm_rope_scaling_check() -> None:
    """Neutralize vLLM 0.8.x's strict `rope_scaling` conflict check.

    Qwen2.5-VL ships a hub config with both legacy `type=mrope` and modern
    `rope_type=default` in `rope_scaling`. vLLM's `patch_rope_scaling_dict`
    hard-raises on the conflict *inside* `get_config`, before any
    `hf_overrides` are consulted. The legacy `type` is informational; dropping
    it is the upstream-recommended fix. We monkey-patch the check to strip the
    conflicting key rather than raise.
    """
    try:
        from vllm.transformers_utils import config as _vcfg
    except Exception:
        return
    original = getattr(_vcfg, "patch_rope_scaling_dict", None)
    if original is None or getattr(original, "_vrm_patched", False):
        return

    def _lenient(rope_scaling: dict) -> None:
        if isinstance(rope_scaling, dict) and "rope_type" in rope_scaling and "type" in rope_scaling:
            rope_scaling.pop("type", None)
        # Delegate to the original implementation for any remaining normalization.
        original(rope_scaling)

    _lenient._vrm_patched = True  # type: ignore[attr-defined]
    _vcfg.patch_rope_scaling_dict = _lenient


def generate_responses(
    records: Sequence[Record],
    *,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    n_per_prompt: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 8192,
) -> list[list[str]]:
    """Returns one inner list of n_per_prompt strings per record."""
    _patch_vllm_rope_scaling_check()
    from vllm import LLM, SamplingParams

    # enforce_eager=True disables torch.compile + cudagraph capture, which
    # have proven fragile on Qwen2.5-VL in vLLM 0.8.5 (engine core crashes
    # silently during inductor compile). Filter pass@K is IO-bound anyway.
    # max_model_len is capped to keep KV cache reasonable (Qwen2.5-VL's
    # default 128K context blows up profiling on a single H200).
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 4},
        enforce_eager=True,
        max_model_len=16384,
        gpu_memory_utilization=0.85,
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
