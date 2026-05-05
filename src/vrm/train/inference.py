"""vLLM batch inference helper used by the difficulty filter and Stage 3 sampling.

vLLM and torch are import-guarded -- non-GPU environments (CI, dev laptops)
won't crash on import.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from vrm.data.schema import Record

# vLLM V1 spawns a worker subprocess via multiprocessing. Default start method
# is `fork`, which inherits all open boto3/asyncio/HTTP connections from the
# parent process and produces silent crashes on Qwen2.5-VL (engine_core dies
# during multimodal profiling with no stderr). `spawn` avoids it by starting
# from a clean interpreter. Verified: probe LLM in isolation works on `fork`,
# but a vrm.data.build run that has already opened R2 + downloaded 1M images
# crashes consistently. Set BEFORE any vLLM import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# LLM is configured with limit_mm_per_prompt={"image": 1}; records with more
# images are truncated to the first N below.
MAX_IMAGES_PER_PROMPT = 1


def _to_chat_template(rec: Record) -> str:
    """Qwen2.5-VL chat template.

    The processor expects one `<|vision_start|><|image_pad|><|vision_end|>`
    block per image, placed inside the user turn before the text. vLLM
    replaces each `<|image_pad|>` with the real image feature tokens and
    fails with 'Expected N prompt updates corresponding to N image items'
    when the placeholders are missing.
    """
    n_images = min(len(rec.images), MAX_IMAGES_PER_PROMPT)
    image_prefix = "<|vision_start|><|image_pad|><|vision_end|>" * n_images
    parts = []
    first_user_patched = False
    for m in rec.messages:
        if m.role == "assistant":
            continue
        content = m.content
        if not first_user_patched and m.role == "user" and image_prefix:
            content = image_prefix + content
            first_user_patched = True
        parts.append(f"<|im_start|>{m.role}\n{content}<|im_end|>")
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


# Module-level singleton: vLLM model load + KV cache profiling costs ~60s
# and 15.6 GiB VRAM. The difficulty filter calls generate_responses once per
# record, so rebuilding the engine every call would be catastrophic (and
# was the root cause of the "silent restart" behavior we were debugging).
_LLM_CACHE: dict[str, Any] = {}


def _get_llm(model_id: str) -> Any:
    if model_id in _LLM_CACHE:
        return _LLM_CACHE[model_id]
    _patch_vllm_rope_scaling_check()
    from vllm import LLM

    # enforce_eager=True disables torch.compile + cudagraph capture, which
    # have proven fragile on Qwen2.5-VL in vLLM 0.8.5 (engine core crashes
    # silently during inductor compile). Filter pass@K is IO-bound anyway.
    # max_model_len must fit worst-case multimodal embeddings: Qwen2.5-VL
    # reserves up to ~16K tokens per image (65K for 4 images). Cap to
    # limit_mm_per_prompt={"image": 1} so single-image prompts -- the vast
    # majority of the corpus -- profile cleanly; records with more images
    # will be truncated by vLLM, which is acceptable for a difficulty-only
    # pass. mm_processor_kwargs caps per-image tokens to keep prefill sane.
    # Pass use_fast=True so HF does not pick the slow image processor (warns in
    # logs; some slow-processor + vLLM paths are crash-prone on RunPod).
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "use_fast": True,
        },
        enforce_eager=True,
        max_model_len=32768,
        gpu_memory_utilization=0.85,
    )
    _LLM_CACHE[model_id] = llm
    return llm


def generate_responses(
    records: Sequence[Record],
    *,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    n_per_prompt: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 8192,
) -> list[list[str]]:
    """Returns one inner list of n_per_prompt strings per record."""
    from vllm import SamplingParams

    llm = _get_llm(model_id)
    sp = SamplingParams(n=n_per_prompt, temperature=temperature, top_p=1.0, max_tokens=max_tokens)
    prompts = []
    for r in records:
        imgs = _load_images(r.images[:MAX_IMAGES_PER_PROMPT])
        entry: dict = {"prompt": _to_chat_template(r)}
        if imgs:
            entry["multi_modal_data"] = {"image": imgs}
        prompts.append(entry)
    outputs = llm.generate(prompts, sp)
    return [[o.text for o in out.outputs] for out in outputs]
