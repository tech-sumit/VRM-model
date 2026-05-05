"""Multimodal VL inference for pass@K filtering and Stage 3 rejection sampling.

Primary path (default): **Hugging Face Transformers** + Qwen2.5-VL — the same
stack the model is developed for, without vLLM's engine / attention-backend
matrix on RunPod.

Optional path: ``VRM_VL_BACKEND=vllm`` for higher throughput when the host
stack is known-good (see pod-entrypoint vLLM env block).

Non-GPU environments import this module without loading torch models until
``generate_responses`` runs.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vrm.data.schema import Record

# vLLM only: spawn avoids fork-after-R2/asyncio crashes. No-op when using Transformers.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MAX_IMAGES_PER_PROMPT = 1


def _vl_backend() -> str:
    """transformers | vllm (aliases: hf, huggingface)."""
    raw = (
        os.environ.get("VRM_VL_BACKEND")
        or os.environ.get("VRM_FILTER_INFERENCE")  # legacy name
        or "transformers"
    )
    b = raw.strip().lower()
    if b in ("hf", "huggingface"):
        return "transformers"
    return b


def _to_chat_template(rec: Record) -> str:
    """Qwen2.5-VL plain-text chat prefix for **vLLM** (manual vision tokens).

    HF Transformers uses ``apply_chat_template`` on structured messages instead.
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


def _record_to_qwen_hf_messages(rec: Record) -> list[dict[str, Any]]:
    """Build Qwen2.5-VL multimodal chat messages for ``apply_chat_template``."""
    imgs = rec.images[:MAX_IMAGES_PER_PROMPT]
    user_content: list[dict[str, Any]] = []
    for p in imgs:
        user_content.append({"type": "image", "image": str(Path(p).resolve())})

    out: list[dict[str, Any]] = []
    for m in rec.messages:
        if m.role == "assistant":
            continue
        if m.role == "system":
            out.append({"role": "system", "content": m.content})
        elif m.role == "user":
            user_content.append({"type": "text", "text": m.content})

    if not any(x.get("type") == "text" for x in user_content):
        user_content.append({"type": "text", "text": rec.user_text() or ""})

    out.append({"role": "user", "content": user_content})
    return out


def _load_images(paths: list[str]) -> list[Any]:
    from PIL import Image

    return [Image.open(p).convert("RGB") for p in paths]


def _patch_vllm_rope_scaling_check() -> None:
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
        original(rope_scaling)

    _lenient._vrm_patched = True  # type: ignore[attr-defined]
    _vcfg.patch_rope_scaling_dict = _lenient


_LLM_CACHE: dict[str, Any] = {}
_HF_VL_CACHE: dict[str, tuple[Any, Any]] = {}


def _get_hf_vl(model_id: str) -> tuple[Any, Any]:
    if model_id in _HF_VL_CACHE:
        return _HF_VL_CACHE[model_id]

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("VRM_VL_BACKEND=transformers requires CUDA (no GPU visible).")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    _HF_VL_CACHE[model_id] = (model, processor)
    return model, processor


def _get_llm(model_id: str) -> Any:
    if model_id in _LLM_CACHE:
        return _LLM_CACHE[model_id]
    _patch_vllm_rope_scaling_check()
    from vllm import LLM

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


def _generate_responses_transformers(
    records: Sequence[Record],
    *,
    model_id: str,
    n_per_prompt: int,
    temperature: float,
    max_tokens: int,
) -> list[list[str]]:
    import torch

    model, processor = _get_hf_vl(model_id)
    device = next(model.parameters()).device
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise ImportError(
            "VRM_VL_BACKEND=transformers requires optional dependency `qwen-vl-utils` "
            "(see pyproject.toml train extras)."
        ) from e

    out_rows: list[list[str]] = []
    temp = max(float(temperature), 1e-5)

    for rec in records:
        messages = _record_to_qwen_hf_messages(rec)
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        proc_inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        proc_inputs = proc_inputs.to(device)
        in_len = int(proc_inputs["input_ids"].shape[1])

        comps: list[str] = []
        for _ in range(n_per_prompt):
            seed = int(torch.randint(0, 2**31 - 1, (1,), device="cpu").item())
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            with torch.inference_mode():
                gen_ids = model.generate(
                    **proc_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temp,
                    top_p=1.0,
                    pad_token_id=pad_id,
                    generator=gen,
                )
            new_tokens = gen_ids[0, in_len:]
            text_out = processor.tokenizer.decode(
                new_tokens.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            comps.append(text_out)
        out_rows.append(comps)

    return out_rows


def _generate_responses_vllm(
    records: Sequence[Record],
    *,
    model_id: str,
    n_per_prompt: int,
    temperature: float,
    max_tokens: int,
) -> list[list[str]]:
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


def generate_responses(
    records: Sequence[Record],
    *,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    n_per_prompt: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 8192,
) -> list[list[str]]:
    """Return one inner list of ``n_per_prompt`` completion strings per record.

    Backend: ``VRM_VL_BACKEND`` (default ``transformers``) or legacy
    ``VRM_FILTER_INFERENCE``.
    """
    backend = _vl_backend()
    if backend == "vllm":
        return _generate_responses_vllm(
            records,
            model_id=model_id,
            n_per_prompt=n_per_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if backend == "transformers":
        return _generate_responses_transformers(
            records,
            model_id=model_id,
            n_per_prompt=n_per_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported VRM_VL_BACKEND={backend!r}; use transformers or vllm")
