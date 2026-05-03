#!/usr/bin/env bash
# Smoke-test base Qwen2.5-VL-7B-Instruct inference.
# On a non-GPU host, only the HF model card is fetched; full inference runs in-pod.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[smoke] No GPU detected -- verifying model card downloads from HF (no inference)."
    uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info('Qwen/Qwen2.5-VL-7B-Instruct')
print(f'OK: model={info.modelId} sha={info.sha[:8]} downloads={info.downloads}')
"
    exit 0
fi

# In-GPU smoke: reference inference on a single image.
uv run --extra train python -c "
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

print('CUDA available:', torch.cuda.is_available(), 'device count:', torch.cuda.device_count())
model_id = 'Qwen/Qwen2.5-VL-7B-Instruct'
proc = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map='cuda:0',
)
print('Loaded:', model.config.model_type, '| hidden:', model.config.hidden_size)
print('Smoke OK.')
"
