# Sub-plan 06 — Stage 1 SFT (LLaMA-Factory wrapper)

> Index: [`../plan.md`](../plan.md) · Depends on: [01 docker](01-docker-images.md) · [05 filter+distill](05-data-filter-distill.md) · Status: ☐

**Goal:** Run Stage 1 SFT cold-start (spec §4.1) by composing a LLaMA-Factory YAML config and spawning `llamafactory-cli train`. Two recipe variants: full FT (default) and LoRA r=128 (memory fallback). Produces a checkpoint pushed to HF Hub.

**Architecture:** Thin Python wrapper that:
1. Pulls latest filtered+distilled SFT shards from HF Hub.
2. Renders the LLaMA-Factory YAML (templated with run-name, paths, batch size, epochs).
3. Spawns `llamafactory-cli train <yaml>` with FSDP/DeepSpeed launcher.
4. On completion, pushes checkpoint to HF Hub via `vrm.infra.hf_hub`.
5. Posts `completed` webhook to GH.

**Tech Stack:** LLaMA-Factory 0.9 · DeepSpeed ZeRO-3 · TRL · transformers 4.45 · WandB.

---

### Task 1: SFT YAML config templates

**Files:**
- Create: `projects/VRM-model/configs/stage1_sft_full.yaml`
- Create: `projects/VRM-model/configs/stage1_sft_lora.yaml`
- Create: `projects/VRM-model/configs/_deepspeed_zero3.json`

- [ ] **Step 1: Write the full-FT YAML (consumed by `llamafactory-cli train`)**

`projects/VRM-model/configs/stage1_sft_full.yaml`:

```yaml
# LLaMA-Factory Stage 1 SFT (full fine-tune) — VRM-7B spec §4.1
# Templated fields are substituted by vrm.train.stage1_sft at run time:
#   ${OUTPUT_DIR}, ${DATA_DIR}, ${RUN_NAME}, ${HF_REPO}

### model
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true   # spec §4.1 — vision encoder frozen
deepspeed: configs/_deepspeed_zero3.json

### dataset
dataset_dir: ${DATA_DIR}
dataset: vrm_sft
template: qwen2_vl
cutoff_len: 8192
max_samples: null
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 5
save_steps: 500
plot_loss: true
overwrite_output_dir: true
report_to: wandb
run_name: ${RUN_NAME}

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16   # 1 * 16 * 8 GPUs = 128 global batch (spec §4.1)
learning_rate: 2.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
optim: adamw_torch
weight_decay: 0.05
adam_beta1: 0.9
adam_beta2: 0.95

### push to hub on save
push_to_hub: true
hub_model_id: ${HF_REPO}
hub_strategy: every_save
hub_private_repo: true
```

- [ ] **Step 2: Write the LoRA fallback YAML**

`projects/VRM-model/configs/stage1_sft_lora.yaml`:

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.05
freeze_vision_tower: true

### dataset
dataset_dir: ${DATA_DIR}
dataset: vrm_sft
template: qwen2_vl
cutoff_len: 8192
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 5
save_steps: 500
plot_loss: true
overwrite_output_dir: true
report_to: wandb
run_name: ${RUN_NAME}

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true

### push
push_to_hub: true
hub_model_id: ${HF_REPO}
hub_strategy: every_save
hub_private_repo: true
```

- [ ] **Step 3: DeepSpeed ZeRO-3 config**

`projects/VRM-model/configs/_deepspeed_zero3.json`:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": { "enabled": true },
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": { "warmup_min_lr": 0, "warmup_max_lr": "auto",
                "warmup_num_steps": "auto", "total_num_steps": "auto" }
  }
}
```

- [ ] **Step 4: Commit**

```bash
git add projects/VRM-model/configs/stage1_sft_full.yaml \
        projects/VRM-model/configs/stage1_sft_lora.yaml \
        projects/VRM-model/configs/_deepspeed_zero3.json
git commit -m "vrm: add Stage 1 SFT YAML configs (full FT primary + LoRA fallback) + DS ZeRO-3"
```

---

### Task 2: SFT driver (`vrm/train/stage1_sft.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/train/stage1_sft.py`
- Create: `projects/VRM-model/tests/unit/test_stage1_sft_render.py`

- [ ] **Step 1: Failing test for YAML templating**

`projects/VRM-model/tests/unit/test_stage1_sft_render.py`:

```python
from pathlib import Path

from vrm.train.stage1_sft import render_yaml


def test_render_substitutes_placeholders(tmp_path, repo_root):
    src = repo_root / "configs/stage1_sft_full.yaml"
    out = render_yaml(
        src,
        output_dir=str(tmp_path / "out"),
        data_dir=str(tmp_path / "data"),
        run_name="test-run",
        hf_repo="tech-sumit/vrm-7b-sft-test",
    )
    text = Path(out).read_text()
    assert "test-run" in text
    assert str(tmp_path / "out") in text
    assert "tech-sumit/vrm-7b-sft-test" in text
    assert "${OUTPUT_DIR}" not in text
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/train/stage1_sft.py`:

```python
"""Stage 1 SFT driver. Renders LLaMA-Factory YAML and spawns `llamafactory-cli train`."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from string import Template

import click

from vrm.infra.hf_hub import model_repo_id
from vrm.infra.webhook import post_status


def render_yaml(src: Path, *, output_dir: str, data_dir: str, run_name: str, hf_repo: str) -> Path:
    """Substitute ${OUTPUT_DIR}, ${DATA_DIR}, ${RUN_NAME}, ${HF_REPO} in a YAML template."""
    raw = Path(src).read_text()
    rendered = Template(raw).safe_substitute(
        OUTPUT_DIR=output_dir, DATA_DIR=data_dir, RUN_NAME=run_name, HF_REPO=hf_repo,
    )
    out_path = Path(output_dir) / f"_rendered_{Path(src).name}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)
    return out_path


def _materialize_dataset_info(data_dir: Path) -> None:
    """LLaMA-Factory needs a `dataset_info.json` describing local datasets.
    We register `vrm_sft` pointing at all parquet shards in data_dir.
    """
    shards = sorted(p.name for p in data_dir.glob("shard-*.parquet"))
    info = {
        "vrm_sft": {
            "file_name": shards,
            "formatting": "sharegpt",  # records have role/content lists in `messages`
            "columns": {
                "messages": "messages",
                "images": "images",
            },
        }
    }
    (data_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))


@click.command()
@click.option("--config", required=True, type=click.Path(path_type=Path))
@click.option("--data-version", required=True, help="HF dataset suffix, e.g. v3")
@click.option("--run-name", required=True)
@click.option("--mode", type=click.Choice(["full", "lora"]), default="full")
@click.option("--data-dir-override", type=click.Path(path_type=Path), default=None,
              help="If set, skip HF download and use a local dir of parquet shards (for smoke runs)")
def main(config: Path, data_version: str, run_name: str, mode: str, data_dir_override: Path | None) -> None:
    workspace = Path(os.environ.get("WORKSPACE", "/workspace/data"))
    data_dir = data_dir_override or workspace / f"sft-{data_version}"
    output_dir = workspace / "outputs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_repo = model_repo_id("sft", run_name)

    if not data_dir_override:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from vrm.infra.hf_hub import dataset_repo_id  # noqa: PLC0415
        repo = dataset_repo_id("sft", data_version)
        snapshot_download(repo, repo_type="dataset", local_dir=str(data_dir),
                          token=os.environ.get("HF_TOKEN"))

    _materialize_dataset_info(data_dir)
    rendered = render_yaml(config, output_dir=str(output_dir), data_dir=str(data_dir),
                            run_name=run_name, hf_repo=hf_repo)

    post_status("started", task="sft", run_name=run_name,
                payload={"hf_repo": hf_repo, "data_version": data_version, "mode": mode})

    nproc = int(os.environ.get("VRM_GPU_COUNT_TRAIN", "8"))
    cmd = ["llamafactory-cli", "train", str(rendered)]
    print(f"[stage1_sft] launching: FORCE_TORCHRUN=1 NPROC={nproc} {' '.join(cmd)}", flush=True)
    env = {**os.environ, "FORCE_TORCHRUN": "1", "NPROC_PER_NODE": str(nproc)}
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        post_status("failure", task="sft", run_name=run_name, payload={"exit_code": rc})
        raise SystemExit(rc)

    post_status("completed", task="sft", run_name=run_name,
                payload={"hf_repo": hf_repo, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run, commit**

```bash
make test
git add projects/VRM-model/src/vrm/train/stage1_sft.py projects/VRM-model/tests/unit/test_stage1_sft_render.py
git commit -m "vrm: add Stage 1 SFT driver (LLaMA-Factory wrapper with templated YAML)"
```

---

### Task 3: Smoke run (1% data, 100 steps, single H200) — manual checklist

This is a manual verification step, not code; copy/paste the commands.

- [ ] **Step 1: Provision a single-H200 secure pod**

```bash
cd projects/VRM-model
export VRM_GPU_COUNT_TRAIN=1
make train-sft DATA_VERSION=smoke RUN_NAME=sft-smoke-$(date +%Y%m%d-%H%M%S)
```

- [ ] **Step 2: SSH into the pod and tail logs**

```bash
# RunPod console → Connect → SSH command
tail -f /workspace/data/outputs/sft-smoke-*/runs/*/events.out.tfevents.*
```

- [ ] **Step 3: Verify**

- Loss decreases from ~3 → ~1.5 over the first 100 steps.
- W&B run appears with curves.
- `outputs/.../trainer_state.json` shows `epoch ≈ 0.05`.

- [ ] **Step 4: Tear down the pod**

```bash
uv run vrm runpod destroy <POD_ID>
```

- [ ] **Step 5: Capture the W&B URL and the HF checkpoint URL in `docs/runbook.md`**

(Done in sub-plan 13.)

---

## Done when

- [ ] `make test` includes the YAML templating test.
- [ ] Smoke run hit 100 steps without OOM, loss decreased, checkpoint appeared on HF Hub.
- [ ] Sub-plan 09 (eval) can use the smoke checkpoint as input.
