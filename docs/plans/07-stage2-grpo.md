# Sub-plan 07 — Stage 2 GRPO (TRL `GRPOTrainer` + vLLM rollout)

> Index: [`../plan.md`](../plan.md) · Depends on: [03 verifiers](03-data-schema-verifiers.md) · [06 SFT](06-stage1-sft.md) · Status: ☐

**Goal:** Run Stage 2 GRPO RL post-training (spec §4.2) using TRL's `GRPOTrainer` with our deterministic verifier-based reward. Frozen vision encoder, frozen connector, LLM-only updates. Rollout via vLLM (10× faster than HF generate). Produces a final RL'd checkpoint pushed to HF Hub.

**Architecture:** Single Python entrypoint that:
1. Loads the post-SFT checkpoint as both `model` and frozen `ref_model`.
2. Loads the RL parquet shards from HF Hub.
3. Wires our `score()` reward function as TRL's `reward_funcs`.
4. Configures GRPO with spec §4.2 hyperparameters: `num_generations=8`, `temperature=1.0`, `epsilon=0.2`, `epsilon_high=0.28`, `beta=0.001`, dynamic-sampling via `mask_truncated_completions=True`, `loss_type="dapo"`, `scale_rewards="batch"`, vLLM colocate rollout.
5. Periodic eval every 200 steps via VLMEvalKit (sub-plan 09).
6. Snapshots every 100 steps to HF Hub.

**Tech Stack:** TRL ≥0.11 · transformers 4.45 · vLLM 0.6 · DeepSpeed ZeRO-3 · accelerate.

---

### Task 1: GRPO YAML config

**Files:**
- Create: `projects/VRM-model/configs/stage2_grpo.yaml`

- [ ] **Step 1: Write the YAML**

`projects/VRM-model/configs/stage2_grpo.yaml`:

```yaml
# TRL GRPO config — VRM-7B spec §4.2 (DAPO settings on)
model:
  name_or_path: ${SFT_CHECKPOINT}    # set at runtime
  freeze_vision_tower: true
  freeze_multi_modal_projector: true # spec §4.2: connector frozen too
  attn_implementation: flash_attention_2
  torch_dtype: bfloat16

dataset:
  hf_repo: ${RL_HF_REPO}             # set at runtime, e.g. tech-sumit/vrm-7b-rl-v1
  split: train

generation:
  max_prompt_length: 4096
  max_completion_length: 8192
  num_generations: 8
  temperature: 1.0
  top_p: 1.0

grpo:
  beta: 0.001                        # KL coefficient (spec §4.2)
  epsilon: 0.2                       # clip lower (spec §4.2)
  epsilon_high: 0.28                 # clip higher (DAPO clip-higher)
  num_iterations: 1                  # spec §4.2: inner PPO epochs = 1
  loss_type: "dapo"                  # token-level loss summed (DAPO)
  scale_rewards: "batch"             # group-relative within prompt-group
  mask_truncated_completions: true   # DAPO dynamic sampling: drop equal-reward groups
  reward_weights: [0.1, 0.9]         # format · accuracy

training:
  output_dir: ${OUTPUT_DIR}
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32    # global batch ~256 (1 * 32 * 8 GPUs)
  num_generations_per_device: 8
  learning_rate: 1.0e-6
  lr_scheduler_type: constant
  warmup_steps: 0
  max_steps: 1500
  logging_steps: 1
  save_steps: 100
  eval_steps: 200
  bf16: true
  gradient_checkpointing: true
  optim: adamw_torch
  report_to: wandb
  push_to_hub: true
  hub_model_id: ${HF_REPO}
  hub_strategy: every_save
  hub_private_repo: true

vllm:
  enabled: true
  mode: colocate                     # share GPUs with training (no separate server)
  gpu_memory_utilization: 0.45       # leave room for training engine
  tensor_parallel_size: 1
  max_model_len: 12288               # prompt 4K + completion 8K
```

- [ ] **Step 2: Commit**

```bash
git add projects/VRM-model/configs/stage2_grpo.yaml
git commit -m "vrm: add Stage 2 GRPO YAML (DAPO-flavored, vLLM colocate)"
```

---

### Task 2: Reward function adapter

**Files:**
- Create: `projects/VRM-model/src/vrm/train/reward.py`
- Create: `projects/VRM-model/tests/unit/test_reward_adapter.py`

- [ ] **Step 1: Failing test**

`projects/VRM-model/tests/unit/test_reward_adapter.py`:

```python
from vrm.train.reward import make_reward_funcs


def _good(answer="42"):
    think = " ".join(["x"] * 60)
    return f"<think>{think}</think><answer>{answer}</answer>"


def test_reward_funcs_returns_two_callables():
    fns = make_reward_funcs()
    assert len(fns) == 2  # format, accuracy


def test_reward_funcs_signal_is_correct():
    fmt_fn, acc_fn = make_reward_funcs()
    completions = [_good("42"), _good("WRONG"), "no format"]
    golds = [
        {"verifier": "exact_numeric", "answer": "42", "tolerance": 0.0},
        {"verifier": "exact_numeric", "answer": "42", "tolerance": 0.0},
        {"verifier": "exact_numeric", "answer": "42", "tolerance": 0.0},
    ]
    fmt_rewards = fmt_fn(completions=completions, gold=golds)
    acc_rewards = acc_fn(completions=completions, gold=golds)
    assert fmt_rewards == [1.0, 1.0, 0.0]
    assert acc_rewards == [1.0, 0.0, 0.0]
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/train/reward.py`:

```python
"""TRL `GRPOTrainer` reward-function adapter.

TRL expects callables with signature `fn(completions, **kwargs) -> list[float]`,
where `**kwargs` includes any extra columns from the dataset.
We expose two callables (format, accuracy) so TRL can apply per-component weights
via the `reward_weights` config (matching the 0.1·format + 0.9·accuracy of spec §3.3).
"""
from __future__ import annotations

from typing import Callable, Sequence

from vrm.data.verifiers import REGISTRY
from vrm.data.verifiers.format import extract_answer, has_valid_format

RewardFn = Callable[..., list[float]]


def _format_reward(*, completions: Sequence[str], **_: object) -> list[float]:
    return [1.0 if has_valid_format(c) else 0.0 for c in completions]


def _accuracy_reward(*, completions: Sequence[str], gold: Sequence[dict], **_: object) -> list[float]:
    out: list[float] = []
    for c, g in zip(completions, gold):
        if not has_valid_format(c):
            out.append(0.0)
            continue
        fn = REGISTRY.get(g.get("verifier"))
        if fn is None:
            out.append(0.0)
            continue
        out.append(fn(extract_answer(c), g))
    return out


def make_reward_funcs() -> list[RewardFn]:
    """Returns [format_reward, accuracy_reward], matched to reward_weights=[0.1, 0.9]."""
    return [_format_reward, _accuracy_reward]
```

- [ ] **Step 3: Run, commit**

```bash
make test
git add projects/VRM-model/src/vrm/train/reward.py projects/VRM-model/tests/unit/test_reward_adapter.py
git commit -m "vrm: add TRL GRPO reward-function adapter (format + accuracy)"
```

---

### Task 3: GRPO driver (`vrm/train/stage2_grpo.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/train/stage2_grpo.py`

- [ ] **Step 1: Implement (no unit test — TRL is heavy; smoke runs in-pod)**

`projects/VRM-model/src/vrm/train/stage2_grpo.py`:

```python
"""Stage 2 GRPO driver. Loads SFT checkpoint, wires verifier reward, calls TRL `GRPOTrainer`."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click
import yaml

from vrm.infra.hf_hub import dataset_repo_id, model_repo_id
from vrm.infra.webhook import post_status
from vrm.train.reward import make_reward_funcs


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _build_dataset(repo: str, split: str):
    """Loads the RL dataset and shapes each row so TRL can pass it to reward fns.
    Each TRL example needs a `prompt` (multimodal-aware string) and a `gold` dict
    with the verifier metadata.
    """
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(repo, split=split)

    def _to_prompt(rec: dict) -> dict:
        from vrm.data.schema import Record  # noqa: PLC0415

        r = Record.model_validate(rec)
        # Build the chat prompt up to (but not including) the assistant turn.
        chat = []
        for m in r.messages:
            if m.role == "assistant":
                continue
            chat.append({"role": m.role, "content": m.content})
        gold = {"verifier": r.verifier, "answer": r.answer, "tolerance": r.tolerance,
                "answer_type": r.answer_type}
        return {
            "prompt": chat,
            "gold": gold,
            "images": r.images,
        }

    return ds.map(_to_prompt, remove_columns=ds.column_names)


@click.command()
@click.option("--config", required=True, type=click.Path(path_type=Path))
@click.option("--sft-checkpoint", required=True)
@click.option("--data-version", required=True)
@click.option("--run-name", required=True)
def main(config: Path, sft_checkpoint: str, data_version: str, run_name: str) -> None:
    import torch  # noqa: PLC0415
    from accelerate.utils import is_main_process  # noqa: PLC0415
    from transformers import AutoProcessor, AutoModelForVision2Seq  # noqa: PLC0415
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

    cfg = _load_yaml(config)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace/data"))
    output_dir = workspace / "outputs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rl_repo = dataset_repo_id("rl", data_version)
    hf_repo = model_repo_id("grpo", run_name)

    post_status("started", task="grpo", run_name=run_name,
                payload={"sft_checkpoint": sft_checkpoint, "rl_repo": rl_repo, "hf_repo": hf_repo})

    processor = AutoProcessor.from_pretrained(sft_checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        sft_checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation=cfg["model"].get("attn_implementation", "flash_attention_2"),
        trust_remote_code=True,
    )
    if cfg["model"].get("freeze_vision_tower"):
        for p in model.visual.parameters():
            p.requires_grad = False
    if cfg["model"].get("freeze_multi_modal_projector"):
        for p in model.visual.merger.parameters():
            p.requires_grad = False

    train_ds = _build_dataset(rl_repo, cfg["dataset"]["split"])

    grpo_cfg = GRPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_steps=cfg["training"]["warmup_steps"],
        max_steps=cfg["training"]["max_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        eval_steps=cfg["training"]["eval_steps"],
        bf16=True,
        gradient_checkpointing=True,
        report_to=cfg["training"]["report_to"],
        push_to_hub=True,
        hub_model_id=hf_repo,
        hub_strategy=cfg["training"]["hub_strategy"],
        hub_private_repo=True,
        # GRPO-specific
        num_generations=cfg["generation"]["num_generations"],
        max_prompt_length=cfg["generation"]["max_prompt_length"],
        max_completion_length=cfg["generation"]["max_completion_length"],
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        beta=cfg["grpo"]["beta"],
        epsilon=cfg["grpo"]["epsilon"],
        epsilon_high=cfg["grpo"]["epsilon_high"],
        num_iterations=cfg["grpo"]["num_iterations"],
        loss_type=cfg["grpo"]["loss_type"],
        scale_rewards=cfg["grpo"]["scale_rewards"],
        mask_truncated_completions=cfg["grpo"]["mask_truncated_completions"],
        reward_weights=cfg["grpo"]["reward_weights"],
        # vLLM colocate
        use_vllm=cfg["vllm"]["enabled"],
        vllm_mode=cfg["vllm"]["mode"],
        vllm_gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
        vllm_tensor_parallel_size=cfg["vllm"]["tensor_parallel_size"],
        vllm_max_model_len=cfg["vllm"]["max_model_len"],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=make_reward_funcs(),
        args=grpo_cfg,
        train_dataset=train_ds,
    )

    try:
        trainer.train()
    except Exception as e:  # noqa: BLE001
        post_status("failure", task="grpo", run_name=run_name, payload={"error": str(e)[:1000]})
        raise

    if is_main_process(trainer.accelerator.process_index):
        post_status("completed", task="grpo", run_name=run_name,
                    payload={"hf_repo": hf_repo, "max_steps": cfg["training"]["max_steps"]})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add projects/VRM-model/src/vrm/train/stage2_grpo.py
git commit -m "vrm: add Stage 2 GRPO driver (TRL GRPOTrainer + vLLM colocate + verifier reward)"
```

---

### Task 4: Smoke run (50 steps on 1 H200, no eval)

- [ ] **Step 1: Override config for smoke**

```bash
cd projects/VRM-model
mkdir -p configs/_smoke
cat > configs/_smoke/stage2_grpo.yaml <<'EOF'
# Same as configs/stage2_grpo.yaml but with max_steps: 50, eval_steps: 100, num_generations: 4
EOF
```

(Copy the body from `configs/stage2_grpo.yaml`, change `max_steps`, `eval_steps`, `num_generations`, and `gpu_memory_utilization: 0.5` for single-GPU.)

- [ ] **Step 2: Launch smoke pod**

```bash
export VRM_GPU_COUNT_TRAIN=1
make train-grpo \
    SFT_CHECKPOINT=tech-sumit/vrm-7b-sft-smoke-XXX \
    DATA_VERSION=smoke \
    RUN_NAME=grpo-smoke-$(date +%Y%m%d-%H%M%S)
```

- [ ] **Step 3: Verify in W&B**

- `train/reward` curve appears.
- `train/completion_length` curve appears.
- KL stays well below 1.0.
- No OOM.

- [ ] **Step 4: Tear down pod, capture observations in `docs/runbook.md`**

---

## Done when

- [ ] `make test` includes the reward adapter test.
- [ ] Smoke run hit 50 steps without OOM, reward curve rose, no KL collapse.
- [ ] Sub-plan 11 (CD train-grpo workflow) can call this driver via the pod entrypoint.
