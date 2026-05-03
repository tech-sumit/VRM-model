# Sub-plan 08 — Stage 3 (optional) Rejection-Sampled SFT

> Index: [`../plan.md`](../plan.md) · Depends on: [06 Stage 1](06-stage1-sft.md) · [07 Stage 2](07-stage2-grpo.md) · Status: ☐ (optional)

**Goal:** After Stage 2 RL converges, sample G=16 responses per problem on the SFT pool with the post-RL model, keep only correct + format-valid responses, run one more SFT epoch on this curated set. Spec §4.3 estimates +1-3 points on hard benchmarks (MathVision, OlympiadBench).

**Architecture:** Two scripts:
1. `vrm.train.stage3_sample` — uses vLLM to generate G=16 completions per record, scores with our verifiers, keeps correct ones, writes a parquet shard `vrm-7b-stage3-{run_name}` to HF Hub.
2. Reuse Stage 1 SFT driver (`vrm.train.stage1_sft`) with the new dataset and a different YAML (`configs/stage3_rejection_sft.yaml`) — same machinery, different inputs.

**Tech Stack:** vLLM · TRL/LLaMA-Factory · same as Stage 1.

---

### Task 1: Rejection-sampling generator (`vrm/train/stage3_sample.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/train/stage3_sample.py`
- Create: `projects/VRM-model/tests/unit/test_stage3_filter.py`

- [ ] **Step 1: Failing test for the keep-fn logic**

```python
# projects/VRM-model/tests/unit/test_stage3_filter.py
from vrm.train.stage3_sample import keep_correct_responses


def _good(answer="42"):
    think = " ".join(["x"] * 60)
    return f"<think>{think}</think><answer>{answer}</answer>"


def test_keep_correct_filters_to_format_and_correct():
    rec = {"verifier": "exact_numeric", "answer": "42", "tolerance": 0.0}
    candidates = [_good("42"), _good("WRONG"), "no fmt", _good("42")]
    kept = keep_correct_responses(rec, candidates)
    assert kept == [_good("42"), _good("42")]


def test_keep_correct_returns_empty_when_none_match():
    rec = {"verifier": "exact_numeric", "answer": "42", "tolerance": 0.0}
    candidates = ["no fmt", _good("WRONG")]
    assert keep_correct_responses(rec, candidates) == []
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/train/stage3_sample.py`:

```python
"""Stage 3: rejection-sampled SFT data generation.

Uses the post-RL model (vLLM) to generate G completions per Stage-1 SFT record,
keeps verifier-correct + format-valid completions, writes augmented parquet shards.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import click
import pyarrow as pa
import pyarrow.parquet as pq

from vrm.data.schema import Message, Record
from vrm.data.verifiers import REGISTRY
from vrm.data.verifiers.format import extract_answer, has_valid_format
from vrm.infra.hf_hub import dataset_repo_id, upload_dataset_shards


def keep_correct_responses(gold: dict, completions: Sequence[str]) -> list[str]:
    fn = REGISTRY[gold["verifier"]]
    out = []
    for c in completions:
        if not has_valid_format(c):
            continue
        if fn(extract_answer(c), gold) == 1.0:
            out.append(c)
    return out


def _generate_with_vllm(records: list[Record], *, model_id: str, n: int = 16) -> list[list[str]]:
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    llm = LLM(model=model_id, dtype="bfloat16", limit_mm_per_prompt={"image": 4})
    sp = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=8192)
    prompts = []
    for r in records:
        text = "\n".join(f"<|im_start|>{m.role}\n{m.content}<|im_end|>" for m in r.messages
                         if m.role != "assistant") + "\n<|im_start|>assistant\n"
        prompts.append({
            "prompt": text,
            "multi_modal_data": {"image": [Image.open(p).convert("RGB") for p in r.images]},
        })
    outputs = llm.generate(prompts, sp)
    return [[o.text for o in out.outputs] for out in outputs]


@click.command()
@click.option("--rl-checkpoint", required=True)
@click.option("--source-data-version", required=True, help="SFT data version to sample from")
@click.option("--out-data-version", required=True, help="New version to publish")
@click.option("--n", default=16, show_default=True, help="completions per record")
@click.option("--limit", type=int, default=None)
def main(rl_checkpoint: str, source_data_version: str, out_data_version: str, n: int, limit: int | None) -> None:
    workspace = Path(os.environ.get("WORKSPACE", "/workspace/data"))
    src_dir = workspace / f"sft-{source_data_version}"
    out_dir = workspace / f"sft-{out_data_version}-stage3"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[Record] = []
    for shard in sorted(src_dir.glob("shard-*.parquet")):
        for row in pq.read_table(shard).to_pylist():
            records.append(Record.model_validate(row))
            if limit and len(records) >= limit:
                break
        if limit and len(records) >= limit:
            break

    completions_per_record = _generate_with_vllm(records, model_id=rl_checkpoint, n=n)

    out_buf: list[dict] = []
    for rec, comps in zip(records, completions_per_record):
        gold = {"verifier": rec.verifier, "answer": rec.answer, "tolerance": rec.tolerance}
        kept = keep_correct_responses(gold, comps)
        if not kept:
            continue
        # One row per kept completion (data augmentation) — capped to 4 per record to limit dataset blow-up.
        for k in kept[:4]:
            new = rec.model_copy(update={
                "messages": list(rec.messages) + [Message(role="assistant", content=k)],
            })
            out_buf.append(json.loads(new.model_dump_json()))

    shard_idx = 0
    for i in range(0, len(out_buf), 5000):
        pq.write_table(pa.Table.from_pylist(out_buf[i:i + 5000]),
                       out_dir / f"shard-{shard_idx:05d}.parquet")
        shard_idx += 1

    repo = dataset_repo_id("sft-stage3", out_data_version)
    url = upload_dataset_shards(out_dir, repo)
    click.echo(f"published {len(out_buf)} records to {repo}: {url}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run, commit**

```bash
make test
git add projects/VRM-model/src/vrm/train/stage3_sample.py projects/VRM-model/tests/unit/test_stage3_filter.py
git commit -m "vrm: add Stage 3 rejection-sampling generator (vLLM-backed)"
```

---

### Task 2: Stage 3 SFT YAML (mostly identical to Stage 1 full FT, just smaller LR)

**Files:**
- Create: `projects/VRM-model/configs/stage3_rejection_sft.yaml`

- [ ] **Step 1: Write the YAML (lower LR, fewer epochs)**

`projects/VRM-model/configs/stage3_rejection_sft.yaml`:

```yaml
### model
model_name_or_path: ${SFT_CHECKPOINT}    # the post-RL checkpoint (we re-SFT on top of it)
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
deepspeed: configs/_deepspeed_zero3.json

### dataset (the new rejection-sampled set)
dataset_dir: ${DATA_DIR}
dataset: vrm_sft
template: qwen2_vl
cutoff_len: 8192
preprocessing_num_workers: 16
overwrite_cache: true

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 5
save_steps: 200
plot_loss: true
overwrite_output_dir: true
report_to: wandb
run_name: ${RUN_NAME}

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-6                  # half of Stage 1 — we're closer to optimum
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
optim: adamw_torch
weight_decay: 0.05

### push
push_to_hub: true
hub_model_id: ${HF_REPO}
hub_strategy: every_save
hub_private_repo: true
```

- [ ] **Step 2: Commit**

```bash
git add projects/VRM-model/configs/stage3_rejection_sft.yaml
git commit -m "vrm: add Stage 3 rejection-SFT YAML (lower LR, post-RL checkpoint as base)"
```

---

### Task 3: End-to-end Stage 3 invocation (manual, optional)

- [ ] **Step 1: Generate the rejection-sampled dataset**

```bash
# Inside an 8×H200 pod
python -m vrm.train.stage3_sample \
    --rl-checkpoint tech-sumit/vrm-7b-grpo-2026-05-15 \
    --source-data-version v1 \
    --out-data-version v1-stage3 \
    --n 16
```

- [ ] **Step 2: Run SFT on the new dataset (reuses Stage 1 driver)**

```bash
# pod-entrypoint dispatches `rejection` to vrm.train.stage1_sft (config file differs)
make train-rejection \
    GRPO_CHECKPOINT=tech-sumit/vrm-7b-grpo-2026-05-15 \
    DATA_VERSION=v1-stage3 \
    RUN_NAME=stage3-2026-05-20
```

- [ ] **Step 3: Eval the resulting model**

```bash
make eval CHECKPOINT=tech-sumit/vrm-7b-rejection-stage3-2026-05-20 SUITE=full
```

Expect +1-3 points on MathVision and OlympiadBench-Vision vs the post-Stage-2 model.

---

## Done when

- [ ] `make test` includes the keep-fn test.
- [ ] Stage 3 generation script runs end-to-end on a smoke subset.
- [ ] Stage 3 fine-tuning produces a new HF checkpoint.
- [ ] Eval shows non-regression on MathVista; ideally +1-3 on MathVision.
