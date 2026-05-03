"""Driver for Stage 2 GRPO training via TRL `GRPOTrainer`."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import click
import yaml

from vrm.infra.hf_hub import dataset_repo_id, model_repo_id, upload_checkpoint
from vrm.infra.webhook import post_status
from vrm.train.reward import make_reward_funcs


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _build_dataset(repo_id: str, num_proc: int):
    from datasets import load_dataset

    ds = load_dataset(repo_id, split="train", num_proc=num_proc)

    # TRL expects 'prompt' (chat-format list) and the verifier columns flattened.
    def _row(rec):
        prompt = [m for m in rec["messages"] if m["role"] != "assistant"]
        return {
            "prompt": prompt,
            "images": rec["images"],
            "answer": rec["answer"],
            "verifier": rec["verifier"],
            "tolerance": rec.get("tolerance", 0.0),
        }

    return ds.map(_row, remove_columns=ds.column_names, num_proc=num_proc)


@click.command()
@click.option("--config", type=click.Path(path_type=Path), required=True)
@click.option("--sft-checkpoint", required=True, help="HF repo id from Stage 1")
@click.option("--data-version", required=True)
@click.option("--run-name", required=True)
def main(config: Path, sft_checkpoint: str, data_version: str, run_name: str) -> None:
    cfg = _load_yaml(config)

    out_dir = Path(cfg["training"]["output_dir"].replace("${OUTPUT_DIR}", "")) or Path(
        f"/workspace/data/runs/grpo-{run_name}"
    )
    out_dir = Path(f"/workspace/data/runs/grpo-{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_dataset = dataset_repo_id("rl", data_version)

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import GRPOConfig, GRPOTrainer

    processor = AutoProcessor.from_pretrained(sft_checkpoint, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        sft_checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg["model"].get("attn_implementation", "flash_attention_2"),
    )
    if cfg["model"].get("freeze_vision_tower"):
        for p in model.visual.parameters():
            p.requires_grad = False
    if cfg["model"].get("freeze_multi_modal_projector"):
        for p in model.visual.merger.parameters():
            p.requires_grad = False

    train_ds = _build_dataset(rl_dataset, num_proc=cfg["dataset"].get("num_proc", 16))

    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir),
        run_name=run_name,
        learning_rate=cfg["training"]["learning_rate"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        max_steps=cfg["training"]["max_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        logging_steps=cfg["training"]["logging_steps"],
        bf16=cfg["training"]["bf16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        report_to=cfg["training"]["report_to"],
        num_generations=cfg["grpo"]["num_generations"],
        max_completion_length=cfg["generation"]["max_new_tokens"],
        temperature=cfg["generation"]["temperature"],
        beta=cfg["grpo"]["beta_kl"],
        epsilon=cfg["grpo"]["clip_lower"],
        epsilon_high=cfg["grpo"]["clip_upper"],
        loss_type=cfg["grpo"]["loss_aggregation"],
        use_vllm=cfg["vllm"]["enabled"],
        vllm_gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
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
    except Exception as e:
        post_status("failure", task="grpo", run_name=run_name, payload={"error": str(e)})
        sys.exit(1)

    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    repo_id = model_repo_id("grpo", run_name)
    commit = upload_checkpoint(final_dir, repo_id)
    shutil.rmtree(out_dir, ignore_errors=False)
    post_status(
        "completed",
        task="grpo",
        run_name=run_name,
        payload={"checkpoint": repo_id, "commit": commit},
    )


if __name__ == "__main__":
    main()
