"""Driver for Stage 1 SFT via LLaMA-Factory.

1. Download data shards from HF Hub (or use local --data-dir-override).
2. Materialize a `dataset_info.json` so LLaMA-Factory can find them.
3. Render the YAML with run-specific paths.
4. Subprocess `llamafactory-cli train ...`.
5. On success, upload checkpoint to HF Hub and webhook GitHub.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from vrm.infra.hf_hub import dataset_repo_id, model_repo_id, upload_checkpoint
from vrm.infra.webhook import post_status


def render_yaml(template: str, vars_: dict[str, str]) -> str:
    out = template
    for k, v in vars_.items():
        out = out.replace(f"${{{k}}}", v)
    return out


def _materialize_dataset_info(parquet_dir: Path) -> Path:
    info_dir = parquet_dir.parent
    info_path = info_dir / "dataset_info.json"
    info = {
        "vrm_sft": {
            "file_name": "*.parquet",
            "subset": "default",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
            "folder": str(parquet_dir),
        }
    }
    info_path.write_text(json.dumps(info, indent=2))
    return info_dir


def _download_dataset_to(local: Path, repo_id: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local),
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=["*.parquet", "*.json"],
    )


@click.command()
@click.option("--config", type=click.Path(path_type=Path), required=True)
@click.option("--data-version", required=True)
@click.option("--run-name", required=True)
@click.option(
    "--mode",
    type=click.Choice(["full", "lora", "rejection"]),
    default="full",
    show_default=True,
)
@click.option(
    "--data-dir-override",
    type=click.Path(path_type=Path),
    default=None,
    help="Use local parquet dir instead of downloading from HF",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("/workspace/data/runs"),
    show_default=True,
)
def main(
    config: Path,
    data_version: str,
    run_name: str,
    mode: str,
    data_dir_override: Path | None,
    output_dir: Path,
) -> None:
    stage = "sft" if mode != "rejection" else "rejection"
    out_dir = output_dir / f"{stage}-{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if data_dir_override:
        data_dir = data_dir_override
    else:
        data_dir = Path("/workspace/data/datasets") / f"vrm-sft-{data_version}"
        repo = dataset_repo_id("sft", data_version)
        _download_dataset_to(data_dir, repo)

    info_dir = _materialize_dataset_info(data_dir)

    rendered = render_yaml(
        Path(config).read_text(),
        {
            "OUTPUT_DIR": str(out_dir),
            "DATA_DIR": str(info_dir),
            "RUN_NAME": run_name,
        },
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=out_dir) as f:
        f.write(rendered)
        cfg_path = Path(f.name)

    cmd = ["llamafactory-cli", "train", str(cfg_path)]
    click.echo(f"[stage1_sft] cmd: {' '.join(cmd)}")
    env = os.environ | {
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "vrm-7b"),
        "WANDB_RUN_NAME": run_name,
    }
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        post_status(
            "failure",
            task=stage,
            run_name=run_name,
            payload={"return_code": rc},
        )
        sys.exit(rc)

    repo_id = model_repo_id(stage, run_name)
    final_dir = out_dir
    # If the trainer wrote a `final/` subdir use it; otherwise use the run dir.
    if (final_dir / "final").exists():
        final_dir = final_dir / "final"
    commit = upload_checkpoint(final_dir, repo_id)
    shutil.rmtree(out_dir, ignore_errors=False)  # keep volume clean
    post_status(
        "completed",
        task=stage,
        run_name=run_name,
        payload={"checkpoint": repo_id, "commit": commit},
    )


if __name__ == "__main__":
    main()
