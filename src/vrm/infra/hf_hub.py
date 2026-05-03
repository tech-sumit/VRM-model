"""Minimal HuggingFace Hub helpers -- repo-id naming + checkpoint upload."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

DEFAULT_ORG = os.environ.get("HF_ORG", "tech-sumit")


def dataset_repo_id(stage: str, data_version: str, org: str = DEFAULT_ORG) -> str:
    """e.g. dataset_repo_id('sft', 'v3') -> 'tech-sumit/vrm-7b-sft-v3'."""
    return f"{org}/vrm-7b-{stage}-{data_version}"


def model_repo_id(stage: str, run_name: str, org: str = DEFAULT_ORG) -> str:
    return f"{org}/vrm-7b-{stage}-{run_name}"


def upload_checkpoint(
    local_dir: Path,
    repo_id: str,
    token: str | None = None,
    *,
    private: bool = True,
) -> str:
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    create_repo(
        repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=api.token,
    )
    commit = api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="model")
    return getattr(commit, "commit_url", str(commit))


def upload_dataset_shards(local_dir: Path, repo_id: str, token: str | None = None) -> str:
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    create_repo(
        repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=api.token,
    )
    commit = api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="dataset")
    return getattr(commit, "commit_url", str(commit))
