"""Promote a private VRM training checkpoint to the public HF model repo.

Invoked by `.github/workflows/vrm-release.yml` on annotated tag push. Reads
the source repo from the tag message (`source=<hf-repo-id>`) and:

  1. Snapshot-downloads the source private repo (config + weights).
  2. Creates the public destination repo if needed.
  3. Uploads everything to the public repo.
  4. Optionally tags the destination repo with the same version tag.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def promote(
    source: str,
    destination: str,
    *,
    tag: str | None,
    token: str,
    work_dir: Path,
) -> str:
    api = HfApi(token=token)
    api.create_repo(destination, repo_type="model", private=False, exist_ok=True, token=token)

    print(f"[promote] downloading {source} -> {work_dir}", flush=True)
    local = Path(
        snapshot_download(
            repo_id=source,
            repo_type="model",
            local_dir=str(work_dir),
            token=token,
        )
    )

    print(f"[promote] uploading {local} -> {destination}", flush=True)
    commit = api.upload_folder(
        folder_path=str(local),
        repo_id=destination,
        repo_type="model",
        commit_message=f"Promote from {source}" + (f" ({tag})" if tag else ""),
    )

    if tag:
        try:
            api.create_tag(destination, tag=tag, repo_type="model")
        except Exception as e:
            print(f"[promote] tag creation failed (non-fatal): {e}", file=sys.stderr)

    try:
        api.update_repo_visibility(destination, private=False)
    except Exception as e:
        print(f"[promote] visibility update failed (non-fatal): {e}", file=sys.stderr)

    return getattr(commit, "commit_url", str(commit))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Private source HF repo id")
    p.add_argument("--destination", required=True, help="Public destination HF repo id")
    p.add_argument("--tag", default=None, help="Optional version tag to apply")
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/vrm-promote"),
        help="Snapshot download dir",
    )
    args = p.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var is required", file=sys.stderr)
        sys.exit(2)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    url = promote(
        args.source,
        args.destination,
        tag=args.tag,
        token=token,
        work_dir=args.work_dir,
    )
    print(url)


if __name__ == "__main__":
    main()
