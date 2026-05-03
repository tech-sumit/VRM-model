"""Verify HuggingFace Hub is set up for VRM training.

Checks:
  1. HF_TOKEN has write scope on the target org.
  2. Target org is accessible.
  3. (Optional) pre-create private model + dataset repos so training pods
     can push on first attempt without permission errors.

Usage:
    HF_TOKEN=... uv run python scripts/bootstrap-hf.py \\
        [--org tech-sumit] [--create-repos]
"""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--org", default=os.environ.get("HF_ORG", "tech-sumit"))
    p.add_argument(
        "--create-repos",
        action="store_true",
        help="Pre-create placeholder model/dataset repos so training can push immediately.",
    )
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var is required", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=token)

    # 1. whoami
    try:
        me = api.whoami()
        print(f"[hf] authenticated as: {me.get('name')} ({me.get('type')})", file=sys.stderr)
    except Exception as e:
        print(f"[hf] whoami failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. write-scope token?
    token_info = me.get("auth", {}).get("accessToken", {})
    role = token_info.get("role", "unknown")
    if role not in ("write", "admin", "fineGrained"):
        print(
            f"[hf] WARNING: token role is '{role}', need 'write' or finegrained-with-write",
            file=sys.stderr,
        )

    # 3. org membership
    orgs = {o.get("name") for o in me.get("orgs", [])}
    if args.org not in orgs and args.org != me.get("name"):
        print(
            f"[hf] ERROR: not a member of '{args.org}' (member of: {sorted(orgs)})",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[hf] org access OK: {args.org}", file=sys.stderr)

    # 4. optional pre-create
    if args.create_repos:
        to_create = [
            ("model", f"{args.org}/vrm-7b-sft-bootstrap"),
            ("dataset", f"{args.org}/vrm-sft-bootstrap"),
            ("dataset", f"{args.org}/vrm-rl-bootstrap"),
        ]
        for kind, repo_id in to_create:
            try:
                api.create_repo(repo_id, repo_type=kind, private=True, exist_ok=True)
                print(f"[hf] ensured private {kind}: {repo_id}", file=sys.stderr)
            except Exception as e:
                print(f"[hf] create_repo failed for {repo_id}: {e}", file=sys.stderr)
                sys.exit(1)

    print("OK")


if __name__ == "__main__":
    main()
