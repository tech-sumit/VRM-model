"""Create (or reuse) a 2 TB network volume for VRM training in US-GA-2.

Usage:
    RUNPOD_API_KEY=... uv run python scripts/bootstrap-runpod-volume.py \\
        [--name vrm-7b-volume] [--size-gb 2000] [--region US-GA-2]

Prints the volume ID on success (paste into GH variable VRM_NETWORK_VOLUME_ID).
If a volume with the same name already exists in the region, reuses it.
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx

RUNPOD_API = "https://rest.runpod.io/v1"


def _auth_headers() -> dict[str, str]:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("RUNPOD_API_KEY env var is required", file=sys.stderr)
        sys.exit(2)
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _find_existing(name: str, region: str, client: httpx.Client) -> str | None:
    r = client.get(f"{RUNPOD_API}/networkvolumes", headers=_auth_headers())
    r.raise_for_status()
    for vol in r.json():
        if vol.get("name") == name and vol.get("dataCenterId", "").upper() == region.upper():
            return vol.get("id")
    return None


def _create(name: str, size_gb: int, region: str, client: httpx.Client) -> str:
    body = {"name": name, "size": size_gb, "dataCenterId": region}
    r = client.post(f"{RUNPOD_API}/networkvolumes", headers=_auth_headers(), json=body)
    if r.status_code >= 300:
        print(f"ERROR {r.status_code}: {r.text}", file=sys.stderr)
        sys.exit(1)
    return r.json()["id"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="vrm-7b-volume")
    p.add_argument("--size-gb", type=int, default=2000)
    p.add_argument("--region", default="US-GA-2")
    args = p.parse_args()

    with httpx.Client(timeout=30.0) as client:
        existing = _find_existing(args.name, args.region, client)
        if existing:
            print(f"[bootstrap-runpod-volume] reusing existing volume: {existing}", file=sys.stderr)
            print(existing)
            return
        print(
            f"[bootstrap-runpod-volume] creating {args.size_gb}GB volume '{args.name}' in {args.region}",
            file=sys.stderr,
        )
        new_id = _create(args.name, args.size_gb, args.region, client)
        print(new_id)


if __name__ == "__main__":
    main()
