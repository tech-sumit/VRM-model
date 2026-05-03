"""Cloudflare R2 client + shard checkpoint helpers.

R2 is an S3-compatible object store; we use it as the durable checkpoint
store for the dataprep pipeline. Layout:

    r2://{bucket}/vrm/{data_version}/{stage}/{source}/shard-NNNNN.parquet
    r2://{bucket}/vrm/{data_version}/{stage}/{source}/images/<name>.jpg
    r2://{bucket}/vrm/{data_version}/{stage}/{source}/_state.json
    r2://{bucket}/vrm/{data_version}/{stage}/_recipe_state.json

On boot the build pipeline scans the prefix to find which sources are
partially or fully done, then resumes each one from its recorded
``last_row_index``. This lets a crashed pod (budget tripwire, OOM, DC
reclaim, schema bug mid-source) pick up where it left off instead of
re-downloading + re-normalizing gigabytes.

Credentials come from four env vars (all required in production):
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


@dataclass(frozen=True)
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @classmethod
    def from_env(cls) -> R2Config | None:
        keys = {
            "account_id": os.environ.get("R2_ACCOUNT_ID"),
            "access_key_id": os.environ.get("R2_ACCESS_KEY_ID"),
            "secret_access_key": os.environ.get("R2_SECRET_ACCESS_KEY"),
            "bucket": os.environ.get("R2_BUCKET"),
        }
        if not all(keys.values()):
            return None
        return cls(**keys)  # type: ignore[arg-type]

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


class R2Client:
    """Thin wrapper around boto3 for R2. Safe to pickle config but not client."""

    def __init__(self, cfg: R2Config):
        self.cfg = cfg
        self._client = boto3.client(
            "s3",
            endpoint_url=cfg.endpoint_url,
            aws_access_key_id=cfg.access_key_id,
            aws_secret_access_key=cfg.secret_access_key,
            region_name="auto",
            config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "adaptive"}),
        )

    # ---- basic ops ------------------------------------------------------

    def put_file(self, local_path: Path, key: str, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        self._client.upload_file(str(local_path), self.cfg.bucket, key, ExtraArgs=extra or None)

    def put_bytes(self, data: bytes, key: str, content_type: str | None = None) -> None:
        kwargs: dict[str, Any] = {"Bucket": self.cfg.bucket, "Key": key, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        self._client.put_object(**kwargs)

    def get_bytes(self, key: str) -> bytes | None:
        try:
            resp = self._client.get_object(Bucket=self.cfg.bucket, Key=key)
            return resp["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise

    def list_prefix(self, prefix: str) -> list[str]:
        """Returns all object keys under ``prefix`` (paginated; handles >1000)."""
        keys: list[str] = []
        kw = {"Bucket": self.cfg.bucket, "Prefix": prefix}
        while True:
            resp = self._client.list_objects_v2(**kw)
            for obj in resp.get("Contents", []) or []:
                keys.append(obj["Key"])
            if not resp.get("IsTruncated"):
                break
            kw["ContinuationToken"] = resp["NextContinuationToken"]
        return keys

    def download_to(self, key: str, local_path: Path) -> bool:
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self.cfg.bucket, key, str(local_path))
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return False
            raise

    # ---- checkpoint helpers --------------------------------------------

    def source_prefix(self, data_version: str, stage: str, source: str) -> str:
        return f"vrm/{data_version}/{stage}/{source}"

    def read_state(self, data_version: str, stage: str, source: str) -> dict[str, Any]:
        """Return the last saved _state.json for a (stage, source), or {} if absent."""
        key = f"{self.source_prefix(data_version, stage, source)}/_state.json"
        raw = self.get_bytes(key)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def write_state(self, data_version: str, stage: str, source: str, state: dict[str, Any]) -> None:
        key = f"{self.source_prefix(data_version, stage, source)}/_state.json"
        self.put_bytes(
            json.dumps(state, separators=(",", ":")).encode("utf-8"),
            key,
            content_type="application/json",
        )

    def list_shards(self, data_version: str, stage: str, source: str) -> list[str]:
        """Return sorted list of shard keys that already exist in R2."""
        prefix = f"{self.source_prefix(data_version, stage, source)}/"
        return sorted(k for k in self.list_prefix(prefix) if k.endswith(".parquet") and "/shard-" in k)


def get_client() -> R2Client | None:
    """Factory: return an R2Client if credentials are in env, else None."""
    cfg = R2Config.from_env()
    return R2Client(cfg) if cfg else None


def download_stage_to_local(
    r2: R2Client,
    *,
    data_version: str,
    stage: str,
    local_root: Path,
    sources: list[str] | None = None,
    include_images: bool = True,
) -> dict[str, int]:
    """Mirror an entire stage's R2 prefix to local disk.

    Used by downstream stages (filter, distill) to pull the prior stage's
    output before running. Returns counts so caller can sanity-check.

    Layout on disk mirrors R2 (per-source subdirs):
        local_root/<source>/shard-*.parquet
        local_root/<source>/images/*.jpg
        local_root/<source>/_state.json
    """
    local_root.mkdir(parents=True, exist_ok=True)
    prefix = f"vrm/{data_version}/{stage}/"
    keys = r2.list_prefix(prefix)
    n_parquet = 0
    n_images = 0
    n_state = 0
    for key in keys:
        rel = key[len(prefix) :]
        parts = rel.split("/")
        if not parts or not parts[0]:
            continue
        source = parts[0]
        if sources is not None and source not in sources:
            continue
        local_path = local_root / rel
        if not include_images and "/images/" in key:
            continue
        if local_path.exists():
            # Already fetched -- skip re-download for incremental resume.
            if key.endswith(".parquet"):
                n_parquet += 1
            elif "/images/" in key:
                n_images += 1
            elif key.endswith("_state.json"):
                n_state += 1
            continue
        if r2.download_to(key, local_path):
            if key.endswith(".parquet"):
                n_parquet += 1
            elif "/images/" in key:
                n_images += 1
            elif key.endswith("_state.json"):
                n_state += 1
    return {"parquet_shards": n_parquet, "images": n_images, "state_files": n_state}
