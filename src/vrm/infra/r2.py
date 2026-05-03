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
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    download_workers = int(os.environ.get("VRM_R2_DOWNLOAD_WORKERS", "32"))

    def _classify(key: str) -> tuple[str | None, str, bool]:
        """Return (source, kind, should_download). kind in {parquet,tar,loose,state,skip}."""
        rel = key[len(prefix) :]
        parts = rel.split("/")
        if not parts or not parts[0]:
            return None, "skip", False
        source = parts[0]
        if sources is not None and source not in sources:
            return None, "skip", False
        is_image_tar = key.endswith("-images.tar")
        is_loose_image = "/images/" in key
        if not include_images and (is_image_tar or is_loose_image):
            return None, "skip", False
        if key.endswith(".parquet"):
            return source, "parquet", True
        if is_image_tar:
            return source, "tar", True
        if is_loose_image:
            return source, "loose", True
        if key.endswith("_state.json"):
            return source, "state", True
        return source, "skip", False

    # Partition keys by work type and skip any that are already on disk.
    todo: list[tuple[str, Path, str, str]] = []  # (key, local_path, source, kind)
    image_tars: list[tuple[str, Path]] = []
    n = {"parquet": 0, "tar": 0, "loose": 0, "state": 0}
    for key in keys:
        source, kind, should_download = _classify(key)
        if not should_download or source is None:
            continue
        local_path = local_root / key[len(prefix) :]
        if local_path.exists() and local_path.stat().st_size > 0:
            n[kind] += 1
            if kind == "tar":
                image_tars.append((source, local_path))
            continue
        todo.append((key, local_path, source, kind))

    tars_lock = threading.Lock()

    def _fetch(item: tuple[str, Path, str, str]) -> tuple[str, bool]:
        key, local_path, source, kind = item
        ok = r2.download_to(key, local_path)
        if ok and kind == "tar":
            with tars_lock:
                image_tars.append((source, local_path))
        return kind, ok

    if todo:
        with ThreadPoolExecutor(max_workers=download_workers) as ex:
            for kind, ok in (f.result() for f in as_completed(ex.submit(_fetch, it) for it in todo)):
                if ok:
                    n[kind] += 1

    def _extract(task: tuple[str, Path]) -> int:
        source, tar_path = task
        images_dir = local_root / source / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(images_dir)
        except Exception:  # pragma: no cover - corrupt tar; downstream skips records
            return 0
        return len(list(images_dir.glob("*")))

    # Extract each image tar into the source's images/ subdir so downstream
    # code sees loose JPEGs just like the old layout.
    if image_tars:
        with ThreadPoolExecutor(max_workers=min(8, len(image_tars))) as ex:
            for count in ex.map(_extract, image_tars):
                n["loose"] += count

    n_parquet = n["parquet"]
    n_images = n["loose"]
    n_state = n["state"]
    n_tars = n["tar"]

    return {
        "parquet_shards": n_parquet,
        "images_loose": n_images,
        "image_tars": n_tars,
        "state_files": n_state,
    }
