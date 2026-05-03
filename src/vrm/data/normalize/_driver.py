"""Drive a registered normalizer over a HF dataset, writing parquet shards."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from vrm.data.normalize import REGISTRY


def _write_shard(records: list[dict[str, Any]], out_path: Path) -> None:
    if not records:
        return
    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_path)


def normalize_dataset(
    raw: Iterable[dict],
    *,
    source: str,
    out_dir: Path,
    shard_size: int = 5000,
) -> dict[str, int]:
    spec = REGISTRY[source]
    out_dir.mkdir(parents=True, exist_ok=True)
    in_count = 0
    out_count = 0
    shard_idx = 0
    buf: list[dict[str, Any]] = []
    for raw_rec in raw:
        in_count += 1
        rec = spec.normalize(dict(raw_rec))
        if rec is None:
            continue
        buf.append(json.loads(rec.model_dump_json()))
        out_count += 1
        if len(buf) >= shard_size:
            _write_shard(buf, out_dir / f"shard-{shard_idx:05d}.parquet")
            shard_idx += 1
            buf = []
    if buf:
        _write_shard(buf, out_dir / f"shard-{shard_idx:05d}.parquet")
    return {
        "records_in": in_count,
        "records_out": out_count,
        "shards": shard_idx + (1 if buf else 0),
    }
