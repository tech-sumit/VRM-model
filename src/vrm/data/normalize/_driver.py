"""Drive a registered normalizer over a HF dataset, writing parquet shards.

Many HF datasets expose image columns as PIL.Image objects (via the
``datasets.Image`` feature). Our normalizers and the canonical Record
schema expect string paths. This driver acts as a shim: before calling
each normalizer, it detects PIL images in the raw row, persists them to
``out_dir / "images" / <source>-<index>-<column>.jpg``, and replaces the
field with a relative path. That keeps normalizers simple and keeps
Record.images as list[str].
"""

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


def _is_pil(x: Any) -> bool:
    return hasattr(x, "save") and hasattr(x, "mode") and hasattr(x, "size")


def _persist_pil_fields(
    raw: dict[str, Any], *, images_dir: Path, rec_index: int, source: str
) -> dict[str, Any]:
    """Return a copy of raw with PIL values replaced by relative paths."""
    out = dict(raw)
    img_idx = 0
    for k, v in list(out.items()):
        if _is_pil(v):
            fname = f"{source}-{rec_index:07d}-{img_idx}.jpg"
            fpath = images_dir / fname
            v2 = v if v.mode in ("RGB", "L") else v.convert("RGB")
            v2.save(fpath, format="JPEG", quality=90)
            out[k] = f"images/{fname}"
            img_idx += 1
        elif isinstance(v, list) and v and all(_is_pil(x) for x in v):
            paths = []
            for x in v:
                fname = f"{source}-{rec_index:07d}-{img_idx}.jpg"
                fpath = images_dir / fname
                x2 = x if x.mode in ("RGB", "L") else x.convert("RGB")
                x2.save(fpath, format="JPEG", quality=90)
                paths.append(f"images/{fname}")
                img_idx += 1
            out[k] = paths
    return out


def normalize_dataset(
    raw: Iterable[dict],
    *,
    source: str,
    out_dir: Path,
    shard_size: int = 5000,
) -> dict[str, int]:
    spec = REGISTRY[source]
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    in_count = 0
    out_count = 0
    shard_idx = 0
    buf: list[dict[str, Any]] = []
    for raw_rec in raw:
        in_count += 1
        try:
            raw_with_paths = _persist_pil_fields(
                dict(raw_rec), images_dir=images_dir, rec_index=in_count - 1, source=source
            )
        except Exception:
            continue
        rec = spec.normalize(raw_with_paths)
        if rec is None:
            continue
        d = json.loads(rec.model_dump_json())
        # pyarrow cannot infer schema for always-empty dict/struct columns.
        # Serialize freeform metadata to a JSON string so the parquet schema
        # stays stable across sources and batches.
        d["metadata"] = json.dumps(d.get("metadata") or {}, separators=(",", ":"))
        buf.append(d)
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
