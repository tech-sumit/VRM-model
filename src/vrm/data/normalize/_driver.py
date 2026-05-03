"""Drive a registered normalizer over a HF dataset, writing parquet shards.

Two behaviors that make the dataprep pipeline crash-safe:

  * Each shard (default 500 records) is flushed to local disk and -- if an
    R2Client is provided -- uploaded to R2 immediately, along with an
    updated _state.json. A pod loss past this point only replays the
    most-recent 500 records rather than the whole source.
  * On boot, the caller can pass ``resume_from_row`` to skip the N raw
    rows already processed in a prior run. The shard index continues
    from where the prior run left off so R2 keys remain monotonic.

PIL image handling: many HF datasets encode images inline via the
``datasets.Image`` feature. We detect PIL values in each raw row,
persist them to ``<out_dir>/images/`` (and upload to R2 if configured),
and replace the field with a relative path string so Record.images
stays ``list[str]``.
"""

from __future__ import annotations

import contextlib
import json
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from vrm.data.normalize import REGISTRY
from vrm.infra.r2 import R2Client

DEFAULT_SHARD_SIZE = 500


def _write_shard(records: list[dict[str, Any]], out_path: Path) -> None:
    if not records:
        return
    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_path)


def _is_pil(x: Any) -> bool:
    return hasattr(x, "save") and hasattr(x, "mode") and hasattr(x, "size")


def _persist_pil_fields(
    raw: dict[str, Any],
    *,
    images_dir: Path,
    rec_index: int,
    source: str,
    shard_image_names: list[str],
) -> dict[str, Any]:
    """Return a copy of raw with PIL values replaced by local relative paths.

    Each saved image filename is also appended to ``shard_image_names`` so
    the caller can tar them up at shard-flush time. Per-image R2 uploads
    are intentionally NOT done here -- they get bundled into one tar per
    shard to avoid R2 rate-limiting on same-prefix PutObject storms
    (observed: chartqa issuing 38K tiny puts to images/ got throttled,
    dropping throughput to ~22 rec/min. Tar-per-shard is ~500x fewer puts).
    """
    out = dict(raw)
    img_idx = 0

    def _save_one(pil_obj: Any) -> str:
        nonlocal img_idx
        fname = f"{source}-{rec_index:07d}-{img_idx}.jpg"
        fpath = images_dir / fname
        pil = pil_obj if pil_obj.mode in ("RGB", "L") else pil_obj.convert("RGB")
        pil.save(fpath, format="JPEG", quality=90)
        shard_image_names.append(fname)
        img_idx += 1
        return f"images/{fname}"

    for k, v in list(out.items()):
        if _is_pil(v):
            out[k] = _save_one(v)
        elif isinstance(v, list) and v and all(_is_pil(x) for x in v):
            out[k] = [_save_one(x) for x in v]
    return out


def _pack_shard_images(
    images_dir: Path,
    image_names: list[str],
    tar_path: Path,
) -> int:
    """Bundle a shard's images into a single tar. Returns byte size."""
    if not image_names:
        tar_path.touch()
        return 0
    with tarfile.open(tar_path, "w") as tar:
        for fname in image_names:
            fpath = images_dir / fname
            if fpath.exists():
                tar.add(fpath, arcname=fname)
    return tar_path.stat().st_size


def normalize_dataset(
    raw: Iterable[dict],
    *,
    source: str,
    out_dir: Path,
    shard_size: int = DEFAULT_SHARD_SIZE,
    r2: R2Client | None = None,
    data_version: str | None = None,
    stage: str = "normalized",
    start_shard_idx: int = 0,
    start_row_offset: int = 0,
    total_rows_hint: int | None = None,
) -> dict[str, int]:
    """Normalize a stream of raw rows to parquet shards.

    Args:
        raw: iterator of raw dicts from HF load_dataset.
        source: registry key; selects the normalizer + R2 sub-prefix.
        out_dir: local working directory for this source.
        shard_size: records per parquet shard (default 500).
        r2: if set, each shard is also uploaded to R2 + state.json updated.
        data_version: required when r2 is set (builds R2 prefix).
        stage: R2 sub-prefix, e.g. "normalized" / "filtered" / "distilled".
        start_shard_idx: starting index for shard filenames (resume support).
        start_row_offset: number of raw rows already processed upstream --
            added to in_count so _state.json reports true absolute row index.
        total_rows_hint: informational only; written to _state.json.
    """
    spec = REGISTRY[source]
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    r2_src_prefix = (
        f"{r2.source_prefix(data_version, stage, source)}"  # type: ignore[union-attr]
        if (r2 is not None and data_version is not None)
        else None
    )

    in_count = 0
    out_count = 0
    shard_idx = start_shard_idx
    buf: list[dict[str, Any]] = []
    shard_image_names: list[str] = []

    def _flush() -> None:
        nonlocal buf, shard_idx, shard_image_names
        if not buf:
            return
        shard_path = out_dir / f"shard-{shard_idx:05d}.parquet"
        _write_shard(buf, shard_path)

        # Pack this shard's images into one tar (zero images -> zero-byte
        # tar, still uploaded so downstream can distinguish "no images" from
        # "missing state"). Single PutObject per shard instead of per image
        # avoids R2 same-prefix rate limiting.
        tar_path = out_dir / f"shard-{shard_idx:05d}-images.tar"
        tar_bytes = _pack_shard_images(images_dir, shard_image_names, tar_path)

        if r2 is not None and r2_src_prefix is not None:
            try:
                r2.put_file(
                    shard_path,
                    f"{r2_src_prefix}/{shard_path.name}",
                    content_type="application/octet-stream",
                )
                if tar_bytes > 0:
                    r2.put_file(
                        tar_path,
                        f"{r2_src_prefix}/{tar_path.name}",
                        content_type="application/x-tar",
                    )
                r2.write_state(
                    data_version,  # type: ignore[arg-type]
                    stage,
                    source,
                    {
                        "last_row_index": start_row_offset + in_count,
                        "shards_written": shard_idx + 1,
                        "records_in": in_count,
                        "records_out": out_count,
                        "total_rows_hint": total_rows_hint,
                    },
                )
            except Exception:
                # Shard + tar stay on local disk; next resume re-writes.
                pass
        buf = []
        shard_image_names = []
        shard_idx += 1

    for raw_rec in raw:
        in_count += 1
        try:
            raw_with_paths = _persist_pil_fields(
                dict(raw_rec),
                images_dir=images_dir,
                rec_index=start_row_offset + in_count - 1,
                source=source,
                shard_image_names=shard_image_names,
            )
        except Exception:
            continue
        rec = spec.normalize(raw_with_paths)
        if rec is None:
            continue
        d = json.loads(rec.model_dump_json())
        # pyarrow can't infer schema for always-empty struct columns;
        # serialize metadata to a JSON string for stable columnar schema.
        d["metadata"] = json.dumps(d.get("metadata") or {}, separators=(",", ":"))
        buf.append(d)
        out_count += 1
        if len(buf) >= shard_size:
            _flush()

    _flush()

    # Write a terminal state so resume logic knows this source is done.
    if r2 is not None and data_version is not None:
        with contextlib.suppress(Exception):
            r2.write_state(
                data_version,
                stage,
                source,
                {
                    "last_row_index": start_row_offset + in_count,
                    "shards_written": shard_idx,
                    "records_in": in_count,
                    "records_out": out_count,
                    "total_rows_hint": total_rows_hint,
                    "done": True,
                },
            )

    return {
        "records_in": in_count,
        "records_out": out_count,
        "shards": shard_idx - start_shard_idx,
        "final_shard_idx": shard_idx,
    }
