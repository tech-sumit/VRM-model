"""pass@K difficulty filter (spec §3.2).

Per spec: keep problems where 0.1 <= pass@K <= 0.85. We separate the heavy
generation step from the cheap accounting:

- `generate_responses(prompts, model, k)` lives in vrm.train.inference (vLLM-backed)
- `compute_difficulty(responses, gold)` is pure-Python (testable here)
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vrm.data.schema import Record
from vrm.data.verifiers import score
from vrm.infra.r2 import R2Client


def compute_difficulty(responses: list[str], gold: dict) -> float:
    """Fraction of K responses that are accuracy=1.0."""
    if not responses:
        return 0.0
    correct = sum(1 for r in responses if score(gold, r)["accuracy"] == 1.0)
    return correct / len(responses)


def keep_in_band(pass_at_k: float, lo: float = 0.1, hi: float = 0.85) -> bool:
    return lo <= pass_at_k <= hi


def filter_shards(
    in_dir: Path,
    out_dir: Path,
    *,
    difficulty_provider: Callable[[Record], float],
    lo: float = 0.1,
    hi: float = 0.85,
    shard_size: int = 500,
    r2: R2Client | None = None,
    data_version: str | None = None,
) -> dict[str, float]:
    """Stream parquet shards, keep records whose pass@K is in [lo, hi].

    When ``r2`` + ``data_version`` are set, each output shard is uploaded
    to R2 at ``vrm/{data_version}/filtered/all/shard-NNNNN.parquet`` and a
    running ``_state.json`` is written so the stage is resumable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    in_count = 0
    out_count = 0

    existing = sorted(out_dir.glob("shard-*.parquet"))
    shard_idx = len(existing)
    if r2 is not None and data_version is not None:
        state = r2.read_state(data_version, "filtered", "all")
        # Treat `done=true` with records_in=0 as a corrupt checkpoint from an
        # aborted earlier run (e.g. started before normalized data was on disk)
        # -- otherwise the stage would skip forever on subsequent pods.
        if state.get("done") and int(state.get("records_in", 0)) > 0:
            return {
                "records_in": float(state.get("records_in", 0)),
                "records_out": float(state.get("records_out", 0)),
                "kept_pct": float(state.get("kept_pct", 0.0)),
                "resumed": 1.0,
            }
        shard_idx = max(shard_idx, int(state.get("shards_written", 0)))

    buf: list[dict] = []

    def _flush() -> None:
        nonlocal buf, shard_idx
        if not buf:
            return
        shard_path = out_dir / f"shard-{shard_idx:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(buf), shard_path)
        if r2 is not None and data_version is not None:
            with contextlib.suppress(Exception):
                r2.put_file(
                    shard_path,
                    f"vrm/{data_version}/filtered/all/{shard_path.name}",
                    content_type="application/octet-stream",
                )
                r2.write_state(
                    data_version,
                    "filtered",
                    "all",
                    {
                        "shards_written": shard_idx + 1,
                        "records_in": in_count,
                        "records_out": out_count,
                        "kept_pct": (out_count / in_count if in_count else 0.0),
                    },
                )
        shard_idx += 1
        buf = []

    # _run_filter renames flattened inputs to "{source}-shard-*.parquet", so
    # match any .parquet in the input dir rather than pinning to "shard-*".
    for shard_path in sorted(in_dir.glob("*.parquet")):
        table = pq.read_table(shard_path)
        for row in table.to_pylist():
            in_count += 1
            rec = Record.model_validate(row)
            p = difficulty_provider(rec)
            if not keep_in_band(p, lo, hi):
                continue
            row["difficulty"] = p
            buf.append(row)
            out_count += 1
            if len(buf) >= shard_size:
                _flush()
    _flush()

    # Only write the terminal "done" marker when we actually processed input.
    # An empty run usually means normalized data wasn't on disk yet -- we want
    # the next pod to retry rather than short-circuit.
    if r2 is not None and data_version is not None and in_count > 0:
        with contextlib.suppress(Exception):
            r2.write_state(
                data_version,
                "filtered",
                "all",
                {
                    "shards_written": shard_idx,
                    "records_in": in_count,
                    "records_out": out_count,
                    "kept_pct": (out_count / in_count if in_count else 0.0),
                    "done": True,
                },
            )

    return {
        "records_in": float(in_count),
        "records_out": float(out_count),
        "kept_pct": (out_count / in_count if in_count else 0.0),
    }
