"""pass@K difficulty filter (spec §3.2).

Per spec: keep problems where 0.1 <= pass@K <= 0.85. We separate the heavy
generation step from the cheap accounting:

- `generate_responses(prompts, model, k)` lives in vrm.train.inference (vLLM-backed)
- `compute_difficulty(responses, gold)` is pure-Python (testable here)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vrm.data.schema import Record
from vrm.data.verifiers import score


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
    shard_size: int = 5000,
) -> dict[str, float]:
    """Stream parquet shards, keep records whose pass@K is in [lo, hi]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    in_count = 0
    out_count = 0
    shard_idx = 0
    buf: list[dict] = []
    for shard_path in sorted(in_dir.glob("shard-*.parquet")):
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
                pq.write_table(
                    pa.Table.from_pylist(buf),
                    out_dir / f"shard-{shard_idx:05d}.parquet",
                )
                shard_idx += 1
                buf = []
    if buf:
        pq.write_table(pa.Table.from_pylist(buf), out_dir / f"shard-{shard_idx:05d}.parquet")
    return {
        "records_in": float(in_count),
        "records_out": float(out_count),
        "kept_pct": (out_count / in_count if in_count else 0.0),
    }
