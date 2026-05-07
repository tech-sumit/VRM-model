"""pass@K difficulty filter (spec §3.2).

Per spec: keep problems where 0.1 <= pass@K <= 0.85. We separate the heavy
generation step from the cheap accounting:

- `generate_responses(prompts, model, k)` lives in vrm.train.inference (vLLM-backed)
- `compute_difficulty(responses, gold)` is pure-Python (testable here)
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq

from vrm.data.schema import Record
from vrm.data.verifiers import score
from vrm.infra.r2 import R2Client


def _filter_log_every() -> int:
    raw = os.environ.get("VRM_FILTER_LOG_EVERY", "20").strip().lower()
    if raw in ("0", "", "never", "off"):
        return 0
    try:
        return max(1, int(raw))
    except ValueError:
        return 20


def _filter_checkpoint_every() -> int:
    """Throttle R2 cursor writes after **drop** rows (buffer empty). 0 = never (flush-only)."""
    raw = os.environ.get("VRM_FILTER_CHECKPOINT_EVERY", "1").strip().lower()
    if raw in ("0", "", "never", "off"):
        return 0
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _state_payload(
    *,
    next_shard_idx: int,
    in_count: int,
    out_count: int,
    done: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "shards_written": next_shard_idx,
        "records_in": in_count,
        "records_out": out_count,
        "resume_scanned_in": in_count,
        "kept_pct": (out_count / in_count if in_count else 0.0),
    }
    if done:
        payload["done"] = True
    return payload


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

    Row-level resume: ``resume_scanned_in`` is the number of input rows for
    which work is durable: kept rows have been flushed to R2, or (when the
    output buffer is empty) dropped rows are checkpointed. Tune how often we
    write the cursor after drops with ``VRM_FILTER_CHECKPOINT_EVERY`` (default
    ``1``; ``0`` = only on output shard flush + terminal done).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_every = _filter_log_every()
    checkpoint_every = _filter_checkpoint_every()

    existing = sorted(out_dir.glob("shard-*.parquet"))
    shard_idx = len(existing)
    resume_skip = 0
    out_count = 0
    if r2 is not None and data_version is not None:
        state = r2.read_state(data_version, "filtered", "all")
        # Treat `done=true` with records_in=0 as a corrupt checkpoint from an
        # aborted earlier run (e.g. started before normalized data was on disk)
        # -- otherwise the stage would skip forever on subsequent pods.
        if state.get("done") and int(state.get("records_in", 0)) > 0:
            click.echo(
                "[filter] R2 checkpoint says stage already completed (skipped): "
                f"records_in={state.get('records_in')} records_out={state.get('records_out')}",
                err=True,
            )
            return {
                "records_in": float(state.get("records_in", 0)),
                "records_out": float(state.get("records_out", 0)),
                "kept_pct": float(state.get("kept_pct", 0.0)),
                "resumed": 1.0,
            }
        shard_idx = max(shard_idx, int(state.get("shards_written", 0)))
        rs = state.get("resume_scanned_in")
        if rs is None:
            rs = state.get("records_in", 0)
        resume_skip = int(rs or 0)
        out_count = int(state.get("records_out", 0))

    in_count = 0
    input_paths = sorted(in_dir.glob("*.parquet"))
    total_input_rows = 0
    for _p in input_paths:
        total_input_rows += int(pq.read_metadata(_p).num_rows)
    if resume_skip > total_input_rows:
        click.echo(
            f"[filter] WARN: resume_skip_rows={resume_skip} > total_input_rows={total_input_rows}; clamping",
            err=True,
        )
        resume_skip = total_input_rows

    band = f"[{lo:g}, {hi:g}]"
    ck_msg = f"{checkpoint_every}" if checkpoint_every > 0 else "off (flush-only)"
    click.echo(
        f"[filter] start: inputs={len(input_paths)} parquet(s), shard_size={shard_size}, "
        f"starting_output_shard_idx={shard_idx}, resume_skip_rows={resume_skip}, "
        f"R2_cursor_every={ck_msg}, pass@K_band={band}, "
        f"log_every={log_every} (env VRM_FILTER_LOG_EVERY; 0=quiet until flush/done)",
        err=True,
    )

    buf: list[dict] = []
    t0 = time.monotonic()

    def _maybe_persist_safe_cursor() -> None:
        """Persist resume_scanned_in only when no kept rows wait in memory (``buf``)."""
        if r2 is None or data_version is None or buf:
            return
        if checkpoint_every <= 0:
            return
        if in_count % checkpoint_every != 0:
            return
        with contextlib.suppress(Exception):
            r2.write_state(
                data_version,
                "filtered",
                "all",
                _state_payload(
                    next_shard_idx=shard_idx,
                    in_count=in_count,
                    out_count=out_count,
                    done=False,
                ),
            )

    def _flush() -> None:
        nonlocal buf, shard_idx
        if not buf:
            return
        nrows = len(buf)
        shard_path = out_dir / f"shard-{shard_idx:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(buf), shard_path)
        r2_ok = False
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
                    _state_payload(
                        next_shard_idx=shard_idx + 1,
                        in_count=in_count,
                        out_count=out_count,
                        done=False,
                    ),
                )
                r2_ok = True
        click.echo(
            f"[filter] flushed shard-{shard_idx:05d}: +{nrows} kept rows "
            f"(scanned_total={in_count}, kept_total={out_count}; R2_checkpoint={'ok' if r2_ok else 'n/a/local'})",
            err=True,
        )
        shard_idx += 1
        buf = []

    # _run_filter renames flattened inputs to "{source}-shard-*.parquet", so
    # match any .parquet in the input dir rather than pinning to "shard-*".
    for shard_path in input_paths:
        table = pq.read_table(shard_path)
        n_rows_shard = table.num_rows
        click.echo(
            f"[filter] input parquet {shard_path.name} ({n_rows_shard} rows); scanned_so_far={in_count}",
            err=True,
        )
        shard_t0 = time.monotonic()
        for row_i, row in enumerate(table.to_pylist(), start=1):
            in_count += 1
            if in_count <= resume_skip:
                continue
            if in_count == resume_skip + 1:
                click.echo(
                    "[filter] first-record inference beginning (VL load happens on first call; may take minutes)",
                    err=True,
                )
            rec = Record.model_validate(row)
            p = difficulty_provider(rec)
            if log_every > 0 and (in_count == resume_skip + 1 or in_count % log_every == 0):
                elapsed = time.monotonic() - t0
                rpm = (in_count / elapsed * 60.0) if elapsed > 0 else 0.0
                k_pct = (100.0 * out_count / in_count) if in_count else 0.0
                sec_per_scan = elapsed / in_count if in_count else 0.0
                in_b = keep_in_band(p, lo, hi)
                shard_elapsed = time.monotonic() - shard_t0
                click.echo(
                    f"[filter] progress scanned={in_count} kept={out_count} keep_rate_so_far={k_pct:.2f}% "
                    f"out_shards_written={shard_idx} shard_row={row_i}/{n_rows_shard} "
                    f"last_p={p:.4f} in_band={in_b} input_shard={shard_path.name} "
                    f"wall_s={elapsed:.1f} s_per_scan={sec_per_scan:.2f} shard_wall_s={shard_elapsed:.1f} "
                    f"rpm_scanned={rpm:.2f}",
                    err=True,
                )
            if not keep_in_band(p, lo, hi):
                _maybe_persist_safe_cursor()
                continue
            row["difficulty"] = p
            buf.append(row)
            out_count += 1
            if len(buf) >= shard_size:
                _flush()
        shard_wall = time.monotonic() - shard_t0
        click.echo(
            f"[filter] finished parquet {shard_path.name} scanned_total={in_count} kept_total={out_count} "
            f"this_shard_wall_s={shard_wall:.1f}",
            err=True,
        )
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
                _state_payload(
                    next_shard_idx=shard_idx,
                    in_count=in_count,
                    out_count=out_count,
                    done=True,
                ),
            )

    wall = time.monotonic() - t0
    click.echo(
        f"[filter] done: scanned={in_count} kept={out_count} "
        f"kept_pct={(100.0 * out_count / in_count) if in_count else 0.0:.2f}% "
        f"output_shards={shard_idx} wall_s={wall:.1f}",
        err=True,
    )

    return {
        "records_in": float(in_count),
        "records_out": float(out_count),
        "kept_pct": (out_count / in_count if in_count else 0.0),
    }
