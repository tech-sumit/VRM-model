"""Stage 3: sample N candidates per prompt from the post-RL model, keep
verifier-correct ones, and write a new SFT-format parquet for fine-tuning.

This is the rejection-sampling step from spec §4.3. After this script writes
the new dataset, run `vrm.train.stage1_sft --mode rejection --config
configs/stage3_rejection_sft.yaml ...` to fine-tune on it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq

from vrm.data.schema import Message, Record
from vrm.data.verifiers import REGISTRY
from vrm.data.verifiers.format import extract_answer, has_valid_format


def _generate_with_vllm(
    records: Sequence[Record],
    *,
    model_id: str,
    n: int = 16,
) -> list[list[str]]:
    from vrm.train.inference import generate_responses

    return generate_responses(
        records,
        model_id=model_id,
        n_per_prompt=n,
        temperature=1.0,
    )


def keep_correct_responses(rec: Record, completions: Sequence[str]) -> list[str]:
    fn = REGISTRY.get(rec.verifier)
    if fn is None:
        return []
    out: list[str] = []
    for c in completions:
        if not has_valid_format(c):
            continue
        gold = {"answer": rec.answer, "tolerance": rec.tolerance}
        if fn(extract_answer(c), gold) == 1.0:
            out.append(c)
    return out


@click.command()
@click.option("--in-dataset", required=True, help="HF dataset id (e.g. tech-sumit/vrm-7b-sft-v1)")
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--model-id", required=True, help="post-RL HF model id")
@click.option("--n-per-prompt", default=16, show_default=True)
@click.option("--limit", type=int, default=None)
def main(
    in_dataset: str,
    out_dir: Path,
    model_id: str,
    n_per_prompt: int,
    limit: int | None,
) -> None:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(in_dataset, split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    records = [Record.model_validate(r) for r in ds]
    completions_per = _generate_with_vllm(records, model_id=model_id, n=n_per_prompt)

    out_rows: list[dict] = []
    in_count = len(records)
    out_count = 0
    for rec, comps in zip(records, completions_per, strict=False):
        kept = keep_correct_responses(rec, comps)
        if not kept:
            continue
        chosen = max(kept, key=len)
        new_messages = [*rec.messages, Message(role="assistant", content=chosen)]
        new_rec = rec.model_copy(update={"messages": new_messages})
        out_rows.append(json.loads(new_rec.model_dump_json()))
        out_count += 1
    if out_rows:
        pq.write_table(pa.Table.from_pylist(out_rows), out_dir / "shard-00000.parquet")
    click.echo(f"rejection sample: in={in_count} out={out_count}")


if __name__ == "__main__":
    main()
