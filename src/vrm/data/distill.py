"""Teacher distillation -- Claude + GPT-4o ensemble.

For each Record, ask each teacher for a `<think>...</think><answer>...</answer>`
solution. Pick the longest CoT that passes the deterministic verifier. Drop
records where neither teacher produced a verifier-correct response.

Reads input parquet shards (output of pass@K filter), writes augmented parquet
shards where each record has its `messages` list extended with the chosen
assistant turn.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import Sequence
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import retry, stop_after_attempt, wait_exponential

from vrm.data.schema import Message, Record
from vrm.data.verifiers import REGISTRY
from vrm.data.verifiers.format import extract_answer, has_valid_format

PROMPT_PREFIX = (
    "You are solving a visual reasoning problem. "
    "Think step-by-step inside <think>...</think> tags, "
    "then give your final concise answer inside <answer>...</answer> tags. "
    "Do not output anything outside the tags."
)


def pick_best_completion(rec: Record, completions: Sequence[str]) -> str | None:
    """Returns the longest format-valid + verifier-correct completion, or None."""
    fn = REGISTRY.get(rec.verifier)
    if fn is None:
        return None
    correct = []
    for c in completions:
        if not has_valid_format(c):
            continue
        gold = {
            "verifier": rec.verifier,
            "answer": rec.answer,
            "tolerance": rec.tolerance,
        }
        if fn(extract_answer(c), gold) == 1.0:
            correct.append(c)
    if not correct:
        return None
    return max(correct, key=len)


def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
async def _ask_claude(client, model: str, rec: Record) -> str:
    parts: list[dict] = [{"type": "text", "text": f"{PROMPT_PREFIX}\n\n{rec.user_text()}"}]
    for img in rec.images[:4]:
        parts.insert(
            0,
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _b64_image(img),
                },
            },
        )
    msg = await client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": parts}],
    )
    return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
async def _ask_openai(client, model: str, rec: Record) -> str:
    parts: list[dict] = [{"type": "text", "text": f"{PROMPT_PREFIX}\n\n{rec.user_text()}"}]
    for img in rec.images[:4]:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_b64_image(img)}"},
            }
        )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": parts}],
        max_tokens=4096,
    )
    return resp.choices[0].message.content or ""


async def _distill_one(
    rec: Record,
    *,
    anthropic_client,
    openai_client,
    claude_model: str,
    gpt_model: str,
) -> str | None:
    completions = await asyncio.gather(
        _ask_claude(anthropic_client, claude_model, rec),
        _ask_openai(openai_client, gpt_model, rec),
        return_exceptions=True,
    )
    cleaned = [c for c in completions if isinstance(c, str)]
    return pick_best_completion(rec, cleaned)


async def distill_shards(in_dir: Path, out_dir: Path, *, concurrency: int = 16) -> dict[str, int]:
    import anthropic
    import openai

    anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    claude_model = os.environ.get("TEACHER_MODEL_CLAUDE", "claude-opus-4-7-thinking-high")
    gpt_model = os.environ.get("TEACHER_MODEL_OPENAI", "gpt-5.4-medium")

    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    in_count = 0
    out_count = 0
    shard_idx = 0
    buf: list[dict] = []

    async def _wrap(rec: Record) -> Record | None:
        async with sem:
            best = await _distill_one(
                rec,
                anthropic_client=anthropic_client,
                openai_client=openai_client,
                claude_model=claude_model,
                gpt_model=gpt_model,
            )
            if best is None:
                return None
            new_messages = [*list(rec.messages), Message(role="assistant", content=best)]
            return rec.model_copy(update={"messages": new_messages})

    for shard_path in sorted(in_dir.glob("shard-*.parquet")):
        table = pq.read_table(shard_path)
        records = [Record.model_validate(r) for r in table.to_pylist()]
        in_count += len(records)
        results = await asyncio.gather(*[_wrap(r) for r in records])
        for rec in results:
            if rec is None:
                continue
            buf.append(json.loads(rec.model_dump_json()))
            out_count += 1
            if len(buf) >= 5000:
                pq.write_table(
                    pa.Table.from_pylist(buf),
                    out_dir / f"shard-{shard_idx:05d}.parquet",
                )
                shard_idx += 1
                buf = []

    if buf:
        pq.write_table(pa.Table.from_pylist(buf), out_dir / f"shard-{shard_idx:05d}.parquet")
    return {"records_in": in_count, "records_out": out_count}


@click.command()
@click.option("--in-dir", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--concurrency", default=16, show_default=True)
@click.option("--recipe", default=None, help="Recipe YAML (informational; logged only)")
@click.option("--data-version", default=None, help="Logged for traceability")
def main(
    in_dir: Path,
    out_dir: Path,
    concurrency: int,
    recipe: str | None,
    data_version: str | None,
) -> None:
    """Distill teacher CoT for each record. Reads filtered parquet, writes augmented parquet."""
    if recipe:
        click.echo(f"[distill] recipe={recipe} data_version={data_version}")
    result = asyncio.run(distill_shards(in_dir, out_dir, concurrency=concurrency))
    click.echo(f"distilled: {result}")


if __name__ == "__main__":
    main()
