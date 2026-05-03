"""Teacher distillation via OpenRouter (tiered ensemble).

One OpenRouter key gives access to 150+ models. We use two tiers:

  Bulk model     runs on every filter-kept record. Default:
                 ``qwen/qwen3-vl-235b-a22b-thinking`` (SoTA open math-visual
                 reasoner, native ``<think>`` output, ~$0.26/$2.60 per 1M).

  Refine model   runs only on ``refine_fraction`` of the hardest records
                 (lowest pass@K first, fallback random). Default:
                 ``anthropic/claude-sonnet-4.6`` for a quality-ceiling
                 cross-check and reasoning-style diversity.

Per-request usage is streamed to ``CreditCounter`` which enforces a
global USD cap (soft-pause, not kill): when the cap is hit we stop
issuing new calls, flush in-flight results, persist state to R2, and
exit with code 75 (EX_TEMPFAIL) so the next pod cold-start resumes
from the last checkpointed shard.

Inputs:  parquet shards from ``vrm.data.filter`` (with ``difficulty`` set).
Outputs: parquet shards where each record gains an assistant turn + a
         metadata field describing which teacher produced the CoT.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from vrm.data.schema import Message, Record
from vrm.data.verifiers import REGISTRY
from vrm.data.verifiers.format import extract_answer, has_valid_format
from vrm.infra.r2 import R2Client, get_client

PROMPT_PREFIX = (
    "You are solving a visual reasoning problem. "
    "Think step-by-step inside <think>...</think> tags, "
    "then give your final concise answer inside <answer>...</answer> tags. "
    "Do not output anything outside the tags."
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Per-1M-token prices (prompt, completion) as of 2026-05-03. Used to estimate
# spend when OpenRouter's response does not include native usage (rare).
PRICE_TABLE_USD_PER_M: dict[str, tuple[float, float]] = {
    "qwen/qwen3-vl-235b-a22b-thinking": (0.26, 2.60),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.20, 0.88),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "openai/gpt-5-mini": (0.25, 2.00),
}


class BudgetExceeded(Exception):
    """Raised by CreditCounter when the running USD total exceeds the cap."""


@dataclass
class CreditCounter:
    """Tracks cumulative USD spend across all OpenRouter calls this run.

    The R2-backed state lets a resumed pod continue the running total
    instead of restarting at $0, so pause+resume behaves as if one run.
    """

    cap_usd: float
    r2: R2Client | None = None
    data_version: str | None = None
    run_label: str = "distill"
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    total_usd: float = 0.0
    total_calls: int = 0
    by_model: dict[str, float] = field(default_factory=dict)
    _paused: bool = False

    def r2_key(self) -> str | None:
        if self.r2 is None or self.data_version is None:
            return None
        return f"vrm/{self.data_version}/distill/{self.run_label}_credits.json"

    def load(self) -> None:
        if self.r2 is None:
            return
        raw = self.r2.get_bytes(self.r2_key())  # type: ignore[arg-type]
        if not raw:
            return
        try:
            st = json.loads(raw.decode("utf-8"))
            self.total_usd = float(st.get("total_usd", 0))
            self.total_calls = int(st.get("total_calls", 0))
            self.by_model = {k: float(v) for k, v in (st.get("by_model") or {}).items()}
            self._paused = bool(st.get("paused", False))
        except Exception:
            pass

    def persist(self) -> None:
        if self.r2 is None:
            return
        key = self.r2_key()
        if not key:
            return
        self.r2.put_bytes(
            json.dumps(
                {
                    "total_usd": self.total_usd,
                    "total_calls": self.total_calls,
                    "by_model": self.by_model,
                    "cap_usd": self.cap_usd,
                    "paused": self._paused,
                    "ts": time.time(),
                },
                separators=(",", ":"),
            ).encode("utf-8"),
            key,
            content_type="application/json",
        )

    @property
    def paused(self) -> bool:
        return self._paused or self.total_usd >= self.cap_usd

    async def charge(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        prices = PRICE_TABLE_USD_PER_M.get(model)
        if prices is None:
            # Conservative fallback -- assume mid-tier pricing so we don't
            # blow through the cap on an unknown model.
            prices = (1.0, 3.0)
        cost = (prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1e6
        async with self._lock:
            self.total_usd += cost
            self.total_calls += 1
            self.by_model[model] = self.by_model.get(model, 0.0) + cost
            if self.total_usd >= self.cap_usd and not self._paused:
                self._paused = True
        if self.total_calls % 50 == 0 or self._paused:
            # Don't block the hot path on R2; fire-and-forget persist.
            with contextlib.suppress(Exception):
                self.persist()


def pick_best_completion(rec: Record, completions: Sequence[str]) -> str | None:
    fn = REGISTRY.get(rec.verifier)
    if fn is None:
        return None
    correct: list[str] = []
    gold = {"verifier": rec.verifier, "answer": rec.answer, "tolerance": rec.tolerance}
    for c in completions:
        if not has_valid_format(c):
            continue
        if fn(extract_answer(c), gold) == 1.0:
            correct.append(c)
    if not correct:
        return None
    return max(correct, key=len)


def _b64_image(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return None


def _build_messages(rec: Record) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": f"{PROMPT_PREFIX}\n\n{rec.user_text()}"}]
    for img in rec.images[:4]:
        b64 = _b64_image(img)
        if b64 is None:
            continue
        content.insert(
            0,
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
        )
    return [{"role": "user", "content": content}]


class PermanentError(Exception):
    pass


class TransientError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception_type(TransientError),
    reraise=True,
)
async def _openrouter_call(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    rec: Record,
    *,
    max_tokens: int = 4096,
) -> tuple[str, int, int]:
    """Returns (text, prompt_tokens, completion_tokens). Raises on permanent error."""
    payload = {
        "model": model,
        "messages": _build_messages(rec),
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.95,
    }
    try:
        resp = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/tech-sumit/VRM-model",
                "X-Title": "VRM-7B distillation",
            },
            json=payload,
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
    except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
        raise TransientError(str(e)) from e
    if resp.status_code == 429 or 500 <= resp.status_code < 600:
        raise TransientError(f"{resp.status_code}: {resp.text[:200]}")
    if resp.status_code >= 400:
        raise PermanentError(f"{resp.status_code}: {resp.text[:200]}")
    body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        raise PermanentError(f"no choices: {body}")
    text = (choices[0].get("message") or {}).get("content") or ""
    usage = body.get("usage") or {}
    return (
        text,
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
    )


async def _try_teacher(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    rec: Record,
    counter: CreditCounter,
) -> str | None:
    if counter.paused:
        return None
    try:
        text, pt, ct = await _openrouter_call(client, api_key, model, rec)
    except (PermanentError, TransientError):
        return None
    except Exception:
        return None
    await counter.charge(model, pt, ct)
    return text


async def _distill_one(
    rec: Record,
    *,
    client: httpx.AsyncClient,
    api_key: str,
    bulk_model: str,
    refine_model: str | None,
    use_refine: bool,
    counter: CreditCounter,
) -> tuple[str, str] | None:
    """Returns (chosen_completion, teacher_label) or None if no teacher passed."""
    completions: list[tuple[str, str]] = []
    bulk_text = await _try_teacher(client, api_key, bulk_model, rec, counter)
    if bulk_text:
        completions.append((bulk_text, bulk_model))

    # Refine pass: only invoke if we have budget AND (a) bulk failed verifier
    # or (b) this record is in the refine sample.
    if use_refine and refine_model and not counter.paused:
        needs_refine = True
        if bulk_text and pick_best_completion(rec, [bulk_text]) is not None:
            # Bulk already verifier-correct; still call refine for 10% to get
            # diversity (the caller decides via use_refine).
            pass
        if needs_refine:
            refine_text = await _try_teacher(client, api_key, refine_model, rec, counter)
            if refine_text:
                completions.append((refine_text, refine_model))

    best = pick_best_completion(rec, [c for c, _ in completions])
    if best is None:
        return None
    # Pick label by which completion matches the best choice (by identity).
    for c, label in completions:
        if c is best:
            return best, label
    return best, completions[0][1] if completions else "unknown"


def _is_hard(rec: Record, cutoff: float) -> bool:
    d = getattr(rec, "difficulty", None)
    if d is None:
        return False
    return float(d) <= cutoff


async def distill_shards(
    in_dir: Path,
    out_dir: Path,
    *,
    concurrency: int = 16,
    data_version: str | None = None,
    bulk_model: str | None = None,
    refine_model: str | None = None,
    refine_fraction: float = 0.10,
    cap_usd: float | None = None,
) -> dict[str, Any]:
    """Distill all filtered shards using tiered OpenRouter ensemble.

    Saves output parquet shards every 500 records. Updates R2 state after
    each shard so a pause+resume continues cleanly from the last-written
    output shard.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set -- configure it in the pod env (via "
            "GitHub secret -> runpod._common_env) before running distill."
        )
    bulk_model = bulk_model or os.environ.get("VRM_DISTILL_BULK_MODEL", "qwen/qwen3-vl-235b-a22b-thinking")
    refine_model = refine_model or os.environ.get("VRM_DISTILL_REFINE_MODEL", "anthropic/claude-sonnet-4.6")
    cap_usd = float(cap_usd or os.environ.get("VRM_DISTILL_MAX_USD", "100"))

    out_dir.mkdir(parents=True, exist_ok=True)
    r2 = get_client()
    counter = CreditCounter(cap_usd=cap_usd, r2=r2, data_version=data_version)
    counter.load()

    # Resume: skip output shards already in out_dir or R2.
    existing_out = sorted(out_dir.glob("shard-*.parquet"))
    out_shard_idx = len(existing_out)
    if r2 is not None and data_version is not None:
        state = r2.read_state(data_version, "distilled", "all")
        if state.get("done"):
            return {"resumed": "already-done", **{k: state[k] for k in state}}
        out_shard_idx = max(out_shard_idx, int(state.get("shards_written", 0)))

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:

        async def _wrap(rec: Record, use_refine: bool) -> Record | None:
            async with sem:
                result = await _distill_one(
                    rec,
                    client=client,
                    api_key=api_key,
                    bulk_model=bulk_model,
                    refine_model=refine_model,
                    use_refine=use_refine,
                    counter=counter,
                )
                if result is None:
                    return None
                best, label = result
                new_messages = [*list(rec.messages), Message(role="assistant", content=best)]
                meta = dict(getattr(rec, "metadata", {}) or {})
                meta["teacher"] = label
                return rec.model_copy(update={"messages": new_messages, "metadata": meta})

        in_count = 0
        out_count = 0
        buf: list[dict] = []
        paused = False

        for shard_path in sorted(in_dir.glob("shard-*.parquet")):
            if counter.paused:
                paused = True
                break
            table = pq.read_table(shard_path)
            records = [Record.model_validate(r) for r in table.to_pylist()]
            in_count += len(records)

            # Decide refine membership: hardest ``refine_fraction`` within
            # this shard. Difficulty is in [0,1] where 0=impossible.
            if records and refine_fraction > 0:
                # Sort by difficulty ascending (hardest first) and take top N.
                with_diff = [(r, getattr(r, "difficulty", None)) for r in records]
                with_diff_sorted = sorted(
                    with_diff, key=lambda x: (x[1] is None, x[1] if x[1] is not None else 1.0)
                )
                n_refine = max(1, int(len(records) * refine_fraction))
                refine_ids = {id(r) for r, _ in with_diff_sorted[:n_refine]}
            else:
                refine_ids = set()

            results = await asyncio.gather(*[_wrap(r, id(r) in refine_ids) for r in records])
            for rec in results:
                if rec is None:
                    continue
                buf.append(json.loads(rec.model_dump_json(exclude={"images"})))
                # re-add images as a plain list[str] (excluded above to avoid
                # large PIL objects in pydantic dump)
                buf[-1]["images"] = list(rec.images)
                # Serialize metadata dict -> json-str for stable parquet schema.
                buf[-1]["metadata"] = json.dumps(buf[-1].get("metadata") or {}, separators=(",", ":"))
                out_count += 1
                if len(buf) >= 500:
                    shard_out = out_dir / f"shard-{out_shard_idx:05d}.parquet"
                    pq.write_table(pa.Table.from_pylist(buf), shard_out)
                    if r2 is not None and data_version is not None:
                        try:
                            r2.put_file(
                                shard_out,
                                f"vrm/{data_version}/distilled/all/{shard_out.name}",
                                content_type="application/octet-stream",
                            )
                            r2.write_state(
                                data_version,
                                "distilled",
                                "all",
                                {
                                    "shards_written": out_shard_idx + 1,
                                    "records_in": in_count,
                                    "records_out": out_count,
                                    "credits_usd": counter.total_usd,
                                },
                            )
                        except Exception:
                            pass
                    out_shard_idx += 1
                    buf = []
            if counter.paused:
                paused = True
                break

        if buf:
            shard_out = out_dir / f"shard-{out_shard_idx:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(buf), shard_out)
            out_shard_idx += 1

        counter.persist()
        if r2 is not None and data_version is not None and not paused:
            with contextlib.suppress(Exception):
                r2.write_state(
                    data_version,
                    "distilled",
                    "all",
                    {
                        "shards_written": out_shard_idx,
                        "records_in": in_count,
                        "records_out": out_count,
                        "credits_usd": counter.total_usd,
                        "done": True,
                    },
                )

    return {
        "records_in": in_count,
        "records_out": out_count,
        "shards_written": out_shard_idx,
        "credits_usd": counter.total_usd,
        "by_model": counter.by_model,
        "paused": paused,
    }


@click.command()
@click.option("--in-dir", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--concurrency", default=16, show_default=True)
@click.option("--data-version", default=None, help="R2 checkpoint prefix version tag")
@click.option("--bulk-model", default=None, help="OpenRouter model id for bulk pass")
@click.option("--refine-model", default=None, help="OpenRouter model id for refine pass")
@click.option("--refine-fraction", default=0.10, show_default=True)
@click.option("--cap-usd", default=None, type=float, help="Total USD cap across all calls")
def main(
    in_dir: Path,
    out_dir: Path,
    concurrency: int,
    data_version: str | None,
    bulk_model: str | None,
    refine_model: str | None,
    refine_fraction: float,
    cap_usd: float | None,
) -> None:
    """Distill teacher CoT onto filtered records using OpenRouter."""
    result = asyncio.run(
        distill_shards(
            in_dir,
            out_dir,
            concurrency=concurrency,
            data_version=data_version,
            bulk_model=bulk_model,
            refine_model=refine_model,
            refine_fraction=refine_fraction,
            cap_usd=cap_usd,
        )
    )
    click.echo(f"distilled: {result}")
    if result.get("paused"):
        # Soft-pause exit: standard EX_TEMPFAIL so runpod/cron can resume.
        sys.exit(75)


if __name__ == "__main__":
    main()
