# Sub-plan 05 — pass@K filter + teacher distillation

> Index: [`../plan.md`](../plan.md) · Depends on: [04 normalize](04-data-normalize.md) · Status: ☐

**Goal:** Two pipeline stages over normalized parquet shards: (a) the **difficulty filter** that runs base Qwen2.5-VL-7B inference K=8 times per problem and keeps `0.1 ≤ pass@K ≤ 0.85` (per spec §3.2), and (b) the **teacher distillation** that asks Claude + GPT-4o for `<think>...</think><answer>...</answer>` solutions and keeps verifier-correct CoTs (spec §3.1).

**Architecture:** Both stages read parquet shards, do work in parallel, write filtered/augmented parquet shards to a new versioned output. The base-model inference uses vLLM (one process running on the dataprep pod's available GPU when training is idle, or on the train pod after training). The teacher-API path is CPU-only, runs on the dataprep pod, and uses async httpx + tenacity for backoff.

**Tech Stack:** vllm · anthropic SDK · openai SDK · tenacity · asyncio · pyarrow.

---

### Task 1: pass@K filter (`vrm/data/filter.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/data/filter.py`
- Create: `projects/VRM-model/tests/unit/test_filter.py`

- [ ] **Step 1: Failing test (mock the inference fn so test runs without GPU)**

`projects/VRM-model/tests/unit/test_filter.py`:

```python
import pyarrow as pa
import pyarrow.parquet as pq

from vrm.data.filter import compute_difficulty, keep_in_band


def test_compute_difficulty_counts_correct():
    responses = ["<think>" + "x " * 60 + "</think><answer>72</answer>"] * 4
    responses += ["<think>" + "x " * 60 + "</think><answer>WRONG</answer>"] * 4
    gold = {"verifier": "exact_numeric", "answer": "72", "tolerance": 0.0}
    p = compute_difficulty(responses, gold)
    assert abs(p - 0.5) < 1e-6


def test_keep_in_band_thresholds():
    assert keep_in_band(0.5, lo=0.1, hi=0.85)
    assert not keep_in_band(0.05, lo=0.1, hi=0.85)
    assert not keep_in_band(0.95, lo=0.1, hi=0.85)
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/data/filter.py`:

```python
"""pass@K difficulty filter (spec §3.2).

Per spec: keep problems where 0.1 ≤ pass@K ≤ 0.85.
We separate the heavy generation step from the cheap accounting:
- `generate_responses(prompts, model, k)` lives in vrm.train.inference (depends on vLLM)
- `compute_difficulty(responses, gold)` is pure-Python (testable here)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

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
    difficulty_provider,  # callable: Record -> float (pass@K)
    lo: float = 0.1,
    hi: float = 0.85,
    shard_size: int = 5000,
) -> dict[str, int]:
    """Stream parquet shards from in_dir, keep records whose pass@K is in [lo, hi], write to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    in_count, out_count, shard_idx = 0, 0, 0
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
                pq.write_table(pa.Table.from_pylist(buf), out_dir / f"shard-{shard_idx:05d}.parquet")
                shard_idx += 1
                buf = []
    if buf:
        pq.write_table(pa.Table.from_pylist(buf), out_dir / f"shard-{shard_idx:05d}.parquet")
    return {"records_in": in_count, "records_out": out_count, "kept_pct": (out_count / in_count if in_count else 0.0)}
```

- [ ] **Step 3: Run, commit**

```bash
make test
git add projects/VRM-model/src/vrm/data/filter.py projects/VRM-model/tests/unit/test_filter.py
git commit -m "vrm: add pass@K difficulty filter (pure-Python, GPU-decoupled)"
```

---

### Task 2: vLLM-backed inference for the filter (`vrm/train/inference.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/train/inference.py`

- [ ] **Step 1: Implement (no unit test — vLLM-heavy; integration on pod only)**

`projects/VRM-model/src/vrm/train/inference.py`:

```python
"""vLLM batch inference helper used by the difficulty filter.

This module is import-guarded — vLLM and torch are only imported when needed,
so non-GPU environments (CI, dev laptops) don't crash on import.
"""
from __future__ import annotations

from typing import Sequence

from vrm.data.schema import Record


def generate_responses(
    records: Sequence[Record],
    *,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    n_per_prompt: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 8192,
) -> list[list[str]]:
    """Returns a list of length len(records); each inner list has n_per_prompt strings."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415 — lazy import

    llm = LLM(model=model_id, tensor_parallel_size=1, dtype="bfloat16",
              limit_mm_per_prompt={"image": 4})
    sp = SamplingParams(n=n_per_prompt, temperature=temperature, top_p=1.0, max_tokens=max_tokens)

    prompts = []
    for r in records:
        # vLLM multimodal API: dict with prompt + multi_modal_data
        prompts.append({
            "prompt": _to_chat_template(r),
            "multi_modal_data": {"image": _load_images(r.images)},
        })
    outputs = llm.generate(prompts, sp)
    return [[o.text for o in out.outputs] for out in outputs]


def _to_chat_template(rec: Record) -> str:
    # Concatenate via Qwen2.5-VL chat template; processor handles vision tokens at inference time.
    parts = []
    for m in rec.messages:
        parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _load_images(paths: list[str]) -> list:
    from PIL import Image  # noqa: PLC0415
    return [Image.open(p).convert("RGB") for p in paths]
```

- [ ] **Step 2: Commit**

```bash
git add projects/VRM-model/src/vrm/train/inference.py
git commit -m "vrm: add vLLM batch inference for difficulty filter (lazy import)"
```

---

### Task 3: Teacher distillation (`vrm/data/distill.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/data/distill.py`
- Create: `projects/VRM-model/tests/unit/test_distill.py`

- [ ] **Step 1: Failing test (mocks the API clients via responses lib)**

`projects/VRM-model/tests/unit/test_distill.py`:

```python
import pytest

from vrm.data.distill import pick_best_completion
from vrm.data.schema import Record


def _rec():
    return Record(
        id="x", images=["/tmp/x.png"],
        messages=[
            {"role": "system", "content": "..."},
            {"role": "user", "content": "<image>\nq"},
        ],
        answer="42", answer_type="numeric", verifier="exact_numeric", tolerance=0.0,
        source="test",
    )


def test_pick_best_completion_prefers_correct_with_longer_think():
    rec = _rec()
    short_think = "<think>" + "x " * 60 + "</think><answer>42</answer>"
    long_think = "<think>" + "x " * 200 + "</think><answer>42</answer>"
    wrong = "<think>" + "x " * 100 + "</think><answer>43</answer>"
    chosen = pick_best_completion(rec, [short_think, wrong, long_think])
    assert chosen == long_think


def test_pick_best_completion_returns_none_if_all_wrong():
    rec = _rec()
    wrong = "<think>" + "x " * 60 + "</think><answer>WRONG</answer>"
    assert pick_best_completion(rec, [wrong, wrong]) is None


def test_pick_best_completion_returns_none_if_no_format():
    rec = _rec()
    no_fmt = "Just 42"
    assert pick_best_completion(rec, [no_fmt]) is None
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/data/distill.py`:

```python
"""Teacher distillation — Claude + GPT-4o ensemble.

For each Record, ask each teacher for a `<think>...</think><answer>...</answer>` solution.
Pick the longest CoT that passes the deterministic verifier. Drop records where neither
teacher produced a verifier-correct response.

Reads input parquet shards (output of pass@K filter), writes augmented parquet shards
where each record has its `messages` list extended with the chosen assistant turn.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Sequence

import click
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import retry, stop_after_attempt, wait_exponential

from vrm.data.schema import Message, Record
from vrm.data.verifiers import score
from vrm.data.verifiers.format import has_valid_format

PROMPT_PREFIX = (
    "You are solving a visual reasoning problem. "
    "Think step-by-step inside <think>...</think> tags, "
    "then give your final concise answer inside <answer>...</answer> tags. "
    "Do not output anything outside the tags."
)


def pick_best_completion(rec: Record, completions: Sequence[str]) -> str | None:
    """Returns the longest format-valid + verifier-correct completion, or None."""
    correct = []
    for c in completions:
        if not has_valid_format(c):
            continue
        s = score({"verifier": rec.verifier, "answer": rec.answer, "tolerance": rec.tolerance}, c)
        if s["accuracy"] == 1.0:
            correct.append(c)
    if not correct:
        return None
    return max(correct, key=len)


# --- API clients (async; lazy import to avoid hard dep in tests) ---

def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
async def _ask_claude(client, model: str, rec: Record) -> str:
    parts = [{"type": "text", "text": f"{PROMPT_PREFIX}\n\n{rec.user_text()}"}]
    for img in rec.images[:4]:
        parts.insert(0, {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _b64_image(img)},
        })
    msg = await client.messages.create(
        model=model, max_tokens=4096,
        messages=[{"role": "user", "content": parts}],
    )
    return "".join(block.text for block in msg.content if block.type == "text")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
async def _ask_openai(client, model: str, rec: Record) -> str:
    parts = [{"type": "text", "text": f"{PROMPT_PREFIX}\n\n{rec.user_text()}"}]
    for img in rec.images[:4]:
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64_image(img)}"}})
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": parts}],
        max_tokens=4096,
    )
    return resp.choices[0].message.content or ""


async def _distill_one(rec: Record, *, anthropic_client, openai_client, claude_model: str, gpt_model: str) -> str | None:
    completions = await asyncio.gather(
        _ask_claude(anthropic_client, claude_model, rec),
        _ask_openai(openai_client, gpt_model, rec),
        return_exceptions=True,
    )
    cleaned = [c for c in completions if isinstance(c, str)]
    return pick_best_completion(rec, cleaned)


async def distill_shards(in_dir: Path, out_dir: Path, *, concurrency: int = 16) -> dict[str, int]:
    import anthropic  # noqa: PLC0415
    import openai  # noqa: PLC0415

    anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    claude_model = os.environ.get("TEACHER_MODEL_CLAUDE", "claude-opus-4-7-thinking-high")
    gpt_model = os.environ.get("TEACHER_MODEL_OPENAI", "gpt-5.4-medium")

    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    in_count, out_count, shard_idx = 0, 0, 0
    buf: list[dict] = []

    async def _wrap(rec: Record) -> Record | None:
        async with sem:
            best = await _distill_one(rec, anthropic_client=anthropic_client, openai_client=openai_client,
                                       claude_model=claude_model, gpt_model=gpt_model)
            if best is None:
                return None
            new_messages = list(rec.messages) + [Message(role="assistant", content=best)]
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
                pq.write_table(pa.Table.from_pylist(buf), out_dir / f"shard-{shard_idx:05d}.parquet")
                shard_idx += 1
                buf = []

    if buf:
        pq.write_table(pa.Table.from_pylist(buf), out_dir / f"shard-{shard_idx:05d}.parquet")
    return {"records_in": in_count, "records_out": out_count}


@click.command()
@click.option("--in-dir", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--concurrency", default=16, show_default=True)
def main(in_dir: Path, out_dir: Path, concurrency: int) -> None:
    """Distill teacher CoT for each record. Reads filtered parquet, writes augmented parquet."""
    result = asyncio.run(distill_shards(in_dir, out_dir, concurrency=concurrency))
    click.echo(f"distilled: {result}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run unit test, commit**

```bash
make test
git add projects/VRM-model/src/vrm/data/distill.py projects/VRM-model/tests/unit/test_distill.py
git commit -m "vrm: add Claude+GPT-4o ensemble teacher distillation"
```

---

### Task 4: Recipe YAML driving the full data pipeline

**Files:**
- Create: `projects/VRM-model/configs/data/sft_recipe.yaml`
- Create: `projects/VRM-model/configs/data/rl_recipe.yaml`
- Create: `projects/VRM-model/src/vrm/data/recipe.py`
- Create: `projects/VRM-model/tests/unit/test_recipe.py`

- [ ] **Step 1: Write the SFT recipe**

`projects/VRM-model/configs/data/sft_recipe.yaml`:

```yaml
# SFT data recipe — spec §3.1
name: sft
target_size: 280000
# Filter band per spec §3.2
difficulty_lo: 0.1
difficulty_hi: 0.85
# Sources, with per-source caps (rejection-sample to cap)
sources:
  - source: mavis
    cap: 80000
  - source: mathv360k
    cap: 60000
  - source: vision_r1_cold
    cap: 50000
  - source: geo170k
    cap: 30000
  - source: chartqa
    cap: 30000
  - source: mm_eureka
    cap: 30000
distillation:
  enabled: true
  concurrency: 16
  ensemble: [claude, openai]
```

`projects/VRM-model/configs/data/rl_recipe.yaml`:

```yaml
# RL data recipe — spec §3.2
name: rl
target_size: 110000
difficulty_lo: 0.1
difficulty_hi: 0.85
sources:
  - source: mm_eureka
    cap: 54000
  - source: geometry3k
    cap: 2100
  - source: mathvista
    cap: 5100
  - source: we_math
    cap: 6500
  - source: geoqa
    cap: 8000
  - source: tabmwp
    cap: 10000
  - source: chartqa
    cap: 10000
distillation:
  enabled: false
```

- [ ] **Step 2: Recipe loader + validation**

`projects/VRM-model/src/vrm/data/recipe.py`:

```python
"""Recipe schema and loader."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class SourceCap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    cap: int


class DistillationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    concurrency: int = 16
    ensemble: list[str] = []


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    target_size: int
    difficulty_lo: float
    difficulty_hi: float
    sources: list[SourceCap]
    distillation: DistillationConfig


def load_recipe(path: Path) -> Recipe:
    return Recipe.model_validate(yaml.safe_load(path.read_text()))
```

`projects/VRM-model/tests/unit/test_recipe.py`:

```python
from pathlib import Path

from vrm.data.recipe import load_recipe


def test_load_sft_recipe(repo_root):
    rec = load_recipe(repo_root / "configs/data/sft_recipe.yaml")
    assert rec.name == "sft"
    assert rec.distillation.enabled is True
    assert {s.source for s in rec.sources} >= {"mavis", "mathv360k"}


def test_load_rl_recipe(repo_root):
    rec = load_recipe(repo_root / "configs/data/rl_recipe.yaml")
    assert rec.name == "rl"
    assert rec.distillation.enabled is False
```

- [ ] **Step 3: Commit**

```bash
make test
git add projects/VRM-model/configs/data/ projects/VRM-model/src/vrm/data/recipe.py projects/VRM-model/tests/unit/test_recipe.py
git commit -m "vrm: add SFT and RL data recipes (YAML) + Pydantic loader"
```

---

## Done when

- [ ] `make test` includes the filter, distill, and recipe tests.
- [ ] `vrm data normalize`, `vrm data filter`, `python -m vrm.data.distill` are runnable.
- [ ] Sub-plan 11 (CD data-build workflow) can call these modules end-to-end inside the dataprep pod.
