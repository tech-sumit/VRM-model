# Sub-plan 04 — Source-dataset normalizers

> Index: [`../plan.md`](../plan.md) · Depends on: [03 data schema](03-data-schema-verifiers.md) · Status: ☐

**Goal:** One module per source dataset that converts raw HuggingFace records into our `Record` schema. A single registry-driven CLI dispatches `vrm-data normalize <source> --output-dir <dir>`.

**Architecture:** Each `vrm/data/normalize/<source>.py` exposes a `normalize(raw: dict) -> Record | None` function (returns None to drop malformed records). A registry module (`normalize/__init__.py`) maps source names to (HF dataset id, split, normalize_fn). A driver streams the dataset and writes parquet shards.

**Tech Stack:** datasets · pyarrow · pydantic · click.

---

### Task 1: Common interface + registry

**Files:**
- Create: `projects/VRM-model/src/vrm/data/normalize/__init__.py`
- Create: `projects/VRM-model/src/vrm/data/normalize/_base.py`
- Create: `projects/VRM-model/tests/unit/test_normalize_base.py`

- [ ] **Step 1: Failing test (registry contract)**

`projects/VRM-model/tests/unit/test_normalize_base.py`:

```python
import pytest

from vrm.data.normalize import REGISTRY, NormalizeSpec


def test_all_registered_sources_have_specs():
    expected = {"mavis", "mathv360k", "vision_r1_cold", "geo170k", "chartqa",
                "mm_eureka", "geometry3k", "mathvista", "we_math", "geoqa", "tabmwp"}
    assert expected.issubset(set(REGISTRY.keys()))


def test_spec_fields_populated():
    for name, spec in REGISTRY.items():
        assert isinstance(spec, NormalizeSpec)
        assert spec.hf_id, f"{name} missing hf_id"
        assert callable(spec.normalize), f"{name} missing normalize fn"
```

- [ ] **Step 2: Implement base + empty registry**

`projects/VRM-model/src/vrm/data/normalize/_base.py`:

```python
"""Common types for dataset normalizers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from vrm.data.schema import Record

NormalizeFn = Callable[[dict], Record | None]


@dataclass(frozen=True)
class NormalizeSpec:
    hf_id: str
    split: str
    normalize: NormalizeFn
    image_column: str | None = None  # if HF dataset already gives PIL images
    # Optional: a sub-config name passed to load_dataset(name=...)
    config: str | None = None
    # Default verifier name (each normalizer can override per-record)
    default_verifier: str = "exact_numeric"


SYSTEM_PROMPT = (
    "You are a careful visual reasoner. Solve step-by-step. "
    "Put your reasoning in <think>...</think> and your final answer in <answer>...</answer>."
)
```

`projects/VRM-model/src/vrm/data/normalize/__init__.py`:

```python
"""Registry of all source dataset normalizers."""
from __future__ import annotations

from vrm.data.normalize._base import NormalizeSpec
from vrm.data.normalize.chartqa import SPEC as chartqa_spec
from vrm.data.normalize.geo170k import SPEC as geo170k_spec
from vrm.data.normalize.geometry3k import SPEC as geometry3k_spec
from vrm.data.normalize.geoqa import SPEC as geoqa_spec
from vrm.data.normalize.mathv360k import SPEC as mathv360k_spec
from vrm.data.normalize.mathvista import SPEC as mathvista_spec
from vrm.data.normalize.mavis import SPEC as mavis_spec
from vrm.data.normalize.mm_eureka import SPEC as mm_eureka_spec
from vrm.data.normalize.tabmwp import SPEC as tabmwp_spec
from vrm.data.normalize.vision_r1_cold import SPEC as vision_r1_cold_spec
from vrm.data.normalize.we_math import SPEC as we_math_spec

REGISTRY: dict[str, NormalizeSpec] = {
    "mavis": mavis_spec,
    "mathv360k": mathv360k_spec,
    "vision_r1_cold": vision_r1_cold_spec,
    "geo170k": geo170k_spec,
    "chartqa": chartqa_spec,
    "mm_eureka": mm_eureka_spec,
    "geometry3k": geometry3k_spec,
    "mathvista": mathvista_spec,
    "we_math": we_math_spec,
    "geoqa": geoqa_spec,
    "tabmwp": tabmwp_spec,
}

__all__ = ["NormalizeSpec", "REGISTRY"]
```

- [ ] **Step 3: Run (FAILS — submodules missing). That's expected; task 2 fixes.**

---

### Task 2: Stub one normalizer per source — start with MAVIS as the template

**Files (one per source — pattern repeats):**
- Create: `projects/VRM-model/src/vrm/data/normalize/mavis.py`
- Create: `projects/VRM-model/tests/unit/test_normalize_mavis.py`

- [ ] **Step 1: Test for the MAVIS normalizer using a hand-crafted raw record**

`projects/VRM-model/tests/unit/test_normalize_mavis.py`:

```python
from vrm.data.normalize.mavis import normalize


RAW = {
    "id": "mavis_5",
    "image": "/tmp/img5.png",
    "question": "In triangle ABC, find angle A.",
    "answer": "72",
    "answer_type": "numeric",
}


def test_mavis_basic():
    rec = normalize(RAW)
    assert rec is not None
    assert rec.id == "mavis_5"
    assert rec.source == "mavis"
    assert rec.answer == "72"
    assert rec.answer_type == "numeric"
    assert rec.verifier == "exact_numeric"
    assert rec.images == ["/tmp/img5.png"]
    assert "<image>" in rec.user_text()


def test_mavis_drops_record_without_image():
    assert normalize({**RAW, "image": None}) is None


def test_mavis_drops_record_without_answer():
    assert normalize({**RAW, "answer": ""}) is None
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/data/normalize/mavis.py`:

```python
"""MAVIS-Instruct normalizer.

HF dataset: PKU-Alignment/MAVIS-Instruct (or similar mirror).
Fields used: id, image, question, answer, answer_type.
"""
from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    image = raw.get("image")
    answer = (raw.get("answer") or "").strip()
    if not image or not answer:
        return None

    answer_type = raw.get("answer_type") or "numeric"
    verifier = {
        "numeric": "exact_numeric",
        "multiple_choice": "normalize_choice",
        "latex_math": "math_equal",
        "span": "span_match",
    }.get(answer_type, "exact_numeric")

    question = raw.get("question") or ""
    return Record(
        id=str(raw.get("id") or raw.get("uid") or hash(question)),
        images=[str(image)],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{question}"),
        ],
        answer=answer,
        answer_type=answer_type,  # type: ignore[arg-type]
        verifier=verifier,  # type: ignore[arg-type]
        tolerance=0.001 if answer_type == "numeric" else 0.0,
        source="mavis",
    )


SPEC = NormalizeSpec(
    hf_id="PKU-Alignment/MAVIS-Instruct",
    split="train",
    normalize=normalize,
    default_verifier="exact_numeric",
)
```

- [ ] **Step 3: Run, commit**

```bash
make test
git add projects/VRM-model/src/vrm/data/normalize/_base.py \
        projects/VRM-model/src/vrm/data/normalize/__init__.py \
        projects/VRM-model/src/vrm/data/normalize/mavis.py \
        projects/VRM-model/tests/unit/test_normalize_base.py \
        projects/VRM-model/tests/unit/test_normalize_mavis.py
git commit -m "vrm: add normalizer base + MAVIS as the template"
```

---

### Task 3: Add the remaining 10 normalizers (one task each, repeat the pattern)

For each of `mathv360k`, `vision_r1_cold`, `geo170k`, `chartqa`, `mm_eureka`, `geometry3k`, `mathvista`, `we_math`, `geoqa`, `tabmwp`:

**Files (per source):**
- Create: `projects/VRM-model/src/vrm/data/normalize/<source>.py`
- Create: `projects/VRM-model/tests/unit/test_normalize_<source>.py`

- [ ] **Step 1: Look up the actual HF dataset card for each source**

Find each on HF Hub; note the schema (column names, dtypes, splits). Common community mirrors:
| Source | HF dataset (verify before coding) |
|---|---|
| mathv360k | `Math-LLaVA/MathV360K` |
| vision_r1_cold | `Osilly/Vision-R1-cold-distill` |
| geo170k | `Luckyjhg/Geo170K` |
| chartqa | `HuggingFaceM4/ChartQA` |
| mm_eureka | `FanqingM/MM-Eureka-Dataset` |
| geometry3k | `InfiMM/Geometry3K` |
| mathvista | `AI4Math/MathVista` |
| we_math | `We-Math/We-Math` |
| geoqa | `Luckyjhg/GeoQA` (or `unimelb-nlp/GeoQA+`) |
| tabmwp | `TabMWP/TabMWP` |

- [ ] **Step 2: Use this pattern (adapted from MAVIS)**

```python
# projects/VRM-model/src/vrm/data/normalize/<source>.py
from __future__ import annotations

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec
from vrm.data.schema import Message, Record


def normalize(raw: dict) -> Record | None:
    # ... map raw fields → Record per source schema ...
    return Record(
        id=str(raw["id"]),
        images=[raw["image"]],
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"<image>\n{raw['question']}"),
        ],
        answer=str(raw["answer"]).strip(),
        answer_type="numeric",          # adjust per source
        verifier="exact_numeric",       # adjust per source
        tolerance=0.001,
        source="<source>",
    )


SPEC = NormalizeSpec(
    hf_id="<HF dataset id>",
    split="train",
    normalize=normalize,
    default_verifier="exact_numeric",
)
```

- [ ] **Step 3: Test pattern**

```python
# projects/VRM-model/tests/unit/test_normalize_<source>.py
from vrm.data.normalize.<source> import normalize


def test_basic_record():
    raw = {"id": "x", "image": "/tmp/x.png", "question": "...", "answer": "42"}
    rec = normalize(raw)
    assert rec is not None
    assert rec.source == "<source>"


def test_drops_missing_fields():
    assert normalize({"id": "x", "image": None, "question": "q", "answer": "1"}) is None
```

- [ ] **Step 4: After all 10 added, run full test + commit**

```bash
make test
git add projects/VRM-model/src/vrm/data/normalize/ projects/VRM-model/tests/unit/test_normalize_*.py
git commit -m "vrm: add normalizers for mathv360k/vision_r1_cold/geo170k/chartqa/mm_eureka/geometry3k/mathvista/we_math/geoqa/tabmwp"
```

---

### Task 4: Driver script — stream HF dataset, write parquet shards

**Files:**
- Create: `projects/VRM-model/src/vrm/data/normalize/_driver.py`
- Create: `projects/VRM-model/tests/integration/test_normalize_driver.py`

- [ ] **Step 1: Failing integration test (uses a tiny dummy dataset)**

```python
# projects/VRM-model/tests/integration/test_normalize_driver.py
import pytest
from datasets import Dataset

from vrm.data.normalize._driver import normalize_dataset


@pytest.mark.integration
def test_normalize_dataset_writes_parquet(tmp_path):
    raw = Dataset.from_list([
        {"id": "1", "image": "/tmp/x.png", "question": "q", "answer": "42"},
        {"id": "2", "image": "/tmp/y.png", "question": "q", "answer": "13"},
        {"id": "3", "image": None, "question": "q", "answer": "0"},  # dropped
    ])
    out = normalize_dataset(raw, source="mavis", out_dir=tmp_path, shard_size=10)
    assert out["records_in"] == 3
    assert out["records_out"] == 2
    assert (tmp_path / "shard-00000.parquet").exists()
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/data/normalize/_driver.py`:

```python
"""Drive a registered normalizer over a HF dataset, writing parquet shards."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

from vrm.data.normalize import REGISTRY


def _write_shard(records: list[dict[str, Any]], out_path: Path) -> None:
    if not records:
        return
    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_path)


def normalize_dataset(raw: Dataset | Iterable[dict], *, source: str, out_dir: Path, shard_size: int = 5000) -> dict[str, int]:
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
    return {"records_in": in_count, "records_out": out_count, "shards": shard_idx + (1 if buf or in_count else 0)}
```

- [ ] **Step 3: Run**

```bash
make test-integration
```

- [ ] **Step 4: Commit**

```bash
git add projects/VRM-model/src/vrm/data/normalize/_driver.py projects/VRM-model/tests/integration/test_normalize_driver.py
git commit -m "vrm: add normalizer driver (HF dataset → parquet shards)"
```

---

### Task 5: Wire into top-level CLI

**Files:**
- Modify: `projects/VRM-model/src/vrm/cli.py`
- Create: `projects/VRM-model/src/vrm/data/cli.py`

- [ ] **Step 1: Write the data CLI group**

`projects/VRM-model/src/vrm/data/cli.py`:

```python
"""`vrm data ...` CLI subgroup."""
from __future__ import annotations

from pathlib import Path

import click
from datasets import load_dataset

from vrm.data.normalize import REGISTRY
from vrm.data.normalize._driver import normalize_dataset


@click.group()
def data() -> None:
    """Data pipeline commands."""


@data.command("normalize")
@click.option("--source", required=True, type=click.Choice(sorted(REGISTRY.keys())))
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--shard-size", default=5000, show_default=True)
@click.option("--limit", type=int, default=None, help="Cap N records (for smoke runs)")
def normalize(source: str, out_dir: Path, shard_size: int, limit: int | None) -> None:
    spec = REGISTRY[source]
    ds = load_dataset(spec.hf_id, name=spec.config, split=spec.split, streaming=False)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    result = normalize_dataset(ds, source=source, out_dir=out_dir, shard_size=shard_size)
    click.echo(f"normalized {source}: {result}")
```

- [ ] **Step 2: Register on top-level CLI**

Modify `src/vrm/cli.py` — add `from vrm.data.cli import data` and `main.add_command(data, name="data")`.

- [ ] **Step 3: Verify**

```bash
uv run vrm data --help
```

Expected: shows `normalize` subcommand listing all 11 sources.

- [ ] **Step 4: Commit**

```bash
git add projects/VRM-model/src/vrm/data/cli.py projects/VRM-model/src/vrm/cli.py
git commit -m "vrm: add `vrm data normalize` CLI"
```

---

## Done when

- [ ] All 11 normalizers exist with at least 2 unit tests each.
- [ ] `vrm data normalize --source mavis --out-dir /tmp/x --limit 10` works (assuming HF auth and dataset access).
- [ ] Integration test for the driver passes.
- [ ] Sub-plan 05 (filter + distill) can call `vrm data normalize` and operate on the parquet shards.
