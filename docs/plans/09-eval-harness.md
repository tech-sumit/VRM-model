# Sub-plan 09 — Eval harness (VLMEvalKit + custom comparator)

> Index: [`../plan.md`](../plan.md) · Depends on: [06 SFT](06-stage1-sft.md) (need a checkpoint to eval) · Status: ☐

**Goal:** A repeatable eval pipeline that runs VLMEvalKit on any HF checkpoint, parses the results, and emits a markdown report with deltas vs the base model and vs the previous checkpoint. Includes a negative-control suite (DocVQA, ChartQA) to detect base-capability regression per spec §5.

**Architecture:** Three tightly-scoped modules:
1. `vrm.eval.run_vlmevalkit` — spawns `python -m vlmeval.run` with our model wrapper + suite YAML, captures the JSON output.
2. `vrm.eval.parse_results` — converts VLMEvalKit JSON into a uniform `EvalReport` dataclass + markdown.
3. `vrm.eval.compare` — diffs two `EvalReport`s and produces a delta-markdown.

**Tech Stack:** VLMEvalKit · pandas · jinja2 (markdown templating) · pydantic.

---

### Task 1: Eval suite YAML configs

**Files:**
- Create: `projects/VRM-model/configs/eval/full.yaml`
- Create: `projects/VRM-model/configs/eval/quick.yaml`
- Create: `projects/VRM-model/configs/eval/negative_control.yaml`

- [ ] **Step 1: Full suite (spec §5 headline benchmarks)**

`projects/VRM-model/configs/eval/full.yaml`:

```yaml
name: full
benchmarks:
  - MathVista_MINI
  - MathVerse_MINI
  - MathVision
  - MMMU_Pro
  - WeMath
  - BLINK
  - LogicVista
  - OlympiadBench-Vision
seeds: [42, 1337, 2024]   # spec §5: report mean ± std across 3 seeds
batch_size: 8
max_new_tokens: 8192
temperature: 0.0          # eval is deterministic
```

- [ ] **Step 2: Quick suite (CI smoke; just MathVista)**

`projects/VRM-model/configs/eval/quick.yaml`:

```yaml
name: quick
benchmarks:
  - MathVista_MINI
seeds: [42]
batch_size: 8
max_new_tokens: 4096
temperature: 0.0
```

- [ ] **Step 3: Negative-control (capability-regression detector)**

`projects/VRM-model/configs/eval/negative_control.yaml`:

```yaml
name: negative_control
benchmarks:
  - DocVQA_VAL
  - ChartQA_TEST
seeds: [42]
batch_size: 8
max_new_tokens: 1024
temperature: 0.0
```

- [ ] **Step 4: Commit**

```bash
git add projects/VRM-model/configs/eval/
git commit -m "vrm: add eval suite YAML configs (full / quick / negative_control)"
```

---

### Task 2: VLMEvalKit driver (`vrm/eval/run_vlmevalkit.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/eval/run_vlmevalkit.py`

- [ ] **Step 1: Implement (no unit test — VLMEvalKit invocation is integration-only)**

`projects/VRM-model/src/vrm/eval/run_vlmevalkit.py`:

```python
"""Spawn VLMEvalKit on a checkpoint, collect JSON results.

VLMEvalKit's `python -m vlmeval.run` accepts:
    --model <name>       # we pass our checkpoint via VLMEVALKIT_MODELS env hack
    --data <bench>       # one or more dataset names
    --work-dir <dir>
    --reuse              # reuse cached predictions
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import click
import yaml

from vrm.infra.webhook import post_status


def _write_model_config(checkpoint: str, work_dir: Path) -> Path:
    """VLMEvalKit reads model configs via $VLMEVALKIT_CONFIG; we generate one inline."""
    cfg = {
        "vrm-7b-eval": {
            "class": "Qwen2VLChat",  # VLMEvalKit's wrapper for Qwen2/2.5-VL
            "model_path": checkpoint,
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
    }
    p = work_dir / "vrm_model_config.json"
    p.write_text(json.dumps(cfg, indent=2))
    return p


@click.command()
@click.option("--checkpoint", required=True)
@click.option("--suite", required=True, help="full | quick | negative_control (or a YAML path)")
@click.option("--run-name", required=True)
def main(checkpoint: str, suite: str, run_name: str) -> None:
    workspace = Path(os.environ.get("WORKSPACE", "/workspace/data"))
    work_dir = workspace / "eval" / run_name
    work_dir.mkdir(parents=True, exist_ok=True)

    suite_path = Path(suite) if Path(suite).exists() else \
                 Path(__file__).resolve().parents[3] / "configs/eval" / f"{suite}.yaml"
    cfg = yaml.safe_load(suite_path.read_text())

    model_cfg = _write_model_config(checkpoint, work_dir)

    post_status("started", task="eval", run_name=run_name,
                payload={"checkpoint": checkpoint, "suite": cfg["name"], "benchmarks": cfg["benchmarks"]})

    env = {**os.environ, "VLMEVALKIT_CONFIG": str(model_cfg)}
    failures: dict[str, str] = {}
    for bench in cfg["benchmarks"]:
        cmd = [
            "python", "-m", "vlmeval.run",
            "--model", "vrm-7b-eval",
            "--data", bench,
            "--work-dir", str(work_dir / bench),
            "--reuse",
        ]
        print(f"[eval] {bench}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            failures[bench] = f"exit_code={rc}"

    summary = {
        "checkpoint": checkpoint, "suite": cfg["name"], "work_dir": str(work_dir),
        "failures": failures,
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if failures:
        post_status("failure", task="eval", run_name=run_name, payload=summary)
        sys.exit(1)

    post_status("completed", task="eval", run_name=run_name, payload=summary)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add projects/VRM-model/src/vrm/eval/run_vlmevalkit.py
git commit -m "vrm: add VLMEvalKit driver (spawns per-benchmark + writes summary.json)"
```

---

### Task 3: Result parser (`vrm/eval/parse_results.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/eval/parse_results.py`
- Create: `projects/VRM-model/tests/unit/test_parse_results.py`

- [ ] **Step 1: Failing test (with a sample VLMEvalKit JSON)**

```python
# projects/VRM-model/tests/unit/test_parse_results.py
import json

from vrm.eval.parse_results import EvalReport, parse_results_dir


SAMPLE_MATHVISTA = {
    "MathVista_MINI": {
        "Overall": {"acc": 67.4, "n": 1000},
        "math": {"acc": 70.0, "n": 500},
    }
}


def test_parse_one_benchmark(tmp_path):
    bench_dir = tmp_path / "MathVista_MINI"
    bench_dir.mkdir()
    (bench_dir / "vrm-7b-eval_MathVista_MINI_acc.json").write_text(json.dumps(SAMPLE_MATHVISTA))
    report = parse_results_dir(tmp_path, checkpoint="tech-sumit/vrm-7b-x", suite="quick")
    assert isinstance(report, EvalReport)
    assert report.checkpoint == "tech-sumit/vrm-7b-x"
    assert report.results["MathVista_MINI"]["Overall"] == 67.4


def test_parse_handles_missing_results(tmp_path):
    report = parse_results_dir(tmp_path, checkpoint="x", suite="quick")
    assert report.results == {}
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/eval/parse_results.py`:

```python
"""Parse VLMEvalKit per-benchmark JSON outputs into a uniform EvalReport."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalReport:
    checkpoint: str
    suite: str
    # benchmark → category → score (% accuracy)
    results: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [f"# Eval report — `{self.checkpoint}` ({self.suite})", ""]
        for bench, cats in sorted(self.results.items()):
            lines.append(f"## {bench}")
            lines.append("| Category | Accuracy (%) |")
            lines.append("|---|---|")
            for cat, score in sorted(cats.items()):
                lines.append(f"| {cat} | {score:.2f} |")
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({"checkpoint": self.checkpoint, "suite": self.suite, "results": self.results}, indent=2)


def parse_results_dir(work_dir: Path, *, checkpoint: str, suite: str) -> EvalReport:
    """Walks a VLMEvalKit work dir of the form work_dir/<bench>/{model}_{bench}_acc.json."""
    rep = EvalReport(checkpoint=checkpoint, suite=suite)
    for bench_dir in sorted(work_dir.iterdir()) if work_dir.exists() else []:
        if not bench_dir.is_dir():
            continue
        # Find any *_acc.json file inside
        for jf in bench_dir.glob("*_acc.json"):
            try:
                blob = json.loads(jf.read_text())
                # VLMEvalKit returns a dict keyed by benchmark name, then category → {acc, n}
                for bench_name, cats in blob.items():
                    rep.results[bench_name] = {k: float(v["acc"]) for k, v in cats.items()
                                                if isinstance(v, dict) and "acc" in v}
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return rep
```

- [ ] **Step 3: Run, commit**

```bash
make test
git add projects/VRM-model/src/vrm/eval/parse_results.py projects/VRM-model/tests/unit/test_parse_results.py
git commit -m "vrm: add EvalReport + VLMEvalKit JSON → markdown parser"
```

---

### Task 4: Comparator (`vrm/eval/compare.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/eval/compare.py`
- Create: `projects/VRM-model/tests/unit/test_compare.py`

- [ ] **Step 1: Failing test**

```python
# projects/VRM-model/tests/unit/test_compare.py
from vrm.eval.compare import diff_reports
from vrm.eval.parse_results import EvalReport


def test_diff_reports_basic():
    a = EvalReport(checkpoint="base", suite="quick", results={"MathVista_MINI": {"Overall": 60.0}})
    b = EvalReport(checkpoint="post", suite="quick", results={"MathVista_MINI": {"Overall": 67.4}})
    md = diff_reports(a, b)
    assert "MathVista_MINI" in md
    assert "+7.40" in md  # delta
    assert "60.00" in md  # baseline
    assert "67.40" in md  # candidate


def test_diff_reports_handles_missing_benchmark():
    a = EvalReport(checkpoint="base", suite="quick", results={"X": {"O": 50.0}})
    b = EvalReport(checkpoint="post", suite="quick", results={"Y": {"O": 60.0}})
    md = diff_reports(a, b)
    assert "X" in md and "Y" in md
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/eval/compare.py`:

```python
"""Diff two EvalReports into a markdown delta table."""
from __future__ import annotations

from vrm.eval.parse_results import EvalReport


def diff_reports(baseline: EvalReport, candidate: EvalReport) -> str:
    lines = [
        f"# Eval delta — `{candidate.checkpoint}` vs `{baseline.checkpoint}`",
        f"Suite: `{candidate.suite}`",
        "",
        "| Benchmark | Category | Baseline | Candidate | Δ |",
        "|---|---|---|---|---|",
    ]
    benches = sorted(set(baseline.results) | set(candidate.results))
    for b in benches:
        cats = sorted(set(baseline.results.get(b, {})) | set(candidate.results.get(b, {})))
        for c in cats:
            base = baseline.results.get(b, {}).get(c)
            cand = candidate.results.get(b, {}).get(c)
            if base is not None and cand is not None:
                delta = cand - base
                marker = "🟢" if delta > 0 else ("🔴" if delta < -0.5 else "⚪")
                lines.append(f"| {b} | {c} | {base:.2f} | {cand:.2f} | {marker} {delta:+.2f} |")
            elif cand is not None:
                lines.append(f"| {b} | {c} | — | {cand:.2f} | (new) |")
            else:
                lines.append(f"| {b} | {c} | {base:.2f} | — | (missing) |")
    return "\n".join(lines)
```

- [ ] **Step 3: Run, commit**

```bash
make test
git add projects/VRM-model/src/vrm/eval/compare.py projects/VRM-model/tests/unit/test_compare.py
git commit -m "vrm: add eval comparator (markdown delta table with regression markers)"
```

---

### Task 5: Wire `vrm eval` CLI subcommands

**Files:**
- Modify: `projects/VRM-model/src/vrm/cli.py`
- Create: `projects/VRM-model/src/vrm/eval/cli.py`

- [ ] **Step 1: Subgroup**

`projects/VRM-model/src/vrm/eval/cli.py`:

```python
"""`vrm eval ...` CLI subgroup."""
from __future__ import annotations

from pathlib import Path

import click

from vrm.eval.compare import diff_reports
from vrm.eval.parse_results import parse_results_dir


@click.group()
def eval_() -> None:
    """Eval pipeline commands."""


@eval_.command("parse")
@click.option("--work-dir", type=click.Path(path_type=Path), required=True)
@click.option("--checkpoint", required=True)
@click.option("--suite", default="full")
def parse_cmd(work_dir: Path, checkpoint: str, suite: str) -> None:
    rep = parse_results_dir(work_dir, checkpoint=checkpoint, suite=suite)
    click.echo(rep.to_markdown())


@eval_.command("compare")
@click.option("--baseline-dir", type=click.Path(path_type=Path), required=True)
@click.option("--candidate-dir", type=click.Path(path_type=Path), required=True)
@click.option("--baseline-name", default="baseline")
@click.option("--candidate-name", default="candidate")
@click.option("--suite", default="full")
def compare_cmd(baseline_dir: Path, candidate_dir: Path, baseline_name: str, candidate_name: str, suite: str) -> None:
    a = parse_results_dir(baseline_dir, checkpoint=baseline_name, suite=suite)
    b = parse_results_dir(candidate_dir, checkpoint=candidate_name, suite=suite)
    click.echo(diff_reports(a, b))
```

- [ ] **Step 2: Register**

In `src/vrm/cli.py`, add `from vrm.eval.cli import eval_` and `main.add_command(eval_, name="eval")`.

- [ ] **Step 3: Verify, commit**

```bash
uv run vrm eval --help
make test
git add projects/VRM-model/src/vrm/eval/cli.py projects/VRM-model/src/vrm/cli.py
git commit -m "vrm: expose eval subcommands (parse, compare) under vrm CLI"
```

---

## Done when

- [ ] `make test` includes parse + compare tests.
- [ ] `vrm eval --help` shows parse + compare.
- [ ] An eval pod can be launched (`make eval CHECKPOINT=... SUITE=quick`) and produces summary.json + per-benchmark dirs.
- [ ] Sub-plan 12 (CD eval workflow) can call `python -m vrm.eval.run_vlmevalkit` end-to-end.
