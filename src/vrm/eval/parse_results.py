"""Parse VLMEvalKit JSON output dirs into a structured report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EvalReport:
    checkpoint: str
    suite: str
    metrics: dict[str, float] = field(default_factory=dict)
    raw_paths: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# Eval report: `{self.checkpoint}` -- suite=`{self.suite}`",
            "",
            "| Benchmark | Metric | Value |",
            "|---|---|---|",
        ]
        for k in sorted(self.metrics.keys()):
            bench, metric = ([*k.split(".", 1), "overall"])[:2]
            lines.append(f"| {bench} | {metric} | {self.metrics[k]:.4f} |")
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _flatten(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def parse_results_dir(
    work_dir: Path,
    *,
    checkpoint: str,
    suite: str,
) -> EvalReport:
    rep = EvalReport(checkpoint=checkpoint, suite=suite)
    if not work_dir.exists():
        return rep
    for json_path in sorted(work_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        bench = json_path.parent.name
        flat = _flatten(data, prefix=bench)
        if flat:
            rep.metrics.update(flat)
            rep.raw_paths.append(str(json_path.relative_to(work_dir)))
    return rep
