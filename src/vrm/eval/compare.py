"""Generate a markdown delta report between two eval reports."""

from __future__ import annotations

import json
from pathlib import Path

import click


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


@click.command()
@click.option(
    "--baseline",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to baseline report.json (e.g. base Qwen2.5-VL-7B)",
)
@click.option(
    "--current",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to current report.json (post-RL VRM checkpoint)",
)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(baseline: Path, current: Path, out: Path) -> None:
    b = _load(baseline)
    c = _load(current)
    keys = sorted(set(b.get("metrics", {}).keys()) | set(c.get("metrics", {}).keys()))
    lines = [
        f"# Eval comparison: `{c.get('checkpoint', '?')}` vs baseline `{b.get('checkpoint', '?')}`",
        f"Suite: `{c.get('suite', '?')}` (baseline `{b.get('suite', '?')}`)",
        "",
        "| Benchmark/Metric | Baseline | Current | Δ |",
        "|---|---:|---:|---:|",
    ]
    for k in keys:
        bv = b.get("metrics", {}).get(k)
        cv = c.get("metrics", {}).get(k)
        bv_s = f"{bv:.4f}" if bv is not None else "—"
        cv_s = f"{cv:.4f}" if cv is not None else "—"
        if bv is None or cv is None:
            d_s = "—"
        else:
            d = cv - bv
            sign = "+" if d >= 0 else ""
            d_s = f"{sign}{d:.4f}"
        lines.append(f"| {k} | {bv_s} | {cv_s} | {d_s} |")
    out.write_text("\n".join(lines) + "\n")
    click.echo(str(out))


if __name__ == "__main__":
    main()
