"""Run VLMEvalKit (canonical VLM eval harness) for a checkpoint x suite.

Subprocess-driven: shells out to `vlmutil` / `python -m vlmeval.run` so we
inherit upstream improvements without re-implementing benchmark scaffolding.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click
import yaml

from vrm.eval.parse_results import parse_results_dir
from vrm.infra.webhook import post_status


def _load_suite(suite: str, repo_root: Path) -> dict:
    return yaml.safe_load((repo_root / "configs/eval" / f"{suite}.yaml").read_text())


@click.command()
@click.option("--checkpoint", required=True, help="HF model repo id")
@click.option("--suite", required=True, help="full | quick | negative_control")
@click.option("--run-name", required=True)
@click.option(
    "--repo-root",
    type=click.Path(path_type=Path),
    default=Path("/workspace/vrm"),
    show_default=True,
)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("/workspace/data/eval"),
    show_default=True,
)
def main(checkpoint: str, suite: str, run_name: str, repo_root: Path, work_dir: Path) -> None:
    cfg = _load_suite(suite, repo_root)
    work_dir = Path(work_dir) / run_name
    work_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = cfg["benchmarks"]
    failures: list[str] = []
    for bench in benchmarks:
        cmd = [
            "python",
            "-m",
            "vlmeval.run",
            "--model",
            "QwenVL2_5",
            "--data",
            bench,
            "--work-dir",
            str(work_dir),
            "--mode",
            "all",
            "--reuse",
            "--api-nproc",
            "8",
            "--model-path",
            checkpoint,
        ]
        click.echo(f"[eval] cmd: {' '.join(cmd)}")
        rc = subprocess.call(cmd, env=os.environ.copy())
        if rc != 0:
            failures.append(bench)

    report = parse_results_dir(work_dir, checkpoint=checkpoint, suite=suite)
    (work_dir / "report.json").write_text(report.to_json())
    (work_dir / "report.md").write_text(report.to_markdown())

    status = "completed" if not failures else "failure"
    post_status(
        status,  # type: ignore[arg-type]
        task="eval",
        run_name=run_name,
        payload={
            "suite": suite,
            "checkpoint": checkpoint,
            "report_path": str(work_dir / "report.md"),
            "failures": failures,
            "metrics": report.metrics,
        },
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
