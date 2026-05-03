"""`vrm eval ...` CLI subgroup."""

from __future__ import annotations

from pathlib import Path

import click

from vrm.eval.compare import main as compare_main
from vrm.eval.parse_results import parse_results_dir


@click.group("eval")
def eval_() -> None:
    """Eval pipeline commands."""


eval_.add_command(compare_main, name="compare")


@eval_.command("parse")
@click.option("--work-dir", type=click.Path(path_type=Path), required=True)
@click.option("--checkpoint", required=True, help="HF model repo id (for the report header)")
@click.option("--suite", required=True, help="Eval suite name (full|quick|negative_control)")
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for report.json (Markdown sibling at <out>.md)",
)
def parse(work_dir: Path, checkpoint: str, suite: str, out: Path) -> None:
    """Parse a VLMEvalKit work-dir into a structured EvalReport (json + md)."""
    rep = parse_results_dir(work_dir, checkpoint=checkpoint, suite=suite)
    out.write_text(rep.to_json())
    md_path = out.with_suffix(".md")
    md_path.write_text(rep.to_markdown())
    click.echo(f"wrote {out} and {md_path}")
