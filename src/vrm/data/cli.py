"""`vrm data ...` CLI subgroup."""

from __future__ import annotations

from pathlib import Path

import click

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
    from datasets import load_dataset

    spec = REGISTRY[source]
    ds = load_dataset(spec.hf_id, name=spec.config, split=spec.split, streaming=False)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    result = normalize_dataset(
        (dict(r) for r in ds),
        source=source,
        out_dir=out_dir,
        shard_size=shard_size,
    )
    click.echo(f"normalized {source}: {result}")


@data.command("filter")
@click.option("--in-dir", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--lo", type=float, default=0.1, show_default=True)
@click.option("--hi", type=float, default=0.85, show_default=True)
@click.option("--pass-k", type=int, default=8, show_default=True)
@click.option(
    "--base-model-id",
    default="Qwen/Qwen2.5-VL-7B-Instruct",
    show_default=True,
)
def filter_(
    in_dir: Path,
    out_dir: Path,
    lo: float,
    hi: float,
    pass_k: int,
    base_model_id: str,
) -> None:
    """Run pass@K difficulty filter (vLLM-backed) over normalized parquet shards."""
    from vrm.data.build import _difficulty_provider_factory
    from vrm.data.filter import filter_shards

    provider = _difficulty_provider_factory(base_model_id, pass_k)
    result = filter_shards(in_dir, out_dir, difficulty_provider=provider, lo=lo, hi=hi)
    click.echo(f"filtered: {result}")


@data.command("build")
@click.option("--recipe", "recipe_paths", multiple=True, required=True, type=click.Path(path_type=Path))
@click.option("--data-version", required=True)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("/workspace/data/build"),
    show_default=True,
)
@click.option("--include-distillation/--no-distillation", default=True, show_default=True)
@click.option("--upload/--no-upload", default=True, show_default=True)
@click.pass_context
def build(
    ctx: click.Context,
    recipe_paths: tuple[Path, ...],
    data_version: str,
    work_dir: Path,
    include_distillation: bool,
    upload: bool,
) -> None:
    """End-to-end data build: normalize -> filter -> distill -> upload."""
    from vrm.data.build import main as build_main

    ctx.invoke(
        build_main,
        recipe_paths=recipe_paths,
        data_version=data_version,
        work_dir=work_dir,
        base_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        pass_k=8,
        include_distillation=include_distillation,
        upload=upload,
    )
