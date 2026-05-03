"""End-to-end data-build pipeline: normalize -> filter -> distill -> upload.

Invoked from the dataprep pod with one or more recipe YAML files. For each
recipe:
    1. For every source listed in `recipe.sources`, normalize the upstream HF
       dataset into parquet shards under `<work>/normalized/<source>/`,
       capped at `cap` records per source.
    2. Compute pass@K difficulty per record (vLLM batch inference) and keep
       records with `lo <= pass@K <= hi`.
    3. (If `recipe.distillation.enabled`) ask Claude + GPT-4o for solutions
       and pick the best verifier-passing completion per record.
    4. Upload the final parquet shards to the HF dataset repo.

This is the single entry the dataprep pod calls; pod-entrypoint.sh wires
`VRM_TASK=dataprep` -> `python -m vrm.data.build --recipe ... --data-version ...`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from vrm.data.distill import distill_shards
from vrm.data.filter import filter_shards
from vrm.data.normalize import REGISTRY
from vrm.data.normalize._driver import normalize_dataset
from vrm.data.recipe import Recipe, load_recipe
from vrm.data.schema import Record
from vrm.infra.hf_hub import dataset_repo_id, upload_dataset_shards
from vrm.infra.webhook import post_status


def _normalize_one_source(source: str, cap: int, *, out_dir: Path) -> dict[str, int]:
    from datasets import load_dataset

    spec = REGISTRY[source]
    ds = load_dataset(spec.hf_id, name=spec.config, split=spec.split, streaming=False)
    n = min(cap, len(ds))
    ds = ds.select(range(n))
    return normalize_dataset(
        (dict(r) for r in ds),
        source=source,
        out_dir=out_dir,
        shard_size=5000,
    )


def _difficulty_provider_factory(model_id: str, k: int):
    """Returns a callable Record -> pass@K, lazy-loading vLLM only when called."""

    cache: dict[str, object] = {}

    def _provider(rec: Record) -> float:
        if "llm" not in cache:
            from vrm.train.inference import generate_responses

            cache["fn"] = generate_responses
        fn = cache["fn"]
        comps = fn([rec], model_id=model_id, n_per_prompt=k)[0]  # type: ignore[operator]
        from vrm.data.filter import compute_difficulty

        return compute_difficulty(
            comps,
            {
                "verifier": rec.verifier,
                "answer": rec.answer,
                "tolerance": rec.tolerance,
            },
        )

    return _provider


def build_one_recipe(
    recipe: Recipe,
    *,
    work_dir: Path,
    data_version: str,
    base_model_id: str,
    pass_k: int,
    include_distillation: bool,
    upload: bool,
) -> dict[str, int]:
    norm_dir = work_dir / "normalized"
    filt_dir = work_dir / "filtered"
    distill_dir = work_dir / "distilled"

    n_in = 0
    n_out = 0
    for sc in recipe.sources:
        out = norm_dir / sc.source
        out.mkdir(parents=True, exist_ok=True)
        result = _normalize_one_source(sc.source, sc.cap, out_dir=out)
        n_in += result["records_in"]
        n_out += result["records_out"]

    norm_flat = work_dir / "normalized_flat"
    norm_flat.mkdir(parents=True, exist_ok=True)
    seen = 0
    for src_dir in sorted(norm_dir.iterdir()):
        for shard in sorted(src_dir.glob("shard-*.parquet")):
            (norm_flat / f"{src_dir.name}-{shard.name}").write_bytes(shard.read_bytes())
            seen += 1

    provider = _difficulty_provider_factory(base_model_id, pass_k)
    filter_result = filter_shards(
        norm_flat,
        filt_dir,
        difficulty_provider=provider,
        lo=recipe.difficulty_lo,
        hi=recipe.difficulty_hi,
    )

    final_dir = filt_dir
    if include_distillation and recipe.distillation.enabled:
        distill_result = asyncio.run(
            distill_shards(filt_dir, distill_dir, concurrency=recipe.distillation.concurrency)
        )
        final_dir = distill_dir
    else:
        distill_result = {
            "records_in": int(filter_result["records_out"]),
            "records_out": int(filter_result["records_out"]),
        }

    if upload:
        repo_id = dataset_repo_id(recipe.name, data_version)
        upload_dataset_shards(final_dir, repo_id)

    return {
        "normalized_in": n_in,
        "normalized_out": n_out,
        "filtered_kept": int(filter_result["records_out"]),
        "distilled_kept": int(distill_result["records_out"]),
    }


@click.command()
@click.option(
    "--recipe",
    "recipe_paths",
    multiple=True,
    required=True,
    type=click.Path(path_type=Path),
)
@click.option("--data-version", required=True)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("/workspace/data/build"),
    show_default=True,
)
@click.option(
    "--base-model-id",
    default="Qwen/Qwen2.5-VL-7B-Instruct",
    show_default=True,
    help="Model used for pass@K difficulty filter inference",
)
@click.option("--pass-k", default=8, show_default=True)
@click.option(
    "--include-distillation/--no-distillation",
    default=True,
    show_default=True,
    help="Run teacher distillation (Claude+GPT) after filter",
)
@click.option(
    "--upload/--no-upload",
    default=True,
    show_default=True,
    help="Upload final shards to HF dataset repo",
)
def main(
    recipe_paths: tuple[Path, ...],
    data_version: str,
    work_dir: Path,
    base_model_id: str,
    pass_k: int,
    include_distillation: bool,
    upload: bool,
) -> None:
    """Build dataset shards from one or more recipes."""
    run_name = f"dataprep-{data_version}"
    summary: dict[str, object] = {}
    try:
        for rp in recipe_paths:
            recipe = load_recipe(rp)
            click.echo(f"[build] recipe={recipe.name} sources={[s.source for s in recipe.sources]}")
            sub_work = work_dir / recipe.name
            sub_work.mkdir(parents=True, exist_ok=True)
            result = build_one_recipe(
                recipe,
                work_dir=sub_work,
                data_version=data_version,
                base_model_id=base_model_id,
                pass_k=pass_k,
                include_distillation=include_distillation,
                upload=upload,
            )
            summary[recipe.name] = result
            click.echo(f"[build] {recipe.name}: {result}")
    except Exception as e:
        post_status(
            "failure",
            task="dataprep",
            run_name=run_name,
            payload={"error": str(e), "summary": summary},
        )
        raise

    post_status(
        "completed",
        task="dataprep",
        run_name=run_name,
        payload={
            "data_version": data_version,
            "summary": summary,
            "datasets": [dataset_repo_id(name, data_version) for name in summary],
            "include_distillation": include_distillation,
        },
    )


if __name__ == "__main__":
    main()
