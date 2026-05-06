"""End-to-end data-build pipeline: normalize -> filter -> distill -> upload.

Invoked from the dataprep pod with one or more recipe YAML files. For each
recipe:
    1. For every source listed in `recipe.sources`, normalize the upstream HF
       dataset into parquet shards under `<work>/normalized/<source>/`,
       capped at `cap` records per source.
    2. Compute pass@K difficulty per record (VL inference via Transformers by
       default; set ``VRM_VL_BACKEND=vllm`` for vLLM) and keep records with
       `lo <= pass@K <= hi`.
    3. (If `recipe.distillation.enabled`) ask Claude + GPT-4o for solutions
       and pick the best verifier-passing completion per record.
    4. Upload the final parquet shards to the HF dataset repo.

This is the single entry the dataprep pod calls; pod-entrypoint.sh wires
`VRM_TASK=dataprep` -> `python -m vrm.data.build --recipe ... --data-version ...`.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import click

from vrm.data.distill import distill_shards
from vrm.data.filter import filter_shards
from vrm.data.normalize import REGISTRY
from vrm.data.normalize._driver import DEFAULT_SHARD_SIZE, normalize_dataset
from vrm.data.recipe import Recipe, load_recipe
from vrm.data.schema import Record
from vrm.infra.hf_hub import dataset_repo_id, upload_dataset_shards
from vrm.infra.r2 import R2Client, download_stage_to_local, get_client
from vrm.infra.webhook import post_status

STAGE_ALL = "all"
STAGE_NORMALIZE = "normalize"
STAGE_FILTER = "filter"
STAGE_DISTILL = "distill"
VALID_STAGES = (STAGE_ALL, STAGE_NORMALIZE, STAGE_FILTER, STAGE_DISTILL)


def _normalize_one_source(
    source: str,
    cap: int,
    *,
    out_dir: Path,
    r2: R2Client | None,
    data_version: str,
) -> dict[str, Any]:
    from datasets import load_dataset

    spec = REGISTRY[source]
    load_kwargs: dict[str, Any] = {
        "name": spec.config,
        "split": spec.split,
        "streaming": False,
        "verification_mode": "no_checks",
    }
    if getattr(spec, "data_files", None):
        load_kwargs["data_files"] = spec.data_files
    ds = load_dataset(spec.hf_id, **load_kwargs)
    n = min(cap, len(ds))

    # Resume support: if R2 already has shards for this (data_version,
    # source), skip the raw rows we've already normalized.
    start_row = 0
    start_shard_idx = 0
    if r2 is not None:
        state = r2.read_state(data_version, "normalized", source)
        if state.get("done"):
            # Fully done -- nothing to do this run.
            return {
                "records_in": int(state.get("records_in", 0)),
                "records_out": int(state.get("records_out", 0)),
                "shards": int(state.get("shards_written", 0)),
                "final_shard_idx": int(state.get("shards_written", 0)),
                "resumed": "skipped",
            }
        start_row = int(state.get("last_row_index", 0))
        start_shard_idx = int(state.get("shards_written", 0))
        if start_row >= n:
            return {
                "records_in": int(state.get("records_in", 0)),
                "records_out": int(state.get("records_out", 0)),
                "shards": int(state.get("shards_written", 0)),
                "final_shard_idx": int(state.get("shards_written", 0)),
                "resumed": "already-covered-cap",
            }

    ds = ds.select(range(start_row, n))
    raw_result = normalize_dataset(
        (dict(r) for r in ds),
        source=source,
        out_dir=out_dir,
        shard_size=DEFAULT_SHARD_SIZE,
        r2=r2,
        data_version=data_version,
        stage="normalized",
        start_shard_idx=start_shard_idx,
        start_row_offset=start_row,
        total_rows_hint=n,
    )
    result: dict[str, Any] = dict(raw_result)
    if start_row:
        result["resumed"] = f"from_row={start_row}"
    return result


def _difficulty_provider_factory(model_id: str, k: int):
    """Returns callable Record -> pass@K (lazy VL inference via ``generate_responses``)."""

    cache: dict[str, object] = {}

    def _provider(rec: Record) -> float:
        if "llm" not in cache:
            click.echo(
                f"[filter] wiring VL inference (lazy init): model_id={model_id!r} pass_k={k}",
                err=True,
            )
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


def _run_normalize(
    recipe: Recipe,
    *,
    norm_dir: Path,
    data_version: str,
    r2: R2Client | None,
) -> dict[str, int]:
    n_in = 0
    n_out = 0
    for sc in recipe.sources:
        out = norm_dir / sc.source
        out.mkdir(parents=True, exist_ok=True)
        result = _normalize_one_source(sc.source, sc.cap, out_dir=out, r2=r2, data_version=data_version)
        click.echo(f"[build] normalized {sc.source}: {result}")
        n_in += result["records_in"]
        n_out += result["records_out"]
    return {"normalized_in": n_in, "normalized_out": n_out}


def _run_filter(
    recipe: Recipe,
    *,
    norm_dir: Path,
    filt_dir: Path,
    data_version: str,
    base_model_id: str,
    pass_k: int,
    r2: R2Client | None,
) -> dict[str, float]:
    # Flatten normalized per-source shards into one directory so filter
    # iterates in a single pass. When R2 is set, download any missing
    # source shards that aren't already on this pod's local disk.
    if r2 is not None:
        sources = [sc.source for sc in recipe.sources]
        report = download_stage_to_local(
            r2,
            data_version=data_version,
            stage="normalized",
            local_root=norm_dir,
            sources=sources,
            include_images=True,
        )
        click.echo(f"[build] downloaded normalized from R2: {report}")

    norm_flat = norm_dir.parent / "normalized_flat"
    click.echo(
        f"[build] filter flatten into {norm_flat} (symlink/copy shards + images from {norm_dir}) …",
        err=True,
    )
    norm_flat.mkdir(parents=True, exist_ok=True)
    # Symlink every source's images into a single flat dir so that record
    # paths of the form "images/<src>-<idx>-<i>.jpg" resolve when filter runs
    # with cwd=norm_flat. Image filenames already carry the source prefix, so
    # collisions across sources are impossible.
    flat_images = norm_flat / "images"
    flat_images.mkdir(parents=True, exist_ok=True)
    for src_dir in sorted(p for p in norm_dir.iterdir() if p.is_dir()):
        for shard in sorted(src_dir.glob("*.parquet")):
            dst = norm_flat / f"{src_dir.name}-{shard.name}"
            if not dst.exists():
                dst.write_bytes(shard.read_bytes())
        src_images = src_dir / "images"
        if src_images.is_dir():
            for img in src_images.iterdir():
                link = flat_images / img.name
                if not link.exists():
                    try:
                        link.symlink_to(img.resolve())
                    except OSError:
                        # Symlink not permitted on some filesystems: hardlink,
                        # then copy as final fallback.
                        try:
                            os.link(img, link)
                        except OSError:
                            link.write_bytes(img.read_bytes())

    n_flat_parquet = len(list(norm_flat.glob("*.parquet")))
    click.echo(
        f"[build] filter flatten done: {n_flat_parquet} parquet(s) in {norm_flat}",
        err=True,
    )

    provider = _difficulty_provider_factory(base_model_id, pass_k)
    prev_cwd = os.getcwd()
    os.chdir(norm_flat)
    try:
        return filter_shards(
            norm_flat,
            filt_dir,
            difficulty_provider=provider,
            lo=recipe.difficulty_lo,
            hi=recipe.difficulty_hi,
            r2=r2,
            data_version=data_version,
        )
    finally:
        os.chdir(prev_cwd)


def _run_distill(
    recipe: Recipe,
    *,
    filt_dir: Path,
    distill_dir: Path,
    data_version: str,
    r2: R2Client | None,
) -> dict[str, Any]:
    if r2 is not None:
        report = download_stage_to_local(
            r2,
            data_version=data_version,
            stage="filtered",
            local_root=filt_dir,
            include_images=False,  # images referenced via record paths downloaded separately
        )
        click.echo(f"[build] downloaded filtered from R2: {report}")

    if not recipe.distillation.enabled:
        return {"records_in": 0, "records_out": 0, "skipped": True}

    # filt_dir is structured as <source>/shard-*.parquet when downloaded
    # from R2; flatten into a single directory for distill iteration.
    filt_flat = filt_dir.parent / "filtered_flat"
    filt_flat.mkdir(parents=True, exist_ok=True)
    for p in filt_dir.rglob("shard-*.parquet"):
        dst = filt_flat / (p.parent.name + "-" + p.name if p.parent != filt_dir else p.name)
        if not dst.exists():
            dst.write_bytes(p.read_bytes())
    # Also copy any "all/" single-stage shards (filter output has all/).
    for p in (filt_dir / "all").glob("shard-*.parquet") if (filt_dir / "all").exists() else []:
        dst = filt_flat / p.name
        if not dst.exists():
            dst.write_bytes(p.read_bytes())

    return asyncio.run(
        distill_shards(
            filt_flat,
            distill_dir,
            concurrency=recipe.distillation.concurrency,
            data_version=data_version,
        )
    )


def build_one_recipe(
    recipe: Recipe,
    *,
    work_dir: Path,
    data_version: str,
    base_model_id: str,
    pass_k: int,
    include_distillation: bool,
    upload: bool,
    stage: str = STAGE_ALL,
) -> dict[str, Any]:
    if stage not in VALID_STAGES:
        raise ValueError(f"stage={stage!r} not in {VALID_STAGES}")

    norm_dir = work_dir / "normalized"
    filt_dir = work_dir / "filtered"
    distill_dir = work_dir / "distilled"
    r2 = get_client()
    if r2 is None:
        click.echo(
            "[build] WARNING: R2_* env vars not set -- running WITHOUT durable "
            "checkpointing. A pod crash will lose all progress.",
            err=True,
        )

    summary: dict[str, Any] = {"stage": stage}

    if stage in (STAGE_ALL, STAGE_NORMALIZE):
        summary |= _run_normalize(recipe, norm_dir=norm_dir, data_version=data_version, r2=r2)
        if stage == STAGE_NORMALIZE:
            return summary

    if stage in (STAGE_ALL, STAGE_FILTER):
        filter_result = _run_filter(
            recipe,
            norm_dir=norm_dir,
            filt_dir=filt_dir,
            data_version=data_version,
            base_model_id=base_model_id,
            pass_k=pass_k,
            r2=r2,
        )
        summary["filtered_kept"] = int(filter_result["records_out"])
        summary["filter_kept_pct"] = float(filter_result.get("kept_pct", 0.0))
        if stage == STAGE_FILTER:
            return summary

    if stage in (STAGE_ALL, STAGE_DISTILL):
        final_dir = distill_dir
        if include_distillation and recipe.distillation.enabled:
            distill_result = _run_distill(
                recipe,
                filt_dir=filt_dir,
                distill_dir=distill_dir,
                data_version=data_version,
                r2=r2,
            )
            summary["distilled_kept"] = int(distill_result.get("records_out", 0))
            summary["credits_usd"] = float(distill_result.get("credits_usd", 0.0))
            if distill_result.get("paused"):
                summary["distill_paused"] = True
        else:
            final_dir = filt_dir
            summary["distilled_kept"] = summary.get("filtered_kept", 0)

        if upload:
            repo_id = dataset_repo_id(recipe.name, data_version)
            upload_dataset_shards(final_dir, repo_id)
            summary["uploaded_to"] = repo_id

    return summary


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
    help="Upload final shards to HF dataset repo (only in distill/all stages)",
)
@click.option(
    "--stage",
    type=click.Choice(list(VALID_STAGES), case_sensitive=False),
    default=STAGE_ALL,
    show_default=True,
    help=(
        "Which pipeline stage to run. 'normalize' (CPU), 'filter' (GPU: Transformers "
        "default, or vLLM if VRM_VL_BACKEND=vllm), "
        "'distill' (CPU + OpenRouter), or 'all' for the full pipeline on one pod."
    ),
)
def main(
    recipe_paths: tuple[Path, ...],
    data_version: str,
    work_dir: Path,
    base_model_id: str,
    pass_k: int,
    include_distillation: bool,
    upload: bool,
    stage: str,
) -> None:
    """Build dataset shards from one or more recipes."""
    run_name = f"dataprep-{stage}-{data_version}"
    task_label = f"dataprep-{stage}"
    summary: dict[str, Any] = {}
    try:
        for rp in recipe_paths:
            recipe = load_recipe(rp)
            click.echo(
                f"[build] stage={stage} recipe={recipe.name} sources={[s.source for s in recipe.sources]}"
            )
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
                stage=stage,
            )
            summary[recipe.name] = result
            click.echo(f"[build] {recipe.name}: {result}")
    except Exception as e:
        post_status(
            "failure",
            task=task_label,
            run_name=run_name,
            payload={"error": str(e), "summary": summary},
        )
        raise

    post_status(
        "completed",
        task=task_label,
        run_name=run_name,
        payload={
            "data_version": data_version,
            "stage": stage,
            "summary": summary,
        },
    )


if __name__ == "__main__":
    main()
