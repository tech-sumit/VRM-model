import json
from pathlib import Path

from vrm.eval.parse_results import parse_results_dir


def test_parse_results_flattens_nested_metrics(tmp_path: Path):
    bench_dir = tmp_path / "MathVista_MINI"
    bench_dir.mkdir()
    (bench_dir / "MathVista_MINI_acc.json").write_text(
        json.dumps({"Overall": 0.523, "by_category": {"algebra": 0.61, "geometry": 0.48}})
    )
    rep = parse_results_dir(tmp_path, checkpoint="ckpt", suite="full")
    assert rep.checkpoint == "ckpt"
    assert rep.suite == "full"
    assert rep.metrics["MathVista_MINI.Overall"] == 0.523
    assert rep.metrics["MathVista_MINI.by_category.algebra"] == 0.61


def test_to_markdown_renders_table(tmp_path: Path):
    bench_dir = tmp_path / "MathVerse_MINI"
    bench_dir.mkdir()
    (bench_dir / "MathVerse_MINI.json").write_text(json.dumps({"acc": 0.42}))
    rep = parse_results_dir(tmp_path, checkpoint="ckpt", suite="full")
    md = rep.to_markdown()
    assert "ckpt" in md
    assert "MathVerse_MINI" in md
    assert "0.4200" in md
