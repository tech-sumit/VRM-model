"""End-to-end test for `vrm eval parse` CLI (works without GPU)."""

import json
from pathlib import Path

from click.testing import CliRunner

from vrm.cli import main


def test_eval_parse_writes_json_and_md(tmp_path: Path):
    bench_dir = tmp_path / "MathVista_MINI"
    bench_dir.mkdir()
    (bench_dir / "MathVista_MINI_acc.json").write_text(
        json.dumps({"Overall": 0.523, "by_category": {"algebra": 0.61}})
    )
    out = tmp_path / "report.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "eval",
            "parse",
            "--work-dir",
            str(tmp_path),
            "--checkpoint",
            "tech-sumit/vrm-7b-grpo-r1",
            "--suite",
            "full",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["checkpoint"] == "tech-sumit/vrm-7b-grpo-r1"
    assert data["metrics"]["MathVista_MINI.Overall"] == 0.523
    md = (out.with_suffix(".md")).read_text()
    assert "tech-sumit/vrm-7b-grpo-r1" in md
