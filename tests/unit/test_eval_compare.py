import json
from pathlib import Path

from click.testing import CliRunner

from vrm.eval.compare import main as compare_main


def test_compare_writes_delta_report(tmp_path: Path):
    base = tmp_path / "base.json"
    cur = tmp_path / "cur.json"
    out = tmp_path / "delta.md"
    base.write_text(
        json.dumps(
            {
                "checkpoint": "qwen-base",
                "suite": "full",
                "metrics": {"MathVista.Overall": 0.50, "BLINK.acc": 0.40},
            }
        )
    )
    cur.write_text(
        json.dumps(
            {
                "checkpoint": "vrm-7b-grpo",
                "suite": "full",
                "metrics": {"MathVista.Overall": 0.62, "BLINK.acc": 0.55, "NEW.foo": 0.1},
            }
        )
    )
    runner = CliRunner()
    res = runner.invoke(compare_main, ["--baseline", str(base), "--current", str(cur), "--out", str(out)])
    assert res.exit_code == 0
    md = out.read_text()
    assert "vrm-7b-grpo" in md
    assert "+0.1200" in md
    assert "+0.1500" in md
    assert "—" in md
