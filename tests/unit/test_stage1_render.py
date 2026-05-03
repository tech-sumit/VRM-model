from pathlib import Path

from vrm.train.stage1_sft import render_yaml


def test_render_substitutes_all_placeholders(repo_root: Path):
    template = (repo_root / "configs/stage1_sft_full.yaml").read_text()
    out = render_yaml(
        template,
        {
            "OUTPUT_DIR": "/runs/sft-r1",
            "DATA_DIR": "/datasets/vrm-sft-v1",
            "RUN_NAME": "sft-r1",
        },
    )
    assert "${OUTPUT_DIR}" not in out
    assert "${DATA_DIR}" not in out
    assert "${RUN_NAME}" not in out
    assert "/runs/sft-r1" in out
    assert "sft-r1" in out
