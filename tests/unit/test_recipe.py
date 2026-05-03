from pathlib import Path

from vrm.data.recipe import load_recipe


def test_load_sft_recipe(repo_root: Path):
    rec = load_recipe(repo_root / "configs/data/sft_recipe.yaml")
    assert rec.name == "sft"
    assert rec.distillation.enabled is True
    assert {s.source for s in rec.sources} >= {"mavis", "mathv360k"}


def test_load_rl_recipe(repo_root: Path):
    rec = load_recipe(repo_root / "configs/data/rl_recipe.yaml")
    assert rec.name == "rl"
    assert rec.distillation.enabled is False
