"""Tests for the data-build orchestrator (orchestration logic, no network/GPU)."""

import importlib


def test_build_module_imports_and_exposes_main():
    mod = importlib.import_module("vrm.data.build")
    assert hasattr(mod, "main"), "vrm.data.build must expose a Click `main` command"
    assert hasattr(mod, "build_one_recipe")
    assert hasattr(mod, "_difficulty_provider_factory")


def test_build_main_help():
    from click.testing import CliRunner

    from vrm.data.build import main as build_main

    runner = CliRunner()
    res = runner.invoke(build_main, ["--help"])
    assert res.exit_code == 0
    for flag in (
        "--recipe",
        "--data-version",
        "--include-distillation",
        "--upload",
        "--pass-k",
    ):
        assert flag in res.output


def test_difficulty_provider_factory_returns_callable():
    from vrm.data.build import _difficulty_provider_factory

    p = _difficulty_provider_factory("Qwen/Qwen2.5-VL-7B-Instruct", k=8)
    assert callable(p)
