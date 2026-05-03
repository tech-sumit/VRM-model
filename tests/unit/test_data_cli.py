"""Tests for `vrm data ...` CLI subgroup wiring (no network/GPU)."""

from click.testing import CliRunner

from vrm.cli import main


def test_data_subgroup_lists_normalize_filter_build():
    runner = CliRunner()
    result = runner.invoke(main, ["data", "--help"])
    assert result.exit_code == 0
    assert "normalize" in result.output
    assert "filter" in result.output
    assert "build" in result.output


def test_data_filter_help_shows_lo_hi_pass_k():
    runner = CliRunner()
    result = runner.invoke(main, ["data", "filter", "--help"])
    assert result.exit_code == 0
    assert "--lo" in result.output
    assert "--hi" in result.output
    assert "--pass-k" in result.output


def test_data_build_help_shows_recipe_and_distill_toggle():
    runner = CliRunner()
    result = runner.invoke(main, ["data", "build", "--help"])
    assert result.exit_code == 0
    assert "--recipe" in result.output
    assert "--include-distillation" in result.output
    assert "--data-version" in result.output


def test_eval_subgroup_lists_compare_and_parse():
    runner = CliRunner()
    result = runner.invoke(main, ["eval", "--help"])
    assert result.exit_code == 0
    assert "compare" in result.output
    assert "parse" in result.output
