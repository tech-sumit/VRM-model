from click.testing import CliRunner

from vrm.cli import main


def test_version_command_outputs_semver():
    runner = CliRunner()
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    assert result.output.strip().count(".") == 2  # x.y.z
