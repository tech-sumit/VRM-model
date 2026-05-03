"""VRM CLI entrypoint. Subcommands attached lazily."""

from __future__ import annotations

import click

from vrm import __version__


@click.group(help="VRM-7B training & evaluation toolchain")
@click.version_option(__version__)
def main() -> None:
    """Top-level CLI group."""


@main.command(help="Print package version and exit.")
def version() -> None:
    click.echo(__version__)


def _attach_subcommands() -> None:
    """Attach subgroups; deferred to allow optional submodule failures during dev."""
    try:
        from vrm.infra.runpod import cli as runpod_cli

        main.add_command(runpod_cli, name="runpod")
    except ImportError:
        pass
    try:
        from vrm.data.cli import data as data_cli

        main.add_command(data_cli, name="data")
    except ImportError:
        pass
    try:
        from vrm.eval.cli import eval_ as eval_cli

        main.add_command(eval_cli, name="eval")
    except ImportError:
        pass


_attach_subcommands()


if __name__ == "__main__":
    main()
