"""CLI smoke tests."""

from satark.core.cli.app import app
from typer.testing import CliRunner

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "satark" in result.stdout


def test_list_plugins() -> None:
    result = runner.invoke(app, ["list-plugins"])
    assert result.exit_code == 0
    assert "insider" in result.stdout
