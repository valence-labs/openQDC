import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.parametrize("commands", ["datasets"])
def test_cli(commands):
    from openqdc.cli import app

    result = runner.invoke(app, [commands])
    assert result.exit_code == 0
