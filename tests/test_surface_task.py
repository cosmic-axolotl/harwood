"""harwood surface on the command line: the SVG of the mesh, and IRAF's messages
when the limits leave nothing to draw (a warning, as IRAF prints and draws nothing)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from harwood.cli.main import app

sys.path.insert(0, str(Path(__file__).parent / "data" / "iraf_contour"))
from ct_inputs import write  # noqa: E402


@pytest.fixture
def run(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    return lambda *args: runner.invoke(app, list(args))


def test_the_mesh_is_drawn(run, tmp_path):
    result = run("surface", "gal.fits", "--label", "--device", "gal.svg")
    assert result.exit_code == 0, result.output
    svg = (tmp_path / "gal.svg").read_text(encoding="utf-8")
    assert svg.count("<polygon") > 1000 and "X-AXIS" in svg


def test_limits_that_leave_nothing_are_said(run, tmp_path):
    result = run("surface", "gal.fits", "--floor", "1e6", "--device", "none.svg")
    assert result.exit_code == 0, result.output
    assert "Entire image is at or below specified floor" in result.output
    assert not (tmp_path / "none.svg").exists()
