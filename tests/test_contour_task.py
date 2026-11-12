"""harwood contour on the command line: the SVG it writes, IRAF's message and the
line of its levels, and the constant image refused with IRAF's reason."""

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


def test_a_map_is_drawn_with_its_labels(run, tmp_path):
    result = run("contour", "gal.fits", "--device", "gal.svg")
    assert result.exit_code == 0, result.output
    assert "Image will be block averaged by 3 in x and 3 in y" in result.output
    svg = (tmp_path / "gal.svg").read_text(encoding="utf-8")
    assert "contoured from" in svg and "<polyline" in svg


def test_a_constant_image_is_refused(run):
    result = run("contour", "flat.fits", "--device", "flat.svg")
    assert result.exit_code != 0
    assert "constant valued array" in result.output
