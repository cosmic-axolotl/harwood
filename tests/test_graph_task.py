"""harwood graph on the command line: a list piped in, files and images named, the
SVG it writes without a screen, and IRAF's complaints as warnings that say what to
do. What it draws is IRAF's (tests/test_plotgraph_iraf.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from harwood.cli.main import app

sys.path.insert(0, str(Path(__file__).parent / "data" / "iraf_graph"))
from gr_inputs import write  # noqa: E402


@pytest.fixture
def run(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    def invoke(*args, **kwargs):
        return runner.invoke(app, list(args), **kwargs)

    return invoke


def test_a_list_piped_in_is_plotted(run, tmp_path):
    result = run("graph", "--device", "piped.svg", "--pointmode", input="1 10\n2 30\n3 20\n")
    assert result.exit_code == 0, result.output
    svg = (tmp_path / "piped.svg").read_text(encoding="utf-8")
    assert svg.count("<polyline") >= 3          # three boxes


def test_lists_and_images_together(run, tmp_path):
    result = run("graph", "two.txt,spec.fits", "--wcs", "world", "--device", "both.svg")
    assert result.exit_code == 0, result.output
    svg = (tmp_path / "both.svg").read_text(encoding="utf-8")
    assert "a spectrum" in svg and "Wavelength (Angstroms)" in svg
    assert 'stroke-dasharray="6 4"' in svg      # the second curve is dashed
    assert "only 1 arg; two.txt, line 7" in result.output
    assert "every line needs both" in result.output


def test_nothing_to_read_stops_with_irafs_reason(run):
    result = run("graph", "nothere.txt", "--device", "x.svg")
    assert result.exit_code != 0
    assert "No curves read" in result.output


def test_irafs_device_is_refused(run):
    result = run("graph", "one.txt", "--append")
    assert result.exit_code != 0
    assert "one graph" in result.output
