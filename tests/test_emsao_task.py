"""harwood emsao on the command line: the report on the terminal and appended to
emsao.log, and RVSAO's stop when a line list is missing. What it computes is
RVSAO's (tests/test_emsao_rvsao.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from harwood.cli.main import app

sys.path.insert(0, str(Path(__file__).parent / "data" / "iraf_emsao"))
from em_inputs import write  # noqa: E402


@pytest.fixture
def run(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    return lambda *args: runner.invoke(app, list(args))


def test_the_report_goes_to_the_terminal_and_the_log(run, tmp_path):
    result = run("emsao", "ngc.fits", "--vel-corr", "none")
    assert result.exit_code == 0, result.output
    assert "Emission vel =   779.6" in result.output
    log = (tmp_path / "emsao.log").read_text(encoding="utf-8")
    assert log.startswith("harwood rvsao.emsao 2.8.5 ")
    assert "  Ha   6562.82" in log
    run("emsao", "ngc.fits", "--vel-corr", "none")
    assert (tmp_path / "emsao.log").read_text(encoding="utf-8").count("File: ngc.fits") == 2


def test_a_missing_line_list_stops_it(run):
    result = run("emsao", "ngc.fits", "--emlines", "nothere.dat")
    assert result.exit_code != 0
    assert "Cannot open file (rvsao$lib/nothere.dat)" in result.output
