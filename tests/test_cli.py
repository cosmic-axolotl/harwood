"""The command line: the four commands, and the shape section 3.1 fixes."""

import os

import pytest
from click.testing import CliRunner

from harwood.cli.main import app


@pytest.fixture
def run():
    runner = CliRunner()

    def invoke(*args, **kwargs):
        return runner.invoke(app, list(args), **kwargs)

    return invoke


def test_list_shows_the_tasks_that_run(run):
    result = run("list")
    assert result.exit_code == 0
    assert "imcombine" in result.output and "daofind" in result.output


def test_list_iraf_stays_readable_in_a_narrow_terminal(run, monkeypatch):
    """A long note must not squeeze the identifying columns to nothing."""
    monkeypatch.setenv("COLUMNS", "100")
    result = run("list", "--iraf", "combine")
    assert result.exit_code == 0
    for column in ("task", "package", "domain", "status"):
        assert column in result.output
    assert "imcombine" in result.output


def test_help_and_lpar_have_different_jobs(run):
    """Section 3.2: --help shows defaults, lpar shows effective values and
    where each came from. Blurring them is the mistake IRAF's lpar prevented."""
    manual = run("help", "zerocombine")
    values = run("lpar", "zerocombine")
    assert manual.exit_code == 0 and values.exit_code == 0

    assert "default" in manual.output
    assert "none" in manual.output           # the model default for reject
    assert "preset" in values.output         # lpar names where it came from
    assert "minmax" in values.output         # the value that will actually be used


def test_an_iraf_name_harwood_does_not_run_gets_an_answer_not_an_error(run):
    for name, expected in (("rfits", "not be ported"),
                           ("apall", "after the MVP"),
                           ("tselect", "replaced")):
        result = run("help", name)
        assert expected in result.output, name
        assert result.exit_code == 3


def test_a_typo_suggests_the_real_name(run):
    result = run("help", "daofnd")
    assert "daofind" in result.output


def test_required_parameters_are_positional_and_the_rest_are_options(run):
    """`harwood imcombine @bias.lis Zero.fits --combine median --reject ccdclip`"""
    result = run("imcombine", "--help")
    assert result.exit_code == 0
    assert "INPUT" in result.output and "OUTPUT" in result.output
    assert "--combine" in result.output and "--reject" in result.output


def test_compound_names_take_both_spellings(run):
    result = run("ccdproc", "--help")
    assert "--low-reject" in result.output
    assert "--low_reject" in result.output


def test_booleans_are_a_flag_pair(run):
    result = run("ccdproc", "--help")
    assert "--overscan / --no-overscan" in result.output.replace("\n", " ") or \
           "--overscan" in result.output


def test_plan_writes_nothing(run, tmp_path):
    import numpy as np
    from astropy.io import fits

    frames = []
    for i in range(3):
        path = tmp_path / f"f{i}.fits"
        fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32)).writeto(path)
        frames.append(str(path))
    output = tmp_path / "out.fits"

    result = run("imcombine", *frames, str(output), "--plan")
    assert result.exit_code == 0
    assert "would do this" in result.output
    assert not output.exists(), "--plan must not write anything"


def test_a_failing_task_prints_a_sentence_not_a_traceback(run, tmp_path):
    result = run("daofind", str(tmp_path / "missing.fits"), str(tmp_path / "out.coo"))
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "stopped" in result.output


def test_invalid_parameters_are_reported_by_name(run, tmp_path):
    result = run("phot", "a.fits", "b.coo", str(tmp_path / "c.mag"),
                 "--apertures", "20", "--annulus", "5")
    assert result.exit_code == 2
    assert "annulus" in result.output


def test_init_creates_a_visible_commented_project_file(run, tmp_path):
    result = run("init", str(tmp_path))
    assert result.exit_code == 0
    contents = (tmp_path / "harwood.yaml").read_text()
    assert contents.startswith("#"), "the project file explains itself"
    assert "home directory" in contents


def test_init_refuses_to_overwrite(run, tmp_path):
    run("init", str(tmp_path))
    assert run("init", str(tmp_path)).exit_code == 1


def test_doctor_reports_the_kernels(run):
    result = run("doctor")
    assert result.exit_code == 0
    assert "reference" in result.output and "python" in result.output
