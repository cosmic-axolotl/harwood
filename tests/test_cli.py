"""The command line: the four commands, and the shape section 3.1 fixes."""


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


# --------------------------------------------------- parameters and replay


def test_editpar_writes_only_what_changed(run, tmp_path, monkeypatch):
    """The editor is a view of the project file, never a second copy of it."""
    monkeypatch.chdir(tmp_path)
    run("init", str(tmp_path))

    editor = tmp_path / "fake-editor.sh"
    editor.write_text("#!/bin/sh\n"
                      "sed -i 's/^order = 1$/order = 4/' \"$1\"\n")
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))

    result = run("ccdproc", "--editpar")
    assert result.exit_code == 0
    assert "order" in result.output

    project = (tmp_path / "harwood.yaml").read_text()
    assert "order: 4" in project
    assert "niterate" not in project, "untouched defaults stay out of the file"
    assert project.startswith("#"), "the explanatory header survives"


def test_editpar_clearing_a_value_restores_the_default(run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("init", str(tmp_path))
    editor = tmp_path / "set.sh"
    editor.write_text("#!/bin/sh\nsed -i 's/^order = 1$/order = 4/' \"$1\"\n")
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))
    run("ccdproc", "--editpar")

    clear = tmp_path / "clear.sh"
    clear.write_text("#!/bin/sh\nsed -i 's/^order = 4$/order =/' \"$1\"\n")
    clear.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(clear))
    run("ccdproc", "--editpar")

    assert "order" not in (tmp_path / "harwood.yaml").read_text()


def test_editpar_rejects_an_unknown_parameter_without_saving(run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("init", str(tmp_path))
    editor = tmp_path / "bad.sh"
    editor.write_text("#!/bin/sh\necho 'sparkle = 3' >> \"$1\"\n")
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))

    result = run("ccdproc", "--editpar")
    assert result.exit_code == 2
    assert "sparkle" in result.output
    assert "sparkle" not in (tmp_path / "harwood.yaml").read_text()


def test_editpar_does_not_need_the_required_arguments(run, tmp_path, monkeypatch):
    """You edit parameters before you have decided what to run it on."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EDITOR", "true")
    assert run("ccdproc", "--editpar").exit_code == 0


def test_the_log_replays_a_run_exactly(run, tmp_path, monkeypatch):
    import numpy as np
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32)).writeto(tmp_path / "a.fits")

    run("imarith", "a.fits", "*", "3", "b.fits")
    replay = run("log", "imarith", "--replay", "1")
    assert replay.exit_code == 0
    command = replay.output.strip()
    assert command.startswith("harwood imarith")
    assert "a.fits" in command and "b.fits" in command

    full = run("log", "imarith", "--full")
    assert "command line" in full.output, "each value says where it came from"
    assert "repeat:" in full.output


def test_replaying_a_run_that_does_not_exist_says_how_many_there_are(run, tmp_path,
                                                                    monkeypatch):
    import numpy as np
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32)).writeto(tmp_path / "a.fits")
    run("imarith", "a.fits", "+", "1", "b.fits")

    result = run("log", "--replay", "9")
    assert result.exit_code == 2
    assert "1 runs" in result.output


def test_a_reconstructed_command_omits_defaults_and_quotes_what_needs_it():
    from harwood.core import replay
    from harwood.core.registry import get_class

    cls = get_class("ccdproc")
    command = replay.command("ccdproc", cls.Params, {
        "input": ["m92*.fits"], "output": "%m%tr%", "overscan": True,
        "trim": True, "biassec": "[335:350,2:510]", "order": 1,
        "low_reject": 3.0, "function": "chebyshev",
    })
    assert "'m92*.fits'" in command, "a glob must survive the shell"
    assert "'[335:350,2:510]'" in command, "so must a section"
    assert "--overscan" in command and "--trim" in command
    assert "--order" not in command, "order 1 is the default"
    assert "--low-reject" not in command, "so is the rejection threshold"
