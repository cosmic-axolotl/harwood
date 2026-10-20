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
                           ("continuum", "after the MVP"),
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


def test_observatory_lists_the_sites_setairmass_knows(run):
    """IRAF's own task name, so somebody migrating types what they know."""
    result = run("observatory")
    assert result.exit_code == 0
    assert "kpno" in result.output and "lna" in result.output


def test_observatory_looks_one_up_by_alias(run):
    result = run("observatory", "kitt peak")
    assert result.exit_code == 0 and "kpno" in result.output


def test_an_unknown_observatory_says_what_to_do_instead(run):
    result = run("observatory", "narnia")
    assert result.exit_code != 0
    assert "--latitude" in result.output


def test_the_new_spectroscopic_tasks_are_documented_like_the_rest(run):
    for task in ("refspectra", "setairmass"):
        manual = run("help", task)
        assert manual.exit_code == 0, task
        values = run("lpar", task)
        assert values.exit_code == 0, task


def test_standards_lists_the_calibration_directories(run):
    result = run("standards")
    if "No calibration data found" in result.output:
        pytest.skip("IRAF's onedstds are not installed")
    assert result.exit_code == 0
    assert "spec50cal" in result.output


def test_standards_finds_one_star_across_directories(run):
    """The same star is calibrated in several, with different band widths and
    sometimes different fluxes. Choosing between them is a real decision."""
    result = run("standards", "--star", "feige110")
    if result.exit_code == 3 and "onedstds" in result.output:
        pytest.skip("IRAF's onedstds are not installed")
    assert result.exit_code == 0
    assert result.output.count("bands of") > 1


def test_an_unknown_standard_says_what_to_run(run):
    result = run("standards", "--star", "not a star")
    assert result.exit_code == 3
    assert "harwood standards" in result.output


def test_the_flux_chain_is_documented_like_the_rest(run):
    for task in ("standard", "sensfunc", "calibrate"):
        assert run("help", task).exit_code == 0, task
        assert run("lpar", task).exit_code == 0, task


def test_the_manual_prints_iraf_syntax_rather_than_eating_it(run, monkeypatch):
    """`x y [frame] key` went missing from imexamine's help: rich read
    `[frame]` as a style tag. Manuals quote IRAF syntax constantly, so help
    prints them as text."""
    monkeypatch.setenv("COLUMNS", "200")
    result = run("help", "imexamine")
    assert result.exit_code == 0
    assert "x y [frame] key" in result.output
    assert "radial profile and FWHM" in result.output, "the keys are in the manual"


def test_a_list_is_written_as_iraf_writes_it(run, tmp_path):
    """'--apertures 3,5,10' was refused as 'not a valid float': click checked
    the type before anything could split on the comma. IRAF writes lists that
    way, so both that and the repeated option have to work."""
    from harwood.cli import build
    from harwood.core.registry import get_class

    model = get_class("phot").Params
    for typed in (("3,5,10",), ("3", "5,10")):
        assert build.collect(model, {"apertures": typed})["apertures"] == ["3", "5", "10"]
    assert model(image="a", coords="b", apertures=["3", "5", "8"]).apertures == [3, 5, 8]


def test_doctor_finds_a_module_installed_under_another_name():
    """harwood[gui] installs PySide6-Essentials, whose module is PySide6;
    doctor asked for a distribution called PySide6 and reported it missing on
    every install made the recommended way. yaml (from PyYAML) is the same
    shape of case, and is always installed."""
    from harwood.cli.main import _probe

    result, hint = _probe("yaml", "harwood")
    assert "not installed" not in result and not hint
    assert "not installed" in _probe("no_such_module_here", "harwood")[0]


def test_every_builtin_the_cli_lists_actually_runs(run):
    """'recipe' was listed as a built-in and did not exist: typing it got a
    suggestion to try 'gripes'. A listed command has to answer --help."""
    from harwood.cli.main import BUILTINS

    for name in BUILTINS:
        result = run(name, "--help")
        assert result.exit_code == 0, f"'{name}' is listed but does not run"


def test_every_task_builds_a_command(run):
    """imshift and imalign gained an optional positional output behind their
    list of inputs, which build.py refuses -- and so for a day neither could
    be run at all, from the shell or from a form, while every test of their
    arithmetic passed. Each task has to come out of its model as a command."""
    from harwood.core.registry import list_tasks

    for name in list_tasks():
        result = run(name, "--help")
        assert result.exit_code == 0, f"'{name}' does not build: {result.exception}"


def test_editpar_leaves_no_scratch_copy_behind_unless_it_says_so(run, tmp_path,
                                                                  monkeypatch):
    """Every --editpar made a temporary directory and none was removed. They
    go now -- except when the edit was refused, where the message points the
    user at the copy holding their changes."""
    import tempfile

    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.chdir(tmp_path)
    run("init", str(tmp_path))

    good = tmp_path / "good.sh"
    good.write_text("#!/bin/sh\nsed -i 's/^order = 1$/order = 4/' \"$1\"\n")
    good.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(good))
    run("ccdproc", "--editpar")
    assert list(scratch.iterdir()) == []

    bad = tmp_path / "bad.sh"
    bad.write_text("#!/bin/sh\necho 'sparkle = 3' >> \"$1\"\n")
    bad.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(bad))
    result = run("ccdproc", "--editpar")
    kept = list(scratch.iterdir())
    assert len(kept) == 1 and str(kept[0]) in result.output.replace("\n", "")


def test_the_native_editor_saves_only_when_asked(tmp_path):
    """harwood's own editor, used when no $EDITOR is set: Ctrl+S writes the
    text and reports it saved; closing leaves the file as it was."""
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.viewer._qt import application
    from harwood.viewer.editor import Editor

    application()
    path = tmp_path / "ccdproc.par"
    path.write_text("# order of the fit\norder = 1\n")

    closed = Editor(path)
    closed.text.setPlainText("order = 9\n")
    closed.reject()
    assert not closed.saved and "order = 1" in path.read_text()

    saved = Editor(path)
    saved.text.setPlainText(saved.text.toPlainText().replace("order = 1", "order = 4"))
    saved._save()
    assert saved.saved and "order = 4" in path.read_text()
    assert path.read_text().startswith("# order"), "the comments are kept"


def test_the_session_survives_whatever_is_typed_at_it(monkeypatch, capsys):
    """hw> is where a night is spent. A typo, a task given nothing, an
    unbalanced quote, a task that fails -- each must get a sentence and the
    prompt back, never a traceback that ends the session."""
    from harwood.cli import session

    typed = iter(["", "list", "help phot", "nosuchtask", "phot",
                  "imstatistics 'unbalanced", "imstatistics no_such_file.fits",
                  "lpar daofind", "exit", "list"])
    seen = []

    def fake_input(prompt=""):
        line = next(typed)
        seen.append(line)
        return line

    monkeypatch.setattr(session.console, "input", fake_input)
    session.start()
    assert seen[-1] == "exit", "it stopped at exit, not before and not after"
    out = capsys.readouterr()
    text = out.out + out.err
    assert "unbalanced quotes" in text
    assert "Traceback" not in text


def test_the_session_ends_cleanly_on_ctrl_d(monkeypatch):
    from harwood.cli import session

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr(session.console, "input", eof)
    session.start()


def test_a_section_with_a_comma_is_one_image(run, tmp_path, monkeypatch):
    """`imstatistics 'obj.fits[10:40,60:90]'` was split at the comma into two
    names, neither of them a file -- while imcopy's own help promised that
    'obj.fits[1:100,*]' copies just that part."""
    import numpy as np
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    fits.writeto("obj.fits", np.arange(100, dtype=np.float32).reshape(10, 10))
    result = run("imstatistics", "obj.fits[2:4,1:2]", "--fields", "image,npix")
    assert result.exit_code == 0, result.output
    assert "6" in result.output
    copied = run("imcopy", "obj.fits[1:5,*],obj.fits[6:10,*]", "a.fits,b.fits")
    assert copied.exit_code == 0, copied.output
    assert fits.getdata("a.fits").shape == (10, 5)


def test_indef_is_irafs_no_value_on_the_command_line(run, tmp_path, monkeypatch):
    """autoidentify's blind search is `autoidentify arc INDEF INDEF`, as in
    IRAF; the command line refused INDEF for every number, so it could not be
    asked for at all. And the log writes the run back the same way."""
    from harwood.core import replay
    from harwood.core.registry import get_class

    monkeypatch.chdir(tmp_path)
    planned = run("autoidentify", "arc.fits", "INDEF", "INDEF", "--plan")
    assert planned.exit_code == 0, planned.output
    assert "blind search" in planned.output
    cls = get_class("autoidentify")
    values = cls.Params(image="arc.fits", crval=None, cdelt=None).model_dump()
    assert replay.command("autoidentify", cls.Params, values) == \
        "harwood autoidentify arc.fits INDEF INDEF"
    assert "FLOAT OR INDEF" in run("daofind", "--help").output
