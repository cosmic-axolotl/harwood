"""Where results go when nobody says, and what goes with them."""

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import output
from harwood.core.registry import get_task


@pytest.fixture
def frames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(2)
    for index in (1, 2):
        fits.PrimaryHDU((100.0 + rng.normal(0, 2, (12, 16))).astype(np.float32),
                        fits.Header({"IMAGETYP": "object", "EXPTIME": 30.0})
                        ).writeto(tmp_path / f"a{index}.fits")
    return tmp_path


def test_the_run_directory_is_named_for_the_task_and_the_day():
    from datetime import date

    when = date(2026, 9, 13)
    assert output.name_for("ccdproc", when) == "hrwd_ccdproc_2026_09_13"
    assert output.name_for("ccdproc", when, suffix=2) == "hrwd_ccdproc_2026_09_13_2"


def test_a_second_run_the_same_day_gets_its_own_directory(frames):
    first = output.directory("ccdproc", frames)
    token = output.begin("ccdproc", frames)
    second = output.directory()
    output.end(token)
    assert first != second
    assert second.name.endswith("_2")


def test_nothing_is_created_until_something_is_written(frames):
    token = output.begin("imheader", frames)
    assert output.current() is None
    output.end(token)
    assert not (frames / "output").exists()


def test_files_keep_the_names_of_the_frames_they_came_from(frames):
    """`hrwd_ccdproc_2026_09_13.fits` for seven inputs would name none of them."""
    from harwood.core import images

    token = output.begin("imcopy", frames)
    try:
        specs = images.expand("a*.fits")
        destinations = images.outputs_for(specs, "")
    finally:
        output.end(token)

    assert [p.name for p in destinations] == ["a1.fits", "a2.fits"]
    assert destinations[0].parent.name.startswith("hrwd_imcopy_")


def test_a_run_that_was_told_where_to_write_leaves_no_directory(frames):
    result = get_task("imcopy").run(input=["a1.fits"], output="mine.fits")
    assert result.outputs[0].name == "mine.fits"
    assert not (frames / "output").exists()


def test_the_record_says_what_made_the_files(frames):
    import yaml

    token = output.begin("imarith", frames)
    try:
        result = get_task("imarith").run(operand1="a1.fits", op="*",
                                         operand2="2", result="")
        written = output.record("imarith", {"operand1": "a1.fits", "op": "*"},
                                result, "harwood imarith a1.fits '*' 2",
                                {"operand1": "command line"})
    finally:
        output.end(token)

    assert written is not None
    document = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert document["task"] == "imarith"
    assert document["command"].startswith("harwood imarith")
    assert document["origins"]["operand1"] == "command line"
    assert document["outputs"] == ["a1.fits"]
    assert written.read_text(encoding="utf-8").startswith("#"), "it explains itself"


def test_the_record_carries_each_step_with_its_explanation(frames, tmp_path):
    """The directory is the deliverable: results, and how they were made."""
    import yaml

    from harwood.core.task import Step, TaskResult

    result = TaskResult(
        outputs=[tmp_path / "x.fits"],
        steps=[Step("overscan", "legendre order 1", why="because the amplifier",
                    measured={"level": 505.4},
                    before={"median": 548.0, "sigma": 93.0, "npix": 100},
                    after={"median": 42.5, "sigma": 93.0, "npix": 100}),
               Step("trim", skipped=True, reason="TRIM already in the header")],
    )
    token = output.begin("ccdproc", frames)
    try:
        output.directory()
        written = output.record("ccdproc", {}, result, "harwood ccdproc x")
    finally:
        output.end(token)

    steps = yaml.safe_load(written.read_text(encoding="utf-8"))["steps"]
    assert steps[0]["why"] == "because the amplifier"
    assert steps[0]["measured"]["level"] == "505.4"
    assert steps[0]["before"]["npix"] == 100, "a count stays a count"
    assert steps[1]["skipped"] == "TRIM already in the header"


def test_the_record_keeps_a_notice_and_what_to_do_about_it(frames, tmp_path):
    """A task's notices are Advice; the record keeps them as text. (lineid on
    a spectrum with unnamed features stopped at the end, after its work was
    done, because run.yaml could not hold an Advice.)"""
    import yaml

    from harwood.core import advice
    from harwood.core.task import TaskResult

    result = TaskResult(outputs=[tmp_path / "x.csv"],
                        warnings=[advice.note("4 of 4 features are not in the catalogue.",
                                              "nothing, usually.")])
    token = output.begin("lineid", frames)
    try:
        output.directory()
        written = output.record("lineid", {}, result, "harwood lineid x")
    finally:
        output.end(token)
    assert yaml.safe_load(written.read_text(encoding="utf-8"))["notes"] == [
        "note: 4 of 4 features are not in the catalogue. -- try: nothing, usually."]


def test_output_lives_with_the_reduction_not_with_the_installation(frames):
    """An installation is shared, replaced on upgrade, and often not writable;
    a global place for results is the same mistake as a global place for
    parameters, which section 2 exists to prevent."""
    import harwood

    directory = output.directory("ccdproc", frames)
    assert directory.is_relative_to(frames)
    package = __import__("pathlib").Path(harwood.__file__).parent
    assert not directory.is_relative_to(package)


def test_an_optional_positional_after_a_variadic_one_is_refused():
    """It could never be omitted: one argument goes to the optional slot and
    the list comes back empty. There is no way to write it that is not a trap."""
    from pydantic import Field

    from harwood.cli import build
    from harwood.core.task import TaskParams

    class Bad(TaskParams):
        input: list[str] = Field(..., description="inputs")
        output: str = Field("", description="output",
                            json_schema_extra={"positional": True})

    with pytest.raises(ValueError, match="optional positional"):
        build.parameters_for(Bad)


def test_daofind_and_phot_default_to_the_run_directory(frames, tmp_path):
    """The file keeps the frame's name with IRAF's extension: obj.fits gives
    obj.coo, which is the only name that says which frame it belongs to."""
    import numpy as np
    from astropy.io import fits

    rng = np.random.default_rng(4)
    ny, nx = 60, 60
    yy, xx = np.mgrid[0:ny, 0:nx]
    field = np.full((ny, nx), 100.0) + rng.normal(0, 2, (ny, nx))
    for x, y in ((15.0, 20.0), (40.0, 45.0), (30.0, 12.0)):
        field += 3000.0 * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.4 ** 2)))
    fits.PrimaryHDU(field.astype(np.float32),
                    fits.Header({"EXPTIME": 10.0})).writeto(frames / "obj.fits")

    token = output.begin("daofind", frames)
    try:
        found = get_task("daofind").run(image="obj.fits", fwhmpsf=3.0, threshold=4.0)
    finally:
        output.end(token)

    assert found.outputs[0].name == "obj.coo"
    assert found.outputs[0].parent.name.startswith("hrwd_daofind_")

    token = output.begin("phot", frames)
    try:
        measured = get_task("phot").run(image="obj.fits", coords=found.outputs[0],
                                        apertures=[4.0], annulus=8.0, dannulus=4.0)
    finally:
        output.end(token)
    assert measured.outputs[0].name == "obj.mag"


def test_a_named_output_still_wins(frames, tmp_path):
    import numpy as np
    from astropy.io import fits

    fits.PrimaryHDU(np.full((20, 20), 5.0, dtype=np.float32)).writeto(
        frames / "flat.fits")
    result = get_task("imarith").run(operand1="flat.fits", op="+", operand2="1",
                                     result="chosen.fits")
    assert result.outputs[0].name == "chosen.fits"
    assert not (frames / "output").exists()


def test_a_section_does_not_leak_into_the_derived_name(frames):
    """`obj.fits[SCI,1]` must not produce a file called `obj[SCI,1].coo`."""
    token = output.begin("daofind", frames)
    try:
        name = output.named_for("obj.fits[SCI,1][10:20,*]", ".coo")
    finally:
        output.end(token)
    assert name.name == "obj.coo"


# ------------------------------------------------------- the one window


def test_a_panel_docks_into_the_window_when_there_is_one():
    """Section 8.2: the whole environment in one window. A radial profile in a
    different window from the star it came from is most of what is wrong with
    working across three windows."""
    import numpy as np

    from harwood.viewer import _qt
    from harwood.viewer.implot_panel import ImplotPanel
    from harwood.viewer.window import HarwoodWindow

    _qt.application()
    window = HarwoodWindow()
    try:
        assert _qt.window() is window, "the window announces itself"
        panel = ImplotPanel(np.zeros((10, 10)), "test")
        window.open_panel(panel, "implot")
        assert window.views.count() == 2 and window.views.currentWidget() is panel

        window.close_panel(panel)
        assert window.views.count() == 1
    finally:
        _qt.set_window(None)
        window.close()


def test_a_panel_opens_on_its_own_when_there_is_no_window():
    """Run from a shell, `harwood implot` is a window of its own -- the panels
    do not know which is happening."""
    from harwood.viewer import _qt

    _qt.set_window(None)
    assert _qt.window() is None


def test_the_prompt_runs_the_same_commands_as_the_shell(tmp_path, monkeypatch):
    import numpy as np
    from astropy.io import fits

    from harwood.viewer import _qt
    from harwood.viewer.window import HarwoodWindow

    monkeypatch.chdir(tmp_path)
    fits.PrimaryHDU(np.full((12, 12), 7.0, dtype=np.float32)).writeto(
        tmp_path / "a.fits")

    _qt.application()
    window = HarwoodWindow()
    try:
        window.prompt.setText("imstatistics a.fits --fields image,mean")
        window._run_command()
        text = window.session.toPlainText()
        assert "hw> imstatistics" in text
        assert "7" in text, "the task's own output lands in the log"
    finally:
        _qt.set_window(None)
        window.close()


def test_a_failing_command_does_not_end_the_session(tmp_path, monkeypatch):
    from harwood.viewer import _qt
    from harwood.viewer.window import HarwoodWindow

    monkeypatch.chdir(tmp_path)
    _qt.application()
    window = HarwoodWindow()
    try:
        window.prompt.setText("imstatistics nothing-here.fits")
        window._run_command()
        window.prompt.setText("help phot")
        window._run_command()
        assert "aperture" in window.session.toPlainText().lower()
    finally:
        _qt.set_window(None)
        window.close()


def test_display_from_the_prompt_loads_into_this_window(tmp_path, monkeypatch):
    """Opening a second window would be two of everything, and a second event
    loop that never returns."""
    import numpy as np
    from astropy.io import fits

    from harwood.viewer import _qt
    from harwood.viewer.window import HarwoodWindow

    monkeypatch.chdir(tmp_path)
    fits.PrimaryHDU(np.arange(144, dtype=np.float32).reshape(12, 12)).writeto(
        tmp_path / "b.fits")

    _qt.application()
    window = HarwoodWindow()
    try:
        window.prompt.setText("display b.fits --no-wait")
        window._run_command()
        assert 1 in window.frames, "loaded into frame 1 of this window"
        assert window.frames[1].data.shape == (12, 12)
    finally:
        _qt.set_window(None)
        window.close()


def test_a_file_called_output_is_left_alone(frames):
    """The course's lists are called input and output; a file of that name stays,
    and the runs go to hrwd_output beside it."""
    (frames / "output").write_text("a1_bt.fits\n", encoding="utf-8")
    folder = output.directory("fixpix", frames)
    assert folder.parent.name == "hrwd_output"
    assert (frames / "output").read_text(encoding="utf-8") == "a1_bt.fits\n"
