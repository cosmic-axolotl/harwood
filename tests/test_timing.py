"""How long a task will take: the record, the estimate, and the clock.

The estimate is held to two promises: it comes from this computer's own runs
and from nothing else -- a task that has never run here gets no number --
and it scales with the size of the work. The window is held to running a
task beside itself, so that the clock can move at all, while the panels a
task opens still open on the window's thread and hand their answer back.
"""

import threading
import time

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import timing


def _runs(*entries):
    for pixels, seconds in entries:
        timing.record("ccdproc", timing.Size(1, pixels), seconds)


def test_a_task_that_never_ran_here_gets_no_estimate():
    assert timing.estimate("ccdproc", timing.Size(3, 10**6)) is None


def test_the_estimate_scales_with_the_pixels():
    _runs((10**6, 2.0))
    guess = timing.estimate("ccdproc", timing.Size(4, 4 * 10**6))
    assert guess.seconds == pytest.approx(8.0)
    assert guess.basis() == "from 1 earlier run on this computer"


def test_runs_of_several_sizes_give_a_fixed_cost_and_a_cost_per_pixel():
    _runs((10**6, 1.5), (2 * 10**6, 2.5), (4 * 10**6, 4.5))      # 0.5 s + 1 s per Mpx
    guess = timing.estimate("ccdproc", timing.Size(1, 8 * 10**6))
    assert guess.seconds == pytest.approx(8.5)
    assert guess.runs == 3


def test_a_task_without_images_is_estimated_by_its_usual_time():
    for seconds in (1.0, 3.0, 2.0):
        timing.record("standards", timing.Size(), seconds)
    assert timing.estimate("standards", timing.Size()).seconds == 2.0


def test_only_recent_runs_count(monkeypatch):
    monkeypatch.setattr(timing, "RECENT", 2)
    _runs((10**6, 100.0), (10**6, 1.0), (10**6, 1.0))       # the slow one was long ago
    assert timing.estimate("ccdproc", timing.Size(1, 10**6)).seconds == 1.0


def test_a_record_that_cannot_be_written_is_not_an_error(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocker))
    timing.record("ccdproc", timing.Size(1, 10), 1.0)
    assert timing.estimate("ccdproc", timing.Size(1, 10)) is None


def test_a_line_cut_short_by_a_crash_is_skipped():
    _runs((10**6, 2.0))
    with timing.path().open("a") as handle:
        handle.write('{"task": "ccdproc", "pix')
    assert timing.estimate("ccdproc", timing.Size(1, 10**6)).seconds == 2.0


def test_the_size_of_a_run_is_read_from_the_headers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("a.fits", "b.fits"):
        fits.writeto(name, np.zeros((20, 30), np.float32))
    (tmp_path / "all.lis").write_text("a.fits\nb.fits\n", encoding="utf-8")
    assert timing.size_of({"input": ["@all.lis"]}) == timing.Size(2, 1200)
    assert timing.size_of({"image": "a.fits[1:10,*]", "output": "c.fits"}).images == 1
    assert timing.size_of({"input": ["missing*.fits"]}) == timing.Size()


@pytest.mark.parametrize("seconds, clock, about", [
    (0.4, "0:00", "under a second"), (7, "0:07", "about 7 s"),
    (161, "2:41", "about 3 min"), (3723, "1:02:03", "about 62 min"),
    (7200, "2:00:00", "about 2.0 h")])
def test_times_are_written_as_a_person_reads_them(seconds, clock, about):
    assert timing.clock(seconds) == clock
    assert timing.about(seconds) == about


# --------------------------------------------------------- from the command line


def test_a_run_is_recorded_and_the_plan_says_how_long_the_next_will_take(
        tmp_path, monkeypatch):
    from click.testing import CliRunner

    from harwood.cli.main import app

    monkeypatch.chdir(tmp_path)
    fits.writeto("a.fits", np.zeros((40, 40), np.float32))
    runner = CliRunner()
    plan = runner.invoke(app, ["imstatistics", "a.fits", "--plan"])
    assert "should take" not in plan.output, "no runs yet, so no number"
    assert runner.invoke(app, ["imstatistics", "a.fits"]).exit_code == 0
    [entry] = timing.load("imstatistics")
    assert (entry["images"], entry["pixels"]) == (1, 1600)
    plan = runner.invoke(app, ["imstatistics", "a.fits", "--plan"])
    assert "It should take" in plan.output and "1 earlier run" in plan.output
    assert timing.current is None, "nothing is left marked as running"


# ----------------------------------------------------------------- the window


@pytest.fixture
def window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from harwood.viewer import _qt, theme
    from harwood.viewer.window import HarwoodWindow

    monkeypatch.chdir(tmp_path)
    fits.writeto(tmp_path / "a.fits", np.arange(16, dtype=np.float32).reshape(4, 4))
    _qt.application()
    w = HarwoodWindow()
    yield w
    _qt.set_window(None)
    w.close()
    theme.forget(w._restyle)


@pytest.fixture
def slow(monkeypatch):
    """imstatistics, slowed down and talking while it works."""
    from harwood.core.registry import get_task

    cls = type(get_task("imstatistics"))
    original = cls.execute
    threads = []

    def execute(self, params):
        threads.append(threading.current_thread())
        print("measuring", flush=True)
        time.sleep(1.2)
        return original(self, params)

    monkeypatch.setattr(cls, "execute", execute)
    return threads


def _watch(window, every=150):
    """What the window showed while a command ran."""
    from PySide6 import QtCore

    seen = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: seen.append(
        (window.clock.isVisible(), window.clock_label.text(),
         "measuring" in window.session.toPlainText())))
    timer.start(every)
    return timer, seen


def test_a_task_runs_beside_the_window_with_a_clock(window, slow):
    window.show()
    timer, seen = _watch(window)
    ok, printed = window.run_line("imstatistics a.fits --fields mean")
    timer.stop()
    assert ok and "7.5" in printed
    assert slow[0] is not threading.main_thread(), "the window's thread stayed free"
    assert any(shown and "⏳ imstatistics" in text for shown, text, _ in seen)
    assert any("no estimate yet" in text for _, text, _ in seen)
    assert any(output for shown, _, output in seen if shown), \
        "what the task printed reached the session while it ran"
    assert not window.clock.isVisible(), "and the clock goes when it is done"


def test_the_second_run_has_an_estimate(window, slow):
    window.show()
    window.run_line("imstatistics a.fits --fields mean")
    timer, seen = _watch(window)
    window.run_line("imstatistics a.fits --fields mean")
    timer.stop()
    assert any("left (from 1 earlier run on this computer)" in text
               for _, text, _ in seen)


def test_one_command_at_a_time(window, slow):
    from PySide6 import QtCore

    answers = []
    QtCore.QTimer.singleShot(400, lambda: answers.append(window.run_line("list")))
    window.prompt.setText("help phot")
    QtCore.QTimer.singleShot(500, window._run_command)
    window.run_line("imstatistics a.fits")
    assert answers == [(False, "still running imstatistics\n")]
    assert window.prompt.text() == "help phot", "the line waits at the prompt"


def test_a_panel_opened_from_the_tasks_thread_opens_on_the_windows(window, monkeypatch):
    from PySide6 import QtCore

    from harwood.core.registry import get_task
    from harwood.viewer import icfit_panel
    from harwood.viewer.icfit_panel import IcfitPanel

    cls = type(get_task("imstatistics"))
    original = cls.execute
    came_back = []

    def execute(self, params):
        x = np.arange(20.0)
        came_back.append(icfit_panel.run(x, 2 * x + 1, title="a fit"))
        return original(self, params)

    monkeypatch.setattr(cls, "execute", execute)
    opened = []

    def answer():
        panels = [w for w in window.findChildren(IcfitPanel)]
        if not panels:
            QtCore.QTimer.singleShot(100, answer)
            return
        opened.append((threading.current_thread(), window.clock_label.text()))
        panels[0].key("q")

    QtCore.QTimer.singleShot(100, answer)
    window.show()
    ok, _ = window.run_line("imstatistics a.fits")
    assert ok and came_back and came_back[0] is not None, "the fit reached the task"
    assert opened[0][0] is threading.main_thread()


def test_the_window_refuses_to_be_reached_from_another_thread():
    pytest.importorskip("PySide6")
    from harwood.viewer import _qt

    caught = []
    thread = threading.Thread(target=lambda: caught.append(_raises(_qt.application)))
    thread.start()
    thread.join()
    assert caught == [RuntimeError]


def _raises(function):
    try:
        function()
    except Exception as error:
        return type(error)
    return None


def test_the_tasks_that_work_in_the_window_stay_on_its_thread():
    from harwood.viewer.window import _on_window_thread

    assert _on_window_thread(["display", "a.fits"])
    assert _on_window_thread(["imexamine", "a.fits"])
    assert not _on_window_thread(["ccdproc", "@obj.lis"])
    assert not _on_window_thread(["list"])
