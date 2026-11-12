"""harwood wcslab as a task: the SVG it writes alone and added to the last graph
(contour --no-perimeter, then --append, as the course's exercise 2.2 does), the
grid's values in the result, the refusal with no graph to add to, the display's
overlay (paths and texts, in the image's pixels) and its fallback to a graph
where there is no window."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from harwood.cli.main import app
from harwood.core import display

sys.path.insert(0, str(Path(__file__).parent / "data" / "iraf_wcslab"))
from wcl_inputs import write  # noqa: E402


@pytest.fixture
def run(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HARWOOD_DISPLAY", "headless")
    display.reset()
    runner = CliRunner()
    yield lambda *args: runner.invoke(app, list(args))
    display.reset()


def test_a_grid_alone_is_an_svg_with_irafs_labels(run, tmp_path):
    result = run("wcslab", "wpix.fits", "--device", "grid.svg", "--vl", "0.15",
                 "--vr", "0.85", "--vb", "0.15", "--vt", "0.85")
    assert result.exit_code == 0, result.output
    svg = (tmp_path / "grid.svg").read_text(encoding="utf-8")
    assert "13 28 00 " in svg and "+47 24&#x27;" in svg and "wpix.fits: m51" in svg
    assert "<polyline" in svg
    assert "0:00:05.00" in result.output and "0:01:00.00" in result.output


def test_the_grid_is_added_to_the_contour_map(run, tmp_path):
    assert run("contour", "wpix.fits", "--no-perimeter", "--device", "c.svg").exit_code == 0
    contour = (tmp_path / "c.svg").read_text(encoding="utf-8").count("<polyline")
    result = run("wcslab", "wpix.fits", "--device", "both.svg", "--append")
    assert result.exit_code == 0, result.output
    svg = (tmp_path / "both.svg").read_text(encoding="utf-8")
    assert svg.count("<polyline") > contour and "13 28 00 " in svg


def test_append_with_no_graph_says_what_to_draw_first(run):
    result = run("wcslab", "wpix.fits", "--device", "x.svg", "--append")
    assert result.exit_code != 0
    assert "no graph to draw over" in result.output


def test_without_a_window_imd_is_a_graph(run, tmp_path):
    result = run("wcslab", "polar.fits")
    assert result.exit_code == 0, result.output
    assert "no display window" in result.output
    assert "polar" in result.output
    assert list((tmp_path / "output").rglob("polar.wcslab.svg"))


def test_wcspars_alone_label_a_linear_box(run, tmp_path):
    result = run("wcslab", "--device", "lin.svg", "--log-x1", "0", "--log-x2", "100",
                 "--log-y1", "-5", "--log-y2", "5", "--title", "linear")
    assert result.exit_code == 0, result.output
    svg = (tmp_path / "lin.svg").read_text(encoding="utf-8")
    assert ">20.<" in svg and ">linear<" in svg


class _Screen:
    def __init__(self):
        self.shown, self.marks, self.window = [], [], None

    def show(self, data, header=None, frame=1, **kwargs):
        self.shown.append(frame)

    def mark(self, markers, frame=1):
        self.marks.extend(markers)


def test_on_the_display_the_grid_is_drawn_in_the_images_pixels(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    screen = _Screen()
    monkeypatch.setattr(display, "get", lambda backend=None: screen)
    monkeypatch.setattr(display, "_autodetect", lambda: "native")
    from harwood.core.registry import get_task

    result = get_task("wcslab").run(image="wpix.fits", frame=2)
    assert screen.shown == [2]
    paths = [m for m in screen.marks if m.shape == "path"]
    texts = [m.label for m in screen.marks if m.shape == "text"]
    assert "13 28 00 " in texts and "  h  m  s" in texts
    # the box is the plot's window, the first and last pixels: the lines are where
    # the image's WCS puts them on its own pixels
    box = paths[0].points
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx((1, 512, 1, 512), abs=1e-3)
    assert result.data["graph_type"] == "normal"
