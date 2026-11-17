"""harwood's icon: the h of Planck in the colours of the theme in use, changed with it."""

from __future__ import annotations

import os
import re

import pytest

from harwood.viewer import icon, theme


def test_the_arch_is_the_sun_s_planck_curve():
    """Its highest point is Wien's peak, ~502 nm, at the x-height; it leaves the stem a
    quarter of the way up and ends on the baseline; and it falls slower than it rises."""
    points = icon.arch()
    x_top, y_top = max(points, key=lambda p: p[1])
    peak_x = (icon.PEAK - icon.START) / (icon.END - icon.START) * icon.WIDTH
    assert pytest.approx(501.6, abs=0.1) == icon.PEAK
    assert y_top == pytest.approx(icon.XHEIGHT, rel=1e-3)
    assert x_top == pytest.approx(peak_x, abs=icon.WIDTH / 180)
    assert points[-1][1] == pytest.approx(0.0, abs=1e-9)
    assert 0.2 < points[0][1] / icon.XHEIGHT < 0.3
    assert x_top < icon.WIDTH / 3                  # the peak close to the stem


@pytest.mark.parametrize("name", ["DARK", "LIGHT", "DARK_CB", "LIGHT_CB"])
def test_the_icon_is_drawn_in_the_palette_s_colours(name):
    palette = getattr(theme, name)
    text = icon.svg(palette)
    colours = set(re.findall(r'(?:fill|stroke)="(#[0-9a-fA-F]{6})"', text))
    assert colours == {palette.ground, palette.ink, palette.sky, palette.accent}


def test_the_window_s_icon_follows_the_theme(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from PySide6 import QtGui

    from harwood.viewer._qt import application

    app = application()

    def corner_and_centre():
        image = app.windowIcon().pixmap(64, 64).toImage()
        return (QtGui.QColor(image.pixel(32, 4)).name(),
                {QtGui.QColor(image.pixel(x, y)).name() for x in range(64) for y in range(64)})

    before = theme.current().preferences
    try:
        theme.update(save=False, theme="dark", colorblind=False)
        assert not app.windowIcon().isNull()
        assert app.desktopFileName() == "harwood"
        ground, seen = corner_and_centre()
        assert ground == theme.DARK.ground and theme.DARK.sky in seen
        theme.update(save=False, theme="light")
        ground, seen = corner_and_centre()
        assert ground == theme.LIGHT.ground and theme.LIGHT.sky in seen
        theme.update(save=False, colorblind=True)
        assert theme.LIGHT_CB.sky in corner_and_centre()[1]
    finally:
        theme.update(save=False, theme=before.theme, colorblind=before.colorblind)


# ------------------------------------------------------------ by the hour

def _at(hour, minute=0):
    import time

    return time.mktime((2026, 11, 17, hour, minute, 0, 0, 0, -1))


@pytest.mark.parametrize("night_from,day_from,hour,minute,dark", [
    ("18:00", "06:00", 17, 59, False), ("18:00", "06:00", 18, 0, True),
    ("18:00", "06:00", 23, 30, True), ("18:00", "06:00", 3, 0, True),
    ("18:00", "06:00", 6, 0, False), ("18:00", "06:00", 12, 0, False),
    ("01:00", "05:30", 0, 59, False), ("01:00", "05:30", 4, 0, True),
    ("01:00", "05:30", 5, 30, False)])
def test_night_runs_from_night_from_to_day_from(night_from, day_from, hour, minute, dark):
    prefs = theme.Preferences(theme="hours", night_from=night_from, day_from=day_from)
    assert theme.night(prefs, _at(hour, minute)) is dark


def test_the_night_hours_are_checked_when_read(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "harwood" / "interface.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"theme": "hours", "night_from": "25:00", "day_from": "7:15"}',
                    encoding="utf-8")
    prefs = theme.Preferences.load()
    assert (prefs.theme, prefs.night_from, prefs.day_from) == ("hours", "18:00", "7:15")


def test_by_the_hour_the_look_and_the_icon_turn_at_dusk(monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui

    from harwood.viewer._qt import application

    app = application()
    clock = {"now": _at(17, 59)}
    real = theme.time.localtime
    monkeypatch.setattr(theme.time, "localtime",
                        lambda when=None: real(clock["now"] if when is None else when))
    before = theme.current().preferences
    try:
        theme.update(save=False, theme="hours", night_from="18:00", day_from="06:00",
                     colorblind=False)
        assert not theme.current().dark
        assert not theme.tick() or not theme.current().dark
        clock["now"] = _at(18, 1)
        assert theme.tick() and theme.current().dark
        assert not theme.tick()                       # nothing new until dawn
        corner = QtGui.QColor(app.windowIcon().pixmap(64, 64).toImage().pixel(32, 4)).name()
        assert corner == theme.DARK.ground
        clock["now"] = _at(6, 0)
        assert theme.tick() and not theme.current().dark
    finally:
        theme.update(save=False, theme=before.theme, night_from=before.night_from,
                     day_from=before.day_from, colorblind=before.colorblind)


# ------------------------------------------------------------- the launcher

def test_the_shortcut_installs_a_launcher_whose_icon_follows_the_theme(tmp_path,
                                                                        monkeypatch):
    import sys

    from click.testing import CliRunner

    from harwood.cli.main import app as cli

    if not sys.platform.startswith("linux"):
        pytest.skip("a freedesktop launcher")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    before = theme.current().preferences
    try:
        theme.update(save=False, theme="light", colorblind=False)
        result = CliRunner().invoke(cli, ["shortcut"])
        assert result.exit_code == 0, result.output
        desktop = tmp_path / "applications" / "harwood.desktop"
        folder = tmp_path / "icons" / "hicolor" / "scalable" / "apps"
        text = desktop.read_text(encoding="utf-8")
        for line in (" gui %f", "StartupWMClass=harwood"):
            assert line in text
        light = icon.launcher_icon()
        assert f"Icon={light}" in text and light.parent == folder
        assert theme.LIGHT.ground in light.read_text(encoding="utf-8")
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from harwood.viewer._qt import application

        application()
        theme.update(save=False, theme="dark")
        # a new file under a new name, which the .desktop names; the old one gone, so
        # that GNOME, which keeps an icon by its name, reads the new one
        dark = icon.launcher_icon()
        assert dark != light and not light.exists()
        assert f"Icon={dark}" in desktop.read_text(encoding="utf-8")
        assert theme.DARK.ground in dark.read_text(encoding="utf-8")
        assert [p.name for p in folder.iterdir()] == [dark.name]   # nothing half-written
        result = CliRunner().invoke(cli, ["shortcut", "--remove"])
        assert result.exit_code == 0 and not desktop.exists() and not dark.exists()
    finally:
        theme.update(save=False, theme=before.theme, colorblind=before.colorblind)
