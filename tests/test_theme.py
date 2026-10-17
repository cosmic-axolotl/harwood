"""The look of the window: themes, accessibility modes, text size, export.

Accessibility is held to numbers rather than taste: every text colour on its
ground meets WCAG's 4.5:1, every data colour 3:1, in every theme. And the
picture the window exports is the picture `harwood impicture` writes.
"""

from dataclasses import replace

import numpy as np
import pytest

from harwood.viewer import theme


def _palettes():
    yield "dark", theme.DARK
    yield "light", theme.LIGHT
    yield "dark colour-blind", theme.DARK_CB
    yield "light colour-blind", theme.LIGHT_CB
    yield "cream", replace(theme.LIGHT, ground=theme.CREAM)
    yield "cream colour-blind", replace(theme.LIGHT_CB, ground=theme.CREAM)


@pytest.mark.parametrize("name, palette", list(_palettes()))
def test_every_colour_is_readable_on_its_ground(name, palette):
    for token in ("ink", "muted", "comment"):
        for ground in (palette.ground, palette.panel):
            assert theme.contrast(getattr(palette, token), ground) >= 4.5, (name, token)
    for token in ("accent", "sky", "band", "drop", "note", "warning", "problem",
                  "interstellar", "telluric"):
        assert theme.contrast(getattr(palette, token), palette.ground) >= 3.0, (name, token)
    for colour in palette.series:
        assert theme.contrast(colour, palette.ground) >= 3.0, (name, colour)


def test_the_colour_blind_palettes_are_okabe_ito():
    assert theme.DARK_CB.sky == "#56b4e9" and theme.DARK_CB.accent == "#e69f00"
    assert theme.DARK_CB.drop == "#d55e00"
    # The light variants keep the hue and darken for contrast.
    assert len(set(theme.LIGHT_CB.series)) == len(theme.LIGHT_CB.series)


def test_preferences_are_kept_where_a_person_can_read_them(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    prefs = theme.Preferences(theme="light", dyslexia=True, scale=1.4)
    path = prefs.save()
    assert path == tmp_path / "harwood" / "interface.json"
    assert '"theme": "light"' in path.read_text()
    assert theme.Preferences.load() == prefs


def test_a_broken_preferences_file_falls_back_to_the_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "harwood" / "interface.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert theme.Preferences.load() == theme.Preferences()
    path.write_text('{"theme": "neon", "scale": 99, "unknown": 1}')
    loaded = theme.Preferences.load()
    assert loaded.theme == "dark" and loaded.scale == theme.SCALES[-1]


def test_the_modes_change_what_is_drawn():
    plain = theme.Theme(theme.Preferences(theme="light"))
    assert plain.palette is theme.LIGHT and plain.family[0] == "IBM Plex Sans"
    dyslexic = theme.Theme(theme.Preferences(theme="light", dyslexia=True))
    assert dyslexic.palette.ground == theme.CREAM
    assert dyslexic.family[0] == "Atkinson Hyperlegible"
    assert dyslexic.letter_spacing > 100 and dyslexic.line_height > plain.line_height
    assert "Atkinson Hyperlegible" in dyslexic.stylesheet()
    other = theme.Theme(theme.Preferences(dyslexia=True, dyslexia_font="opendyslexic"))
    assert other.family[0] == "OpenDyslexic"
    system = theme.Theme(theme.Preferences(theme="system"), system_dark=False)
    assert not system.dark


def test_text_follows_the_window_only_when_asked():
    on = theme.Theme(theme.Preferences(scale=1.2, autoscale=True), window_factor=1.5)
    off = theme.Theme(theme.Preferences(scale=1.2, autoscale=False), window_factor=1.5)
    assert on.scale == pytest.approx(1.8) and off.scale == pytest.approx(1.2)


def test_the_fonts_are_bundled_with_their_licences():
    names = {p.name for p in theme.FONTS.iterdir()}
    assert {"AtkinsonHyperlegible-Regular.ttf", "OpenDyslexic-Regular.otf",
            "OFL-AtkinsonHyperlegible.txt", "OFL-OpenDyslexic.txt"} <= names


# --------------------------------------------------------------- the window


@pytest.fixture
def window():
    pytest.importorskip("PySide6")
    from iraf_field import field

    from harwood.viewer._qt import application
    from harwood.viewer.window import HarwoodWindow

    application()
    w = HarwoodWindow()
    w.load(field(), name="field.fits")
    yield w
    w.close()
    theme.forget(w._restyle)


def test_changing_the_theme_restyles_the_window_and_is_saved(window):
    window.set_look(theme="light", colorblind=True)
    assert theme.current().palette is theme.LIGHT_CB
    assert window.graphics.backgroundBrush().color().name() == theme.LIGHT_CB.ground
    assert window._theme_actions["light"].isChecked()
    assert window._look_actions["colorblind"].isChecked()
    assert '"colorblind": true' in theme.Preferences.path().read_text()


def test_examine_marks_follow_the_theme(window):
    window.examine_at("r", 110.0, 53.0)
    marks = window.frames[1].markers
    assert {m.colour for m in marks} <= {"accent", "sky"}, "tokens, resolved when drawn"


def test_the_text_grows_with_the_window(window):
    window.resize(1500, 900)
    window._follow_window_size()
    assert theme.current().window_factor == pytest.approx(1.0)
    window.resize(2400, 1500)
    window._follow_window_size()
    assert theme.current().window_factor > 1.4
    window.set_look(autoscale=False)
    assert theme.current().scale == pytest.approx(theme.current().preferences.scale)


def test_the_window_exports_what_impicture_writes(window, tmp_path):
    from astropy.io import fits
    from iraf_field import field

    from harwood.core.registry import get_task

    window.stretch_box.setCurrentText("asinh")
    window.cmap_box.setCurrentText("cividis")
    saved = window.export_image(tmp_path / "window.png")
    fits.writeto(tmp_path / "field.fits", field())
    get_task("impicture").run(input=[str(tmp_path / "field.fits")],
                           output=str(tmp_path / "task.png"), stretch="asinh",
                           colormap="cividis")
    assert saved.read_bytes() == (tmp_path / "task.png").read_bytes()


@pytest.mark.parametrize("suffix", [".png", ".svg", ".pdf", ".csv"])
def test_a_plot_exports_as_picture_or_numbers(window, tmp_path, suffix):
    window.examine_at("r", 110.0, 53.0)
    path = window.export_plot(tmp_path / f"profile{suffix}")
    assert path.exists() and path.stat().st_size > 100


@pytest.mark.parametrize("suffix", [".jpg", ".tif", ".webp"])
def test_the_image_exports_in_the_other_formats(window, tmp_path, suffix):
    from PySide6 import QtGui

    path = window.export_image(tmp_path / f"field{suffix}")
    image = QtGui.QImage(str(path))
    assert (image.width(), image.height()) == (200, 200)


def test_the_view_exports_with_its_marks(window, tmp_path):
    window.examine_at("r", 110.0, 53.0)
    path = window.export_view(tmp_path / "view.png")
    assert path.exists() and path.stat().st_size > 1000


# ------------------------------------------------------------ the rendering


def test_render_is_the_same_whichever_way_it_is_called():
    from harwood.core import render

    data = np.arange(12.0).reshape(3, 4)
    pixels = render.rgb(data, "minmax", "linear", "grey")
    assert pixels.shape == (3, 4, 3) and pixels.dtype == np.uint8
    assert tuple(pixels[-1, 0]) == (0, 0, 0), "row 1 of the FITS file at the bottom"
    assert tuple(pixels[0, -1]) == (255, 255, 255)
    blank = render.rgb(np.full((2, 2), np.nan), "minmax", "linear", "viridis")
    assert tuple(blank[0, 0]) == tuple(render.lut("viridis")[0])


def test_every_colour_map_runs_from_dark_to_light_except_none():
    from harwood.core import render

    for name in render.COLORBLIND_SAFE:
        table = render.lut(name).astype(float)
        lightness = table @ [0.2126, 0.7152, 0.0722]
        assert np.all(np.diff(lightness) >= -1.0), name


def test_impicture_refuses_what_it_cannot_write(tmp_path):
    from astropy.io import fits

    from harwood.core.registry import get_task

    fits.writeto(tmp_path / "a.fits", np.ones((4, 4), np.float32) * np.arange(4))
    with pytest.raises(ValueError, match="cannot write"):
        get_task("impicture").run(input=[str(tmp_path / "a.fits")],
                               output=str(tmp_path / "a.gif"))
