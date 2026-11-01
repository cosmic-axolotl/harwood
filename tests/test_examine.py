"""imexamine: the same numbers from a key, a cursor file, and the window.

Section 8.2 keeps the measurement out of the viewer so that these three agree
by construction. This file checks that they do -- and that each key answers
the question IRAF taught people to ask with it.
"""

import os

import numpy as np
import pytest

from harwood.core import display, examine
from harwood.core.registry import get_task

FWHM = 3.40
TRUE_X, TRUE_Y = 120.30, 90.70


@pytest.fixture
def star_image(tmp_path):
    """One gaussian star at a known 1-based position, written to disk."""
    from astropy.io import fits

    ny, nx = 200, 240
    sigma = FWHM / 2.3548
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = 4530.0 * np.exp(-(((xx - (TRUE_X - 1))**2 + (yy - (TRUE_Y - 1))**2)
                             / (2 * sigma**2)))
    data += 1030.5 + np.random.default_rng(1).normal(0, 3.0, (ny, nx))
    path = tmp_path / "star.fits"
    fits.writeto(path, data.astype("f4"))
    return path, data


# ---------------------------------------------------------------- the keys


@pytest.mark.parametrize("key", ["r", "a"])
def test_r_and_a_recentre_on_the_star_and_measure_its_width(star_image, key):
    _, data = star_image
    result = examine.examine(data, key, 121.0, 91.0)
    assert result.ok
    assert result.x == pytest.approx(TRUE_X, abs=0.05)
    assert result.y == pytest.approx(TRUE_Y, abs=0.05)
    assert result.values["fwhm"] == pytest.approx(FWHM, abs=0.1)


def test_j_and_k_find_the_centre_along_their_own_axis(star_image):
    """j fits along the line, so it measures x; k along the column, so y."""
    _, data = star_image
    along_line = examine.examine(data, "j", 121.0, 91.0)
    along_column = examine.examine(data, "k", 121.0, 91.0)
    assert along_line.values["centre"] == pytest.approx(TRUE_X, abs=0.05)
    assert along_column.values["centre"] == pytest.approx(TRUE_Y, abs=0.05)
    for result in (along_line, along_column):
        assert result.values["fwhm"] == pytest.approx(FWHM, abs=0.1)
        assert result.values["background"] == pytest.approx(1030.5, abs=3)


def test_x_and_z_read_the_pixels_the_status_bar_would(star_image):
    _, data = star_image
    x = examine.examine(data, "x", 121.0, 91.0)
    assert x.values["value"] == pytest.approx(data[90, 120])

    z = examine.examine(data, "z", 121.0, 91.0)
    assert z.values["columns"] == list(range(116, 127)), "IRAF's x - 5 .. x + 5"
    assert z.values["rows"] == list(range(96, 85, -1)), "top row first, as on screen"
    centre_row = z.line.splitlines()[2 + z.values["rows"].index(91)]
    assert centre_row.split()[0] == "91"
    at = 1 + z.values["columns"].index(121)
    assert float(centre_row.split()[at]) == pytest.approx(data[90, 120], abs=0.05)


def test_m_and_h_describe_the_same_box(star_image):
    _, data = star_image
    box = examine.Settings(ncstat=9, nlstat=9, ncolumns=9, nlines=9, nbins=40)
    stats = examine.examine(data, "m", 50.0, 50.0, box)
    hist = examine.examine(data, "h", 50.0, 50.0, box)
    assert stats.values["npix"] == hist.values["npix"] == 81
    assert hist.values["mode"] == pytest.approx(stats.values["median"], abs=10)


def test_the_defaults_are_iraf_s():
    """rimexam.par, imexamine.par, himexam.par and jimexam.par from IRAF
    2.18, value for value. A FWHM read off harwood should mean what one read
    off IRAF meant, and that starts with the same aperture."""
    s = examine.Settings()
    assert (s.radius, s.buffer, s.width, s.rplot) == (5.0, 5.0, 5.0, 8.0)
    assert (s.fittype, s.beta, s.iterations, s.magzero) == ("moffat", None, 3, 25.0)
    assert (s.ncstat, s.nlstat) == (5, 5)
    assert (s.ncolumns, s.nlines, s.nbins) == (21, 21, 512)
    assert (s.naverage, s.sigma) == (5, 1.0)


def test_radius_is_the_aperture_not_the_plot(star_image):
    """In IRAF `radius` is the aperture the flux is summed in; `rplot` only
    says how far the profile is drawn. The flux of a gaussian inside r is
    the total times 1 - exp(-r^2 / 2 sigma^2)."""
    _, data = star_image
    sigma = FWHM / 2.3548
    total = 4530.0 * 2 * np.pi * sigma ** 2
    fixed = examine.Settings(iterations=1)
    for radius in (3.0, 5.0):
        fixed.radius = radius
        result = examine.examine(data, "a", 121.0, 91.0, fixed)
        expected = total * (1 - np.exp(-radius ** 2 / (2 * sigma ** 2)))
        assert result.values["flux"] == pytest.approx(expected, rel=0.03)

    drawn = examine.examine(data, "r", 121.0, 91.0,
                            examine.Settings(rplot=12, iterations=1))
    assert drawn.plot["x"].max() > 11, "the profile is drawn to rplot"
    assert drawn.values["radius"] == 5.0, "and the aperture did not move"


def test_the_aperture_becomes_three_times_the_direct_fwhm(star_image):
    """IRAF's manual says only a iterates; its code answers r and a with one
    routine, so both do. The code is what people's numbers came from."""
    _, data = star_image
    a = examine.examine(data, "a", 121.0, 91.0)
    r = examine.examine(data, "r", 121.0, 91.0)
    assert a.values["radius"] == pytest.approx(3 * a.values["direct"], rel=0.03)
    assert r.values == pytest.approx(a.values, nan_ok=True, rel=0, abs=0)
    # IRAF's direct FWHM runs a few percent narrow -- 3.28 here, and 3.33 on
    # a star integrated over the pixel -- because its correction for the
    # pixel assumes the light was integrated. harwood reproduces that (the
    # test against IRAF below), so it is checked loosely here.
    assert a.values["direct"] == pytest.approx(FWHM, abs=0.15)
    once = examine.examine(data, "a", 121.0, 91.0, examine.Settings(iterations=1))
    assert once.values["radius"] == 5.0


def test_an_elongated_star_has_an_ellipticity_and_an_angle():
    """e and pa from the second moments; pa in degrees from the x axis."""
    yy, xx = np.mgrid[0:80, 0:80]
    u, v = (xx - 40) * np.cos(np.pi / 4) + (yy - 40) * np.sin(np.pi / 4), \
        -(xx - 40) * np.sin(np.pi / 4) + (yy - 40) * np.cos(np.pi / 4)
    data = 3000 * np.exp(-(u ** 2 / (2 * 3.0 ** 2) + v ** 2 / (2 * 1.5 ** 2))) + 100
    data += np.random.default_rng(2).normal(0, 2, data.shape)
    result = examine.examine(data, "r", 41.0, 41.0, examine.Settings(radius=8))
    assert result.values["e"] > 0.4
    assert result.values["pa"] == pytest.approx(45, abs=3)

    round_star = examine.examine(data * 0 + _round(), "r", 41.0, 41.0)
    assert round_star.values["e"] < 0.05


def _round():
    yy, xx = np.mgrid[0:80, 0:80]
    return 3000 * np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 1.5 ** 2)) + 100


def test_a_fixed_beta_is_kept_and_a_free_one_is_fitted(star_image):
    _, data = star_image
    fixed = examine.examine(data, "r", 121.0, 91.0, examine.Settings(beta=2.5))
    free = examine.examine(data, "r", 121.0, 91.0)
    assert fixed.values["beta"] == 2.5
    assert free.values["beta"] > 2.5, "a gaussian star is a Moffat of large beta"
    assert "MFWHM" in free.line


def test_integer_data_get_no_more_bins_than_values():
    """himexam's autoscale: 512 bins over values spanning 30 is a comb. As in
    IRAF it goes by the pixel type, not by the values."""
    data = np.random.default_rng(3).integers(100, 130, (50, 50)).astype(np.int16)
    result = examine.examine(data, "h", 25.0, 25.0)
    assert result.values["nbins"] <= 30
    assert result.values["npix"] == 21 * 21
    real = examine.examine(data.astype(np.float32), "h", 25.0, 25.0)
    assert real.values["nbins"] == 512


def test_empty_sky_is_not_given_a_fwhm(star_image):
    """A gaussian fits the brightest noise pixel happily. Reporting that as a
    FWHM would put a number on nothing."""
    _, data = star_image
    result = examine.examine(data, "r", 20.0, 20.0)
    assert not result.ok
    assert "noise" in result.line


def test_a_position_off_the_image_says_so(star_image):
    _, data = star_image
    result = examine.examine(data, "r", 500.0, 20.0)
    assert not result.ok and "outside" in result.line


def test_an_unknown_key_is_named_and_not_guessed(star_image):
    _, data = star_image
    result = examine.examine(data, "W", 50.0, 50.0)
    assert not result.ok and "?" in result.line


# --------------------------------------------------------- the cursor file


def test_a_cursor_file_reads_like_iraf_s(tmp_path):
    path = tmp_path / "cur"
    path.write_text("# a comment\n\n10 20 r\n11.5 21.5 2 a   # trailing\n"
                    "30 40\n50 60 2\n")
    assert examine.read_cursor(path, defkey="r") == [
        (10.0, 20.0, 1, "r"), (11.5, 21.5, 2, "a"),
        (30.0, 40.0, 1, "r"), (50.0, 60.0, 2, "r"),       # defkey fills the gap
    ]


def test_a_colon_command_in_a_cursor_file_is_kept_in_order(tmp_path):
    path = tmp_path / "cur"
    path.write_text("10 20 1 :radius 7\n30 40 :defkey m\n50 60\n")
    assert examine.read_cursor(path, defkey="r") == [
        (10.0, 20.0, 1, ":radius 7"), (30.0, 40.0, 1, ":defkey m"),
        (50.0, 60.0, 1, "m"),                         # :defkey moved the default
    ]


def test_imexamine_from_a_cursor_file_gives_the_numbers_and_a_log(star_image, tmp_path):
    path, _ = star_image
    cursor = tmp_path / "cur"
    cursor.write_text("121 91 r\n121 91 j\n")
    log = tmp_path / "imexam.log"

    result = get_task("imexamine").run(input=[str(path)], imagecur=str(cursor),
                                       logfile=str(log))
    r, j = result.data["measurements"]
    assert r["fwhm"] == pytest.approx(FWHM, abs=0.1)
    assert j["centre"] == pytest.approx(TRUE_X, abs=0.05)
    assert [row["key"] for row in result.data["rows"]] == ["r", "j"]

    text = log.read_text()
    assert "[r]" in text and "[j]" in text and "fwhm=  3.398" in text


def test_a_frame_that_was_not_loaded_is_named(star_image, tmp_path):
    path, _ = star_image
    cursor = tmp_path / "cur"
    cursor.write_text("121 91 3 r\n")
    with pytest.raises(ValueError) as raised:
        get_task("imexamine").run(input=[str(path)], imagecur=str(cursor))
    assert "frame 3" in str(raised.value)
    assert raised.value.fix


def test_without_a_screen_imexamine_points_to_the_cursor_file(star_image, monkeypatch):
    path, _ = star_image
    display.reset()
    monkeypatch.setenv("HARWOOD_DISPLAY", "headless")
    monkeypatch.chdir(path.parent)
    try:
        with pytest.raises(ValueError) as raised:
            get_task("imexamine").run(input=[str(path)])
        assert "--imagecur" in raised.value.fix
    finally:
        display.reset()


# --------------------------------------------------------------- the display


def test_display_headless_writes_a_png_and_marks_the_list(star_image, tmp_path,
                                                          monkeypatch):
    from astropy.table import Table

    path, _ = star_image
    coords = tmp_path / "star.coo"
    table = Table({"x": [TRUE_X], "y": [TRUE_Y]})
    table.meta["origin"] = 1
    from harwood.core import catalog

    catalog.write(table, coords)

    display.reset()
    monkeypatch.chdir(tmp_path)
    try:
        result = get_task("display").run(image=str(path), backend="headless",
                                         coords=str(coords))
    finally:
        display.reset()
    assert result.outputs and result.outputs[0].exists()
    marked, = [step for step in result.steps if step.action == "mark"]
    assert marked.measured["marked"] == 1


def test_asking_for_ds9_says_it_is_not_there_yet():
    display.reset()
    with pytest.raises(ValueError) as raised:
        display.get("ds9")
    assert "headless" in raised.value.fix


# ---------------------------------------------------------------- the window


def _window():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from harwood.viewer._qt import application
    from harwood.viewer.window import HarwoodWindow

    application()
    return HarwoodWindow()


def test_the_window_and_the_cursor_file_agree(star_image):
    """Pressing r over the star and reading `121 91 r` from a file are the
    same measurement, to the last digit."""
    _, data = star_image
    window = _window()
    window.load(data, name="star")
    pressed = window.examine_at("r", 121.0, 91.0)
    read = examine.examine(data, "r", 121.0, 91.0)
    assert pressed.values == pytest.approx(read.values, nan_ok=True, rel=0, abs=0)
    assert window.measurements[-1] == ("star", pressed)
    assert any(m.tag == "examine" for m in window.frames[1].markers)


def test_the_task_s_parameters_reach_the_window(star_image, tmp_path):
    _, data = star_image
    window = _window()
    window.load(data, name="star")
    log = tmp_path / "imexam.log"
    window.configure_examine(examine.Settings(radius=8, fittype="moffat"), str(log))

    assert window.param_widgets["radius"].text() == "8"
    assert window.param_widgets["fittype"].currentText() == "moffat"
    window.examine_at("r", 121.0, 91.0)
    assert window.examine_settings.fittype == "moffat"
    assert "[r]" in log.read_text()


def test_n_and_p_walk_the_frames_and_q_ends_the_wait(star_image):
    _, data = star_image
    window = _window()
    for frame in (1, 2, 3):
        window.load(data * frame, frame=frame, name=f"f{frame}")
    window.select_frame(1)

    window._key("n")
    assert window.current == 2
    window._key("p")
    window._key("p")
    assert window.current == 3, "p wraps round from the first frame"

    ended = []
    window.examine_done.connect(lambda: ended.append(True))
    window._key("q")
    assert ended


def test_imexamine_in_the_window_returns_what_was_measured(star_image, monkeypatch):
    """The interactive path end to end: the task opens the image, keys are
    pressed, q ends it, and the measurements come back as the result."""
    from PySide6 import QtCore

    path, _ = star_image
    display.reset()
    monkeypatch.setenv("HARWOOD_DISPLAY", "native")
    window = _window()
    window.show()

    def press():
        window._cursor = (121.0, 91.0)
        window._key("r")
        window._key("a")
        window._key("q")

    # The keys are pressed once the task is actually waiting for them. A timer
    # started any earlier fires inside the display's own event processing,
    # and q would arrive before anyone was listening for it.
    waiting = window.wait_for_examine

    def wait_then_press():
        QtCore.QTimer.singleShot(0, press)
        waiting()

    monkeypatch.setattr(window, "wait_for_examine", wait_then_press)
    try:
        result = get_task("imexamine").run(input=[str(path)], iterations=1)
    finally:
        display.reset()
    keys = [row["key"] for row in result.data["rows"]]
    assert keys == ["r", "a"]
    assert result.data["measurements"][0]["fwhm"] == pytest.approx(FWHM, abs=0.1)


# ----------------------------------------------------------- against IRAF


def _iraf_log(fittype):
    from pathlib import Path

    path = Path(__file__).parent / "data" / f"imexam_iraf_{fittype}.log"
    names = ["x", "y", "wx", "wy", "radius", "mag", "flux", "sky", "peak", "e", "pa"]
    names += (["beta"] if "moffat" in fittype else []) + ["enclosed", "fwhm", "direct"]
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        rows.append(dict(zip(names, (float(v) if v != "INDEF" else float("nan")
                                     for v in line.split()), strict=True)))
    return rows


@pytest.mark.parametrize("fittype", ["moffat", "gaussian", "int_moffat"])
def test_the_a_key_prints_what_iraf_prints(fittype):
    """IRAF 2.18's own imexamine, run on the same pixels (tests/iraf_field.py
    builds them; tests/data holds what IRAF printed).

    Every column agrees to the precision IRAF prints -- two decimals, or four
    significant figures for FLUX, SKY and PEAK -- the profile fits and the
    fits to the enclosed flux included, since both go through IRAF's own
    nlfit (core/nlfit.py) rather than a general-purpose minimiser that would
    go on past where IRAF stops.
    """
    from iraf_field import CURSOR, field

    # int_moffat: the same field rounded to a 16-bit integer image, which
    # starfocus dithers with IRAF's own urand before it measures.
    data = (np.round(field()).astype(np.int16) if fittype.startswith("int")
            else field().astype(float))
    settings = examine.Settings(fittype=fittype.removeprefix("int_"))
    two = 0.0051                                      # half of 0.01, and a hair
    for (x, y), iraf in zip(CURSOR, _iraf_log(fittype), strict=True):
        ours = examine.examine(data, "a", x, y, settings).values
        for name in ("x", "y", "radius", "mag", "e", "fwhm", "direct", "enclosed"):
            assert ours[name] == pytest.approx(iraf[name], abs=two), name
        for name in ("flux", "sky", "peak"):          # %7.4g
            assert ours[name] == pytest.approx(iraf[name], rel=5.01e-4), name
        if "moffat" in fittype:
            assert ours["beta"] == pytest.approx(iraf["beta"], abs=two)


def test_m_j_k_and_z_print_what_iraf_prints():
    """tests/data/imexam_iraf_keys.log is IRAF 2.18's imexamine log for the
    cursor file beside it: m, j, k and z on the six stars, boxes cut by the
    image's edge, and colon commands (:ncstat 6, :nlstat 4) changing the m
    box between keys. Everything agrees to the figures IRAF prints."""
    import re
    from pathlib import Path

    from iraf_field import field

    data = field()
    settings = examine.Settings()
    base = Path(__file__).parent / "data" / "imexam_iraf_keys"
    log = iter(line for line in (base.with_suffix(".log")).read_text().splitlines()
               if not line.startswith("#"))
    for x, y, _, key in examine.read_cursor(base.with_suffix(".cur")):
        if key.startswith(":"):
            examine.colon(settings, key)
            continue
        ours = examine.examine(data, key, x, y, settings).values
        if key == "m":
            section, npix, *numbers = next(log).split()
            assert (ours["section"], ours["npix"]) == (section, int(npix))
            for name, value in zip(("mean", "median", "stddev", "min", "max"), numbers,
                                   strict=True):
                assert ours[name] == pytest.approx(float(value), rel=5.01e-4), name
        elif key in "jk":
            # SPP's %7g keeps seven characters, sign and point included.
            iraf = dict(re.findall(r"(\w+)=\s*([-\d.e+]+)", next(log)))
            for ours_name, name in (("centre", "center"), ("peak", "peak"),
                                    ("background", "bkg")):
                assert ours[ours_name] == pytest.approx(float(iraf[name]), rel=5.01e-5,
                                                        abs=5.01e-4), name
            for name in ("sigma", "fwhm"):                # the log's %5.3f
                assert ours[name] == pytest.approx(float(iraf[name]), abs=5.01e-4)
        else:
            columns = [int(c) for c in next(log).split()]
            assert columns == ours["columns"]
            for row in ours["rows"]:
                line, *values = next(log).split()
                assert int(line) == row
                assert [float(v) for v in values] == pytest.approx(
                    [float(data[row - 1, c - 1]) for c in columns], rel=5.01e-4)


def test_e_contours_at_the_levels_conrec_picks(star_image):
    """ncontours 5 between the box's extremes: the step is the range over
    five, cut to one significant figure, from the minimum up -- IRAF's
    conrec (CLGEN)."""
    _, data = star_image
    result = examine.examine(data, "e", 121.0, 91.0)
    box = data[80:101, 110:131].astype(np.float32)
    step = float((box.max() - box.min()) / 5)
    power = 10.0 ** np.floor(np.log10(step))
    assert result.values["interval"] == pytest.approx(np.trunc(step / power) * power,
                                                      rel=1e-6)
    assert result.values["levels"][0] == pytest.approx(float(box.min()), rel=1e-6)
    assert result.values["levels"][-1] <= box.max()
    assert result.plot["kind"] == "contour" and result.plot["z"].shape == (21, 21)


def test_s_draws_the_box_and_takes_its_angles(star_image):
    _, data = star_image
    settings = examine.Settings()
    examine.colon(settings, ":angh 45")
    result = examine.examine(data, "s", 121.0, 91.0, settings)
    assert result.plot["kind"] == "surface" and result.plot["angh"] == 45.0
    assert result.plot["z"].shape == (21, 21)


def test_colon_commands_find_the_pset_of_the_last_graph(star_image):
    """:naverage after j is jimexam's, after k kimexam's; :sigma is always
    jimexam's (IRAF's k never sees it); after a, the shared names are
    rimexam's; after m, which draws nothing, they do not apply."""
    _, data = star_image
    s = examine.Settings()
    assert examine.colon(s, ":naverage 3") == "Parameter does not apply to current graph"
    examine.examine(data, "k", 121.0, 91.0, s)
    assert examine.colon(s, ":nav 3") == "naverage 3"
    assert (s.knaverage, s.naverage) == (3, 5)
    examine.colon(s, ":sigma 2")
    assert (s.sigma, s.ksigma) == (2.0, 1.0)
    examine.examine(data, "a", 121.0, 91.0, s)
    examine.colon(s, ":width 7")
    assert (s.width, s.kwidth) == (7.0, 10.0)
    assert examine.colon(s, ":radius") == "radius 5"
    examine.examine(data, "e", 121.0, 91.0, s)
    examine.colon(s, ":ncol 31")
    assert (s.encolumns, s.ncolumns) == (31, 21)
    examine.colon(s, ":unlearn")
    assert s.encolumns == 21
    with pytest.raises(ValueError, match="ambiguous"):
        examine.colon(s, ":n")
    with pytest.raises(ValueError) as raised:
        examine.colon(s, ":banner no")
    assert "draws its own" in raised.value.fix


def test_a_colon_command_typed_in_the_window_reaches_the_keys(star_image):
    _, data = star_image
    window = _window()
    window.load(data, name="star")
    window.prompt.setText(":ncstat 9")
    window._run_command()
    assert window.examine_settings.ncstat == 9
    stats = window.examine_at("m", 121.0, 91.0)
    assert stats.values["npix"] == 9 * 5                  # nlstat is still 5
    for key in "es":
        assert window.examine_at(key, 121.0, 91.0).ok
