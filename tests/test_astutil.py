"""Time, place, and the headers that carry them.

setjd, airmass, asttimes and observatory all sit on `core.obstime` and
`core.observatory`, so most of the arithmetic is already checked in
test_refspec. What these cover is the part that is theirs: the heliocentric
correction, and the tasks refusing to guess when they are not told enough.
"""

import numpy as np
import pytest

from harwood.core import observatory, obstime
from harwood.core.registry import get_task

#: DH Cep on the night of the tutorial: a real position, a real date.
HEADER = {"RA": "22:46:32.00", "DEC": "58:02:22.00", "DATE-OBS": "25/01/92",
          "UT": " 2:08:22.00", "EXPTIME": 1200.0, "OBSERVAT": "KPNO",
          "OBJECT": "DHCep 6707"}


@pytest.fixture
def frame(tmp_path, monkeypatch):
    """One frame with the tutorial's header, to edit and stamp."""
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32),
                    fits.Header(HEADER)).writeto(tmp_path / "sp.fits")
    return tmp_path


# ------------------------------------------------------------- the physics


def test_the_heliocentric_correction_stays_inside_the_orbit():
    """It cannot exceed the time light takes to cross the Earth's orbit, which
    is 499 seconds each way. A correction larger than that is a sign error or
    a wrong epoch, not a measurement."""
    site = observatory.find("kpno")
    for day in ("15/01/92", "15/04/92", "15/07/92", "15/10/92"):
        header = dict(HEADER, **{"DATE-OBS": day})
        _, seconds = obstime.heliocentric(header, site.latitude, site.east,
                                          site.altitude)
        assert abs(seconds) < 500.0, day


def test_the_correction_swings_through_a_year_and_comes_back():
    """Six months apart the Earth is on opposite sides of its orbit, so the
    correction for one star has to change sign -- that is the whole effect."""
    site = observatory.find("kpno")
    winter = obstime.heliocentric(dict(HEADER, **{"DATE-OBS": "25/01/92"}),
                                  site.latitude, site.east, site.altitude)[1]
    summer = obstime.heliocentric(dict(HEADER, **{"DATE-OBS": "25/07/92"}),
                                  site.latitude, site.east, site.altitude)[1]
    assert winter * summer < 0, f"{winter:+.1f} s and {summer:+.1f} s"


def test_the_barycentre_and_the_sun_differ_by_a_few_seconds():
    """Jupiter moves the Sun about its own radius from the barycentre, which
    is a couple of light-seconds. Pulsar timing cares; photometry does not."""
    site = observatory.find("kpno")
    helio = obstime.heliocentric(HEADER, site.latitude, site.east,
                                 site.altitude, kind="heliocentric")[0]
    bary = obstime.heliocentric(HEADER, site.latitude, site.east,
                                site.altitude, kind="barycentric")[0]
    difference = abs(helio - bary) * 86400.0
    assert 0.0 < difference < 10.0, f"{difference:.2f} s apart"


def test_sidereal_time_agrees_with_the_airmass_computed_another_way():
    """Two routes to the same sky. The hour angle from sidereal time minus
    right ascension gives sec z directly; obstime.airmass goes through
    astropy's frame transformation. They have to meet."""
    site = observatory.find("kpno")
    lst = obstime.sidereal(HEADER, site.east)
    hour_angle = np.radians((lst - obstime.hours(HEADER["RA"])) * 15.0)
    declination = np.radians(obstime.hours(HEADER["DEC"]))
    latitude = np.radians(site.latitude)

    cos_z = (np.sin(latitude) * np.sin(declination)
             + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle))
    by_hand = obstime.refracted(1.0 / cos_z)
    measured = obstime.airmass(HEADER, site.latitude, site.east, site.altitude)

    assert by_hand == pytest.approx(measured.middle, abs=0.01)


# -------------------------------------------------------------- the tasks


def test_setjd_writes_all_three_dates(frame):
    from astropy.io import fits

    result = get_task("setjd").run(images=["sp.fits"])
    row, = result.data["rows"]

    header = fits.getheader(frame / "sp.fits")
    assert header["JD"] == pytest.approx(2448646.5961, abs=1e-3)
    assert header["HJD"] != header["JD"]
    assert abs(header["HJD"] - header["JD"]) * 86400 < 500
    assert header["LJD"] == int(header["LJD"]), "a whole day"
    assert row["correction s"] == pytest.approx((header["HJD"] - header["JD"])
                                                * 86400, abs=0.2)


def test_setjd_records_the_correction_it_applied(frame):
    from astropy.io import fits

    get_task("setjd").run(images=["sp.fits"])
    history = fits.getheader(frame / "sp.fits")["HISTORY"]
    assert any("setjd" in line and "heliocentric" in line for line in history)


def test_setjd_still_gives_a_julian_date_for_a_bias_frame(tmp_path, monkeypatch):
    """A calibration frame has no sky position, so there is no correction to
    make -- but the JD is still real and still wanted."""
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32), fits.Header(
        {"DATE-OBS": "25/01/92", "UT": "22:43:39.00", "EXPTIME": 0.0,
         "OBSERVAT": "KPNO"})).writeto(tmp_path / "bias.fits")

    result = get_task("setjd").run(images=["bias.fits"])
    row, = result.data["rows"]

    assert row["jd"] > 2448646
    assert row["hjd"] is None
    assert any("expected" in warning for warning in result.warnings)


def test_the_barycentric_switch_reaches_the_header(frame):
    from astropy.io import fits

    get_task("setjd").run(images=["sp.fits"], barycentric=True)
    assert "barycentric" in fits.getheader(frame / "sp.fits").comments["HJD"]


def test_airmass_from_a_frame_matches_what_setairmass_writes(frame):
    both = (get_task("airmass").run(image="sp.fits"),
            get_task("setairmass").run(images=["sp.fits"]))
    assert both[0].data["airmass"] == pytest.approx(
        both[1].data["rows"][0]["airmass"], abs=1e-4)


def test_airmass_from_coordinates_typed_in(frame):
    result = get_task("airmass").run(
        ra="22:46:32.00", dec="58:02:22.00", date="25/01/92", ut="2:08:22.00",
        exptime=1200.0, observatory="kpno")
    assert result.data["airmass"] == pytest.approx(1.565, abs=0.01)
    assert result.data["sidereal"].startswith("03:")


def test_airmass_with_nothing_to_go_on_says_what_it_needs(frame):
    with pytest.raises(ValueError, match="--ra"):
        get_task("airmass").run(observatory="kpno")


def test_a_low_object_is_flagged_for_the_reason_that_matters(frame):
    """Below about two airmasses the atmosphere disperses the image into a
    short spectrum, and a slit that is not at the parallactic angle loses
    part of it. That is a data-quality problem, not just a number."""
    result = get_task("airmass").run(
        ra="22:46:32.00", dec="-20:00:00", date="25/01/92", ut="2:08:22.00",
        observatory="kpno")
    assert result.data["airmass"] > 2.0
    assert any("parallactic" in warning for warning in result.warnings)


def test_asttimes_converts_one_instant_four_ways(frame):
    result = get_task("asttimes").run(image="sp.fits")
    assert result.data["jd"] == pytest.approx(2448646.596, abs=1e-3)
    assert result.data["mjd"] == pytest.approx(result.data["jd"] - 2400000.5)
    assert result.data["epoch"] == pytest.approx(1992.07, abs=0.01)
    assert result.data["sidereal"]


def test_observatory_lists_and_looks_up():
    every = get_task("observatory").run()
    assert len(every.data["rows"]) > 10

    one = get_task("observatory").run(obsid="kitt peak")
    row, = one.data["rows"]
    assert row["code"] == "kpno"
    # Positive west, as IRAF stores it. The column name has to say so, because
    # a sign error here returns a plausible wrong airmass rather than failing.
    assert row["longitude W"] > 0


def test_an_unknown_site_lists_the_ones_that_exist():
    with pytest.raises(ValueError) as raised:
        get_task("observatory").run(obsid="narnia")
    assert "kpno" in raised.value.fix and "--latitude" in raised.value.fix


def test_the_sites_are_irafs_and_a_reduction_can_add_its_own(tmp_path, monkeypatch):
    """IRAF's obsdb.dat ships with harwood; `observatory set` with
    coordinates adds a site in the same format beside harwood.yaml, makes it
    this reduction's default, and every task finds it."""
    from harwood.core import observatory, sample

    monkeypatch.chdir(tmp_path)
    assert len(observatory.SITES) > 70, "IRAF's table, not a short list"
    assert observatory.find("saao").east == pytest.approx(20.8107, abs=1e-3)
    (tmp_path / "harwood.yaml").write_text("params: {}\n", encoding="utf-8")
    get_task("observatory").run(command="set", obsid="skinakas",
                                name="Skinakas Observatory", latitude=35.212,
                                longitude=-24.8989, altitude=1750, timezone=-2)
    assert 'observatory = "skinakas"' in (tmp_path / "obsdb.dat").read_text(encoding="utf-8")
    assert "observatory: skinakas" in (tmp_path / "harwood.yaml").read_text(encoding="utf-8")
    assert observatory.find("skinakas").altitude == 1750
    assert observatory.default().code == "skinakas"
    get_task("observatory").run(command="set", obsid="kpno")
    sample.write(tmp_path)
    from astropy.io import fits

    with fits.open("obj001.fits", mode="update") as hdus:
        del hdus[0].header["OBSERVAT"]
    rows = get_task("setairmass").run(images=["obj001.fits"], update=False).data["rows"]
    assert rows[0]["airmass"] == pytest.approx(1.0343, abs=1e-4), "the default, kpno"
    listed = get_task("observatory").run(command="images", images=["obj001.fits"])
    assert listed.data["rows"][0]["from"] == "this reduction's default"


# --------------------------------------------------------- hedit and imgets


def test_hedit_sets_a_keyword_and_remembers_what_it_was(frame):
    from astropy.io import fits

    get_task("hedit").run(images=["sp.fits"], fields=["OBJECT"],
                          value="'DH Cep'")
    header = fits.getheader(frame / "sp.fits")

    assert header["OBJECT"] == "DH Cep"
    assert any("was DHCep 6707" in line for line in header["HISTORY"])


def test_hedit_types_the_value_the_way_iraf_does(frame):
    """A new keyword takes the type of what is typed: a number is a number,
    yes is the word yes (IRAF's own rule), an expression's value has its
    type, and quotes keep digits as text."""
    from astropy.io import fits

    for number, (value, expected) in enumerate((("2", 2), ("1.5", 1.5), ("yes", "yes"),
                                                ("(yes)", True), ("(3 > 4)", False),
                                                ("'2'", "2"))):
        get_task("hedit").run(images=["sp.fits"], fields=[f"TESTKEY{number}"],
                              value=value, add=True)
        assert fits.getheader(frame / "sp.fits")[f"TESTKEY{number}"] == expected, value


def test_hedit_will_not_invent_a_keyword_without_being_asked(frame):
    """A typo in a keyword name would otherwise silently add a card nobody
    reads, and the file would look fine."""
    from astropy.io import fits

    result = get_task("hedit").run(images=["sp.fits"], fields=["DISPAXSI"],
                                   value="2")
    assert "DISPAXSI" not in fits.getheader(frame / "sp.fits")
    assert any("--add" in warning for warning in result.warnings)
    # The suggestion is a field of its own now, not buried in the prose.
    assert result.warnings[0].fix and result.warnings[0].level == "warning"


def test_hedit_can_show_without_touching_anything(frame):
    from astropy.io import fits

    before = fits.getheader(frame / "sp.fits")["OBJECT"]
    result = get_task("hedit").run(images=["sp.fits"], fields=["OBJECT"],
                                   value="'changed'", update=False)

    assert fits.getheader(frame / "sp.fits")["OBJECT"] == before
    assert result.data["rows"][0]["was"] == before
    assert any("nothing was written" in warning for warning in result.warnings)
    assert not result.outputs


def test_hedit_deletes_and_says_what_was_lost(frame):
    from astropy.io import fits

    get_task("hedit").run(images=["sp.fits"], fields=["OBJECT"], delete=True)
    header = fits.getheader(frame / "sp.fits")

    assert "OBJECT" not in header
    assert any("deleted OBJECT" in line and "DHCep" in line
               for line in header["HISTORY"])


def test_setting_and_deleting_at_once_is_refused(frame):
    with pytest.raises(ValueError, match="cannot"):
        get_task("hedit").run(images=["sp.fits"], fields=["OBJECT"],
                              value="x", delete=True)


def test_hedit_edits_a_whole_night_at_once(tmp_path, monkeypatch):
    """The usual case, and the reason it takes a template: doing it one file
    at a time is how half a set ends up different from the other half."""
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    for index in range(4):
        fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32),
                        fits.Header({"OBJECT": "flat"})).writeto(
                            tmp_path / f"f{index}.fits")

    result = get_task("hedit").run(images=["f*.fits"], fields=["DISPAXIS"],
                                   value="2", add=True)
    assert len(result.outputs) == 4
    for index in range(4):
        assert fits.getheader(tmp_path / f"f{index}.fits")["DISPAXIS"] == 2


def test_imgets_returns_one_value(frame):
    result = get_task("imgets").run(image="sp.fits", param="EXPTIME")
    assert result.data["value"] == "1200.0"
    assert result.data["found"] is True
    assert result.stdout == ["1200.0"]


def test_imgets_falls_back_and_says_it_did(frame):
    result = get_task("imgets").run(image="sp.fits", param="GRATING",
                                    default="unknown")
    assert result.data["value"] == "unknown"
    assert result.data["found"] is False
    assert any("not found (GRATING)" in str(warning) for warning in result.warnings)


def test_imgets_prints_the_value_and_nothing_else(frame):
    """For `$(harwood imgets ...)`: the value alone on stdout, the warning on
    stderr, so a missing keyword is IRAF's 0 and the script still reads it."""
    from click.testing import CliRunner

    from harwood.cli.main import app

    runner = CliRunner()
    found = runner.invoke(app, ["imgets", "sp.fits", "exptime"])
    assert found.exit_code == 0 and found.stdout == "1200.0\n"
    missing = runner.invoke(app, ["imgets", "sp.fits", "GRATING"])
    assert missing.stdout == "0\n"
    assert "not found" in missing.stderr


def test_times_after_the_shipped_table_do_not_go_online(tmp_path, monkeypatch):
    """For a date past astropy's bundled Earth-rotation table, the sidereal
    time and the airmass downloaded IERS-A -- and without a connection each
    call waited half a minute. UT1-UTC and polar motion are below anything an
    airmass or an HJD can show, so they are no longer asked for."""
    import astropy.utils.data

    from harwood.core import sample
    from harwood.core.registry import get_task

    def no_network(*args, **kwargs):
        raise AssertionError("tried to download " + str(args[:1]))

    monkeypatch.setattr(astropy.utils.data, "download_file", no_network)
    monkeypatch.chdir(tmp_path)
    sample.write(tmp_path)                      # observed 2026-09-13 at KPNO
    result = get_task("setairmass").run(images=["obj001.fits"], observatory="kpno",
                                        update=False)
    assert 1.0 < result.data["rows"][0]["airmass"] < 1.1
    assert get_task("asttimes").run(image="obj001.fits").data["sidereal"]
    assert get_task("setjd").run(images=["obj001.fits"], observatory="kpno",
                                 listonly=True).data["rows"][0]["hjd"] > 2461296
