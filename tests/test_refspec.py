"""Time, airmass, and which arc calibrates which exposure.

The chain these cover is `setairmass` -> `refspectra` -> `dispcor`, and the
thing worth proving is the middle one: that an exposure taken between two arcs
gets a wavelength scale interpolated between them, in the right proportion.
The flexure is invented here, so the right answer is known exactly.
"""

import numpy as np
import pytest

from harwood.core import arc, observatory, obstime, refspec

# ------------------------------------------------------------------ obstime


def test_a_time_reads_as_hours_however_it_is_written():
    assert obstime.hours("2:08:22.00") == pytest.approx(2.13944, abs=1e-5)
    assert obstime.hours("12:30") == pytest.approx(12.5)
    assert obstime.hours(3.25) == 3.25
    assert obstime.hours("") is None and obstime.hours(None) is None
    assert obstime.hours("not a time") is None


def test_a_negative_time_stays_negative():
    """`-0:30` is minus half an hour, not minus zero plus half an hour. Summing
    the parts with their own signs gets this wrong whenever the degrees are
    zero, which is exactly where a declination lives."""
    assert obstime.hours("-0:30:00") == pytest.approx(-0.5)


def test_a_two_digit_year_is_nineteen_hundred_and_something():
    """The 1992 headers say 25/01/92, and the FITS rule is that the format
    itself dates the file."""
    assert obstime.date({"DATE-OBS": "25/01/92"}) == (1992, 1, 25)
    assert obstime.date({"DATE-OBS": "1992-01-25T02:08:22"}) == (1992, 1, 25)
    assert obstime.date({"DATE-OBS": ""}) is None


def test_the_middle_of_the_exposure_is_not_its_start():
    header = {"UT": "2:08:22.00", "EXPTIME": 1200.0}
    assert obstime.middle(header) == pytest.approx(2.13944 + 1200 / 7200, abs=1e-5)


def test_a_utmiddle_already_in_the_header_is_believed():
    header = {"UT": "2:08:22.00", "EXPTIME": 1200.0, "UTMIDDLE": "03:00:00.0"}
    assert obstime.middle(header) == pytest.approx(3.0)


def test_the_night_sorts_in_the_order_it_happened():
    """Evening before morning, though 23 is larger than 4."""
    evening, morning = obstime.sortable(23.5), obstime.sortable(4.0)
    assert evening < morning


def test_refraction_shortens_the_path():
    assert obstime.refracted(1.0) == pytest.approx(1.0)
    assert obstime.refracted(2.0) < 2.0
    assert 2.0 - obstime.refracted(2.0) == pytest.approx(0.0055, abs=0.001)


# -------------------------------------------------------------- observatory


def test_a_site_is_found_by_code_alias_or_name():
    assert observatory.find("kpno").code == "kpno"
    assert observatory.find("Kitt Peak").code == "kpno"
    assert observatory.find("pico dos dias").code == "lna"
    assert observatory.find("Laboratorio").code == "lna"
    assert observatory.find("nowhere") is None


def test_longitude_is_flipped_in_exactly_one_place():
    """IRAF counts longitude positive west; every library counts it positive
    east. A sign error here does not fail -- it puts the star on the other side
    of the meridian and returns a plausible airmass."""
    kpno = observatory.find("kpno")
    assert kpno.longitude > 0 and kpno.east < 0
    assert kpno.east == -kpno.longitude
    assert "111.600 W" in kpno.describe()


def test_a_header_naming_its_observatory_is_enough():
    assert observatory.from_header({"OBSERVAT": "KPNO"}).code == "kpno"
    assert observatory.from_header({"TELESCOP": "CTIO"}).code == "ctio"
    assert observatory.from_header({"OBJECT": "kpno"}) is None


def test_the_airmass_rises_over_a_long_exposure():
    """A star setting in the west: the end of the exposure is higher airmass
    than its start, and the effective value sits between them."""
    header = {"RA": "22:30:00", "DEC": "58:00:00", "DATE-OBS": "25/01/92",
              "UT": "3:30:00.00", "EXPTIME": 1200.0}
    site = observatory.find("kpno")
    measured = obstime.airmass(header, site.latitude, site.east, site.altitude)

    assert measured.begin < measured.end
    assert measured.begin < measured.effective < measured.end
    assert measured.effective == pytest.approx(measured.middle, abs=0.01)


def test_no_coordinates_means_no_airmass_rather_than_a_guess():
    site = observatory.find("kpno")
    assert obstime.airmass({"UT": "3:00:00", "EXPTIME": 60.0},
                           site.latitude, site.east, site.altitude) is None


# ---------------------------------------------------------------- refspec


NIGHT = [("arc_early", 1.5), ("arc_late", 4.5)]


def test_interp_weights_by_how_close_each_arc_is():
    """One hour after the first arc and two before the second: two thirds of
    the answer comes from the first."""
    result, = refspec.assign([("obj", 2.5)], NIGHT, "interp")
    assert result.references == ["arc_early", "arc_late"]
    assert result.weights == pytest.approx([2 / 3, 1 / 3])


def test_the_weights_always_sum_to_one():
    for hour in (1.6, 2.0, 3.0, 4.0, 4.4):
        result, = refspec.assign([("obj", hour)], NIGHT, "interp")
        assert sum(result.weights) == pytest.approx(1.0)


def test_average_ignores_the_distance_and_interp_does_not():
    late = [("obj", 4.4)]
    assert refspec.assign(late, NIGHT, "average")[0].weights == pytest.approx([0.5, 0.5])
    assert refspec.assign(late, NIGHT, "interp")[0].weights[1] > 0.9


def test_each_mode_picks_what_its_name_says():
    entry = [("obj", 2.5)]
    assert refspec.assign(entry, NIGHT, "nearest")[0].references == ["arc_early"]
    assert refspec.assign(entry, NIGHT, "preceding")[0].references == ["arc_early"]
    assert refspec.assign(entry, NIGHT, "following")[0].references == ["arc_late"]

    later = [("obj", 4.0)]
    assert refspec.assign(later, NIGHT, "nearest")[0].references == ["arc_late"]
    assert refspec.assign(later, NIGHT, "preceding")[0].references == ["arc_early"]


def test_outside_the_arcs_it_does_not_pretend_to_interpolate():
    """Extrapolating flexure is not interpolating it, and the difference has to
    be visible rather than hidden in a weight of 1.0."""
    result, = refspec.assign([("obj", 5.5)], NIGHT, "interp")
    assert result.references == ["arc_late"]
    assert "rather than an interpolation" in result.reason


def test_an_exposure_with_no_time_gets_no_arc():
    result, = refspec.assign([("obj", None)], NIGHT, "interp")
    assert result.references == []
    assert "does not say when" in result.reason


def test_arcs_with_no_times_at_all_is_an_error_not_an_empty_answer():
    with pytest.raises(ValueError) as raised:
        refspec.assign([("obj", 2.5)], [("arc", None)], "interp")
    assert "UT and EXPTIME" in raised.value.fix


def test_an_unknown_mode_names_the_ones_that_exist():
    with pytest.raises(ValueError) as raised:
        refspec.assign([("obj", 2.5)], NIGHT, "closest")
    assert "interp" in raised.value.fix


def test_the_keywords_round_trip_in_irafs_own_form():
    from astropy.io import fits

    header = fits.Header()
    assignment = refspec.Assignment("obj", 2.5, ["a", "b"], [0.25, 0.75])
    refspec.write_keywords(header, assignment)

    assert header["REFSPEC1"].split()[0] == "a"
    assert all(len(str(card)) <= 80 for card in header.cards)
    names, weights = refspec.read_keywords(header)
    assert names == ["a", "b"] and weights == pytest.approx([0.25, 0.75])


def test_a_single_reference_is_written_without_a_weight():
    """IRAF writes a bare name when there is only one, and so does this: a
    weight of 1.0 on a single arc is noise in the header."""
    from astropy.io import fits

    header = fits.Header()
    refspec.write_keywords(header, refspec.Assignment("obj", 2.5, ["a"], [1.0]))
    assert header["REFSPEC1"] == "a"
    assert refspec.read_keywords(header) == (["a"], [1.0])


def test_writing_an_assignment_clears_the_one_before_it():
    from astropy.io import fits

    header = fits.Header()
    refspec.write_keywords(header, refspec.Assignment("obj", 2.5, ["a", "b"], [0.5, 0.5]))
    refspec.write_keywords(header, refspec.Assignment("obj", 2.5, ["c"], [1.0]))
    assert refspec.read_keywords(header) == (["c"], [1.0])


# ------------------------------------------------------- the tasks together

CRVAL, CDELT, NPIXELS = 6707.0, 0.152, 1024
CENTRE = (NPIXELS + 1) / 2
#: How far the spectrograph flexed between the two arcs, in Angstroms. Half a
#: pixel: small enough to be realistic, large enough that using the wrong arc
#: is visible against the hundredth-Angstrom accuracy of the solution itself.
FLEXURE = 0.5 * CDELT

pytestmark = pytest.mark.skipif(
    not any(d.is_dir() for d in arc.IRAF_LINELISTS),
    reason="IRAF line lists are not installed",
)


def _arc_flux(catalogue, crval, seed=7):
    rng = np.random.default_rng(seed)
    low = crval + (1 - CENTRE) * CDELT
    high = crval + (NPIXELS - CENTRE) * CDELT
    lines = catalogue[(catalogue > low) & (catalogue < high)]
    strong = lines[rng.random(lines.size) < 0.45]

    pixels = (strong - crval) / CDELT + CENTRE
    flux = np.full(NPIXELS, 300.0)
    sigma = 1.7 / 2.3548
    grid = np.arange(1.0, NPIXELS + 1.0)
    for position, amplitude in zip(pixels, 10 ** rng.uniform(2.5, 4.3, pixels.size),
                                   strict=False):
        flux += amplitude * np.exp(-0.5 * ((grid - position) / sigma) ** 2)
    # The rest of the lamp's lines, faint, as in a real lamp.
    faint = lines[~np.isin(lines, strong)]
    for position, amplitude in zip((faint - crval) / CDELT + CENTRE,
                                   10 ** rng.uniform(1.3, 2.0, faint.size), strict=False):
        flux += amplitude * np.exp(-0.5 * ((grid - position) / sigma) ** 2)
    return flux + rng.normal(0, 12, NPIXELS)


@pytest.fixture
def flexing_night(tmp_path, monkeypatch):
    """Two arcs three hours apart with the spectrograph flexing between them,
    and one exposure in the middle."""
    from astropy.io import fits

    from harwood.core import spectrum
    from harwood.core.registry import get_task

    monkeypatch.chdir(tmp_path)
    catalogue = arc.read_linelist("linelists$thorium.dat")[0]

    for name, crval, ut in (("early", CRVAL, "01:00:00.00"),
                            ("late", CRVAL + FLEXURE, "04:00:00.00")):
        flux = _arc_flux(catalogue, crval)
        header = fits.Header({"UT": ut, "EXPTIME": 60.0, "DATE-OBS": "25/01/92",
                              "OBSERVAT": "KPNO", "IMAGETYP": "comp"})
        fits.PrimaryHDU(flux.astype(np.float32), header).writeto(tmp_path / f"{name}.fits")
        get_task("identify").run(image=f"{name}.fits", crval=crval, cdelt=CDELT,
                                 coordlist="linelists$thorium.dat",
                                 function="chebyshev", order=3, niterate=3,
                                 database=tmp_path)

    # The object: an hour after the first arc, two before the second, so
    # interp should weight them two to one.
    star = spectrum.Aperture(
        number=1, centre=50.0, low=-8.0, high=8.0,
        pixel=np.arange(1.0, NPIXELS + 1.0),
        flux=np.full(NPIXELS, 1000.0), error=np.full(NPIXELS, 30.0), title="star")
    header = fits.Header({"UT": "01:30:00.00", "EXPTIME": 3600.0,
                          "DATE-OBS": "25/01/92", "OBSERVAT": "KPNO",
                          "RA": "22:30:00", "DEC": "58:00:00",
                          "IMAGETYP": "object"})
    spectrum.write(tmp_path / "star.ms.fits", [star], header=header)
    return tmp_path, catalogue


def test_refspectra_assigns_by_time_without_being_told_which_is_which(flexing_night):
    from harwood.core.registry import get_task

    work, _ = flexing_night
    result = get_task("refspectra").run(input=["star.ms.fits"], database=work)

    row, = result.data["rows"]
    assert row["arc"] == "early + late"

    # An hour in, two hours to go -- but the arcs are sixty-second exposures
    # and their own middles count too, so it is not exactly two thirds.
    early, late = (float(value) for value in row["weights"].split())
    assert early == pytest.approx(2 / 3, abs=0.005)
    assert early + late == pytest.approx(1.0)


def test_dispcor_needs_no_argument_once_refspectra_has_run(flexing_night):
    """The point of the keywords: the file says which arc calibrates it."""
    from harwood.core import spectrum
    from harwood.core.registry import get_task

    work, _ = flexing_night
    get_task("refspectra").run(input=["star.ms.fits"], database=work)
    get_task("dispcor").run(input="star.ms.fits", output="star.w.fits",
                            database=work)

    calibrated = spectrum.read(work / "star.w.fits")[0]
    assert calibrated.wavelength is not None


def test_the_interpolated_scale_lands_where_the_weights_say(flexing_night):
    """The number this whole chain exists to get right.

    The two arcs differ by half a pixel of flexure. The exposure is a third of
    the way between them, so its wavelength scale should sit a third of the
    way -- not on either arc, and not halfway.
    """
    from harwood.core import spectrum
    from harwood.core.registry import get_task

    work, _ = flexing_night
    get_task("refspectra").run(input=["star.ms.fits"], database=work)

    scales = {}
    for label, reference in (("early", "early"), ("late", "late"), ("interp", "")):
        get_task("dispcor").run(input="star.ms.fits", output=f"{label}.fits",
                                reference=reference, database=work)
        scales[label] = spectrum.read(work / f"{label}.fits")[0].wavelength

    separation = float(np.median(scales["late"] - scales["early"]))
    assert separation == pytest.approx(FLEXURE, abs=0.005), \
        "the two arcs differ by the flexure that was put in"

    offset = float(np.median(scales["interp"] - scales["early"]))
    assert offset == pytest.approx(separation / 3, abs=0.005), \
        f"a third of the way, not {offset / separation:.2f} of it"


def test_the_arcs_and_their_weights_are_reported_not_just_used(flexing_night):
    from harwood.core.registry import get_task

    work, _ = flexing_night
    get_task("refspectra").run(input=["star.ms.fits"], database=work)
    result = get_task("dispcor").run(input="star.ms.fits", output="w.fits",
                                     database=work)

    solution, = [step for step in result.steps if step.action == "solution"]
    assert "early (0.66" in solution.measured["arcs"]
    assert "late (0.33" in solution.measured["arcs"]
    assert any("early" in line and "late" in line for line in result.history)


def test_dispcor_with_no_assignment_says_how_to_get_one(flexing_night):
    from harwood.core.registry import get_task

    work, _ = flexing_night
    with pytest.raises(ValueError) as raised:
        get_task("dispcor").run(input="star.ms.fits", output="w.fits",
                                database=work)
    assert "refspectra" in raised.value.fix, "the suggestion names the way out"


def test_refspectra_will_not_overwrite_an_assignment_by_accident(flexing_night):
    from harwood.core.registry import get_task

    work, _ = flexing_night
    get_task("refspectra").run(input=["star.ms.fits"], database=work)
    again = get_task("refspectra").run(input=["star.ms.fits"], select="nearest",
                                       database=work)
    assert any("--override" in warning for warning in again.warnings)
    assert not again.outputs


def test_setairmass_replaces_the_shutter_opening_value_and_says_so(flexing_night):
    """An hour-long exposure: the airmass at the start is not the airmass the
    light travelled through, and the task has to make that visible rather than
    quietly changing a number."""
    from astropy.io import fits

    from harwood.core.registry import get_task

    work, _ = flexing_night
    with fits.open(work / "star.ms.fits", mode="update") as hdus:
        hdus[0].header["AIRMASS"] = 1.20

    result = get_task("setairmass").run(images=["star.ms.fits"])
    row, = result.data["rows"]

    assert row["was"] == 1.20
    assert row["begin"] < row["airmass"] < row["end"]
    assert any("shutter opened" in warning for warning in result.warnings)

    header = fits.getheader(work / "star.ms.fits")
    assert header["AIRMASS"] == pytest.approx(row["airmass"])
    assert header["UTMIDDLE"] == "02:00:00.0", "an hour's exposure from 01:30"
    assert any("replacing AIRMASS 1.2" in line for line in header["HISTORY"])


def test_setairmass_without_a_site_anywhere_says_which_it_knows():
    from harwood.core.registry import get_task

    with pytest.raises(ValueError) as raised:
        get_task("setairmass").run(images=["anything.fits"], observatory="narnia")
    assert "kpno" in raised.value.fix
