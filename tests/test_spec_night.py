"""The 1992 KPNO spectroscopic night, extracted end to end.

Ten bias frames, dome flats, two comparison lamps and six 1200-second
exposures of DH Cep and AH Cep at 6707 Angstroms, from the course's exercise 5.
Synthetic frames prove the arithmetic; these prove it survives a real
spectrograph, a real trace, and a header written in 1992.

Set HARWOOD_TEST_DATA to the exercises directory to run these; they skip
otherwise.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from harwood.core import spectrum
from harwood.core.registry import get_task

DATA = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser()
SPEC = DATA / "spec"

pytestmark = pytest.mark.skipif(
    not (SPEC / "fsp0019").exists(),
    reason=f"IRAF tutorial spectra not found under {DATA}",
)

#: From the exercise: the frames are 1024 along the dispersion by 101 across
#: after trimming, with the overscan in columns 102 to 133.
BIASSEC, TRIMSEC = "[102:133,1:1024]", "[1:101,1:1024]"
# The detector's, from every header (GAIN, RDNOISE). This test used to guess
# 3 and 5: the extraction's own model agreed with itself, and only the
# variance ccdproc carries, made from the header's gain, showed the errors
# sitting 0.75 below a photon limit computed with the guess -- sqrt(3/5.4).
GAIN, RDNOISE = 5.4, 3.5


@pytest.fixture(scope="module")
def night(tmp_path_factory):
    """Bias-corrected object frames, as exercise 3 would leave them."""
    work = tmp_path_factory.mktemp("spec")
    for name in ("fsp0001", "fsp0002", "fsp0003", "fsp0019", "fsp0020",
                 "fsp0022", "fsp0023"):
        (work / f"{name[1:]}.fits").write_bytes((SPEC / name).read_bytes())

    # ccdproc keeps its subsets file in the working directory: run there, or
    # it lands in whatever directory pytest was started from.
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(work)
        ccdproc = get_task("ccdproc")
        ccdproc.run(input=[str(work / "sp000[123].fits")], output=str(work / "b"),
                    overscan=True, trim=True, biassec=BIASSEC, trimsec=TRIMSEC)
        get_task("zerocombine").run(
            input=[str(work / "b" / "sp000*.fits")], output=work / "Zero.fits",
            reject="minmax", nlow=0, nhigh=1)

        for name in ("sp0019", "sp0020", "sp0022", "sp0023"):
            ccdproc.run(input=[str(work / f"{name}.fits")],
                        output=str(work / f"p{name}.fits"), overscan=True, trim=True,
                        zerocor=True, zero=work / "Zero.fits",
                        biassec=BIASSEC, trimsec=TRIMSEC)
    return work


def extract(work, name, **overrides):
    options = dict(input=str(work / f"p{name}.fits"),
                   output=work / f"{name}.ms.fits", gain=GAIN, readnoise=RDNOISE,
                   nsum=20, t_step=20, background="fit")
    options.update(overrides)
    get_task("apall").run(**options)
    return spectrum.read(work / f"{name}.ms.fits")[0]


def test_the_frame_geometry_is_what_the_exercise_says(night):
    from astropy.io import fits

    data = fits.getdata(night / "psp0019.fits")
    assert data.shape == (1024, 101), "1024 along the dispersion, 101 across"


def test_one_star_is_found_on_the_slit(night):
    star = extract(night, "sp0019")
    assert 40 < star.centre < 70, "a long-slit exposure of a single star"
    assert 4 < star.width < 20, f"aperture {star.width:.1f} px wide"


def test_the_trace_finds_a_real_drift(night):
    """The camera has distortion and the spectrograph flexes; the spectrum does
    not sit on one column, and the amount it moves is a property of the
    instrument rather than of the star."""
    from harwood.core import apextract, images

    data, _ = images.read(night / "psp0019.fits")
    profile, line = apextract.getdata(apextract.frame(data, 2), nsum=20)
    aperture = apextract.Aperture(1, apextract.find(profile, nfind=1)[0], line=line)
    apextract.trace(data, 2, [aperture], line, step=20, nsum=20)
    fit = aperture.trace.fit

    assert fit is not None and fit.model is not None
    assert fit.rms < 0.1, "a bright spectrum traces to a fraction of a pixel"
    ends = aperture.trace(np.array([1.0, 1024.0]))
    assert 0.5 < abs(ends[1] - ends[0]) < 5.0, \
        f"drift of {ends[1] - ends[0]:+.2f} px end to end"


def test_the_star_sits_in_the_same_place_all_night(night):
    """Four exposures over two hours: the guiding held, and the extraction
    agrees that it did."""
    centres = [extract(night, name).centre
               for name in ("sp0019", "sp0020", "sp0022", "sp0023")]
    assert max(centres) - min(centres) < 0.5, f"centres {centres}"


def test_the_error_estimate_matches_the_repeatability(night):
    """The check that makes the uncertainties mean something.

    Two 1200-second exposures of one star, minutes apart. The ratio of their
    fluxes is not one -- transparency changes -- but its *scatter* is the
    repeatability, and it has to agree with what the extraction claims the
    uncertainties are. An error bar nobody has checked against a repeat
    measurement is decoration.
    """
    first = extract(night, "sp0022", weights="variance")
    second = extract(night, "sp0023", weights="variance")

    good = (np.isfinite(first.flux) & np.isfinite(second.flux)
            & (first.flux > 200) & (second.flux > 200))
    assert good.sum() > 500, "most of the spectrum is usable"

    ratio = first.flux[good] / second.flux[good]
    measured = float(np.std(ratio))

    relative = np.hypot(first.error[good] / first.flux[good],
                        second.error[good] / second.flux[good])
    predicted = float(np.median(relative))

    assert 0.5 < measured / predicted < 2.0, (
        f"scatter {measured:.4f} against predicted {predicted:.4f}"
    )


def test_optimal_extraction_has_almost_nothing_to_recover_here(night):
    """On a bright spectrum the two methods agree, and that is the right
    answer rather than a disappointing one.

    Horne's weighting recovers what a plain sum throws away at the edges of
    the profile, where the signal is weak and the read noise is not. These are
    1200-second exposures of a naked-eye-adjacent star at signal-to-noise near
    90 per pixel: the noise is photons, and there is nothing to recover. A
    large gain here would mean the error model was wrong, not that the
    extraction was clever -- which is exactly what an earlier version of this
    test was quietly certifying.
    """
    plain = extract(night, "sp0019", weights="none")
    optimal = extract(night, "sp0019", weights="variance")

    good = np.isfinite(plain.flux) & (plain.flux > 200)
    assert np.median(optimal.flux[good]) == pytest.approx(
        float(np.median(plain.flux[good])), rel=0.01), \
        "weighting must change the noise and not the answer"
    assert not np.allclose(optimal.flux, plain.flux), \
        "identical to the last bit would mean the profile came from the data"

    # IRAF has no sigma for a plain sum. What the weighted one claims can be
    # held against the photon limit instead: on a spectrum this bright it
    # sits just above sqrt(counts), the sky and read noise adding little.
    photon = np.sqrt(optimal.flux[good] / GAIN)
    ratio = float(np.median(optimal.error[good] / photon))
    assert 1.0 < ratio < 1.15, (
        f"sigma {ratio:.3f} x the photon limit: well above it would mean the "
        "error model was wrong; below it, impossible")


def test_the_extracted_file_says_everything_about_itself(night):
    from astropy.io import fits

    extract(night, "sp0019")
    path = night / "sp0019.ms.fits"

    header = fits.getheader(path, ext=1)
    assert header["EXTNAME"] == "SPEC" and header["EXTVER"] == 1
    assert "APCENTER" in header and "APLOW" in header and "APHIGH" in header
    assert header["APTITLE"].startswith("DHCep")

    primary = fits.getheader(path)
    assert primary["HRWDSPEC"] == 1
    assert "psp0019" in primary["HRWDFROM"]
    # A card holds 80 columns; a long path goes to HISTORY, which wraps, rather
    # than losing its end to make room for a comment.
    assert all(len(str(card)) <= 80 for card in primary.cards)
    assert any("extracted from" in line for line in primary["HISTORY"])


def test_wspectext_turns_it_into_something_anyone_can_read(night):
    extract(night, "sp0019", weights="variance")
    get_task("wspectext").run(input=str(night / "sp0019.ms.fits"),
                              output=night / "sp0019.csv")

    lines = (night / "sp0019.csv").read_text(encoding="utf-8").splitlines()
    comments = [line for line in lines if line.startswith("#")]
    rows = [line for line in lines if not line.startswith("#")]

    assert any("DHCep" in line for line in comments)
    assert any("uncertainty on the flux" in line for line in comments)
    assert rows[0].split(",")[:3] == ["pixel", "flux", "error"]
    assert len(rows) == 1025, "a header row and 1024 pixels"

    # The numbers in the text are the numbers in the FITS.
    star = spectrum.read(night / "sp0019.ms.fits")[0]
    first = [float(value) for value in rows[1].split(",")]
    assert first[0] == pytest.approx(star.pixel[0])
    assert first[1] == pytest.approx(star.flux[0], rel=1e-5)


# ------------------------------------------- the night in time, not in files

#: The exposures the tutorial reduces, with the airmass the 1992 acquisition
#: system wrote into each header at the moment the shutter opened.
LOGGED = {"sp0018": 1.12, "sp0019": 1.52, "sp0020": 1.61, "sp0021": 1.72,
          "sp0022": 1.80, "sp0023": 1.91, "sp0024": 2.04}


@pytest.fixture(scope="module")
def stamped(tmp_path_factory):
    """The object frames and both comparison lamps, with airmasses computed."""
    from harwood.core.registry import get_task

    work = tmp_path_factory.mktemp("night")
    for name in LOGGED:
        (work / f"{name}.fits").write_bytes((SPEC / f"f{name}").read_bytes())
    (work / "sp0025.fits").write_bytes((SPEC / "fsp0025").read_bytes())

    get_task("setairmass").run(images=[str(work / "sp00*.fits")],
                               observatory="kpno")
    return work


def test_the_airmass_at_the_start_reproduces_the_1992_header(stamped):
    """The check that says the whole time chain is right.

    Nothing here is fitted to anything: the site comes from a table, the date
    from `25/01/92`, the coordinates from RA and DEC, and the answer has to
    match a number somebody else's software wrote thirty years ago. It does, to
    better than 0.01 airmass across a factor of two in airmass -- which is the
    precision the header itself is quoted to.
    """
    from astropy.io import fits

    for name, logged in LOGGED.items():
        header = fits.getheader(stamped / f"{name}.fits")
        assert header["AMSTART"] == pytest.approx(logged, abs=0.01), name


def test_the_effective_airmass_is_higher_than_the_one_the_observer_logged(stamped):
    """Twenty minutes is long enough for the star to sink, and this is the
    reason setairmass exists rather than the header being enough."""
    from astropy.io import fits

    header = fits.getheader(stamped / "sp0019.fits")
    assert header["AMSTART"] == pytest.approx(1.52, abs=0.01)
    assert header["AIRMASS"] > header["AMSTART"] + 0.03
    assert header["AIRMASS"] < header["AMEND"]
    assert header["UTMIDDLE"].startswith("1992-01-25T02:18"), "02:08:22 plus half of 1200 s"


def test_the_comparison_lamps_bracket_the_night(stamped):
    """The two arcs are not near each other in time -- that is the point of
    there being two -- and every exposure falls between them."""
    from astropy.io import fits

    from harwood.core import obstime

    def when(name):
        return obstime.sortable(obstime.middle(fits.getheader(stamped / f"{name}.fits")))

    first, last = when("sp0018"), when("sp0025")
    assert last - first > 2.5, "nearly three hours apart"
    for name in ("sp0019", "sp0022", "sp0024"):
        assert first < when(name) < last


def test_refspectra_weights_the_two_arcs_across_the_real_night(stamped):
    """Six exposures between two arcs: the weight on the evening arc has to
    fall, exposure by exposure, from nearly all of it to nearly none."""
    import numpy as np

    from harwood.core import spectrum
    from harwood.core.registry import get_task

    names = ["sp0019", "sp0020", "sp0021", "sp0022", "sp0023", "sp0024"]
    for name in names:
        from astropy.io import fits

        header = fits.getheader(stamped / f"{name}.fits")
        ap = spectrum.Aperture(number=1, centre=55.0, low=-8.0, high=8.0,
                               pixel=np.arange(1.0, 1025.0),
                               flux=np.ones(1024), error=np.ones(1024))
        spectrum.write(stamped / f"{name}.ms.fits", [ap], header=header)

    result = get_task("refspectra").run(
        input=[str(stamped / "sp00*.ms.fits")],
        references=[str(stamped / "sp0018.fits"), str(stamped / "sp0025.fits")])

    rows = {row["spectrum"]: row for row in result.data["rows"]}
    assert len(rows) == len(names)

    weights = [float(rows[f"{name}.ms.fits"]["weights"].split()[0])
               for name in names]
    assert all(row["arc"] == "sp0018 + sp0025" for row in rows.values())
    assert weights == sorted(weights, reverse=True), \
        "the evening arc matters less as the night goes on"
    assert weights[0] > 0.6 and weights[-1] < 0.15
