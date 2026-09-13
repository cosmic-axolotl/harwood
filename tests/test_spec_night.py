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
GAIN, RDNOISE = 3.0, 5.0


@pytest.fixture(scope="module")
def night(tmp_path_factory):
    """Bias-corrected object frames, as exercise 3 would leave them."""
    work = tmp_path_factory.mktemp("spec")
    for name in ("fsp0001", "fsp0002", "fsp0003", "fsp0019", "fsp0020",
                 "fsp0022", "fsp0023"):
        (work / f"{name[1:]}.fits").write_bytes((SPEC / name).read_bytes())

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
                   output=work / f"{name}.ms.fits", gain=GAIN, rdnoise=RDNOISE,
                   nsum=20, t_step=20)
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
    profile = apextract.profile(data, dispaxis=2)
    centre = apextract.find(profile, nfind=1, threshold=5.0)[0]["centre"]
    fit = apextract.trace(data, centre, dispaxis=2, step=20, nsum=20)

    assert fit.model is not None, fit.message
    assert fit.rms < 0.1, "a bright spectrum traces to a fraction of a pixel"
    ends = fit(np.array([1.0, 1024.0]))
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


def test_optimal_extraction_beats_the_plain_sum_without_moving_the_flux(night):
    plain = extract(night, "sp0019", weights="none")
    optimal = extract(night, "sp0019", weights="variance")

    good = np.isfinite(plain.flux) & (plain.flux > 200)
    assert np.median(optimal.flux[good]) == pytest.approx(
        float(np.median(plain.flux[good])), rel=0.05), \
        "weighting must change the noise and not the answer"

    plain_snr = float(np.median(plain.flux[good] / plain.error[good]))
    optimal_snr = float(np.median(optimal.flux[good] / optimal.error[good]))
    assert optimal_snr > plain_snr


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
    extract(night, "sp0019")
    get_task("wspectext").run(input=str(night / "sp0019.ms.fits"),
                              output=night / "sp0019.csv")

    lines = (night / "sp0019.csv").read_text().splitlines()
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
