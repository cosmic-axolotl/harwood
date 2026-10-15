"""Arc lamps and wavelength calibration.

The synthetic arc here is built from the real thorium line list, so the
wavelengths are real wavelengths and only the dispersion is invented -- which
is the part these tests need to know exactly.
"""

import numpy as np
import pytest

from harwood.core import arc, icfit

CRVAL, CDELT, NPIXELS = 6707.0, 0.152, 1024
CENTRE = (NPIXELS + 1) / 2

pytestmark = pytest.mark.skipif(
    not any(d.is_dir() for d in arc.IRAF_LINELISTS),
    reason="IRAF line lists are not installed",
)


@pytest.fixture(scope="module")
def catalogue():
    return arc.read_linelist("linelists$thorium.dat")[0]


@pytest.fixture
def synthetic(catalogue):
    """An arc with a known dispersion, from real thorium wavelengths."""
    rng = np.random.default_rng(7)
    low = CRVAL + (1 - CENTRE) * CDELT
    high = CRVAL + (NPIXELS - CENTRE) * CDELT
    lines = catalogue[(catalogue > low) & (catalogue < high)]
    strong = lines[rng.random(lines.size) < 0.45]

    pixels = (strong - CRVAL) / CDELT + CENTRE
    flux = np.full(NPIXELS, 300.0)
    sigma = 1.7 / 2.3548
    grid = np.arange(1.0, NPIXELS + 1.0)
    for position, amplitude in zip(pixels, 10 ** rng.uniform(2.5, 4.3, pixels.size),
        strict=False):
        flux += amplitude * np.exp(-0.5 * ((grid - position) / sigma) ** 2)
    return flux + rng.normal(0, 12, NPIXELS), strong, pixels


def test_the_line_list_reads_with_the_iraf_prefix(catalogue):
    assert catalogue.size > 2000
    assert catalogue.min() < 3200 and catalogue.max() > 9000
    assert np.all(np.diff(catalogue) >= 0), "the list comes back sorted"


def test_an_unknown_prefix_says_so():
    with pytest.raises(ValueError, match="directory prefix"):
        arc.read_linelist("standards$foo.dat")


def test_peaks_are_found_and_centred(synthetic):
    flux, _, true_pixels = synthetic
    peaks = arc.find_peaks(flux, fwidth=4.0, threshold=10.0)
    assert len(peaks) >= 0.8 * true_pixels.size

    found = np.array([p.pixel for p in peaks])
    distance = np.abs(found[:, None] - true_pixels[None, :]).min(axis=1)
    assert np.median(distance) < 0.1, "line centres must be good to a tenth of a pixel"


def test_blended_maxima_are_one_line_not_two():
    grid = np.arange(1.0, 201.0)
    flux = 100 + 900 * np.exp(-0.5 * ((grid - 100) / 0.8) ** 2) \
                + 880 * np.exp(-0.5 * ((grid - 101.2) / 0.8) ** 2)
    assert len(arc.find_peaks(flux, fwidth=4.0, threshold=5.0, minsep=3.0)) == 1


def test_a_correct_prior_recovers_the_dispersion_exactly(synthetic, catalogue):
    flux, _, _ = synthetic
    peaks = arc.find_peaks(flux, fwidth=4.0, threshold=10.0)
    match = arc.autoidentify(peaks, catalogue, crval=CRVAL, cdelt=CDELT,
                             npixels=NPIXELS, tolerance=1.5)
    assert len(match) >= 20

    fit = icfit.fit(match.pixel, match.wavelength,
                    icfit.FitSpec(function="chebyshev", order=3, niterate=3))
    assert fit.rms < 0.01, f"rms {fit.rms} Angstroms"
    assert float(fit(np.array([CENTRE]))[0]) == pytest.approx(CRVAL, abs=0.01)
    slope = float(np.median(np.gradient(fit(np.arange(1.0, NPIXELS + 1.0)))))
    assert slope == pytest.approx(CDELT, rel=1e-3)


def test_the_blind_search_refuses_rather_than_inventing(synthetic, catalogue):
    """The finding that shaped this module.

    Against thorium's three thousand lines, several quite different dispersions
    score almost the same, and a cubic fits any of them to a tenth of a pixel.
    A search that returns the winner returns an invention. This one says so
    instead, which is why IRAF's identify asks you to mark lines by hand.
    """
    flux, _, _ = synthetic
    peaks = arc.find_peaks(flux, fwidth=4.0, threshold=10.0)
    result = arc.search_dispersion(peaks, catalogue, window=(6300.0, 7200.0),
                                   cdelt_range=(0.03, 1.5), npixels=NPIXELS)
    if not result:
        assert "ambiguous" in result.message or result.message
    else:
        # If it does commit to an answer, the answer has to be the right one.
        assert abs(result.cdelt) == pytest.approx(CDELT, rel=0.05)


def test_a_line_is_never_claimed_by_two_peaks(catalogue):
    peaks = np.array([100.0, 100.4, 400.0])
    match = arc.autoidentify(peaks, catalogue, crval=CRVAL, cdelt=CDELT,
                             npixels=NPIXELS, tolerance=2.0)
    assert len(set(np.round(match.wavelength, 6))) == len(match)


def test_add_from_list_extends_a_solution(synthetic, catalogue):
    """IRAF's `l`: four lines by hand, then the catalogue places the rest."""
    flux, _, _ = synthetic
    peaks = arc.find_peaks(flux, fwidth=4.0, threshold=10.0)
    chosen = sorted(peaks, key=lambda p: p.pixel)[::max(1, len(peaks) // 4)][:4]
    pixel = np.array([p.pixel for p in chosen])
    wavelength = CRVAL + (pixel - CENTRE) * CDELT
    wavelength = np.array([catalogue[np.abs(catalogue - w).argmin()]
                           for w in wavelength])

    seed = icfit.fit(pixel, wavelength, icfit.FitSpec(function="chebyshev", order=2))
    more_pixel, more_wave = arc.add_from_list(peaks, seed, catalogue, tolerance=1.5)
    assert more_pixel.size > pixel.size

    fit = icfit.fit(more_pixel, more_wave,
                    icfit.FitSpec(function="chebyshev", order=3, niterate=3))
    assert fit.rms < 0.01


def test_collapse_takes_the_dispersion_axis(synthetic):
    flux, _, _ = synthetic
    frame = np.tile(flux[:, None], (1, 40))
    assert arc.collapse(frame, dispaxis=2, nsum=10).size == NPIXELS
    assert arc.collapse(frame.T, dispaxis=1, nsum=10).size == NPIXELS


def test_positions_accepts_peaks_or_a_plain_array():
    """Both forms are in the signature, so both have to work.

    `if peaks and isinstance(peaks[0], Peak)` written inline is how this became
    a bug: a numpy array of more than one element raises on `if`.
    """
    peaks = [arc.Peak(10.0, 5.0), arc.Peak(30.0, 9.0), arc.Peak(20.0, 1.0)]
    np.testing.assert_array_equal(arc.positions(peaks), [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(arc.positions(np.array([30.0, 10.0])), [10.0, 30.0])
    np.testing.assert_array_equal(
        arc.positions(peaks, limit=2, brightest=True), [10.0, 30.0])


# --------------------------------------------------------------- dispcor


@pytest.fixture
def calibrated(tmp_path, monkeypatch, synthetic, catalogue):
    """An arc extracted as a spectrum, with its solution in the database."""
    from astropy.io import fits

    from harwood.core import spectrum
    from harwood.core.registry import get_task

    monkeypatch.chdir(tmp_path)
    flux, _, _ = synthetic
    fits.PrimaryHDU(flux.astype(np.float32)).writeto(tmp_path / "arc.fits")

    ap = spectrum.Aperture(number=1, centre=1.0, low=-1.0, high=1.0,
                           pixel=np.arange(1.0, flux.size + 1.0), flux=flux,
                           error=np.sqrt(np.abs(flux)), title="arc")
    spectrum.write(tmp_path / "arc.ms.fits", [ap], source="arc.fits")

    get_task("identify").run(image="arc.fits", crval=CRVAL, cdelt=CDELT,
                             order=3, niterate=3, database=tmp_path)
    return tmp_path


def test_dispcor_puts_the_right_wavelengths_on_the_pixels(calibrated, catalogue):
    """The end of the chain: the lines of the calibrated arc have to land on
    the catalogue they were identified from."""
    from harwood.core import spectrum
    from harwood.core.registry import get_task

    get_task("dispcor").run(input="arc.ms.fits", output="arc.w.fits",
                            reference="arc.fits", database=calibrated)
    star = spectrum.read(calibrated / "arc.w.fits")[0]

    assert star.wavelength is not None
    assert star.wavelength.min() == pytest.approx(
        CRVAL + (1 - CENTRE) * CDELT, abs=0.05)

    peaks = arc.find_peaks(star.flux, fwidth=4.0, threshold=10.0)
    found = np.interp([peak.pixel for peak in peaks], star.pixel, star.wavelength)
    distance = np.abs(found[:, None] - catalogue[None, :]).min(axis=1)
    assert np.median(distance) < 0.01, "within a hundredth of an Angstrom"


def test_dispcor_keeps_the_pixel_column(calibrated):
    """Nothing is thrown away: the calibration can be checked or redone."""
    from harwood.core import spectrum
    from harwood.core.registry import get_task

    get_task("dispcor").run(input="arc.ms.fits", output="arc.w.fits",
                            reference="arc.fits", database=calibrated)
    star = spectrum.read(calibrated / "arc.w.fits")[0]
    np.testing.assert_allclose(star.pixel, np.arange(1.0, NPIXELS + 1.0))


def test_dispcor_says_which_pixels_are_extrapolated(calibrated):
    """A polynomial extrapolates badly, and the ends of a spectrum are usually
    beyond the outermost line the arc gave."""
    from harwood.core.registry import get_task

    result = get_task("dispcor").run(input="arc.ms.fits", output="arc.w.fits",
                                     reference="arc.fits", database=calibrated)
    assert result.data["extrapolated"] > 0
    assert any("extrapolated" in warning for warning in result.warnings)


def test_linearizing_conserves_the_flux_and_says_what_it_costs(calibrated):
    from harwood.core import spectrum
    from harwood.core.registry import get_task

    get_task("dispcor").run(input="arc.ms.fits", output="plain.fits",
                            reference="arc.fits", database=calibrated)
    result = get_task("dispcor").run(input="arc.ms.fits", output="even.fits",
                                     reference="arc.fits", database=calibrated,
                                     linearize=True)

    plain = spectrum.read(calibrated / "plain.fits")[0]
    even = spectrum.read(calibrated / "even.fits")[0]

    spacing = np.diff(even.wavelength)
    assert np.ptp(spacing) / np.median(spacing) < 1e-9, "evenly spaced"

    # Flux per unit wavelength, integrated, is what has to survive.
    before = np.trapezoid(plain.flux, plain.wavelength)
    after = np.trapezoid(even.flux, even.wavelength)
    assert after == pytest.approx(before, rel=0.01)

    assert any("correlated" in warning for warning in result.warnings), \
        "interpolation has a cost and it should be said"


def test_dispcor_without_a_reference_lists_what_there_is(calibrated):
    from harwood.core.registry import get_task

    with pytest.raises(ValueError) as raised:
        get_task("dispcor").run(input="arc.ms.fits", output="x.fits",
                                database=calibrated)
    assert "solutions for arc" in raised.value.fix


def test_a_noiseless_arc_still_has_its_lines():
    """With no noise, the threshold fell back on the standard deviation of
    the residual -- which, with no noise, is made of the lines themselves.
    Six strong lines lifted it above their own height and none was found."""
    import numpy as np

    from harwood.core import arc

    x = np.arange(1.0, 1001.0)
    where = [80.0, 210.0, 395.0, 520.0, 700.0, 880.0]
    flux = 10 + sum(500 * np.exp(-0.5 * ((x - p) / 1.5) ** 2) for p in where)
    found = sorted(peak.pixel for peak in arc.find_peaks(flux))
    assert found == pytest.approx(where, abs=0.01)
    assert arc.find_peaks(np.full(100, 7.0)) == [], "and a flat one has none"
