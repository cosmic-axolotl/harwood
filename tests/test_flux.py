"""Flux calibration: standard, sensfunc, calibrate.

These are closed-loop tests. An instrument response and an extinction curve
are invented, a standard star is "observed" through them, and the chain is
asked to take them back out. The true answer is known exactly, so the question
is not whether the numbers look plausible but whether they are right.

The harder half is the second star: a different object, at a different
airmass, calibrated with the response fitted from the first. That is the claim
flux calibration actually makes, and it is the one that fails if the
extinction and the instrument are not properly separated.
"""

import numpy as np
import pytest

from harwood.core import flux

pytestmark = pytest.mark.skipif(
    not any(d.is_dir() for d in flux.IRAF_STANDARDS),
    reason="IRAF's onedstds are not installed",
)

STAR = "bd284211"
CALDIR = "spec50cal"
EXTINCTION = "onedstds$kpnoextinct.dat"

#: The instrument's response, in magnitudes: a broad hump peaking in the
#: middle of the range, which is what a real grating-plus-detector looks like.
#: Invented here so the chain can be asked to recover it.
PEAK, WIDTH, HEIGHT, FLOOR = 5200.0, 1400.0, 2.1, 38.0


def response(wavelength):
    wavelength = np.asarray(wavelength, dtype=float)
    return FLOOR + HEIGHT * np.exp(-0.5 * ((wavelength - PEAK) / WIDTH) ** 2)


@pytest.fixture(scope="module")
def curve():
    return flux.read_extinction(EXTINCTION)


@pytest.fixture(scope="module")
def bands():
    path = flux.find_star(STAR, CALDIR)
    assert path is not None, f"{STAR} should be in {CALDIR}"
    return flux.read_bands(path)


def true_flux(wavelength, bands):
    """The standard's flux density, interpolated between its bands."""
    grid = np.array([band.wavelength for band in bands])
    values = np.array([band.flux for band in bands])
    return np.interp(np.asarray(wavelength, dtype=float), grid, values)


def observe(wavelength, real_flux, curve, airmass, exptime, seed=None):
    """What the telescope would record, given the invented response: the
    response is IRAF's sensitivity, 2.5 log10 of counts per second per
    Angstrom over the flux, so a pixel holds its width's worth."""
    losses = 10.0 ** (-0.4 * flux.extinction_at(wavelength, curve) * airmass)
    width = np.abs(np.gradient(np.asarray(wavelength, dtype=float)))
    counts = real_flux * 10.0 ** (0.4 * response(wavelength)) * exptime * losses * width
    if seed is not None:
        counts = counts + np.random.default_rng(seed).normal(
            0, 0.004 * np.median(counts), counts.size)
    return counts


def make_spectrum(path, wavelength, counts, name, header=None):
    from astropy.io import fits

    from harwood.core import spectrum

    ap = spectrum.Aperture(
        number=1, centre=50.0, low=-8.0, high=8.0,
        pixel=np.arange(1.0, wavelength.size + 1.0), wavelength=wavelength,
        flux=counts, error=np.sqrt(np.abs(counts)) + 1.0, title=name)
    spectrum.write(path, [ap], header=fits.Header(header or {}))
    return path


# ----------------------------------------------------------- the arithmetic


def test_a_calibration_magnitude_becomes_the_published_flux(bands):
    """BD+28 4211 at 3200 A is 9.63 mag in these tables, and the flux that
    corresponds to is about 1.5e-12 erg/cm2/s/A -- the published value."""
    first = bands[0]
    assert first.wavelength == 3200.0
    assert first.flux == pytest.approx(1.515e-12, rel=0.01)


def test_magnitudes_and_fluxes_convert_back(bands):
    for band in bands[::20]:
        back = flux.flux_to_magnitude(band.flux, band.wavelength)
        assert back == pytest.approx(band.magnitude, abs=1e-9)


def test_the_extinction_curve_is_steeper_in_the_blue(curve):
    """Rayleigh scattering goes as the fourth power of the wavelength, so the
    blue end has to cost more magnitudes per airmass than the red."""
    blue, red = flux.extinction_at([3500.0], curve), flux.extinction_at([7000.0], curve)
    assert blue[0] > red[0] > 0
    assert blue[0] > 3 * red[0]


def test_the_extinction_is_interpolated_as_iraf_does(curve):
    """IRAF's intrp: on the table's points the table's values, between them
    a blend of the two neighbouring parabolas -- not a straight line -- and
    past the ends the end parabola continued, which is why calibrate says
    so when a spectrum leaves the table."""
    grid, values = curve
    assert flux.extinction_at([grid[3]], curve)[0] == pytest.approx(values[3], abs=1e-6)
    middle = 0.5 * (grid[3] + grid[4])
    assert flux.extinction_at([middle], curve)[0] != pytest.approx(
        0.5 * (values[3] + values[4]), abs=1e-6)
    beyond = flux.extinction_at([grid[0] - 500.0], curve)[0]
    assert np.isfinite(beyond) and beyond != values[0]


def test_a_band_is_integrated_over_its_whole_width():
    """The trap that cost four per cent.

    Integrating only over the samples that fall inside the band covers about
    one pixel less than the band; dividing that by the *full* width then
    understates every band by the same fraction. At fifty Angstroms and two
    Angstroms per pixel that is four per cent -- a flat offset in the fitted
    response and a flat error in every calibrated flux, and invisible in the
    rms because every band is wrong by the same amount.

    A spectrum of constant counts must measure exactly that many counts per
    Angstrom of band -- the counts per pixel over the pixel's width -- per
    second, whatever the sampling and wherever the band edges fall between
    samples. That is IRAF's std_flux: whole pixels and the fractions of the
    two at the ends.
    """
    from harwood.core import spectrum

    for spacing, offset in ((2.0, 0.0), (2.0, 0.9), (7.0, 3.1), (0.5, 0.0)):
        wavelength = np.arange(4000.0 + offset, 6000.0, spacing)
        ap = spectrum.Aperture(
            number=1, centre=1.0, low=-1.0, high=1.0,
            pixel=np.arange(1.0, wavelength.size + 1.0), wavelength=wavelength,
            flux=np.full(wavelength.size, 500.0),
            error=np.ones(wavelength.size))
        band = flux.Band(wavelength=5000.0, width=50.0, magnitude=10.0)

        measured, = flux.measure(ap, [band], exptime=2.0, airmass=1.0)
        # Single precision, as IRAF sums.
        assert measured.counts == pytest.approx(500.0 / spacing / 2.0, rel=1e-5), \
            f"spacing {spacing}, offset {offset}"


def test_a_band_only_half_covered_is_dropped_not_scaled(bands):
    """A half-covered band looks exactly like a star half as bright, and
    afterwards nothing can tell the two apart."""
    from harwood.core import spectrum

    wavelength = np.linspace(4000.0, 5000.0, 800)
    ap = spectrum.Aperture(number=1, centre=1.0, low=-1.0, high=1.0,
                           pixel=np.arange(1.0, 801.0), wavelength=wavelength,
                           flux=np.ones(800), error=np.ones(800))
    measured = flux.measure(ap, bands, exptime=1.0, airmass=1.0)

    assert measured, "the middle bands are fully covered"
    for item in measured:
        assert item.wavelength - item.width / 2 >= 4000.0
        assert item.wavelength + item.width / 2 <= 5000.0


# ------------------------------------------------------------- the chain


@pytest.fixture
def night(tmp_path, monkeypatch, bands, curve):
    """A standard observed at 1.25 airmasses through the invented response."""
    monkeypatch.chdir(tmp_path)
    wavelength = np.linspace(3300.0, 7500.0, 2100)
    real = true_flux(wavelength, bands)
    counts = observe(wavelength, real, curve, airmass=1.25, exptime=300.0, seed=4)

    make_spectrum(tmp_path / "std.ms.fits", wavelength, counts, STAR,
                  {"OBJECT": STAR, "AIRMASS": 1.25, "EXPTIME": 300.0})
    return tmp_path, wavelength, real


def test_standard_measures_the_bands_it_can_reach(night):
    from harwood.core.registry import get_task

    work, _, _ = night
    result = get_task("standard").run(input="std.ms.fits", output=work / "std",
                                      caldir=CALDIR)

    rows = result.data["rows"]
    assert len(rows) > 40, "most of a 3300-7500 A range in 50 A bands"
    assert all(3300 <= row["wavelength"] <= 7500 for row in rows)


def test_standard_reads_the_star_and_the_airmass_from_the_header(night):
    """Nothing was given on the command line: the name came from OBJECT and
    the airmass from the header setairmass writes."""
    from harwood.core import stdfile
    from harwood.core.registry import get_task

    work, _, _ = night
    get_task("standard").run(input="std.ms.fits", output=work / "std",
                             caldir=CALDIR)

    standard = stdfile.read(work / "std")
    assert {item.star for item in standard.measurements} == {STAR}
    assert all(item.airmass == pytest.approx(1.25) for item in standard.measurements)


def test_sensfunc_recovers_the_response_that_was_put_in(night):
    """The heart of it. The response was invented; the fit has to find it.

    The comparison is against the invented curve at each band, not against the
    fit's own scatter -- a fit can be beautifully self-consistent and wrong.
    """
    from harwood.core import database, icfit
    from harwood.core.registry import get_task

    work, _, _ = night
    get_task("standard").run(input="std.ms.fits", output=work / "std",
                             caldir=CALDIR)
    result = get_task("sensfunc").run(standards=work / "std", sensitivity="s",
                                      database=work, order=7)

    solution = database.read("s", work, prefix="sens")
    fitted = icfit.fit(solution.pixel, solution.wavelength,
                       icfit.FitSpec(function=solution.function,
                                     order=solution.order))
    grid = np.linspace(3400.0, 7400.0, 300)
    assert np.max(np.abs(fitted(grid) - response(grid))) < 0.05, \
        "the fitted response has to be the one that was put in"
    assert result.data["rms"] < 0.02


def test_the_rms_is_reported_as_a_percentage_of_flux(night):
    """Magnitudes hide how large a systematic error is, and this one is the
    floor under every spectrum the solution calibrates."""
    from harwood.core.registry import get_task

    work, _, _ = night
    get_task("standard").run(input="std.ms.fits", output=work / "std",
                             caldir=CALDIR)
    result = get_task("sensfunc").run(standards=work / "std", sensitivity="s",
                                      database=work)

    assert any("per cent in flux" in warning for warning in result.warnings)
    assert any("not reduced by observing the object for longer" in warning
               for warning in result.warnings)


def test_calibrate_returns_the_standard_to_its_published_flux(night):
    """Round trip: observe a star of known flux, then calibrate it back."""
    from harwood.core import spectrum
    from harwood.core.registry import get_task

    work, wavelength, real = night
    get_task("standard").run(input="std.ms.fits", output=work / "std",
                             caldir=CALDIR)
    get_task("sensfunc").run(standards=work / "std", sensitivity="s",
                             database=work, order=7)
    get_task("calibrate").run(input="std.ms.fits", output=work / "std.f.fits",
                              sensitivity="s", extinction=EXTINCTION,
                              database=work)

    got = spectrum.read(work / "std.f.fits")[0]
    inside = (wavelength > 3500) & (wavelength < 7300)
    ratio = got.flux[inside] / real[inside]

    assert np.median(ratio) == pytest.approx(1.0, abs=0.02)
    assert np.std(ratio) < 0.05, "no wavelength-dependent residual"


def test_a_second_star_at_a_different_airmass_comes_out_right(night, bands, curve):
    """The claim flux calibration actually makes.

    A different object, observed at a different airmass, calibrated with the
    response fitted from the standard. If the atmosphere and the instrument
    were not properly separated this is where it shows: the error would be a
    smooth tilt with wavelength, largest in the blue, exactly where an
    extinction mistake lives.
    """
    from astropy.io import fits

    from harwood.core import spectrum
    from harwood.core.registry import get_task

    work, wavelength, _ = night
    get_task("standard").run(input="std.ms.fits", output=work / "std",
                             caldir=CALDIR)
    get_task("sensfunc").run(standards=work / "std", sensitivity="s",
                             database=work, order=7)

    # A blackbody at 9000 K, observed at 1.9 airmasses for twice as long.
    target = 3.0e-13 * (5500.0 / wavelength) ** 4 / (
        np.exp(1.4388e8 / (wavelength * 9000.0)) - 1.0) * 1e4
    counts = observe(wavelength, target, curve, airmass=1.9, exptime=600.0)
    make_spectrum(work / "obj.ms.fits", wavelength, counts, "the object",
                  {"OBJECT": "target", "AIRMASS": 1.9, "EXPTIME": 600.0})

    get_task("calibrate").run(input="obj.ms.fits", output=work / "obj.f.fits",
                              sensitivity="s", extinction=EXTINCTION,
                              database=work)

    got = spectrum.read(work / "obj.f.fits")[0]
    inside = (wavelength > 3500) & (wavelength < 7300)
    ratio = got.flux[inside] / target[inside]

    assert np.median(ratio) == pytest.approx(1.0, abs=0.03)
    # A tilt is the signature of an extinction error; check the ends agree
    # with each other and not just the middle.
    blue = float(np.median(ratio[wavelength[inside] < 4200]))
    red = float(np.median(ratio[wavelength[inside] > 6500]))
    assert abs(blue - red) < 0.05, f"blue {blue:.3f} against red {red:.3f}"

    header = fits.getheader(work / "obj.f.fits")
    assert header["BUNIT"] == flux.FLUX_UNIT


def test_ignoring_the_airmass_shows_up_as_exactly_that_tilt(night, bands, curve):
    """The negative control: if the extinction is *not* corrected, the error
    has to be a blue-to-red tilt. A test that only checks the right answer
    cannot tell a correct pipeline from one whose errors cancel."""
    from harwood.core import spectrum
    from harwood.core.registry import get_task

    work, wavelength, real = night
    get_task("standard").run(input="std.ms.fits", output=work / "std",
                             caldir=CALDIR)
    get_task("sensfunc").run(standards=work / "std", sensitivity="s",
                             database=work, order=7)
    result = get_task("calibrate").run(
        input="std.ms.fits", output=work / "flat.fits", sensitivity="s",
        extinction=EXTINCTION, extinct=False, database=work)

    got = spectrum.read(work / "flat.fits")[0]
    inside = (wavelength > 3500) & (wavelength < 7300)
    ratio = got.flux[inside] / real[inside]
    blue = float(np.median(ratio[wavelength[inside] < 4200]))
    red = float(np.median(ratio[wavelength[inside] > 6500]))

    assert blue < 0.8 * red, "an uncorrected atmosphere eats the blue"
    assert not any("BUNIT" in warning for warning in result.warnings)


# ------------------------------------------------------- when it goes wrong


def test_running_standard_twice_builds_one_fit_from_two_stars(night, bands, curve):
    """The reason the file appends: one star rarely covers the whole range."""
    from harwood.core import stdfile
    from harwood.core.registry import get_task

    work, _, _ = night
    get_task("standard").run(input="std.ms.fits", output=work / "std",
                             caldir=CALDIR)
    first = len(stdfile.read(work / "std"))

    other = flux.stars(CALDIR)
    name = next(item for item in other if item not in (STAR, "names"))
    other_bands = flux.read_bands(flux.find_star(name, CALDIR))
    wavelength = np.linspace(4000.0, 6000.0, 1200)
    real = true_flux(wavelength, other_bands)
    counts = observe(wavelength, real, curve, airmass=1.4, exptime=200.0)
    make_spectrum(work / "two.ms.fits", wavelength, counts, name,
                  {"OBJECT": name, "AIRMASS": 1.4, "EXPTIME": 200.0})

    get_task("standard").run(input="two.ms.fits", output=work / "std",
                             star_name=name, caldir=CALDIR)
    standard = stdfile.read(work / "std")

    assert len(standard) > first, "the second star was appended"
    assert len({item.star for item in standard.measurements}) == 2
    assert len({item.airmass for item in standard.measurements}) == 2


def test_a_misspelt_standard_says_where_it_actually_is(night):
    from harwood.core.registry import get_task

    work, _, _ = night
    with pytest.raises(ValueError, match="feige110|caldir"):
        get_task("standard").run(input="std.ms.fits", output=work / "x",
                                 star_name="feige110", caldir="blackbody")


def test_an_uncalibrated_spectrum_is_refused_with_the_reason(tmp_path, monkeypatch):
    from harwood.core import spectrum
    from harwood.core.registry import get_task

    monkeypatch.chdir(tmp_path)
    ap = spectrum.Aperture(number=1, centre=1.0, low=-1.0, high=1.0,
                           pixel=np.arange(1.0, 101.0), flux=np.ones(100),
                           error=np.ones(100))
    spectrum.write(tmp_path / "raw.fits", [ap])

    with pytest.raises(ValueError) as raised:
        get_task("standard").run(input="raw.fits", output=tmp_path / "x",
                                 star_name=STAR, caldir=CALDIR)
    assert "dispcor" in raised.value.fix


def test_no_airmass_explains_what_it_costs(night):
    from astropy.io import fits

    from harwood.core.registry import get_task

    work, _, _ = night
    with fits.open(work / "std.ms.fits", mode="update") as hdus:
        del hdus[0].header["AIRMASS"]

    with pytest.raises(ValueError) as raised:
        get_task("standard").run(input="std.ms.fits", output=work / "x",
                                 caldir=CALDIR)
    assert "setairmass" in raised.value.fix
    # What it costs, in the message rather than the suggestion: the reader
    # should see why before they see how.
    assert "30 per cent" in str(raised.value)


def test_calibrate_without_a_solution_says_how_to_make_one(night):
    from harwood.core.registry import get_task

    work, _, _ = night
    with pytest.raises(FileNotFoundError) as raised:
        get_task("calibrate").run(input="std.ms.fits", output=work / "x.fits",
                                  sensitivity="nothing", database=work)
    assert "sensitivity function" in str(raised.value)
    assert "standard on a flux standard" in raised.value.fix


def test_the_standards_file_says_what_it_is(night):
    from harwood.core.registry import get_task

    work, _, _ = night
    get_task("standard").run(input="std.ms.fits", output=work / "std",
                             caldir=CALDIR)

    text = (work / "std").read_text()
    assert "wavelength,width,magnitude,flux,counts,airmass,star,used" in text
    assert "erg/cm2/s/Angstrom" in text
    assert "kpnoextinct" in text
    # The formula it is all for, written down where somebody will find it.
    assert "2.5 log10(" in text
