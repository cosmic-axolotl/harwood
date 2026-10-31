"""Finding spectra, following them, adding them up, and writing them down.

Built on a synthetic frame where the trace, the width and the flux are known,
so the answers can be checked rather than merely be plausible. That the
arithmetic is IRAF's, number for number, is test_apall_iraf.py's job; this
file checks it does what a spectrum needs.
"""

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import apextract, icfit, spectrum
from harwood.core.registry import get_task

NY, NX = 400, 60            # 400 along the dispersion, 60 across the slit
CENTRE, WIDTH = 24.0, 2.0   # 1-based centre at the middle of the dispersion
DRIFT = 3.0                 # how far the spectrum wanders, end to end
SKY = 40.0


def synthetic(flux_per_pixel=2000.0, seed=5):
    """A spectrum that drifts across the slit, on a flat sky."""
    rng = np.random.default_rng(seed)
    rows = np.arange(1.0, NY + 1.0)
    across = np.arange(1.0, NX + 1.0)

    # Linear drift, so a first-order trace would do and a third-order one must
    # not invent structure.
    centres = CENTRE + DRIFT * (rows - (NY + 1) / 2) / NY
    profile = np.exp(-0.5 * ((across[None, :] - centres[:, None]) / WIDTH) ** 2)
    profile /= profile.sum(axis=1, keepdims=True)

    frame = SKY + flux_per_pixel * profile
    return frame + rng.normal(0, 3.0, frame.shape), centres


def band(inner, outer, low=-8.0, high=8.0):
    """A background sample `inner` to `outer` pixels outside each edge."""
    return icfit.FitSpec(function="chebyshev", order=1, naverage=-3,
                         sample=f"{low - outer:g}:{low - inner:g},"
                                f"{high + inner:g}:{high + outer:g}")


def traced(frame, low=-8.0, high=8.0, background=None):
    aperture = apextract.Aperture(1, CENTRE, low, high, NY // 2,
                                  background=background or band(4.0, 14.0, low, high))
    points = apextract.trace(frame, 2, [aperture], NY // 2)
    return aperture, points[0]


def test_the_profile_finds_the_spectrum():
    frame, _ = synthetic()
    profile, line = apextract.getdata(apextract.frame(frame, 2))
    found = apextract.find(profile, nfind=1)
    assert line == NY // 2
    assert found == [pytest.approx(CENTRE, abs=0.1)]


def test_two_spectra_closer_than_minsep_are_one():
    across = np.arange(1.0, 61.0)
    blended = ((np.exp(-0.5 * ((across - 30) / 2) ** 2)
                + np.exp(-0.5 * ((across - 36) / 2) ** 2)) * 1000 + 10).astype(np.float32)
    assert len(apextract.find(blended, nfind=5, minsep=8.0)) == 1
    assert len(apextract.find(blended, nfind=5, minsep=2.0)) == 2


def test_the_aperture_is_sized_from_the_profile():
    frame, _ = synthetic()
    profile, _ = apextract.getdata(apextract.frame(frame, 2))
    low, high = apextract.ylevel(profile, CENTRE, 0.05, low=-NX, high=NX)
    # A Gaussian falls to 5% of its peak at 2.45 sigma either side.
    assert -low == pytest.approx(2.45 * WIDTH, abs=0.5)
    assert high == pytest.approx(2.45 * WIDTH, abs=0.5)


def test_the_trace_follows_the_drift():
    """The point of tracing: the spectrum does not sit on one column."""
    frame, centres = synthetic()
    aperture, points = traced(frame)
    fit = aperture.trace.fit
    assert fit is not None and fit.model is not None
    assert fit.rms < 0.2, "a bright spectrum traces to a fraction of a pixel"
    ends = aperture.centre_at(np.array([1.0, float(NY)]))
    assert ends[1] - ends[0] == pytest.approx(DRIFT, abs=0.3)
    np.testing.assert_allclose(aperture.centre_at(points.positions), centres[
        points.positions.astype(int) - 1], atol=0.3)


def test_a_trace_that_cannot_be_followed_leaves_no_points():
    blank = np.full((NY, NX), SKY)
    _, points = traced(blank)
    assert points.positions.size < 2
    assert points.lost, "where it was lost is recorded"


def test_extraction_recovers_the_flux_that_was_put_in():
    frame, _ = synthetic(flux_per_pixel=2000.0)
    aperture, _ = traced(frame)
    out = apextract.extract(apextract.frame(frame, 2), [aperture], background="fit")[0]
    assert np.median(out.spec) == pytest.approx(2000.0, rel=0.03)
    assert np.median(out.sky) == pytest.approx(SKY * 16, rel=0.05), \
        "the background is summed over the 16 px aperture"
    assert out.sigma is None, "IRAF has a sigma only with variance weights"


def _uniform_spectrum(peak, seed=0, length=600, across=60, centre=30.0,
                      fwhm=3.5, sky=20.0, gain=2.0, rdnoise=6.0):
    """A spectrum of constant flux, so the truth is one number.

    Constant on purpose: the measured scatter of the extraction is then the
    extraction's own noise and nothing else, which is what these tests need to
    compare against.
    """
    rng = np.random.default_rng(seed)
    axis = np.arange(across) + 1.0
    sigma = fwhm / 2.3548
    shape = np.exp(-0.5 * ((axis - centre) / sigma) ** 2)
    shape /= shape.sum()
    clean = np.full(length, peak)[:, None] * shape[None, :] + sky
    counts = rng.poisson(np.clip(clean * gain, 0, None)) / gain
    return counts + rng.normal(0, rdnoise / gain, counts.shape)


def _both_ways(frame, gain=2.0, rdnoise=6.0):
    """A perfectly straight spectrum: the trace is not what is being tested."""
    out = {}
    for weights in ("none", "variance"):
        aperture = apextract.Aperture(1, 30.0, -8.0, 8.0, 300,
                                      background=band(10.0, 25.0))
        out[weights] = apextract.extract(apextract.frame(frame, 2), [aperture],
                                         background="fit", weights=weights,
                                         gain=gain, readnoise=rdnoise)[0]
    return out


def test_a_profile_taken_from_the_data_is_not_a_profile():
    """The regression test for a bug that hid for weeks.

    Horne's weighting needs a *model* of the spatial profile. Take the profile
    from the data at that wavelength -- P = D / sum(D) -- and the algebra
    collapses to the plain sum, for every input, while the error shrinks
    because the "profile" has absorbed the noise it was meant to weight
    against. IRAF fits the profile along the dispersion; this checks the
    fluxes differ, which no comparison of flux/error can.
    """
    both = _both_ways(_uniform_spectrum(1500.0, seed=5))
    assert not np.allclose(both["none"].spec, both["variance"].spec), \
        "identical fluxes mean the profile came from the data"


def test_optimal_extraction_recovers_the_flux_that_was_put_in():
    """Weighting must not move the answer. A scheme that improves the
    signal-to-noise *and* shifts the flux has measured something else."""
    truth = 3000.0
    for weights, result in _both_ways(_uniform_spectrum(truth, seed=7)).items():
        assert np.median(result.spec) == pytest.approx(truth, rel=0.02), weights


def test_optimal_extraction_earns_its_keep_on_a_faint_spectrum():
    """And barely at all on a bright one, which is the honest shape of it.

    Horne's method recovers what a plain sum throws away at the edges of the
    profile, where the signal is weak and the read noise is not. On a bright
    spectrum the noise is photons and there is nothing to recover; on a faint
    one it is worth most of a magnitude. Measured scatter, not the reported
    error: a wrong error estimate cannot flatter itself here.
    """
    def gain_at(peak, seed):
        both = _both_ways(_uniform_spectrum(peak, seed=seed))
        scatter = {name: float(np.std(r.spec) / np.median(r.spec))
                   for name, r in both.items()}
        return scatter["none"] / scatter["variance"]

    assert gain_at(20000.0, 3) < 1.10, "nothing to recover from a bright spectrum"
    assert gain_at(400.0, 3) > 1.30, "and a real gain on a faint one"


def test_the_reported_sigma_matches_the_scatter_it_claims_to_describe():
    """An uncertainty nobody has compared with a repeat measurement is
    decoration."""
    result = _both_ways(_uniform_spectrum(2000.0, seed=13))["variance"]
    measured = float(np.std(result.spec))
    claimed = float(np.median(result.sigma))
    assert 0.7 < measured / claimed < 1.4, f"scatter {measured:.1f}, sigma {claimed:.1f}"


def test_the_background_is_taken_from_beside_the_aperture_not_under_it():
    frame, _ = synthetic()
    fair_ap, _ = traced(frame, -6.0, 6.0, band(6.0, 18.0, -6.0, 6.0))
    tight_ap, _ = traced(frame, -6.0, 6.0, band(0.0, 4.0, -6.0, 6.0))
    data = apextract.frame(frame, 2)
    fair = apextract.extract(data, [fair_ap], background="fit")[0]
    tight = apextract.extract(data, [tight_ap], background="fit")[0]
    assert np.median(tight.sky) > np.median(fair.sky), \
        "sky measured on the wings of the profile is sky that includes the star"


# ------------------------------------------------------------ the format


@pytest.fixture
def written(tmp_path):
    pixel = np.arange(1.0, 51.0)
    apertures = [
        spectrum.Aperture(number=1, centre=24.0, low=-5.0, high=5.0, pixel=pixel,
                          flux=pixel * 2, error=np.sqrt(pixel), sky=pixel * 0 + 3,
                          raw=pixel * 2 + 30, title="a star",
                          meta={"EXPTIME": 1200.0}),
        spectrum.Aperture(number=2, centre=40.0, low=-4.0, high=4.0, pixel=pixel,
                          flux=pixel * 5, error=np.sqrt(pixel), sky=pixel * 0 + 3,
                          raw=pixel * 5 + 24, title="another"),
    ]
    return spectrum.write(tmp_path / "spec.ms.fits", apertures,
                          source="frame.fits")


def test_one_table_per_aperture(written):
    with fits.open(written) as hdul:
        names = [hdu.header.get("EXTNAME", "") for hdu in hdul]
    assert names.count("SPEC") == 2
    assert spectrum.count(written) == 2


def test_the_geometry_travels_beside_the_numbers(written):
    header = fits.getheader(written, ext=1)
    assert header["APCENTER"] == pytest.approx(24.0)
    assert header["APLOW"] == pytest.approx(-5.0)
    assert header["EXPTIME"] == pytest.approx(1200.0)
    assert header["EXTVER"] == 1, "so obj.fits[SPEC,1] addresses it"


def test_it_reads_back_the_same(written):
    back = spectrum.read(written)
    assert [aperture.number for aperture in back] == [1, 2]
    assert back[0].title == "a star"
    np.testing.assert_allclose(back[0].flux, np.arange(1.0, 51.0) * 2)
    assert back[1].width == pytest.approx(8.0)


def test_one_aperture_can_be_asked_for(written):
    only = spectrum.read(written, number=2)
    assert len(only) == 1 and only[0].number == 2


def test_a_wrong_number_and_a_wrong_file_say_different_things(written, tmp_path):
    with pytest.raises(ValueError, match="no aperture 9. It holds 1, 2"):
        spectrum.read(written, number=9)

    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(
        tmp_path / "plain.fits")
    with pytest.raises(ValueError, match="holds no extracted spectra"):
        spectrum.read(tmp_path / "plain.fits")


# -------------------------------------------------------------- the tasks


@pytest.fixture
def frame(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data, _ = synthetic()
    fits.PrimaryHDU(data.astype(np.float32),
                    fits.Header({"OBJECT": "test star", "EXPTIME": 600.0})
                    ).writeto(tmp_path / "obj.fits")
    return tmp_path


def test_apall_end_to_end(frame):
    result = get_task("apall").run(input="obj.fits", output="obj.ms.fits",
                                   background="fit", t_step=10, nsum=10)
    assert result.data["apertures"] == 1

    back = spectrum.read(frame / "obj.ms.fits")[0]
    assert back.centre == pytest.approx(CENTRE, abs=0.5)
    assert np.median(back.flux) == pytest.approx(2000.0, rel=0.05)
    assert back.title == "test star"


def test_apall_says_what_to_check_when_it_finds_nothing(frame):
    fits.PrimaryHDU(np.full((100, 40), 5.0, dtype=np.float32)).writeto(
        frame / "blank.fits")
    with pytest.raises(RuntimeError) as raised:
        get_task("apall").run(input="blank.fits", output="x.ms.fits")
    # What is actually worth checking, in the suggestion rather than buried
    # in the sentence that says it failed.
    assert "--dispaxis" in raised.value.fix


def test_apall_explains_every_step_that_touches_the_data(frame):
    result = get_task("apall").run(input="obj.fits", output="obj.ms.fits",
                                   t_step=10, nsum=10)
    for step in result.steps:
        assert step.explained(), f"'{step.action}' does not say what it does"
    trace = next(step for step in result.steps if step.action == "trace")
    assert "drift" in trace.measured


def test_wspectext_writes_every_column_with_what_it_means(frame):
    get_task("apall").run(input="obj.fits", output="obj.ms.fits", t_step=10,
                          nsum=10, background="fit", weights="variance", gain="2.",
                          readnoise="5.")
    get_task("wspectext").run(input="obj.ms.fits", output="obj.csv")

    text = (frame / "obj.csv").read_text()
    assert "# aperture 1 at" in text
    assert "flux: extracted flux, background subtracted" in text

    rows = [line for line in text.splitlines() if not line.startswith("#")]
    assert rows[0].split(",") == ["pixel", "flux", "error", "sky", "raw"]
    assert len(rows) == NY + 1


def test_wspectext_can_be_asked_for_some_columns(frame):
    get_task("apall").run(input="obj.fits", output="obj.ms.fits", t_step=10,
                          nsum=10)
    get_task("wspectext").run(input="obj.ms.fits", output="two.csv",
                              columns="pixel,flux", header=False)
    rows = (frame / "two.csv").read_text().splitlines()
    assert rows[0] == "pixel,flux"

    with pytest.raises(ValueError, match="no column called sparkle"):
        get_task("wspectext").run(input="obj.ms.fits", output="x.csv",
                                  columns="sparkle")


# -------------------------------------------------------------- apedit
# Its keys are held to IRAF's in test_apedit_keys.py; here, that the task takes them up.


def test_apall_uses_what_the_keys_left(frame):
    """The edited apertures are the ones extracted, not the found ones."""
    (frame / "cur").write_text(f"{CENTRE} 0 1 .\n{CENTRE - 9} 0 1 l\n{CENTRE + 9} 0 1 u\n"
                               "0 0 1 q\n")
    result = get_task("apall").run(input="obj.fits", output="obj.ms.fits", edit=True,
                                   t_step=10, nsum=10, gcommands="cur")
    back = spectrum.read(frame / "obj.ms.fits")[0]
    assert back.low == pytest.approx(-9.0, abs=0.7)
    assert back.high == pytest.approx(9.0, abs=0.7)
    assert result.data["apertures"] == 1


def test_deleting_every_aperture_stops_rather_than_extracting_nothing(frame):
    (frame / "cur").write_text(f"{CENTRE} 0 1 d\n0 0 1 q\n")
    with pytest.raises(RuntimeError, match="nothing to extract"):
        get_task("apall").run(input="obj.fits", output="x.ms.fits", edit=True, gcommands="cur")


def test_the_trace_and_a_dispersion_solution_do_not_collide(frame):
    """A trace and a wavelength solution are different things about the same
    frame. IRAF keeps them apart by prefix -- ap and id -- and the reason is
    not tidiness: run apall and identify on one image with a shared name and
    the second quietly overwrites the first.

    Found by running the whole chain on a real night, not by a unit test:
    apall was writing its trace to database/id<image>, under a header that
    called itself a dispersion solution.
    """
    from harwood.core import database
    from harwood.core.registry import get_task

    get_task("apall").run(input="obj.fits", output="obj.ms.fits")

    assert (frame / "database" / "apobj").exists(), "the trace goes under ap"
    assert not (frame / "database" / "idobj").exists(), \
        "id is the dispersion solution's prefix and must stay free"

    # And the file says what it is, rather than claiming to be something else.
    text = (frame / "database" / "apobj").read_text()
    assert text.startswith("# harwood trace")
    assert "One line per aperture" in text

    stored = database.read("obj", frame, prefix="ap")
    assert len(stored) >= 1
