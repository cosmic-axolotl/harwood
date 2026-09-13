"""Finding spectra, following them, adding them up, and writing them down.

Built on a synthetic frame where the trace, the width and the flux are known,
so the answers can be checked rather than merely be plausible. The real KPNO
spectra are exercised in test_m92.py's companion checks.
"""

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import apextract, spectrum
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


def test_the_profile_finds_the_spectrum():
    frame, _ = synthetic()
    spatial = apextract.profile(frame, dispaxis=2)
    found = apextract.find(spatial, nfind=1, threshold=5.0)
    assert len(found) == 1
    assert found[0]["centre"] == pytest.approx(CENTRE, abs=0.4)


def test_two_spectra_closer_than_minsep_are_one():
    across = np.arange(1.0, 61.0)
    blended = (np.exp(-0.5 * ((across - 30) / 2) ** 2)
               + np.exp(-0.5 * ((across - 33) / 2) ** 2)) * 1000 + 10
    assert len(apextract.find(blended, nfind=5, minsep=8.0, threshold=3.0)) == 1
    assert len(apextract.find(blended, nfind=5, minsep=2.0, threshold=3.0)) >= 1


def test_the_aperture_is_sized_from_the_profile():
    frame, _ = synthetic()
    spatial = apextract.profile(frame, dispaxis=2)
    low, high = apextract.edges(spatial, CENTRE, fraction=0.05)

    # A Gaussian falls to 5% of its peak at 2.45 sigma either side.
    expected = 2.45 * WIDTH
    assert -low == pytest.approx(expected, abs=1.5)
    assert high == pytest.approx(expected, abs=1.5)


def test_the_trace_follows_the_drift():
    """The point of tracing: the spectrum does not sit on one column."""
    frame, centres = synthetic()
    fit = apextract.trace(frame, CENTRE, dispaxis=2, step=10, nsum=10)

    assert fit.model is not None, fit.message
    assert fit.rms < 0.2, "a bright spectrum traces to a fraction of a pixel"

    ends = fit(np.array([1.0, float(NY)]))
    assert ends[1] - ends[0] == pytest.approx(DRIFT, abs=0.3)
    np.testing.assert_allclose(fit(np.arange(1.0, NY + 1.0)), centres, atol=0.3)


def test_a_trace_that_cannot_be_followed_says_so_rather_than_returning_a_line():
    blank = np.full((NY, NX), SKY)
    fit = apextract.trace(blank, CENTRE, dispaxis=2, step=20, nsum=10)
    assert fit.model is None
    assert "dispersion axis" in fit.message


def test_extraction_recovers_the_flux_that_was_put_in():
    frame, _ = synthetic(flux_per_pixel=2000.0)
    fit = apextract.trace(frame, CENTRE, dispaxis=2, step=10, nsum=10)
    out = apextract.extract(frame, fit, -8.0, 8.0, dispaxis=2,
                            background=(4.0, 14.0), gain=2.0, rdnoise=5.0)

    flux = out["flux"]
    assert np.median(flux) == pytest.approx(2000.0, rel=0.03)
    assert np.median(out["sky"]) == pytest.approx(SKY, abs=2.0)


def test_optimal_extraction_changes_the_noise_and_not_the_flux():
    """The check that matters: weighting must not move the answer.

    A weighting scheme that improves the signal-to-noise *and* shifts the flux
    has not improved anything -- it has measured something else.
    """
    frame, _ = synthetic(flux_per_pixel=300.0, seed=11)
    fit = apextract.trace(frame, CENTRE, dispaxis=2, step=10, nsum=10)

    plain = apextract.extract(frame, fit, -8.0, 8.0, dispaxis=2,
                              background=(4.0, 14.0), gain=2.0, rdnoise=5.0)
    optimal = apextract.extract(frame, fit, -8.0, 8.0, dispaxis=2,
                                background=(4.0, 14.0), gain=2.0, rdnoise=5.0,
                                weights="variance")

    assert np.median(optimal["flux"]) == pytest.approx(np.median(plain["flux"]),
                                                       rel=0.05)
    plain_snr = np.median(plain["flux"] / plain["error"])
    optimal_snr = np.median(optimal["flux"] / optimal["error"])
    assert optimal_snr > plain_snr, "that is what optimal extraction is for"


def test_the_background_is_taken_from_beside_the_aperture_not_under_it():
    frame, _ = synthetic()
    fit = apextract.trace(frame, CENTRE, dispaxis=2, step=10, nsum=10)

    fair = apextract.extract(frame, fit, -6.0, 6.0, dispaxis=2,
                             background=(6.0, 18.0))
    tight = apextract.extract(frame, fit, -6.0, 6.0, dispaxis=2,
                              background=(0.0, 4.0))
    assert np.median(tight["sky"]) > np.median(fair["sky"]), \
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
                                   gain=2.0, rdnoise=5.0, t_step=10, nsum=10)
    assert result.data["apertures"] == 1

    back = spectrum.read(frame / "obj.ms.fits")[0]
    assert back.centre == pytest.approx(CENTRE, abs=0.5)
    assert np.median(back.flux) == pytest.approx(2000.0, rel=0.05)
    assert back.title == "test star"


def test_apall_says_what_to_check_when_it_finds_nothing(frame):
    fits.PrimaryHDU(np.full((100, 40), 5.0, dtype=np.float32)).writeto(
        frame / "blank.fits")
    with pytest.raises(RuntimeError) as raised:
        get_task("apall").run(input="blank.fits", output="x.ms.fits",
                              threshold=8.0)
    # The two things actually worth checking, in the suggestion rather than
    # buried in the sentence that says it failed.
    assert "--dispaxis" in raised.value.fix
    assert "--threshold" in raised.value.fix


def test_apall_explains_every_step_that_touches_the_data(frame):
    result = get_task("apall").run(input="obj.fits", output="obj.ms.fits",
                                   t_step=10, nsum=10)
    for step in result.steps:
        assert step.explained(), f"'{step.action}' does not say what it does"
    trace = next(step for step in result.steps if step.action == "trace")
    assert "drift" in trace.measured


def test_wspectext_writes_every_column_with_what_it_means(frame):
    get_task("apall").run(input="obj.fits", output="obj.ms.fits", t_step=10,
                          nsum=10)
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


@pytest.fixture
def panel():
    """The profile of a star, with the aperture apfind would have set."""
    from harwood.viewer._qt import application
    from harwood.viewer.aperture_panel import AperturePanel

    application()
    frame, _ = synthetic()
    spatial = apextract.profile(frame, dispaxis=2)
    low, high = apextract.edges(spatial, CENTRE, 0.05)
    return AperturePanel(spatial, [{"centre": CENTRE, "low": low, "high": high}],
                         background=[3.0, 12.0])


def test_an_edge_goes_where_the_cursor_is(panel):
    panel._cursor = CENTRE + 6.0
    panel.key("u")
    assert panel.apertures[0]["high"] == pytest.approx(6.0, abs=0.01)


def test_an_aperture_that_would_not_contain_its_centre_is_refused(panel):
    """It extracts nothing, and nobody means to ask for it."""
    before = dict(panel.apertures[0])
    panel._cursor = CENTRE + 3.0
    panel.key("l")
    assert panel.apertures[0] == before
    assert "below the centre" in panel.status.text()


def test_adding_and_deleting(panel):
    panel._cursor = 45.0
    panel.key("a")
    assert len(panel.apertures) == 2
    assert panel.apertures[panel.selected]["centre"] == pytest.approx(45.0, abs=5)

    panel.key("d")
    assert len(panel.apertures) == 1
    assert "deleted" in panel.status.text()


def test_apertures_stay_in_order_across_the_slit(panel):
    for position in (45.0, 12.0, 38.0):
        panel._cursor = position
        panel.key("a")
    centres = [aperture["centre"] for aperture in panel.apertures]
    assert centres == sorted(centres)


def test_the_background_band_takes_two_presses(panel):
    panel._cursor = CENTRE + 12.0
    panel.key("b")
    assert "Press b again" in panel.status.text()
    assert panel.background == [3.0, 12.0], "nothing changes until the second"

    panel._cursor = CENTRE + 25.0
    panel.key("b")
    inner, outer = panel.background
    assert outer > inner and outer - inner > 5


def test_a_background_band_narrower_than_a_pixel_is_refused(panel):
    panel._cursor = CENTRE + 12.0
    panel.key("b")
    panel._cursor = CENTRE + 12.2
    panel.key("b")
    assert "less than a pixel" in panel.status.text()


def test_recentring_puts_the_aperture_on_the_peak(panel):
    panel.apertures[0]["centre"] = CENTRE + 3.0
    panel.key("g")
    assert panel.apertures[0]["centre"] == pytest.approx(CENTRE, abs=0.6)


def test_colon_commands_set_the_numbers_directly(panel):
    panel.command.setText(":lower -7")
    panel._run_command()
    assert panel.apertures[0]["low"] == pytest.approx(-7.0)

    panel.command.setText(":centre 30")
    panel._run_command()
    assert panel.apertures[0]["centre"] == pytest.approx(30.0)

    panel.command.setText(":sparkle 3")
    panel._run_command()
    assert "no parameter called" in panel.status.text()


def test_apall_uses_what_apedit_returned(frame, monkeypatch):
    """The edited apertures are the ones extracted, not the found ones."""
    from harwood.viewer import aperture_panel

    def edited(profile, apertures, background, ylevel, title=""):
        chosen = [dict(aperture) for aperture in apertures]
        chosen[0]["low"], chosen[0]["high"] = -9.0, 9.0
        return chosen, [5.0, 20.0]

    monkeypatch.setattr(aperture_panel, "run", edited)
    result = get_task("apall").run(input="obj.fits", output="obj.ms.fits",
                                   edit=True, t_step=10, nsum=10)

    back = spectrum.read(frame / "obj.ms.fits")[0]
    assert back.low == pytest.approx(-9.0)
    assert back.high == pytest.approx(9.0)
    assert result.data["apertures"] == 1


def test_deleting_every_aperture_stops_rather_than_extracting_nothing(frame,
                                                                     monkeypatch):
    from harwood.viewer import aperture_panel

    monkeypatch.setattr(aperture_panel, "run",
                        lambda *args, **kwargs: ([], None))
    with pytest.raises(RuntimeError, match="nothing to extract"):
        get_task("apall").run(input="obj.fits", output="x.ms.fits", edit=True)


def test_recentring_converges_from_a_few_pixels_away(panel):
    """A centroid in a box that is not centred on the peak is pulled towards
    the middle of the box; one pass lands short and three do not."""
    for offset in (-4.0, -2.0, 2.0, 4.0):
        panel.apertures[0]["centre"] = CENTRE + offset
        panel.key("g")
        assert panel.apertures[0]["centre"] == pytest.approx(CENTRE, abs=0.4), \
            f"starting {offset:+g} px away"


def test_adding_where_there_is_nothing_puts_it_where_you_pointed(panel):
    """Centroiding is a refinement, not a search: on featureless profile the
    iteration wanders off down whatever slope it finds, and what was asked for
    is an aperture there."""
    panel._cursor = 50.0
    panel.key("a")
    assert panel.apertures[panel.selected]["centre"] == pytest.approx(50.0, abs=1.0)
