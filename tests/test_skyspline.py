"""skyspline and flexure, held to skies put there beforehand, and run on real twilights."""

import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import skyspline, zogy
from harwood.core.registry import get_task

TILT_TOP = 3.0


def sky_at(s):
    """A spectrum with sharp lines and a slow continuum, at any place along it."""
    lines = [(150.3, 3000.0), (310.7, 800.0), (420.1, 5000.0), (560.6, 1500.0),
             (700.2, 4000.0)]
    total = 200.0 + 0.05 * s
    for centre, height in lines:
        total = total + height * np.exp(-0.5 * ((s - centre) / 0.9) ** 2)
    return total


def frame(rng, shape=(800, 90), slit=(15, 74), objects=None, gain=2.0, rdnoise=5.0,
          tilt=TILT_TOP, dropping=0.2, spikes=0):
    rows, columns = shape
    y, x = np.mgrid[0:rows, 0:columns].astype(float)
    middle = 0.5 * (slit[0] + slit[1])
    half = 0.5 * (slit[1] - slit[0])
    s = y + tilt * (x - middle) / half * (1 + 0.3 * (y - rows / 2) / rows)
    fall = 1.0 - dropping * ((x - middle) / half) ** 2
    model = sky_at(s) * fall
    model[:, :slit[0]] = 0.0
    model[:, slit[1] + 1:] = 0.0
    star = np.zeros(shape)
    if objects is not None:
        star = 2000.0 * np.exp(-0.5 * ((x - objects) / 1.5) ** 2)
    electrons = rng.poisson(np.maximum(model + star, 0.0) * gain)
    data = (electrons + rng.normal(0, rdnoise, shape)) / gain
    if spikes:
        at = (rng.integers(0, rows, spikes), rng.integers(slit[0], slit[1] + 1, spikes))
        data[at] += 5000.0
    return data, model, star, s


def test_a_tilt_is_measured_from_an_arc():
    rng = np.random.default_rng(1)
    data, *_ = frame(rng)
    tilt = skyspline.measure_tilt(data, (15, 74))
    y, x = np.mgrid[100:700:100, 20:70:10].astype(float)
    middle = 44.5
    truth = TILT_TOP * (x - middle) / 29.5 * (1 + 0.3 * (y - 400) / 800)
    assert np.abs(tilt(x, y) - truth).max() < 0.5, "good enough for the sky to start from"
    assert tilt.largest == pytest.approx(TILT_TOP, rel=0.2)


def test_what_remains_is_the_noise_where_lines_across_the_slit_leave_the_sky():
    rng = np.random.default_rng(2)
    data, model, _, _ = frame(rng)
    slit = (15, 74)
    none = np.zeros(60, bool)
    tilt = skyspline.measure_tilt(data, slit)
    done = skyspline.fit(data, tilt, slit, none, 2.0, 5.0, 0.5)
    y, x = np.mgrid[200:500:100, 20:70:10].astype(float)
    truth = TILT_TOP * (x - 44.5) / 29.5 * (1 + 0.3 * (y - 400) / 800)
    assert np.abs(tilt(x, y) - truth).max() < 0.08, "the sky has made it better"
    sigma = skyspline.noise(model, 2.0, 5.0)[:, 18:72]
    assert (done.subtracted[:, 18:72] / sigma).std() == pytest.approx(1.0, abs=0.08)
    assert np.abs((done.sky - model)[:, 18:72]).max() < 200, "on 5000, at a line's steep side"
    lines = skyspline.by_lines(data, slit, none)[:, 18:72]
    assert (lines / sigma).std() > 3 * (done.subtracted[:, 18:72] / sigma).std()
    assert done.illumination[44] == pytest.approx(1.0, abs=0.01)
    assert done.illumination[20] == pytest.approx(1 - 0.2 * (24.5 / 29.5) ** 2, abs=0.02)


def test_an_object_stays_and_cosmic_rays_do_not_count():
    rng = np.random.default_rng(3)
    data, model, star, _ = frame(rng, objects=40.0, spikes=120)
    slit = (15, 74)
    objects = skyspline.object_columns(data, slit)
    assert objects[40 - 15] and not objects[20 - 15] and not objects[70 - 15]
    tilt = skyspline.measure_tilt(data, slit, objects=objects)
    done = skyspline.fit(data, tilt, slit, objects, 2.0, 5.0, 0.5)
    assert done.rejected > 80
    on_star = done.subtracted[100:700, 38:43].sum()
    assert on_star == pytest.approx(star[100:700, 38:43].sum(), rel=0.02)
    assert np.abs((done.sky - model)[:, 18:30]).max() < 200


def test_no_tilt_comes_to_lines_and_is_said(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data, *_ = frame(np.random.default_rng(4), tilt=0.0)
    fits.writeto("a.fits", data.astype("f4"))
    done = get_task("skyspline").run(image="a.fits", output="o.fits", dispaxis=2, gain=2.0,
                                     rdnoise=5.0)
    assert done.data["spacing"] == 1.0
    assert any("no better than a fit by lines" in warning.message for warning in done.warnings)
    assert done.data["remains"] < 1.2


def test_the_task_with_the_dispersion_along_the_lines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data, model, *_ = frame(np.random.default_rng(5))
    header = fits.Header({"GAIN": 2.0, "RDNOISE": 5.0, "DISPAXIS": 1})
    fits.writeto("a.fits", data.T.astype("f4"), header)
    fits.writeto("arc.fits", data.T.astype("f4"))
    done = get_task("skyspline").run(image="a.fits", output="o.fits", sky="s.fits",
                                     tilt_frame="arc.fits")
    assert done.data["spacing"] == 0.5
    assert done.data["slit"] == [16, 75]
    assert done.data["remains"] == pytest.approx(1.0, abs=0.1)
    out, sky = fits.getdata("o.fits"), fits.getdata("s.fits")
    assert out.shape == data.T.shape
    assert np.allclose(out + sky, data.T, atol=0.01)
    assert fits.getheader("o.fits")["SKYSPLIN"] == 0.5


def test_a_wrong_gain_is_warned_of(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data, *_ = frame(np.random.default_rng(6))
    fits.writeto("a.fits", data.astype("f4"))
    done = get_task("skyspline").run(image="a.fits", output="o.fits", dispaxis=2, gain=40.0,
                                     rdnoise=1.0)
    assert any("times its noise" in warning.message for warning in done.warnings)


def test_skyspline_says_what_it_needs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data, *_ = frame(np.random.default_rng(7))
    fits.writeto("a.fits", data.astype("f4"))
    fits.writeto("small.fits", data[:100, :20].astype("f4"))
    run = get_task("skyspline").run
    with pytest.raises(ValueError, match="which way the dispersion"):
        run(image="a.fits", output="o.fits")
    with pytest.raises(ValueError, match="small.fits is"):
        run(image="a.fits", output="o.fits", dispaxis=2, tilt_frame="small.fits")
    with pytest.raises(ValueError, match="--object 4"):
        run(image="a.fits", output="o.fits", dispaxis=2, object="4")
    with pytest.raises(ValueError, match="a slit of 3 columns"):
        run(image="a.fits", output="o.fits", dispaxis=2, slit="20:22")


REAL = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser() / "spec"


def _real(name, trim=101):
    biases = [fits.getdata(REAL / f"fsp{number:04d}") for number in range(1, 11)]
    return (fits.getdata(REAL / name).astype(float) - np.median(biases, axis=0))[:, :trim]


@pytest.fixture
def twilight():
    if not (REAL / "fsp0013").exists():
        pytest.skip(f"IRAF tutorial data not found under {REAL}")
    return _real("fsp0013")


def test_the_real_twilight_has_no_tilt_and_no_better_nor_worse_than_lines(twilight, tmp_path,
                                                                          monkeypatch):
    monkeypatch.chdir(tmp_path)
    fits.writeto("a.fits", twilight.astype("f4"))
    done = get_task("skyspline").run(image="a.fits", output="o.fits", sky="s.fits", dispaxis=2,
                                     gain=5.4, rdnoise=3.5)
    assert done.data["slit"] == [33, 76]
    assert done.data["tilt"] < 0.1
    out = fits.getdata("o.fits")
    inner = out[:, 35:73]
    sky = fits.getdata("s.fits")[:, 35:73]
    sigma = skyspline.noise(sky, 5.4, 3.5)
    theirs = skyspline.by_lines(twilight, (31, 75), np.zeros(45, bool))[:, 35:73]
    assert (inner / sigma).std() < (theirs / sigma).std(), "if anything, it does better"


def test_a_frame_with_hardly_any_sky_is_said_to_have_none(tmp_path, monkeypatch):
    """AH Cep, 20 minutes: a bright star on a slit with no sky to speak of."""
    if not (REAL / "fsp0022").exists():
        pytest.skip(f"IRAF tutorial data not found under {REAL}")
    monkeypatch.chdir(tmp_path)
    fits.writeto("a.fits", _real("fsp0022").astype("f4"))
    fits.writeto("arc.fits", _real("fsp0018").astype("f4"))
    done = get_task("skyspline").run(image="a.fits", output="o.fits", dispaxis=2, gain=5.4,
                                     rdnoise=3.5, tilt_frame="arc.fits", slit="32:76")
    assert any("hardly any sky" in warning.message for warning in done.warnings)
    assert done.data["objects"] > 0
    out = fits.getdata("o.fits")
    assert out[:, 50:57].sum() == pytest.approx(_real("fsp0022")[:, 50:57].sum(), rel=0.02)


# ------------------------------------------------------------------------ flexure


@pytest.mark.parametrize("shift", [0.0, 0.37, -1.62, 4.25])
def test_a_shift_put_there_is_found(shift):
    rng = np.random.default_rng(8)
    s = np.arange(800.0)
    a = rng.normal(sky_at(s), 3.0)
    b = rng.normal(sky_at(s - shift), 3.0)
    found = skyspline.flexure(b, a)
    assert found.shift == pytest.approx(shift, abs=0.03)
    assert found.peak > 0.9


def test_frames_with_nothing_in_common_are_warned_of(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(9)
    fits.writeto("a.fits", rng.normal(100, 5, 800).astype("f4"))
    fits.writeto("b.fits", rng.normal(100, 5, 800).astype("f4"))
    done = get_task("flexure").run(image="a.fits", reference="b.fits")
    assert any("not to be trusted" in warning.fix for warning in done.warnings)


def test_the_task_moves_the_frame_s_wavelengths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(10)
    s = np.arange(800.0)
    header = fits.Header({"CRPIX1": 1.0, "CRVAL1": 4000.0, "CDELT1": 2.0})
    fits.writeto("a.fits", rng.normal(sky_at(s - 1.5), 3.0).astype("f4"), header)
    fits.writeto("b.fits", rng.normal(sky_at(s), 3.0).astype("f4"))
    done = get_task("flexure").run(image="a.fits", reference="b.fits", output="o.fits")
    assert done.data["shift"] == pytest.approx(1.5, abs=0.05)
    written = fits.getheader("o.fits")
    assert written["CRPIX1"] == pytest.approx(2.5, abs=0.05)
    assert written["FLEXURE"] == pytest.approx(1.5, abs=0.05)
    assert any("3.0" in str(value) or "+3" in str(value)
               for step in done.steps for value in (step.measured or {}).values())


def test_flexure_says_what_it_needs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame_two = np.random.default_rng(11).normal(100, 5, (100, 80)).astype("f4")
    fits.writeto("a.fits", frame_two)
    fits.writeto("b.fits", frame_two[:50, :40])
    run = get_task("flexure").run
    with pytest.raises(ValueError, match="a.fits is"):
        run(image="a.fits", reference="b.fits", dispaxis=2)
    with pytest.raises(ValueError, match="which way"):
        run(image="a.fits", reference="a.fits")


def test_real_twilights_agree_and_a_shift_put_in_one_is_found(twilight, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = _real("fsp0012")
    fits.writeto("a.fits", twilight.astype("f4"))
    fits.writeto("b.fits", other.astype("f4"))
    same = get_task("flexure").run(image="a.fits", reference="b.fits", dispaxis=2,
                                   columns="35:70")
    assert abs(same.data["shift"]) < 0.3, "frames of one twilight"
    assert same.data["peak"] > 0.9
    moved = zogy.shifted(twilight, 0.0, 1.4)
    fits.writeto("c.fits", moved.astype("f4"))
    found = get_task("flexure").run(image="c.fits", reference="b.fits", dispaxis=2,
                                    columns="35:70")
    assert found.data["shift"] - same.data["shift"] == pytest.approx(1.4, abs=0.08)
