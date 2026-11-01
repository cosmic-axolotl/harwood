"""The methods IRAF has no task for, held to answers known beforehand and, where there is one,
to the implementation others use: skymodel, psfphot, zogy, coadd, drizzle and imcombine's
inverse variance."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage

from harwood.core import catalog, coadd, drizzle, epsf, mef, skymesh, zogy
from harwood.core.registry import get_task


def moffat(dx, dy, fwhm, beta=2.5):
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    return (beta - 1) / (np.pi * alpha ** 2) * (1 + (dx * dx + dy * dy) / alpha ** 2) ** -beta


def mad(values):
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def stars(shape, xs, ys, flux, fwhm, reach=30):
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    image = np.zeros(shape)
    for a, b, f in zip(xs, ys, flux, strict=True):
        x1, x2 = max(0, int(a) - reach), min(nx, int(a) + reach + 1)
        y1, y2 = max(0, int(b) - reach), min(ny, int(b) + reach + 1)
        image[y1:y2, x1:x2] += f * moffat(x[y1:y2, x1:x2] - a, y[y1:y2, x1:x2] - b, fwhm)
    return image


@pytest.fixture(scope="module")
def field():
    """400 stars, 0-based truth, on a sky that varies, with noise of 8."""
    rng = np.random.default_rng(3)
    ny, nx = 400, 480
    y, x = np.mgrid[0:ny, 0:nx]
    sky = 200 + 0.1 * x - 0.05 * y + 15 * np.sin(x / 200.0) * np.cos(y / 150.0)
    xs, ys = rng.uniform(15, nx - 15, 300), rng.uniform(15, ny - 15, 300)
    flux = 10 ** rng.uniform(2.5, 4.8, 300)
    image = rng.normal(sky + stars((ny, nx), xs, ys, flux, 3.5), 8.0)
    return image, sky, xs, ys, flux


# ---------------------------------------------------------------------- the sky


def test_the_mesh_finds_a_sky_known_beforehand(field):
    image, sky, *_ = field
    model = skymesh.model(image, box=64)
    inner = (model.sky - sky)[40:-40, 40:-40]
    assert np.abs(np.median(inner)) < 2.0, "a little high: stars too faint to clip stay in it"
    assert inner.std() < 2.0, "against a noise of 8 in each pixel"
    assert np.median(model.rms) == pytest.approx(8.0, rel=0.25), "faint stars stay in it"


def test_the_mesh_is_box_for_box_what_photutils_finds(field):
    photutils = pytest.importorskip("photutils.background")
    from astropy.stats import SigmaClip

    image = field[0][:384, :448]  # whole boxes: the two cut the edges differently
    ours = skymesh.model(image, box=64, filter_size=1)
    theirs = photutils.Background2D(image, (64, 64), filter_size=(1, 1),
                                    sigma_clip=SigmaClip(3.0, maxiters=10),
                                    bkg_estimator=photutils.SExtractorBackground(),
                                    bkgrms_estimator=photutils.StdBackgroundRMS(),
                                    exclude_percentile=100)
    assert ours.grid.shape == theirs.background_mesh.shape
    assert np.abs(ours.grid - theirs.background_mesh).max() < 0.5


def test_a_box_a_mask_covers_goes_by_its_neighbours():
    image = np.random.default_rng(1).normal(100.0, 5.0, (128, 128))
    mask = np.zeros(image.shape, bool)
    mask[:64, :64] = True
    image[:64, :64] = 1e6
    model = skymesh.model(image, box=64, filter_size=1, mask=mask)
    assert model.replaced == 1
    assert model.sky.max() < 110


def test_a_box_too_small_is_refused():
    with pytest.raises(ValueError, match="a box of 2"):
        skymesh.model(np.zeros((50, 50)), box=2)


# ---------------------------------------------------------------- the effective PSF


@pytest.fixture(scope="module")
def built(field):
    image, sky, xs, ys, flux = field
    use = [(a + 1, b + 1) for a, b, f in zip(xs, ys, flux, strict=True)
           if f > 8000 and 25 < a < image.shape[1] - 25 and 25 < b < image.shape[0] - 25]
    return epsf.build(image, use, sky=sky, size=21, oversampling=2, iterations=6)


def test_the_effective_psf_is_the_stars_own_shape(built):
    y, x = np.mgrid[-10:11, -10:11].astype(float)
    truth = moffat(x, y, 3.5)
    truth /= truth.sum()
    assert np.abs(built.sampled() - truth).max() < 0.004
    assert built.sampled().max() == pytest.approx(truth.max(), rel=0.05)


def test_it_does_not_collapse_with_more_passes(field):
    image, sky, xs, ys, flux = field
    use = [(a + 1, b + 1) for a, b, f in zip(xs, ys, flux, strict=True)
           if f > 8000 and 25 < a < image.shape[1] - 25 and 25 < b < image.shape[0] - 25]
    long = epsf.build(image, use, sky=sky, size=21, oversampling=2, iterations=30)
    y, x = np.mgrid[-10:11, -10:11].astype(float)
    truth = moffat(x, y, 3.5)
    truth /= truth.sum()
    assert np.abs(long.sampled() - truth).max() < 0.01


def test_the_fits_get_the_stars_light_and_place(field, built):
    image, sky, xs, ys, flux = field
    rng = np.random.default_rng(8)
    guess = [(a + 1 + rng.normal(0, 0.3), b + 1 + rng.normal(0, 0.3), None)
             for a, b in zip(xs, ys, strict=True)]
    done, residual = epsf.photometry(image, built, guess, sky)
    fitted = np.array([f.flux for f in done])
    good = np.array([f.ier == 0 for f in done]) & (fitted > 0)
    bright = good & (flux > 3000)
    dmag = -2.5 * np.log10(fitted[bright] / flux[bright])
    assert mad(dmag) < 0.015, "the true PSF gets 0.008 on this crowded field"
    assert abs(np.median(dmag)) < 0.05, "the wings beyond the PSF's box are the zero point"
    dx = np.array([f.x for f in done])[bright] - (xs[bright] + 1)
    assert mad(dx) < 0.03
    assert residual.std() < 12, "with the stars out, not far from the noise of 8"


def test_the_fits_beat_an_aperture_where_stars_are_crowded(field, built):
    image, sky, xs, ys, flux = field
    done, _ = epsf.photometry(image, built, [(a + 1, b + 1, None) for a, b in
                                             zip(xs, ys, strict=True)], sky)
    fitted = np.array([f.flux for f in done])
    y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    aperture = np.array([(image - sky)[(x - a) ** 2 + (y - b) ** 2 <= 16].sum()
                         for a, b in zip(xs, ys, strict=True)])
    faint = (flux > 500) & (flux < 3000) & (fitted > 0) & (aperture > 0)
    psf_scatter = mad(-2.5 * np.log10(fitted[faint] / flux[faint]))
    aperture_scatter = mad(-2.5 * np.log10(aperture[faint] / flux[faint]))
    assert psf_scatter < 0.8 * aperture_scatter


def test_too_few_stars_for_a_psf_are_said(field):
    with pytest.raises(ValueError, match="usable stars"):
        epsf.build(field[0], [(100.0, 100.0)], sky=field[1], size=21)


def test_psfphot_writes_a_table_and_keeps_its_psf(tmp_path, monkeypatch, field):
    image, _, xs, ys, flux = field
    monkeypatch.chdir(tmp_path)
    fits.writeto("f.fits", image.astype("f4"))
    catalog.write(Table({"x": xs + 1, "y": ys + 1}), "c.coo")
    run = get_task("psfphot").run
    first = run(image="f.fits", coords="c.coo", output="f.psf.ecsv", psfout="f.epsf.fits",
                residual="f.res.fits", nstars=40)
    assert first.data["psf_stars"] >= 10
    table = catalog.read("f.psf.ecsv")
    fitted = np.asarray(table["flux"], float)
    bright = (flux > 3000) & (fitted > 0)
    assert mad(-2.5 * np.log10(fitted[bright] / flux[bright])) < 0.02
    good = np.isfinite(table["merr"]) & (fitted > 0)
    assert np.median(np.asarray(table["merr"], float)[good & (flux > 3000)]) < 0.05
    second = run(image="f.fits", coords="c.coo", output="g.psf.ecsv", psf="f.epsf.fits")
    assert second.data["fitted"] == first.data["fitted"]
    assert fits.getdata("f.res.fits").shape == image.shape


def test_an_image_with_a_variance_takes_its_errors_from_it(tmp_path, monkeypatch, field):
    image, _, xs, ys, _ = field
    monkeypatch.chdir(tmp_path)
    variance = np.full(image.shape, 64.0)
    mef.save("v.fits", image, fits.Header(), variance, np.zeros(image.shape, "u2"), "VAR",
             "harwood")
    catalog.write(Table({"x": xs[:50] + 1, "y": ys[:50] + 1}), "c.coo")
    get_task("psfphot").run(image="v.fits", coords="c.coo", output="v.ecsv", nstars=15)
    assert "VAR" in catalog.read("v.ecsv").meta["harwood_merr"]


# ------------------------------------------------------------------------ ZOGY


def scene(fwhm, noise, seed, scale=1.0, transient=None):
    layout = np.random.default_rng(99)
    xs, ys = layout.uniform(10, 246, 150), layout.uniform(10, 246, 150)
    flux = 10 ** layout.uniform(2.5, 4.5, 150)
    y, x = np.mgrid[0:256, 0:256].astype(float)
    sigma = fwhm / 2.3548200450309493
    image = np.zeros((256, 256))
    for a, b, f in zip(xs, ys, flux, strict=True):
        image += f * scale * np.exp(-((x - a) ** 2 + (y - b) ** 2) / (2 * sigma ** 2)) \
            / (2 * np.pi * sigma ** 2)
    if transient:
        a, b, f = transient
        image += f * scale * np.exp(-((x - a) ** 2 + (y - b) ** 2) / (2 * sigma ** 2)) \
            / (2 * np.pi * sigma ** 2)
    return image + np.random.default_rng(seed).normal(0, noise, image.shape)


@pytest.fixture(scope="module")
def subtraction():
    new = scene(4.0, 6.0, 2, scale=1.3, transient=(128.0, 100.0, 800.0))
    reference = scene(2.5, 3.0, 1)
    return zogy.subtract(new, reference, zogy.gaussian(4.0), zogy.gaussian(2.5), 6.0, 3.0,
                         flux_new=1.3, flux_reference=1.0)


def test_the_difference_has_white_noise_of_unit_variance(subtraction):
    d = subtraction.D
    assert mad(d) == pytest.approx(1.0, abs=0.03)
    power = np.abs(np.fft.fft2(d)) ** 2
    high = power[64:192, 64:192].mean()
    low = np.concatenate([power[1:32, 1:32].ravel(), power[-32:, -32:].ravel()]).mean()
    assert high / low == pytest.approx(1.0, abs=0.4), "the same at all scales"


def test_what_has_not_changed_is_gone_and_what_has_is_found(subtraction):
    at = np.unravel_index(np.argmax(subtraction.Scorr), subtraction.Scorr.shape)
    assert at == (100, 128)
    assert subtraction.Scorr[at] > 20
    assert mad(subtraction.Scorr) == pytest.approx(1.0, abs=0.05)
    y, x = np.mgrid[0:256, 0:256]
    away = (x - 128) ** 2 + (y - 100) ** 2 > 12 ** 2
    assert (np.abs(subtraction.Scorr[away]) > 6).sum() < 3, "nothing but the transient"


def test_its_flux_is_what_was_put_there(subtraction):
    alpha, error = subtraction.alpha[100, 128], subtraction.alpha_err[100, 128]
    assert abs(alpha - 800.0) < 3 * error
    assert error < 60


def test_the_score_is_right_at_any_seeing():
    """A blurred reference and a sharp new image, the other way round from above."""
    new = scene(2.5, 3.0, 5, transient=(60.0, 180.0, 2000.0))
    reference = scene(5.0, 8.0, 6)
    done = zogy.subtract(new, reference, zogy.gaussian(2.5), zogy.gaussian(5.0), 3.0, 8.0)
    assert np.unravel_index(np.argmax(done.Scorr), done.Scorr.shape) == (180, 60)
    assert abs(done.alpha[180, 60] - 2000.0) < 3 * done.alpha_err[180, 60]
    assert mad(done.D) == pytest.approx(1.0, abs=0.05)


def test_the_noise_of_the_sources_keeps_stars_out_of_the_score():
    new = scene(3.0, 3.0, 7, scale=1.0)
    reference = scene(3.0, 3.0, 8)
    extra = np.maximum(scene(3.0, 0.001, 9) * 2.0, 0.0)  # bright stars add to the variance
    new = new + np.random.default_rng(10).normal(0, np.sqrt(extra))
    plain = zogy.subtract(new, reference, zogy.gaussian(3.0), zogy.gaussian(3.0), 3.0, 3.0)
    full = zogy.subtract(new, reference, zogy.gaussian(3.0), zogy.gaussian(3.0), 3.0, 3.0,
                         variance_new=9.0 + extra)
    assert (np.abs(full.Scorr) > 5).sum() < (np.abs(plain.Scorr) > 5).sum()


def test_the_astrometric_noise_keeps_a_shifted_star_out():
    y, x = np.mgrid[0:128, 0:128].astype(float)
    star = 20000 * np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / 8.0) / (8 * np.pi)
    moved = 20000 * np.exp(-((x - 64.1) ** 2 + (y - 64) ** 2) / 8.0) / (8 * np.pi)
    rng = np.random.default_rng(4)
    new, reference = moved + rng.normal(0, 3, x.shape), star + rng.normal(0, 3, x.shape)
    psf = zogy.gaussian(4.71)
    plain = zogy.subtract(new, reference, psf, psf, 3.0, 3.0)
    held = zogy.subtract(new, reference, psf, psf, 3.0, 3.0, astrometric=(0.1, 0.1))
    assert np.abs(held.Scorr).max() < np.abs(plain.Scorr).max() / 2


def test_zogy_writes_its_images_and_its_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    new = scene(4.0, 6.0, 2, transient=(128.0, 100.0, 1500.0)) + 300
    fits.writeto("new.fits", new.astype("f4"))
    fits.writeto("ref.fits", (scene(2.5, 3.0, 1) + 100).astype("f4"))
    result = get_task("zogy").run(new="new.fits", reference="ref.fits", output="d.fits",
                                  scorr="s.fits", flux="a.fits", candidates="c.ecsv",
                                  fwhm_new=4.0, fwhm_reference=2.5, threshold=8.0)
    assert result.data["noise"] == pytest.approx(1.0, abs=0.1)
    table = catalog.read("c.ecsv")
    assert any(abs(r["x"] - 129) <= 1 and abs(r["y"] - 101) <= 1 for r in table)
    assert fits.getdata("s.fits").shape == (256, 256)


def test_zogy_finds_the_ratio_from_stars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    layout = np.random.default_rng(99)
    xs, ys = layout.uniform(10, 246, 150), layout.uniform(10, 246, 150)
    catalog.write(Table({"x": xs + 1, "y": ys + 1}), "c.coo")
    fits.writeto("new.fits", (scene(3.0, 3.0, 2, scale=1.7) + 300).astype("f4"))
    fits.writeto("ref.fits", (scene(3.0, 3.0, 1) + 100).astype("f4"))
    result = get_task("zogy").run(new="new.fits", reference="ref.fits", output="d.fits",
                                  fwhm_new=3.0, fwhm_reference=3.0, coords="c.coo")
    assert result.data["zero_ratio"] == pytest.approx(1.7, rel=0.03)


def test_what_is_left_of_the_alignment_is_found_and_taken_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    layout = np.random.default_rng(99)
    xs, ys = layout.uniform(10, 246, 150), layout.uniform(10, 246, 150)
    catalog.write(Table({"x": xs + 1, "y": ys + 1}), "c.coo")
    new = zogy.shifted(scene(3.0, 0.001, 2), 0.3, -0.2)
    new = new + np.random.default_rng(3).normal(0, 3.0, new.shape)
    fits.writeto("new.fits", (new + 300).astype("f4"))
    fits.writeto("ref.fits", (scene(3.0, 3.0, 1) + 100).astype("f4"))
    args = dict(new="new.fits", reference="ref.fits", fwhm_new=3.0, fwhm_reference=3.0,
                coords="c.coo", threshold=6.0, scorr="s.fits")
    done = get_task("zogy").run(output="d.fits", **args)
    assert done.data["offset"] == pytest.approx([0.3, -0.2], abs=0.02)
    left = get_task("zogy").run(output="e.fits", align=False, astrometric_x=0.0, **args)
    assert done.data["candidates"] < left.data["candidates"] / 3


def test_a_smooth_psf_leaves_no_rings_where_seeing_differs(tmp_path, monkeypatch):
    """Through the whole chain: stars, effective PSFs, a transient, against two Moffats."""
    monkeypatch.chdir(tmp_path)
    layout = np.random.default_rng(21)
    xs, ys = layout.uniform(20, 300, 120), layout.uniform(20, 300, 120)
    flux = 10 ** layout.uniform(3.0, 4.6, 120)
    shape = (320, 320)
    noise = np.random.default_rng(1).normal(100, 4, shape)
    reference = stars(shape, xs, ys, flux, 3.0) + noise
    new = stars(shape, xs, ys, flux, 4.2)
    new = new + stars(shape, [150.0], [90.0], [3000.0], 4.2)
    new = new + np.random.default_rng(2).normal(300, 4, shape)
    fits.writeto("new.fits", new.astype("f4"))
    fits.writeto("ref.fits", reference.astype("f4"))
    catalog.write(Table({"x": xs + 1, "y": ys + 1}), "c.coo")
    for name in ("new", "ref"):
        get_task("psfphot").run(image=f"{name}.fits", coords="c.coo", output=f"{name}.ecsv",
                                psfout=f"{name}.epsf.fits", nstars=40, box=48)
    done = get_task("zogy").run(new="new.fits", reference="ref.fits", output="d.fits",
                                psf_new="new.epsf.fits", psf_reference="ref.epsf.fits",
                                coords="c.coo", threshold=6.0, box=48, gain=1.0)
    rows = done.data["rows"]
    assert len(rows) == 1, rows
    assert (rows[0]["x"], rows[0]["y"]) == pytest.approx((151.0, 91.0), abs=1)
    plain = get_task("zogy").run(new="new.fits", reference="ref.fits", output="e.fits",
                                 psf_new="new.epsf.fits", psf_reference="ref.epsf.fits",
                                 coords="c.coo", threshold=6.0, box=48)
    assert len(plain.data["rows"]) > 1, "bright stars, with no noise of theirs in the score"
    assert any("--gain" in str(warning.fix) for warning in plain.warnings)


def test_two_real_exposures_with_nothing_new_give_next_to_nothing(tmp_path, monkeypatch):
    """Two 60 s frames of M92 in V from the tutorial's tape, a few minutes apart."""
    import os
    import shutil
    from pathlib import Path

    data = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser() / "phot"
    if not (data / "fm920004").exists():
        pytest.skip(f"IRAF tutorial data not found under {data}")
    monkeypatch.chdir(tmp_path)
    for tape, name in (("fm920001", "bias"), ("fm920003", "flat"), ("fm920004", "o1"),
                       ("fm920005", "o2")):
        shutil.copy(data / tape, f"{name}.fits")
    fit = dict(function="legendre", order=1, niterate=1, logfile="", gain=1.5, readnoise=5.0)
    run = get_task("ccdproc").run
    run(input=["bias.fits"], output="Zero.fits", overscan=True, trim=True, **fit)
    run(input=["flat.fits"], output="Flat.fits", overscan=True, trim=True, zerocor=True,
        zero="Zero.fits", **fit)
    for name in ("o1", "o2"):
        run(input=[f"{name}.fits"], output=f"r{name}.fits", overscan=True, trim=True,
            zerocor=True, zero="Zero.fits", flatcor=True, flat="Flat.fits", **fit)
    sky = get_task("skymodel").run(image="ro1.fits", output="sky.fits")
    get_task("daofind").run(image="ro1.fits", output="c.coo", fwhmpsf=2.5,
                            sigma=sky.data["rms"], threshold=5.0, epadu=1.5)
    frames = [mef.load(f"r{name}.fits") for name in ("o1", "o2")]
    a, b = frames[0][0], frames[1][0]
    power = np.fft.ifft2(np.fft.fft2(a - np.median(a))
                         * np.conj(np.fft.fft2(b - np.median(b)))).real
    j, i = np.unravel_index(np.argmax(power), power.shape)
    dy, dx = (j if j < a.shape[0] // 2 else j - a.shape[0]), (i if i < a.shape[1] // 2
                                                              else i - a.shape[1])
    get_task("imshift").run(input=["ro2.fits"], output="al2.fits", xshift=float(dx),
                            yshift=float(dy), boundary_type="nearest")
    for image, psf in (("ro1.fits", "ref.epsf.fits"), ("al2.fits", "new.epsf.fits")):
        get_task("psfphot").run(image=image, coords="c.coo", output=f"{image}.ecsv",
                                psfout=psf, datamax=25000.0)
    done = get_task("zogy").run(new="al2.fits", reference="ro1.fits", output="d.fits",
                                psf_new="new.epsf.fits", psf_reference="ref.epsf.fits",
                                coords="c.coo", gain=1.5, threshold=6.0, datamax=25000.0,
                                scorr="s.fits")
    assert done.data["candidates"] <= 3, done.data["rows"]
    score = fits.getdata("s.fits")
    assert 1.4826 * np.median(np.abs(score)) == pytest.approx(1.0, abs=0.12)


def test_zogy_says_what_it_needs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fits.writeto("a.fits", np.random.default_rng(1).normal(0, 1, (64, 64)).astype("f4"))
    fits.writeto("b.fits", np.random.default_rng(2).normal(0, 1, (64, 32)).astype("f4"))
    with pytest.raises(ValueError, match=r"is \(64, 64\) and"):
        get_task("zogy").run(new="a.fits", reference="b.fits", output="d.fits", fwhm_new=3.0,
                             fwhm_reference=3.0)
    fits.writeto("c.fits", np.random.default_rng(3).normal(0, 1, (64, 64)).astype("f4"))
    with pytest.raises(ValueError, match="the PSF of the new"):
        get_task("zogy").run(new="a.fits", reference="c.fits", output="d.fits", box=32)


# ---------------------------------------------------------------- inverse variance


def _frames(tmp_path, noises):
    rng = np.random.default_rng(12)
    names = []
    for k, noise in enumerate(noises):
        name = str(tmp_path / f"v{k}.fits")
        data = 100.0 + rng.normal(0, noise, (200, 200))
        mef.save(name, data, fits.Header(), np.full(data.shape, noise ** 2),
                 np.zeros(data.shape, "u2"), "VAR", "harwood")
        names.append(name)
    return names


def test_inverse_variance_has_the_lowest_noise_and_says_so(tmp_path):
    names = _frames(tmp_path, [1.0, 3.0, 3.0])
    out = tmp_path / "ivar.fits"
    get_task("imcombine").run(input=names, output=out, weight="ivar")
    plain = tmp_path / "plain.fits"
    get_task("imcombine").run(input=names, output=plain)
    ivar = fits.getdata(out)
    assert ivar.std() == pytest.approx(np.sqrt(1 / (1 + 1 / 9 + 1 / 9)), rel=0.05)
    assert fits.getdata(plain).std() == pytest.approx(np.sqrt(19) / 3, rel=0.05)
    _, _, var, _ = mef.load(out)
    assert var.mean() == pytest.approx(1 / (1 + 1 / 9 + 1 / 9), rel=1e-6)


def test_inverse_variance_needs_a_variance(tmp_path):
    names = []
    for k in range(3):
        name = tmp_path / f"p{k}.fits"
        fits.writeto(name, np.ones((8, 8), "f4"))
        names.append(str(name))
    with pytest.raises(ValueError, match="ccdproc"):
        get_task("imcombine").run(input=names, output=tmp_path / "o.fits", weight="ivar")


# ---------------------------------------------------------------------- the coaddition

SEEING = [2.0, 3.5, 5.0, 2.5]
NOISE = [10.0, 8.0, 6.0, 14.0]
ZEROS = [1.0, 0.8, 1.2, 0.6]


def gaussians(xs, ys, flux, fwhm, shape=(256, 256)):
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    sigma = fwhm / 2.3548200450309493
    image = np.zeros(shape)
    for a, b, f in zip(xs, ys, flux, strict=True):
        image += f * np.exp(-((x - a) ** 2 + (y - b) ** 2) / (2 * sigma * sigma)) / (
            2 * np.pi * sigma * sigma)
    return image


@pytest.fixture(scope="module")
def faint():
    """Faint stars at whole pixels, far apart, in four frames that differ in everything."""
    rng = np.random.default_rng(1)
    ys, xs = (grid.ravel() for grid in np.mgrid[14:250:22, 14:250:22])
    flux = np.full(xs.size, 200.0)
    frames = [zero * gaussians(xs, ys, flux, fwhm) + rng.normal(0, noise, (256, 256))
              for fwhm, noise, zero in zip(SEEING, NOISE, ZEROS, strict=True)]
    psfs = [zogy.gaussian(fwhm, 41) for fwhm in SEEING]
    return frames, psfs, xs, ys, coadd.coadd(frames, psfs, NOISE, ZEROS)


def test_the_noise_of_a_coadd_is_white_and_what_the_equations_say():
    rng = np.random.default_rng(2)
    frames = [rng.normal(0, noise, (256, 256)) for noise in NOISE]
    done = coadd.coadd(frames, [zogy.gaussian(fwhm, 41) for fwhm in SEEING], NOISE, ZEROS)
    expected = sum(z * z / n / n for z, n in zip(ZEROS, NOISE, strict=True))
    assert done.sigma == pytest.approx(1 / np.sqrt(expected))
    assert done.image.std() == pytest.approx(done.sigma, rel=0.02)
    power = np.abs(np.fft.fft2(done.image)) ** 2
    fy = np.abs(np.fft.fftfreq(256))[:, None]
    fx = np.abs(np.fft.fftfreq(256))[None, :]
    high = power[np.hypot(fx, fy) > 0.35].mean()
    low = power[(np.hypot(fx, fy) < 0.1) & (np.hypot(fx, fy) > 0)].mean()
    assert high / low == pytest.approx(1.0, abs=0.1), "as much at every frequency"
    assert done.score.std() == pytest.approx(1.0, rel=0.1), "and the score is in sigma"


def test_a_star_scores_the_root_of_the_sum_of_what_each_frame_gives_it(faint):
    frames, psfs, xs, ys, done = faint
    each = [200.0 * zero / noise * np.sqrt((psf ** 2).sum())
            for zero, noise, psf in zip(ZEROS, NOISE, psfs, strict=True)]
    expected = np.sqrt(sum(value ** 2 for value in each))
    assert np.median(done.score[ys, xs]) == pytest.approx(expected, rel=0.03)
    assert np.median(done.score[ys, xs]) > 1.3 * max(each)


def test_a_star_has_the_flux_it_was_given(faint):
    _, _, xs, ys, done = faint
    values = done.flux[ys, xs]
    assert np.median(values) == pytest.approx(200.0, abs=3 * done.flux_err / np.sqrt(xs.size))
    assert values.std() == pytest.approx(done.flux_err, rel=0.2)


def test_it_beats_the_weighted_mean_of_the_pixels(faint):
    frames, psfs, xs, ys, done = faint
    weights = np.array([zero / noise ** 2 for zero, noise in zip(ZEROS, NOISE, strict=True)])
    mean = sum(w * frame for w, frame in zip(weights, frames, strict=True)) / weights.sum()
    # And the filter that is best for that mean, its own PSF
    psf = sum(w * zero * p for w, zero, p in zip(weights, ZEROS, psfs, strict=True))
    psf = psf / psf.sum()
    transform = np.fft.fft2(zogy._centred(psf, mean.shape))
    matched = np.fft.ifft2(np.fft.fft2(mean) * np.conj(transform)).real
    score = matched / (1.4826 * np.median(np.abs(matched - np.median(matched))))
    assert np.median(done.score[ys, xs]) > 1.05 * np.median(score[ys, xs])


def test_one_frame_comes_back_as_itself():
    frame = np.random.default_rng(3).normal(0, 5.0, (128, 128))
    done = coadd.coadd([frame], [zogy.gaussian(3.0, 41)], [5.0], [1.0])
    assert np.abs(done.image - frame).max() < 1e-9
    assert done.sigma == 5.0


def test_a_bad_pixel_takes_its_value_from_the_others(faint):
    frames, psfs, xs, ys, plain = faint
    spoiled = [frame.copy() for frame in frames]
    bad = [np.zeros(frame.shape, bool) for frame in frames]
    spoiled[0][100, 100], bad[0][100, 100] = 1e6, True
    for frame, flags in zip(spoiled, bad, strict=True):
        frame[50, 50], flags[50, 50] = 1e6, True
    done = coadd.coadd(spoiled, psfs, NOISE, ZEROS, bad)
    assert (done.filled, done.unfilled) == (1, 1)
    assert np.abs(done.image).max() < 1000.0
    unguarded = coadd.coadd(spoiled, psfs, NOISE, ZEROS)
    assert np.abs(unguarded.image).max() > 1e4


def test_the_task_finds_the_zero_points_the_shifts_and_the_psfs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(4)
    shape = (300, 300)
    xs, ys = rng.uniform(25, 275, 60), rng.uniform(25, 275, 60)
    flux = 10 ** rng.uniform(4.0, 5.0, 60)
    shifts = [(0.0, 0.0), (0.4, -0.3), (-0.25, 0.35)]
    zeros = [1.0, 0.7, 1.3]
    for number, ((dx, dy), zero, fwhm) in enumerate(zip(shifts, zeros, (3.0, 4.0, 3.5),
                                                        strict=True)):
        image = 100.0 + zero * stars(shape, xs + dx, ys + dy, flux, fwhm)
        fits.writeto(f"f{number}.fits", rng.normal(image, 5.0).astype("f4"))
    Table({"x": xs + 1, "y": ys + 1}).write("c.ecsv", overwrite=True)
    done = get_task("coadd").run(input=["f*.fits"], output="c.fits", coords="c.ecsv",
                                 score="s.fits", flux="fl.fits", psfout="c.epsf.fits")
    assert done.data["zeros"] == pytest.approx(zeros, rel=0.05), "the wings differ with seeing"
    for found, shift in zip(done.data["shifts"], shifts, strict=True):
        assert found == pytest.approx(list(shift), abs=0.03)
    assert done.data["noise"] == pytest.approx(done.data["sigma"], rel=0.3), "the wings too"
    out = fits.getdata("c.fits")
    assert fits.getheader("c.fits")["COADDN"] == 3
    measured = get_task("psfphot").run(image="c.fits", coords="c.ecsv", output="p.ecsv",
                                       psf="c.epsf.fits")
    table = catalog.read("p.ecsv")
    good = np.isfinite(table["flux"].astype(float))
    ratio = np.asarray(table["flux"], float)[good] / flux[good]
    assert measured.data["fitted"] >= 50 and np.median(ratio) == pytest.approx(1.0, abs=0.04)
    assert out.shape == shape


def test_coadd_says_what_it_needs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = np.random.default_rng(5).normal(100, 5.0, (128, 128)).astype("f4")
    fits.writeto("a.fits", frame)
    fits.writeto("b.fits", frame)
    fits.writeto("small.fits", frame[:64, :64])
    run = get_task("coadd").run
    with pytest.raises(ValueError, match="PSFs"):
        run(input=["a.fits", "b.fits"], output="c.fits")
    with pytest.raises(ValueError, match="2 frames and 1 of --fwhm"):
        run(input=["a.fits", "b.fits"], output="c.fits", fwhm="3")
    with pytest.raises(ValueError, match="same shape"):
        run(input=["a.fits", "small.fits"], output="c.fits", fwhm="3,3")
    with pytest.raises(FileNotFoundError):
        run(input=["none*.fits"], output="c.fits", fwhm="3")


def test_two_real_exposures_add_up(tmp_path, monkeypatch):
    """The two 60 s V frames of M92: a star has what each gives it, summed in quadrature."""
    import os
    from pathlib import Path

    data = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser() / "phot"
    if not (data / "m92010.fits").exists():
        pytest.skip(f"IRAF tutorial data not found under {data}")
    monkeypatch.chdir(tmp_path)
    a, header = fits.getdata(data / "m92010.fits", header=True)
    b = fits.getdata(data / "m92011.fits")
    power = np.fft.ifft2(np.fft.fft2(a - np.median(a))
                         * np.conj(np.fft.fft2(b - np.median(b))))
    j, i = np.unravel_index(np.argmax(power.real), a.shape)
    dy, dx = (j if j < a.shape[0] // 2 else j - a.shape[0]), (i if i < a.shape[1] // 2
                                                              else i - a.shape[1])
    fits.writeto("a.fits", a, header)
    fits.writeto("b.fits", np.roll(b, (dy, dx), (0, 1)))
    sky = get_task("skymodel").run(image="a.fits", output="sky.fits")
    get_task("daofind").run(image="a.fits", output="c.coo", fwhmpsf=2.5,
                            sigma=sky.data["rms"], threshold=10.0, epadu=1.5)
    done = get_task("coadd").run(input=["a.fits", "b.fits"], output="s.fits", coords="c.coo",
                                 datamax=25000.0, score="sc.fits", psfout="s.epsf.fits")
    for name in ("a", "b"):
        get_task("coadd").run(input=[f"{name}.fits"], output=f"one{name}.fits", coords="c.coo",
                              datamax=25000.0, score=f"sc{name}.fits", align=False)
    table = catalog.read("c.coo")
    xs, ys = catalog.coordinates(table, "c.coo")
    xs, ys = np.round(xs).astype(int) - 1, np.round(ys).astype(int) - 1
    inside = (xs > 30) & (ys > 30) & (xs < a.shape[1] - 30) & (ys < a.shape[0] - 30)
    xs, ys = xs[inside], ys[inside]

    def peak(name):
        return ndimage.maximum_filter(fits.getdata(name), size=3)[ys, xs]

    together, one, other = peak("sc.fits"), peak("sca.fits"), peak("scb.fits")
    faint_ones = np.median(one) * 3 > one
    gain = together[faint_ones] / np.sqrt(one[faint_ones] ** 2 + other[faint_ones] ** 2)
    assert np.median(gain) == pytest.approx(1.0, abs=0.08)
    assert done.data["noise"] == pytest.approx(done.data["sigma"], rel=0.5)


# ---------------------------------------------------------------------- drizzle

DITHERS = [(a / 4, b / 4) for a in range(4) for b in range(4)]


def narrow(dx, dy, shape=(64, 64), fwhm=1.2, flux=1000.0):
    """A star a camera undersamples, by the mean over each pixel, on no sky."""
    from scipy.special import erf

    sigma = fwhm / 2.3548200450309493 * np.sqrt(2)
    x = np.arange(shape[1] + 1) - 0.5 - (31.3 + dx)
    y = np.arange(shape[0] + 1) - 0.5 - (30.6 + dy)
    across = np.diff(erf(x / sigma)) / 2
    down = np.diff(erf(y / sigma)) / 2
    return flux * np.outer(down, across)


def _quadrilateral_by_counting(xs, ys, size=1500):
    grid = (np.arange(size) + 0.5) / size + 0.5
    x, y = np.meshgrid(grid, grid)
    inside = np.ones(x.shape, bool)
    for at in range(4):
        following = (at + 1) % 4
        inside &= ((xs[following] - xs[at]) * (y - ys[at])
                   - (ys[following] - ys[at]) * (x - xs[at])) > 0
    return inside.mean()


def test_an_overlap_is_what_counting_gives():
    rng = np.random.default_rng(1)
    for _ in range(12):
        angles = np.sort(rng.uniform(0, 2 * np.pi, 4))
        # Convex, and round the output pixel at (1, 1)
        angles = angles[0] + np.array([0, 1, 2, 3]) * np.pi / 2 + rng.uniform(-0.5, 0.5, 4)
        radius = rng.uniform(0.3, 1.2)
        xs, ys = 1 + radius * np.cos(angles), 1 + radius * np.sin(angles)
        counted = _quadrilateral_by_counting(xs, ys)
        assert drizzle.overlap(xs, ys, 1, 1) == pytest.approx(counted, abs=0.002)


def test_a_drop_wholly_inside_and_wholly_outside():
    xs, ys = np.array([0.8, 1.2, 1.2, 0.8]), np.array([0.8, 0.8, 1.2, 1.2])
    assert drizzle.overlap(xs, ys, 1, 1) == pytest.approx(0.16)
    assert drizzle.overlap(xs, ys, 3, 1) == 0.0
    assert drizzle.overlap(xs, ys, 1.5, 1) == pytest.approx(0.08)


@pytest.mark.parametrize("scale, pixfrac", [(0.5, 0.7), (0.5, 1.0), (1.0, 1.0), (0.25, 0.5)])
def test_the_flux_is_kept(scale, pixfrac):
    grid = drizzle.Grid((int(64 / scale), int(64 / scale)), pixfrac, scale)
    for dx, dy in DITHERS:
        grid.add(narrow(dx, dy), drizzle.shifted(dx, dy, scale))
    assert grid.image.sum() == pytest.approx(1000.0, rel=2e-3)
    rows, columns = np.mgrid[0:grid.image.shape[0], 0:grid.image.shape[1]]
    x = ((columns + 0.5) * scale - 0.5) * grid.image
    y = ((rows + 0.5) * scale - 0.5) * grid.image
    assert x.sum() / grid.image.sum() == pytest.approx(31.3, abs=0.01)
    assert y.sum() / grid.image.sum() == pytest.approx(30.6, abs=0.01)


def test_what_the_pixels_lost_comes_back():
    def width(image, scale):
        total = image.sum()
        rows, columns = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        x = columns * scale
        mean = (x * image).sum() / total
        return np.sqrt((image * (x - mean) ** 2).sum() / total)

    grid = drizzle.Grid((128, 128), 0.5, 0.5)
    for dx, dy in DITHERS:
        grid.add(narrow(dx, dy), drizzle.shifted(dx, dy, 0.5))
    # A pixel adds 1/12 to a variance, a drop of 0.5 and a pixel of 0.5 add 1/48 each
    sigma = 1.2 / 2.3548200450309493
    assert width(narrow(0, 0), 1.0) == pytest.approx(np.sqrt(sigma ** 2 + 1 / 12), rel=0.02)
    wanted = np.sqrt(sigma ** 2 + 1 / 12 + 1 / 48 + 1 / 48 + (0.25 ** 2 * 15 / 12) / 4 * 0)
    assert width(grid.image, 0.5) == pytest.approx(wanted, rel=0.06)
    assert grid.image.max() / 0.25 > 1.25 * narrow(0, 0).max(), "the peak, per input pixel"


def test_the_noise_of_blocks_is_the_noise_of_the_frames():
    """A pixel understates it, by about the factor: the one the formula is for is a dither at
    random."""
    rng = np.random.default_rng(2)
    shape = (400, 400)
    for scale, pixfrac in ((1.0, 1.0), (1.0, 0.5)):
        grid = drizzle.Grid((int(400 / scale), int(400 / scale)), pixfrac, scale)
        for _ in range(16):
            dx, dy = rng.uniform(0, 1, 2)
            grid.add(rng.normal(0, 1, shape), drizzle.shifted(dx, dy, scale))
        inner = grid.image[20:-20, 20:-20]
        side = 16
        rows, columns = inner.shape[0] // side, inner.shape[1] // side
        blocks = inner[:rows * side, :columns * side].reshape(rows, side, columns, side)
        blocks = blocks.sum(axis=(1, 3)) / side
        assert blocks.std() == pytest.approx(scale / 4, rel=0.12), "s / root of the frames"
        factor = drizzle.noise_correlation(pixfrac, scale)
        assert blocks.std() / inner.std() == pytest.approx(factor, rel=0.15)


def test_a_rotated_frame_is_what_reproject_s_exact_overlap_gives(tmp_path):
    reproject = pytest.importorskip("reproject")
    from astropy.wcs import WCS

    header = fits.Header()
    for key, value in (("CTYPE1", "RA---TAN"), ("CTYPE2", "DEC--TAN"), ("CRVAL1", 150.0),
                       ("CRVAL2", 30.0), ("CRPIX1", 40.5), ("CRPIX2", 40.5),
                       ("CD1_1", -1.0e-4 * np.cos(0.3)), ("CD1_2", 1.0e-4 * np.sin(0.3)),
                       ("CD2_1", 1.0e-4 * np.sin(0.3)), ("CD2_2", 1.0e-4 * np.cos(0.3))):
        header[key] = value
    output = fits.Header()
    for key, value in (("CTYPE1", "RA---TAN"), ("CTYPE2", "DEC--TAN"), ("CRVAL1", 150.0),
                       ("CRVAL2", 30.0), ("CRPIX1", 80.5), ("CRPIX2", 80.5),
                       ("CD1_1", -0.5e-4), ("CD1_2", 0.0), ("CD2_1", 0.0), ("CD2_2", 0.5e-4)):
        output[key] = value
    data = np.random.default_rng(3).normal(100, 10, (80, 80))
    grid = drizzle.Grid((160, 160), 1.0, 0.5)
    grid.add(data, drizzle.wcs_transform(header, output))
    theirs, footprint = reproject.reproject_exact((data, WCS(header)), WCS(output), (160, 160))
    inner = (footprint > 0.999) & (grid.weight > 0.999)
    assert inner.sum() > 10000
    assert np.abs(grid.image[inner] / 0.25 - theirs[inner]).max() < 0.02


def _dithered_frames(rng, dithers, shape=(120, 120), sky=50.0):
    xs, ys = rng.uniform(15, 105, 25), rng.uniform(15, 105, 25)
    flux = 10 ** rng.uniform(4.0, 5.0, 25)
    for at, (dx, dy) in enumerate(dithers):
        image = sky + at + stars(shape, xs + dx, ys + dy, flux, 1.6)
        fits.writeto(f"f{at}.fits", rng.normal(image, 3.0).astype("f4"))
    Table({"x": xs + 1, "y": ys + 1}).write("c.ecsv", overwrite=True)
    return xs, ys, flux


def test_the_task_measures_the_dithers_and_keeps_the_flux(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dithers = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
    xs, ys, flux = _dithered_frames(np.random.default_rng(4), dithers)
    done = get_task("drizzle").run(input=["f*.fits"], output="d.fits", coords="c.ecsv",
                                   weight="w.fits")
    for found, dither in zip(done.data["shifts"], dithers, strict=True):
        assert found == pytest.approx(list(dither), abs=0.05)
    image = fits.getdata("d.fits")
    assert image.shape == (240, 240)
    header = fits.getheader("d.fits")
    assert header["NDRIZZLE"] == 6 and header["DRIZFRAC"] == 0.8
    assert header["DRIZNCOR"] == pytest.approx(done.data["noise_factor"], rel=1e-5)
    assert done.data["holes"] < 0.001
    single = fits.getdata("f0.fits")
    sky = np.median(image)
    assert sky == pytest.approx(0.25 * np.median(single), rel=0.03)
    half = 8
    level = np.median(single)
    ratios = []
    for x, y in zip(xs, ys, strict=True):
        ix, iy = int(round(x)), int(round(y))
        if not (half < ix < 119 - half and half < iy < 119 - half):
            continue
        box = image[2 * (iy - half):2 * (iy + half + 1), 2 * (ix - half):2 * (ix + half + 1)]
        same = single[iy - half:iy + half + 1, ix - half:ix + half + 1]
        ratios.append((box.sum() - box.size * sky) / (same.sum() - same.size * level))
    assert np.median(ratios) == pytest.approx(1.0, abs=0.02), "what the first frame has"


def test_a_dq_keeps_a_pixel_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = np.full((64, 64), 10.0, "f4")
    fits.writeto("a.fits", frame)
    Table({"dx": [0.0], "dy": [0.0]}).write("s.ecsv")
    spoiled = frame.copy()
    spoiled[30, 30] = 1e5
    fits.writeto("b.fits", spoiled)
    done = get_task("drizzle").run(input=["b.fits"], output="d.fits", shifts="s.ecsv",
                                   scale=1.0, pixfrac=1.0, datamax=1e4, sky="none")
    image = fits.getdata("d.fits")
    assert image.max() == pytest.approx(10.0)
    assert fits.getdata("d.fits")[30, 30] == 0.0, "nothing landed"
    assert done.data["holes"] > 0


def test_holes_and_frames_at_one_phase_are_said(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = np.random.default_rng(5).normal(10, 1, (128, 128)).astype("f4")
    fits.writeto("a.fits", frame)
    fits.writeto("b.fits", frame)
    Table({"dx": [0.0, 1.0], "dy": [0.0, 2.0]}).write("s.ecsv")
    done = get_task("drizzle").run(input=["a.fits", "b.fits"], output="d.fits",
                                   shifts="s.ecsv", scale=0.25, pixfrac=0.1)
    messages = " ".join(warning.message for warning in done.warnings)
    assert "had nothing land" in messages
    assert "same fraction of a pixel" in messages
    assert "correlated" in messages


def test_drizzle_says_what_it_needs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = np.full((64, 64), 10.0, "f4")
    fits.writeto("a.fits", frame)
    fits.writeto("b.fits", frame)
    fits.writeto("small.fits", frame[:32, :32])
    Table({"dx": [0.0], "dy": [0.0]}).write("one.ecsv")
    run = get_task("drizzle").run
    with pytest.raises(ValueError, match="where the frames are"):
        run(input=["a.fits", "b.fits"], output="d.fits")
    with pytest.raises(ValueError, match="1 rows"):
        run(input=["a.fits", "b.fits"], output="d.fits", shifts="one.ecsv")
    with pytest.raises(ValueError, match="same shape"):
        run(input=["a.fits", "small.fits"], output="d.fits", shifts="one.ecsv")


def test_two_real_exposures_keep_their_flux(tmp_path, monkeypatch):
    """The two 60 s V frames of M92, a drift of some pixels and a fraction apart."""
    import os
    from pathlib import Path

    data = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser() / "phot"
    if not (data / "m92010.fits").exists():
        pytest.skip(f"IRAF tutorial data not found under {data}")
    monkeypatch.chdir(tmp_path)
    a, header = fits.getdata(data / "m92010.fits", header=True)
    b = fits.getdata(data / "m92011.fits")
    power = np.fft.ifft2(np.fft.fft2(a - np.median(a))
                         * np.conj(np.fft.fft2(b - np.median(b))))
    j, i = np.unravel_index(np.argmax(power.real), a.shape)
    dy, dx = (j if j < a.shape[0] // 2 else j - a.shape[0]), (i if i < a.shape[1] // 2
                                                              else i - a.shape[1])
    fits.writeto("a.fits", a, header)
    fits.writeto("b.fits", np.roll(b, (dy, dx), (0, 1)))
    sky = get_task("skymodel").run(image="a.fits", output="sky.fits")
    get_task("daofind").run(image="a.fits", output="c.coo", fwhmpsf=2.5,
                            sigma=sky.data["rms"], threshold=10.0, epadu=1.5)
    done = get_task("drizzle").run(input=["a.fits", "b.fits"], output="d.fits",
                                   coords="c.coo", scale=0.5, pixfrac=1.0, datamax=25000.0)
    assert done.data["holes"] < 0.02
    image = fits.getdata("d.fits")
    table = catalog.read("c.coo")
    xs, ys = catalog.coordinates(table, "c.coo")
    level = np.median(a)
    assert np.median(image) == pytest.approx(0.25 * level, rel=0.1)
    ratios = []
    half = 8
    for x, y in zip(xs, ys, strict=True):
        ix, iy = int(round(float(x))) - 1, int(round(float(y))) - 1
        if not (20 < ix < a.shape[1] - 20 and 20 < iy < a.shape[0] - 20):
            continue
        inside = a[iy - half:iy + half + 1, ix - half:ix + half + 1]
        if inside.max() > 25000 or inside.sum() - inside.size * level < 5e3:
            continue
        kx, ky = 2 * ix + 1, 2 * iy + 1
        out = image[ky - 2 * half - 1:ky + 2 * half + 1, kx - 2 * half - 1:kx + 2 * half + 1]
        ratios.append((out.sum() - out.size * np.median(image))
                      / (inside.sum() - inside.size * level))
    assert len(ratios) > 8
    assert np.median(ratios) == pytest.approx(1.0, abs=0.02), ratios
