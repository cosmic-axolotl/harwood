"""End to end: a synthetic night, reduced, measured, checked against truth."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from harwood.core.registry import get_task

FWHM = 3.4
SKY = 120.0
GAIN, RDNOISE = 2.0, 6.5


@pytest.fixture
def night(tmp_path):
    """Bias frames, a flat, and a field of stars at exactly known positions."""
    rng = np.random.default_rng(4242)
    ny, nx, useful = 160, 200, 180
    ramp = np.broadcast_to(1000 + np.linspace(0, 6, ny)[:, None], (ny, nx)).copy()

    def write(name, image_type, exptime, data):
        header = fits.Header({"IMAGETYP": image_type, "EXPTIME": exptime,
                              "BIASSEC": f"[{useful + 1}:{nx},*]",
                              "TRIMSEC": f"[1:{useful},1:{ny}]"})
        path = tmp_path / name
        fits.PrimaryHDU(data.astype(np.float32), header).writeto(path)
        return path

    biases = [write(f"bias{i}.fits", "zero", 0.0,
                    ramp + rng.normal(0, RDNOISE / GAIN, (ny, nx)))
              for i in range(5)]

    x = np.array([40.25, 90.5, 140.75, 60.0])
    y = np.array([50.5, 100.25, 70.0, 130.75])
    flux = np.array([40000.0, 20000.0, 12000.0, 60000.0])
    sigma = FWHM / 2.3548
    yy, xx = np.mgrid[0:ny, 0:nx]
    stars = np.zeros((ny, nx))
    for sx, sy, sf in zip(x, y, flux, strict=False):
        stars += sf / (2 * np.pi * sigma**2) * np.exp(
            -(((xx - (sx - 1))**2 + (yy - (sy - 1))**2) / (2 * sigma**2)))
    stars[:, useful:] = 0.0

    science = ramp + rng.poisson(np.clip((stars + SKY) * GAIN, 0, None)) / GAIN
    science[:, useful:] = ramp[:, useful:] + rng.normal(0, RDNOISE / GAIN,
                                                        (ny, nx - useful))
    obj = write("obj.fits", "object", 60.0, science)
    return {"dir": tmp_path, "biases": biases, "object": obj,
            "x": x, "y": y, "flux": flux}


def test_the_whole_chain_recovers_the_truth(night):
    tmp = night["dir"]

    get_task("ccdproc").run(input=[str(p) for p in night["biases"]],
                            output=tmp / "bp", overscan=True, trim=True)
    get_task("zerocombine").run(input=[str(p) for p in sorted((tmp / "bp").glob("*.fits"))],
                                output=tmp / "Zero.fits", combine="average",
                                reject="minmax")
    get_task("ccdproc").run(input=[str(night["object"])], output=tmp / "obj_p.fits",
                            overscan=True, trim=True, zerocor=True,
                            zero=tmp / "Zero.fits")

    get_task("daofind").run(image=str(tmp / "obj_p.fits"), output=tmp / "stars.ecsv",
                            fwhm=FWHM, threshold=5.0)
    get_task("phot").run(image=str(tmp / "obj_p.fits"),
                                  coords=tmp / "stars.ecsv", output=tmp / "mags.ecsv",
                                  apertures=[6.0], annulus=12.0, dannulus=5.0,
                                  epadu=GAIN, readnoise=RDNOISE, zmag=25.0,
                                  exposure="EXPTIME")

    measured = Table.read(tmp / "mags.ecsv")
    assert len(measured) == len(night["x"]), "every injected star should be found"

    distance = np.hypot(np.asarray(measured["x"])[:, None] - night["x"][None, :],
                        np.asarray(measured["y"])[:, None] - night["y"][None, :])
    match = distance.argmin(axis=1)
    assert distance.min(axis=1).max() < 0.25, "centroids must land on the injected stars"

    true_mag = 25.0 - 2.5 * np.log10(night["flux"][match] / 60.0)
    residual = np.asarray(measured["mag"]) - true_mag
    assert abs(np.median(residual)) < 0.05, f"photometric zero point off: {residual}"


def test_ccdproc_refuses_to_bias_subtract_twice(night):
    tmp = night["dir"]
    get_task("ccdproc").run(input=[str(night["object"])], output=tmp / "once.fits",
                            overscan=True, trim=True)
    result = get_task("ccdproc").run(input=[str(tmp / "once.fits")],
                                     output=tmp / "twice.fits", overscan=True, trim=True)
    assert result.data["steps_skipped"] == 2
    before = fits.getdata(tmp / "once.fits")
    after = fits.getdata(tmp / "twice.fits")
    np.testing.assert_array_equal(before, after)


def test_the_plan_tells_the_truth_about_skipped_steps(night):
    tmp = night["dir"]
    get_task("ccdproc").run(input=[str(night["object"])], output=tmp / "once.fits",
                            overscan=True, trim=True)
    task = get_task("ccdproc")
    plan = task.plan(task.Params(input=[str(tmp / "once.fits")],
                                 output=tmp / "twice.fits", overscan=True, trim=True))
    assert [step.skipped for step in plan if step.action in ("overscan", "trim")] == \
        [True, True]


def test_imcombine_rejects_a_cosmic_ray(night):
    tmp = night["dir"]
    frames = []
    rng = np.random.default_rng(9)
    for i in range(7):
        data = np.full((20, 20), 500.0) + rng.normal(0, 4, (20, 20))
        if i == 3:
            data[10, 10] = 9000.0
        path = tmp / f"c{i}.fits"
        fits.PrimaryHDU(data.astype(np.float32)).writeto(path)
        frames.append(str(path))

    result = get_task("imcombine").run(input=frames, output=tmp / "clean.fits",
                                       combine="average", reject="ccdclip",
                                       gain=GAIN, rdnoise=RDNOISE)
    combined = fits.getdata(tmp / "clean.fits")
    assert combined[10, 10] == pytest.approx(500.0, abs=8.0)
    assert result.data["rejected"] >= 1
    assert result.backend, "the backend that ran must be recorded"


def test_phot_refuses_a_catalogue_whose_pixel_origin_is_unknown(night, tmp_path):
    """Assuming a convention would put a silent one-pixel bias in every magnitude."""
    foreign = tmp_path / "foreign.ecsv"
    Table({"xcentroid": [10.0], "ycentroid": [10.0]}).write(foreign, format="ascii.ecsv")
    with pytest.raises(ValueError, match="origin is unknown"):
        get_task("phot").run(image=str(night["object"]), coords=foreign,
                             output=tmp_path / "out.ecsv")


def test_an_overscan_section_shorter_than_the_frame_still_corrects_all_of_it(tmp_path):
    """Found by the KPNO exercise, which picks biassec [335:350,2:510] on a
    512-row frame: the first and last rows are usually bad, so the section
    stops short of them -- and the correction still has to reach them.

    Before this, the fit was evaluated only at the sampled rows and the
    subtraction failed to broadcast; worse, the fit's x axis counted from 1
    inside the strip rather than using the image's own row numbers, so
    --sample meant something different here than everywhere else.
    """
    import numpy as np
    from astropy.io import fits

    from harwood.core.registry import get_task

    ny, nx = 40, 30
    ramp = np.broadcast_to(100 + np.arange(ny)[:, None] * 0.5, (ny, nx)).copy()
    path = tmp_path / "frame.fits"
    fits.PrimaryHDU(ramp.astype(np.float32)).writeto(path)

    result = get_task("ccdproc").run(
        input=[str(path)], output=tmp_path / "out.fits", overscan=True,
        biassec="[25:30,3:38]", function="legendre", order=2)

    corrected = fits.getdata(tmp_path / "out.fits").astype(float)
    assert corrected.shape == (ny, nx), "the whole frame is corrected, not the sample"
    # The ramp is exactly linear, so an order-2 fit removes it everywhere,
    # including the two rows at each edge the section did not cover.
    assert abs(float(np.median(corrected[0]))) < 0.01
    assert abs(float(np.median(corrected[-1]))) < 0.01
    assert result.data["steps_skipped"] == 0
