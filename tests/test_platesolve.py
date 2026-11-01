"""platesolve, held to fields put there beforehand, and to what it should refuse."""

import io
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from harwood.core import catalog, platesolve
from harwood.core.registry import get_task

SHAPE = (300, 400)
CENTRE = (150.0, 20.0)
SCALE = 0.5


def plate(rotation: float, mirror: bool, scale: float = SCALE):
    t = np.radians(rotation)
    sign = 1.0 if mirror else -1.0
    return scale / 3600.0 * np.array([[sign * np.cos(t), -np.sin(t)],
                                      [sign * np.sin(t), np.cos(t)]])


def field(rng, rotation=33.0, mirror=False, count=220, missing=0.15, extra=0.15, noise=0.2,
          distortion=None, cd=None):
    """A catalogue, and a frame of it: some stars in only one of them, and noise in pixels."""
    cd = plate(rotation, mirror) if cd is None else cd
    crpix = np.array([(SHAPE[1] + 1) / 2, (SHAPE[0] + 1) / 2])
    # Twice the frame each way, so that a good part of the catalogue is off it
    u = rng.uniform(-SHAPE[1], SHAPE[1], count * 4)
    v = rng.uniform(-SHAPE[0], SHAPE[0], count * 4)
    xi, eta = cd @ np.vstack([u, v])
    ra, dec = platesolve.unproject(xi, eta, *CENTRE)
    mags = rng.uniform(12, 18, u.size)
    if distortion is not None:
        u, v = distortion(u, v)
    x, y = u + crpix[0], v + crpix[1]
    on = (x > 1) & (x < SHAPE[1]) & (y > 1) & (y < SHAPE[0])
    seen = on & (rng.uniform(0, 1, u.size) > missing)
    stars = rng.normal(0, noise, (2, seen.sum())) + np.vstack([x[seen], y[seen]])
    # Stars of the frame the catalogue has not: they are as bright as any
    others = int(extra * seen.sum())
    stars = np.hstack([stars, rng.uniform((1, 1), (SHAPE[1], SHAPE[0]), (others, 2)).T])
    frame_mags = np.r_[mags[seen] + rng.normal(0, 0.3, seen.sum()),
                       rng.uniform(12, 18, others)]
    return ra, dec, mags, stars[0], stars[1], frame_mags, cd, crpix, on


def write_field(ra, dec, mags, x, y, frame_mags, header=None):
    fits.writeto("f.fits", np.zeros(SHAPE, "f4"), header, overwrite=True)
    Table({"x": x, "y": y, "mag": frame_mags}).write("c.ecsv", overwrite=True)
    meta = Table({"ra": ra, "dec": dec, "mag": mags})
    meta.write("cat.ecsv", overwrite=True)
    table = catalog.read("c.ecsv")
    table.meta["harwood_convention"] = catalog.CONVENTION
    table.write("c.ecsv", overwrite=True, format="ascii.ecsv")


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


COMMON = dict(coords="c.ecsv", catalog="cat.ecsv", ra=str(CENTRE[0] + 0.005),
              dec=str(CENTRE[1] - 0.005), scale=SCALE * 1.01, scale_error=0.03)


def true_positions(ra, dec, cd, crpix, on, distortion=None):
    xi, eta = platesolve.project(ra[on], dec[on], *CENTRE)
    u, v = np.linalg.inv(cd) @ np.vstack([xi, eta])
    if distortion is not None:
        u, v = distortion(u, v)
    return ra[on], dec[on], u + crpix[0], v + crpix[1]


@pytest.mark.parametrize("rotation, mirror", [(33.0, False), (0.0, True), (181.0, True),
                                              (270.0, False), (92.5, True)])
def test_any_rotation_and_mirror_gives_the_true_wcs(work, rotation, mirror):
    rng = np.random.default_rng(int(rotation) + 1)
    ra, dec, mags, x, y, frame_mags, cd, crpix, on = field(rng, rotation, mirror)
    write_field(ra, dec, mags, x, y, frame_mags)
    done = get_task("platesolve").run(image="f.fits", output="o.fits", matched="m.ecsv",
                                      **COMMON)
    assert done.data["flipped"] == mirror
    assert done.data["scale"] == pytest.approx(SCALE, rel=1e-3)
    assert done.data["significance"] > 30
    assert done.data["rms"] < 0.2, "0.2 pixels of noise a side is 0.14 arcsec"
    wcs = WCS(fits.getheader("o.fits"))
    a, d, tx, ty = true_positions(ra, dec, cd, crpix, on)
    px, py = wcs.all_world2pix(a, d, 1)
    assert np.hypot(px - tx, py - ty).max() < 0.2, "the whole frame, not only the pairs"
    pairs = catalog.read("m.ecsv")
    assert len(pairs) == done.data["pairs"]
    assert np.sqrt(np.mean(pairs["dra"] ** 2 + pairs["ddec"] ** 2)) == pytest.approx(
        done.data["rms"], rel=1e-3)


def test_a_distortion_is_fitted_and_any_tool_reads_it(work):
    def distort(u, v):
        return (u + 2e-4 * u * v + 1e-4 * u * u + 2e-7 * u ** 3,
                v - 1.5e-4 * u * u + 2e-4 * v * v - 1e-7 * v ** 3)

    rng = np.random.default_rng(4)
    ra, dec, mags, x, y, frame_mags, cd, crpix, on = field(rng, 20.0, False, count=400,
                                                           distortion=distort, noise=0.05)
    write_field(ra, dec, mags, x, y, frame_mags)
    plain = get_task("platesolve").run(image="f.fits", **COMMON)
    done = get_task("platesolve").run(image="f.fits", output="o.fits", sip=3, tolerance=2.0,
                                      **COMMON)
    assert done.data["rms"] < plain.data["rms"] / 2
    assert done.data["rms"] < 0.08
    header = fits.getheader("o.fits")
    assert header["CTYPE1"] == "RA---TAN-SIP" and header["A_ORDER"] == 3
    wcs = WCS(header)
    a, d, tx, ty = true_positions(ra, dec, cd, crpix, on, distort)
    # all_world2pix uses the inverse the header has, and all_pix2world the forward
    inside = (tx > 10) & (tx < SHAPE[1] - 10) & (ty > 10) & (ty < SHAPE[0] - 10)
    back_a, back_d = wcs.all_pix2world(tx[inside], ty[inside], 1)
    gap = np.hypot((back_a - a[inside]) * np.cos(np.radians(20.0)), back_d - d[inside])
    gap = gap * 3600.0
    assert np.median(gap) < 0.05
    assert gap.max() < 0.25


@pytest.mark.parametrize("seed", [5, 21, 22, 23])
def test_stars_at_random_are_refused_and_nothing_is_written(work, seed):
    rng = np.random.default_rng(seed)
    ra, dec, mags, _, _, frame_mags, *_ = field(rng)
    x = rng.uniform(1, SHAPE[1], frame_mags.size)
    y = rng.uniform(1, SHAPE[0], frame_mags.size)
    write_field(ra, dec, mags, x, y, frame_mags)
    with pytest.raises(ValueError, match="no solution") as refused:
        get_task("platesolve").run(image="f.fits", output="o.fits", **COMMON)
    assert not Path("o.fits").exists()
    assert "--radius" in refused.value.fix


def test_a_wrong_scale_is_refused(work):
    ra, dec, mags, x, y, frame_mags, *_ = field(np.random.default_rng(6))
    write_field(ra, dec, mags, x, y, frame_mags)
    options = {**COMMON, "scale": 2.0}
    with pytest.raises(ValueError, match="no solution"):
        get_task("platesolve").run(image="f.fits", **options)


def test_a_known_rotation_and_side_save_the_search(work):
    ra, dec, mags, x, y, frame_mags, *_ = field(np.random.default_rng(7), 40.0, False)
    write_field(ra, dec, mags, x, y, frame_mags)
    done = get_task("platesolve").run(image="f.fits", rotation=-41.0, flip="no", **COMMON)
    assert done.data["flipped"] is False
    assert done.data["rotation"] == pytest.approx(-40.0, abs=0.1)


def test_a_header_s_pointing_at_its_equinox_is_read(work):
    from astropy import units as u
    from astropy.coordinates import FK5, SkyCoord
    from astropy.time import Time

    ra, dec, mags, x, y, frame_mags, *_ = field(np.random.default_rng(8), 10.0, True)
    equinox = FK5(equinox=Time(1987.7, format="jyear"))
    place = SkyCoord(*CENTRE, unit="deg").transform_to(equinox)
    header = fits.Header()
    header["RA"] = place.ra.to_string(unit=u.hourangle, sep=":", precision=2)
    header["DEC"] = place.dec.to_string(unit=u.deg, sep=":", precision=1)
    header["EPOCH"] = 1987.7
    write_field(ra, dec, mags, x, y, frame_mags, header)
    options = {key: value for key, value in COMMON.items() if key not in ("ra", "dec")}
    done = get_task("platesolve").run(image="f.fits", **options)
    assert done.data["crval"] == pytest.approx(list(CENTRE), abs=2e-4)


def test_proper_motions_bring_the_catalogue_to_the_frame(work):
    rng = np.random.default_rng(9)
    ra, dec, mags, x, y, frame_mags, *_ = field(rng, 10.0, True, noise=0.05)
    pmra = rng.normal(0, 50, ra.size)
    pmdec = rng.normal(0, 50, ra.size)
    write_field(ra, dec, mags, x, y, frame_mags, fits.Header({"DATE-OBS": "01/09/87"}))
    years = 1987.665 - 2016.0
    Table({"ra": ra - pmra * years / 3.6e6 / np.cos(np.radians(dec)),
           "dec": dec - pmdec * years / 3.6e6, "pmra": pmra, "pmdec": pmdec,
           "mag": mags}).write("cat.ecsv", overwrite=True)
    done = get_task("platesolve").run(image="f.fits", **COMMON)
    assert done.data["rms"] < 0.1
    with pytest.raises(ValueError, match="no solution"):
        get_task("platesolve").run(image="f.fits", epoch=2016.0, **COMMON)


def test_gaia_is_asked_for_the_place_and_the_radius_only(work, monkeypatch):
    ra, dec, mags, x, y, frame_mags, *_ = field(np.random.default_rng(10), 10.0, True)
    write_field(ra, dec, mags, x, y, frame_mags)
    asked = []
    rows = "".join(f"{a:.8f},{d:.8f},,,{m:.3f}\n"
                   for a, d, m in zip(ra, dec, mags, strict=True))

    class Answer(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def answer(request, timeout=0):
        from urllib.parse import parse_qs

        asked.append((request.full_url, parse_qs(request.data.decode())))
        if len(asked) == 1:
            return Answer(b"<html>SERVICE ERROR</html>")
        return Answer(("ra,dec,pmra,pmdec,phot_g_mean_mag\n" + rows).encode())

    monkeypatch.setattr("urllib.request.urlopen", answer)
    options = {key: value for key, value in COMMON.items() if key != "catalog"}
    done = get_task("platesolve").run(image="f.fits", gaia=True, epoch=2020.0, **options)
    assert done.data["significance"] > 20
    assert [url for url, _ in asked] == list(platesolve.GAIA_SERVICES)
    query = asked[-1][1]["QUERY"][0]
    assert "CIRCLE('ICRS',150.005000,19.995000," in query
    assert "frame" not in query.lower()


@pytest.mark.parametrize(
    "options, match",
    [(dict(catalog=None), "which catalogue"), (dict(gaia=True), "which catalogue"),
     (dict(sip=1), "order 1"), (dict(ra="", dec=""), "where the image is"),
     (dict(scale=None), "what scale")],
)
def test_it_says_what_it_needs(work, options, match):
    ra, dec, mags, x, y, frame_mags, *_ = field(np.random.default_rng(11))
    write_field(ra, dec, mags, x, y, frame_mags)
    with pytest.raises(ValueError, match=match):
        get_task("platesolve").run(image="f.fits", **{**COMMON, **options})


def test_a_catalogue_with_no_ra_and_a_place_with_no_stars_are_said(work):
    ra, dec, mags, x, y, frame_mags, *_ = field(np.random.default_rng(12))
    write_field(ra, dec, mags, x, y, frame_mags)
    Table({"a": [1.0]}).write("bad.ecsv", overwrite=True)
    with pytest.raises(ValueError, match="bad.ecsv has a") as told:
        get_task("platesolve").run(image="f.fits", **{**COMMON, "catalog": "bad.ecsv"})
    assert "needs ra and dec" in told.value.fix
    with pytest.raises(ValueError, match="catalogue stars within"):
        get_task("platesolve").run(image="f.fits", **{**COMMON, "ra": "10.0", "dec": "-40.0"})


def test_the_old_wcs_goes_and_what_is_not_wcs_stays(work):
    ra, dec, mags, x, y, frame_mags, *_ = field(np.random.default_rng(13), 15.0, False)
    header = fits.Header({"OBJECT": "M-92 V", "LTV1": 3.0, "CDELT1": 1.0,
                          "WAT0_001": "system=x"})
    write_field(ra, dec, mags, x, y, frame_mags, header)
    get_task("platesolve").run(image="f.fits", output="o.fits", **COMMON)
    written = fits.getheader("o.fits")
    assert written["OBJECT"] == "M-92 V"
    assert all(key not in written for key in ("LTV1", "CDELT1", "WAT0_001"))
    assert written["RADESYS"] == "ICRS" and written["PLATEN"] > 10


def test_the_tutorial_s_frames_are_refused_rather_than_solved_wrongly(work):
    """M92 in V, with a frozen piece of Gaia DR3 around the header's pointing. Nothing in them
    says their scale or orientation, and nothing we tried finds them above chance. If it ever
    does, that is news, and this test should become the real one."""
    data = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser() / "phot"
    if not (data / "m92010.fits").exists():
        pytest.skip(f"IRAF tutorial data not found under {data}")
    here = Path(__file__).parent / "data"
    image, header = fits.getdata(data / "m92010.fits", header=True)
    fits.writeto("a.fits", image, header)
    get_task("daofind").run(image="a.fits", output="c.coo", fwhmpsf=2.5, sigma=7.4,
                            threshold=5.0, epadu=1.5)
    with pytest.raises(ValueError, match="no solution"):
        get_task("platesolve").run(image="a.fits", coords="c.coo", output="o.fits",
                                   catalog=here / "gaia_m92_near.ecsv", scale=0.86,
                                   scale_error=0.03, radius=0.1)
    assert not Path("o.fits").exists()
