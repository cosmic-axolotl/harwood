"""display against IRAF 2.18: z1 and z2, and the grey levels themselves.

IRAF's display talks to an image server over a socket. For these numbers it
was pointed at a stand-in server that accepted the protocol and kept the
frame it was sent, so what is compared is not only the "z1=... z2=..." line
display prints but every grey level it wrote -- linear, log, none and user,
for a real image and a 16-bit one. The results are in
tests/data/iraf_display.json; the images are made here, from an integer
generator, so the test needs neither IRAF nor the course data.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pytest

from harwood.core import pixmask, plio, render

HERE = Path(__file__).parent / "data"
REF = json.loads((HERE / "iraf_display.json").read_text())
PHOT = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser() / "phot"


def lcg(n: int, seed: int) -> np.ndarray:
    out = np.empty(n, dtype=np.float64)
    s = seed
    for k in range(n):
        s = (s * 1103515245 + 12345) % 2147483648
        out[k] = s
    return out / 2147483648.0


def synthetic(name: str) -> np.ndarray:
    shape, seed, kind = {
        "syn_sky": ((300, 400), 7, "sky"), "syn_ramp": ((101, 37), 11, "ramp"),
        "syn_stars": ((256, 256), 3, "stars"), "syn_wide": ((9, 2000), 5, "sky"),
        "syn_1d": ((1500,), 13, "ramp"), "syn_small": ((5, 4), 17, "sky"),
        "syn_ramp_s": ((101, 37), 11, "ramp"),
    }[name]
    u = lcg(int(np.prod(shape)), seed).reshape(shape)
    if kind == "sky":
        a = 1000 + 30 * (u - 0.5) * 3.4
        a.flat[::97] += 5000
    elif kind == "ramp":
        yy, xx = np.indices(shape) if len(shape) == 2 else (0, np.arange(shape[0]))
        a = 10 * xx + 3 * np.asarray(yy) + 50 * u
    else:
        a = 200 + 20 * u
        yy, xx = np.indices(shape)
        for k in range(12):
            cx, cy = (k * 37) % shape[1], (k * 53) % shape[0]
            a += 4000 / (k + 1) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 8.0)
    a = a.astype(np.float32)
    if name.endswith("_s"):
        a = np.round(a / 10).astype(np.int16)
    # md5 of the pixels as the FITS file IRAF read holds them, big-endian
    if hashlib.md5(a.astype(a.dtype.newbyteorder(">")).tobytes()).hexdigest() \
            != REF["md5"][name]:
        # exp() may differ by an ulp between maths libraries; the others are
        # plain arithmetic and do not.
        pytest.skip(f"{name} came out different on this machine")
    return a


def image(name: str) -> np.ndarray:
    if name.startswith("syn_"):
        return synthetic(name)
    from harwood.core import images

    stem, _, section = name.partition("[")
    path = PHOT / f"{stem}.fits"
    if not path.exists():
        pytest.skip(f"course data not found: {path}")
    return images.read(str(path) + (f"[{section}" if section else ""))[0]


def _g(value: float) -> str:
    return f"{value:.7g}"


@pytest.mark.parametrize("case", REF["zscale"],
                         ids=[f"{c['image']}-{c['params']}" for c in REF["zscale"]])
def test_zscale_matches_iraf(case):
    data = image(case["image"])
    params = case["params"]
    nsample = max(100, params.get("nsample", 1000))
    contrast = 0.0 if params.get("zscale") == "no" else params.get("contrast", 0.25)
    z1, z2 = render.zscale(data, contrast, nsample)
    assert (_g(z1), _g(z2)) == (_g(case["z1"]), _g(case["z2"]))


@pytest.mark.parametrize("case", REF["levels"],
                         ids=[f"{c['image']}-{c['variant']}" for c in REF["levels"]])
def test_grey_levels_match_iraf(case, tmp_path):
    from harwood.tasks.tv import DisplayParams, display_limits

    data = synthetic(case["image"])
    lut = tmp_path / "lut2.txt"
    lut.write_text(REF["lut2.txt"])
    params = dict(case["params"])
    for flag in ("zscale", "zrange"):
        if flag in params:
            params[flag] = params[flag] == "yes"
    if "lutfile" in params:
        params["lutfile"] = str(lut)
    chosen = DisplayParams(image="x.fits", **params)
    z1, z2, table, _ = display_limits(chosen, data.astype(float), None)
    assert (_g(z1), _g(z2)) == (_g(case["z1"]), _g(case["z2"]))

    expected = np.frombuffer(zlib.decompress(base64.b64decode(case["levels"])),
                             np.uint8).reshape(case["shape"])
    got = render.device_levels(data, z1, z2, chosen.ztrans, table,
                               short=data.dtype == np.int16)
    assert np.array_equal(np.mod(got, 256), expected)


def test_display_task_reports_iraf_limits(tmp_path, monkeypatch):
    from astropy.io import fits

    import harwood
    from harwood.core import display

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARWOOD_DISPLAY", "headless")
    display.reset()
    fits.PrimaryHDU(synthetic("syn_sky")).writeto("sky.fits")
    case = next(c for c in REF["zscale"]
                if c["image"] == "syn_sky" and c["params"] == {"contrast": 0.5})
    result = harwood.run("display", image="sky.fits", contrast=0.5, wait=False)
    assert (result.data["z1"], result.data["z2"]) == (float(_g(case["z1"])),
                                                     float(_g(case["z2"])))
    display.reset()


def test_display_needs_z1_z2_without_zscale_or_zrange(tmp_path, monkeypatch):
    from astropy.io import fits

    import harwood
    from harwood.core import display

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARWOOD_DISPLAY", "headless")
    display.reset()
    fits.PrimaryHDU(synthetic("syn_ramp")).writeto("ramp.fits")
    with pytest.raises(Exception, match="--z1 and --z2"):
        harwood.run("display", image="ramp.fits", zscale=False, zrange=False, wait=False)
    result = harwood.run("display", image="ramp.fits", zscale=False, zrange=False,
                         z1=10, z2=20, wait=False)
    assert (result.data["z1"], result.data["z2"]) == (10.0, 20.0)
    display.reset()


def test_bad_pixels_are_left_out_of_the_sample():
    data = synthetic("syn_ramp").astype(float)
    spoiled = data.copy()
    spoiled[:10] = 1e6
    bpm = np.zeros(data.shape, dtype=int)
    bpm[:10] = 1
    # Flagged pixels never enter the sample; the ones after them take their place.
    sample = render._zsample(spoiled, 1000, None, bpm)
    assert sample.max() < 1e6 and sample.size == 1000
    assert render.zscale(spoiled, bpm=bpm)[1] < 1e6
    nan = data.copy()
    nan[5, 5] = np.nan
    assert np.isfinite(render._zsample(nan, 1000, None, None)).all()


# ------------------------------------------------------------------ masks


def test_pl_masks_written_by_iraf_decode():
    """mk1.pl, mk3.pl and mk1sec.pl were written by IRAF's imcopy from these
    arrays; mk1sec from the section [11:40,21:50], with LTV1 -10 and LTV2 -20."""
    mk1 = np.zeros((60, 45), np.int16)
    mk1[10:20, 5] = 1
    mk1[30, :] = 3
    mk1[40:45, 20:25] = 7
    mk1[0, 0] = 5000
    mk1[59, 44] = 2
    mk3 = np.zeros((4, 9, 8), np.int16)
    mk3[1, 2, 3] = 1
    mk3[3, :, 7] = 9
    mask, header = plio.read(HERE / "mk1.pl")
    assert np.array_equal(mask, mk1)
    assert header["LTV1"] == -3.0
    assert np.array_equal(plio.read(HERE / "mk3.pl")[0], mk3)
    section, header = plio.read(HERE / "mk1sec.pl")
    assert np.array_equal(section, mk1[20:50, 10:40])
    assert (header["LTV1"], header["LTV2"]) == (-10.0, -20.0)


def test_mask_follows_a_section_in_physical_coordinates():
    """The image is the section [11:40,21:50] of the detector; the mask is
    the whole detector. Matching by LTV gives the mask's pixels under the
    section's, not its first 30 x 30."""
    whole, _ = plio.read(HERE / "mk1.pl")
    header = {"LTV1": -10.0, "LTV2": -20.0}
    got = pixmask.match(whole.astype(np.int32), None, header, (30, 30))
    assert np.array_equal(got, whole[20:50, 10:40])
    # and the other way: a mask of the section, an image of the whole
    part, part_header = plio.read(HERE / "mk1sec.pl")
    back = pixmask.match(part.astype(np.int32), part_header, None, (60, 45))
    assert np.array_equal(back[20:50, 10:40], part)
    assert not back[:20].any() and not back[:, :10].any()


def test_mask_names(tmp_path):
    from astropy.io import fits

    shape = (6, 7)
    assert pixmask.read("", None, shape) is None
    assert pixmask.read("BPM", {}, shape) is None            # no keyword: no mask
    assert not pixmask.read("EMPTY", None, shape).any()
    assert pixmask.read("^EMPTY", None, shape).all()
    section = pixmask.read("[*:3,2:4]", None, shape)
    assert section.sum() == 9 and section[1, 0] == 1 and section[1, 3] == 1
    flagged = np.zeros(shape, np.int16)
    flagged[2, 3] = 4
    flagged[0, 0] = -1
    fits.PrimaryHDU(flagged).writeto(tmp_path / "bad.fits")
    via_keyword = pixmask.read("BPM", {"BPM": "bad.fits"}, shape, tmp_path)
    assert via_keyword[2, 3] == 4 and via_keyword[0, 0] == 0
    assert pixmask.read("!MYMASK", {"MYMASK": "bad.fits"}, shape, tmp_path).sum() == 4
    inverted = pixmask.read("^bad.fits", None, shape, tmp_path)
    assert inverted[2, 3] == 0 and inverted[0, 0] == 1
    (tmp_path / "regions").write_text("2 2 1 5\n3 6 4 4\n7 1\n")
    regions = pixmask.read("regions", None, shape, tmp_path)
    assert regions[0, 1] == 2 and regions[4, 1] == 2       # a column: 2
    assert regions[3, 2] == 3 and regions[3, 5] == 3       # a line: 3
    assert regions[0, 6] == 2                              # one pixel counts as a column


def test_user_lut_as_iraf_builds_it(tmp_path):
    lut = tmp_path / "lut.txt"
    lut.write_text("0 0\n500 100\n1500 200\n")
    table = render.UserLut(lut)
    assert (table.z1, table.z2) == (0.0, 1500.0)
    # dsulut.x: the last entry is the last segment's start plus its slope
    # times 4095 -- the 250 IRAF wrote for every pixel above 200.
    assert table.table[-1] == 250
    assert table.table[0] == 0
