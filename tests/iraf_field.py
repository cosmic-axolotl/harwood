"""A star field built the same way every time, for checking against IRAF.

`tests/data/imexam_iraf_*.log` hold what IRAF 2.18's own imexamine printed
for this field (see `make_reference`). The test rebuilds the field from this
function, so the comparison needs no data files beyond those logs -- and
float32 with a fixed seed makes the pixels identical bit for bit.
"""

import numpy as np

#: x, y (1-based), peak, FWHM, beta, axis ratio, angle (degrees).
STARS = [
    (40.3, 45.7, 9000.0, 1.6, 3.0, 1.0, 0.0),     # narrow: the correction table
    (110.6, 52.2, 6000.0, 2.5, 2.5, 1.0, 0.0),
    (160.2, 140.9, 4000.0, 3.5, 3.5, 1.0, 0.0),
    (60.8, 150.4, 2500.0, 5.0, 2.5, 1.0, 0.0),    # wide: a large aperture
    (130.1, 100.6, 5000.0, 3.0, 3.0, 0.6, 30.0),  # elongated
    (95.5, 170.5, 600.0, 2.8, 3.0, 1.0, 0.0),     # faint
]

#: Where the cursor is put for each star: a pixel or so off, as a hand would.
CURSOR = [(41.0, 45.0), (110.0, 53.0), (161.0, 141.0), (60.0, 151.0),
          (131.0, 100.0), (96.0, 170.0)]


def field() -> np.ndarray:
    ny, nx = 200, 200
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    image = np.full((ny, nx), 100.0)
    for x, y, peak, fwhm, beta, ratio, angle in STARS:
        alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
        t = np.radians(angle)
        u = (xx - x) * np.cos(t) + (yy - y) * np.sin(t)
        v = -(xx - x) * np.sin(t) + (yy - y) * np.cos(t)
        image += peak * (1 + (u / alpha) ** 2 + (v / (alpha * ratio)) ** 2) ** (-beta)
    image += np.random.default_rng(7).normal(0, 5.0, image.shape)
    return image.astype(np.float32)


def dirty_field() -> np.ndarray:
    """field() with what the cleaning and the sky's rejection are for: a hot
    pixel inside each centring box, two pixels from the star, and cosmic-ray
    blobs and a faint neighbour in each sky annulus."""
    image = field().astype(float)
    ny, nx = image.shape
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    rng = np.random.default_rng(11)
    for x, y, *_ in STARS:
        ix, iy = int(round(x)), int(round(y))
        image[iy + 2 - 1, ix + 1 - 1] += 700.0
        for _ in range(3):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(12.0, 18.0)
            bx = int(round(x + radius * np.cos(angle)))
            by = int(round(y + radius * np.sin(angle)))
            image[by - 2:by + 1, bx - 2:bx + 1] += rng.uniform(30.0, 300.0, (3, 3))
        angle = rng.uniform(0, 2 * np.pi)
        fx, fy = x + 15.0 * np.cos(angle), y + 15.0 * np.sin(angle)
        image += 150.0 * np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * 1.3 ** 2))
    return image.astype(np.float32)


def make_reference(directory) -> None:
    """Write the field and the cursor file IRAF is run on."""
    from pathlib import Path

    from astropy.io import fits

    directory = Path(directory)
    fits.writeto(directory / "field.fits", field(), overwrite=True)
    (directory / "cursor").write_text(
        "".join(f"{x} {y} 1 a\n" for x, y in CURSOR))


def fixpix_case():
    """An image and a mask with every shape fixpix decides differently on.

    Mask values: 1 for the ordinary cases; 5 forces a region along lines and
    7 along columns, for linterp=5 cinterp=7.
    """
    rng = np.random.default_rng(11)
    ny, nx = 40, 50
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    image = (200 + 3 * xx + 2 * yy + rng.normal(0, 4, (ny, nx))).astype(np.float32)
    mask = np.zeros((ny, nx), np.int16)
    mask[5:35, 10] = 1            # a bad column: fixed along lines
    mask[20, 15:30] = 1           # a bad row segment: fixed along columns
    mask[8:11, 30:35] = 1         # a 5 wide, 3 tall block
    mask[25, 40] = 1              # one hot pixel: a tie, so along the line
    mask[0:4, 45] = 1             # a column stub at the bottom edge
    mask[30, 0:3] = 1             # a row stub at the left edge
    mask[12:16, 20] = 1           # an L: a column ...
    mask[15, 20:26] = 1           # ... and a row sharing a pixel
    mask[33:36, 30:38] = 5        # wide block, forced along lines
    mask[2:4, 2:8] = 7            # wide block, forced along columns
    return image, mask


def combine_stack():
    """Seven frames of one field for imcombine: different sky levels and
    exposure times, a star, cosmic rays in single frames, a few low pixels.
    Returns [(data, exptime), ...]."""
    rng = np.random.default_rng(23)
    ny, nx = 40, 50
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    star = 2000 * np.exp(-((xx - 25.3) ** 2 + (yy - 20.6) ** 2) / (2 * 1.6 ** 2))
    frames = []
    for k, exptime in enumerate([10.0, 12.0, 10.0, 15.0, 10.0, 20.0, 11.0]):
        level = 300.0 + 40 * k
        gain = exptime / 10.0
        image = (level + star) * gain + rng.normal(0, 8, (ny, nx))
        image[rng.integers(0, ny, 6), rng.integers(0, nx, 6)] += rng.uniform(800, 5000, 6)
        image[rng.integers(0, ny, 2), rng.integers(0, nx, 2)] -= 400
        frames.append((image.astype(np.float32), exptime))
    return frames


def crowded_field():
    """A field for daofind: a hundred gaussian stars, pairs closer than a
    FWHM and a few FWHM apart, stars on and over the edges, cosmic rays (too
    sharp), a galaxy (too broad), a hot column and a sky gradient."""
    rng = np.random.default_rng(31)
    ny, nx = 160, 180
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    image = 200.0 + 0.3 * xx - 0.2 * yy

    def star(x, y, peak, sigma=1.3):
        return peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    for _ in range(100):
        image += star(rng.uniform(-2, nx + 3), rng.uniform(-2, ny + 3),
                      10 ** rng.uniform(1.5, 4.0))
    for x, y, dx, dy in ((40.2, 30.7, 2.1, 0.4), (90.6, 120.3, 4.3, -3.2),
                         (140.1, 60.5, 0.0, 6.0)):          # pairs
        image += star(x, y, 3000.0) + star(x + dx, y + dy, 1200.0)
    for x, y in ((1.2, 80.4), (180.0, 20.6), (70.3, 1.0), (120.7, 159.6),
                 (2.0, 2.0)):                                # on the edges
        image += star(x, y, 2500.0)
    image += star(100.4, 40.6, 600.0, sigma=6.0)             # a galaxy
    image[rng.integers(5, ny - 5, 8), rng.integers(5, nx - 5, 8)] += \
        rng.uniform(500, 4000, 8)                            # cosmic rays
    image[:, 150] += 150.0                                   # a hot column
    image += rng.normal(0, 6.0, image.shape)
    return image.astype(np.float32)


#: Where the frames of align_frames() sit relative to the first, (x, y).
ALIGN_SHIFTS = [(0.0, 0.0), (2.37, -1.64), (-4.71, 3.18), (0.42, 0.93)]


def align_frames():
    """Four exposures of one field for imcentroid and imalign: thirty stars
    and a sky that changes, each frame moved by ALIGN_SHIFTS and with its own
    noise. Returns (frames, coords): the stars' positions in the first frame,
    a hand's worth off -- plus one that leaves the image in the third frame,
    one on blank sky and one on a pair."""
    rng = np.random.default_rng(41)
    ny, nx = 120, 140
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    stars = [(rng.uniform(8, nx - 8), rng.uniform(8, ny - 8), 10 ** rng.uniform(2.5, 3.8))
             for _ in range(30)]
    stars += [(70.4, 60.2, 3000.0), (72.6, 61.0, 1500.0)]      # a pair
    stars += [(3.5, 100.3, 2500.0)]                           # leaves frame 3
    frames = []
    for k, (dx, dy) in enumerate(ALIGN_SHIFTS):
        image = 150.0 + 20 * k + 0.1 * xx
        for x, y, peak in stars:
            image = image + peak * np.exp(-((xx - x - dx) ** 2 + (yy - y - dy) ** 2)
                                          / (2 * 1.4 ** 2))
        image += np.random.default_rng(50 + k).normal(0, 5.0, image.shape)
        frames.append(image.astype(np.float32))
    picks = sorted(range(30), key=lambda i: -stars[i][2])[:12]
    coords = [(round(stars[i][0] + rng.uniform(-1, 1), 1),
               round(stars[i][1] + rng.uniform(-1, 1), 1)) for i in picks]
    coords += [(70.0, 60.0), (3.0, 100.0), (25.0, 115.0)]    # pair, edge, sky
    return frames, coords


def spectra_frame():
    """A long-slit frame for apall: DISPAXIS=2, the dispersion along the
    lines (y), 300 of them, and 100 columns across. Two spectra, one bright
    and one faint, each wandering by a few pixels along the dispersion; a
    sky with a slope across the slit and emission lines along it; absorption
    lines in the brighter star; cosmic rays, some in the apertures; noise for
    a gain of 2 e-/ADU and a read noise of 5 e-."""
    rng = np.random.default_rng(61)
    ny, nx = 300, 100
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    t = (yy - 150.0) / 150.0
    image = 80.0 + 0.3 * (xx - 50.0)
    for line, strength in ((40.0, 300.0), (122.5, 120.0), (233.0, 500.0)):
        image += strength * np.exp(-((yy - line) ** 2) / (2 * 1.3 ** 2))
    for centre, curve, tilt, flux, sigma in ((34.6, 2.5, 1.8, 6000.0, 1.7),
                                             (68.3, -1.5, 1.2, 1500.0, 1.9)):
        position = centre + tilt * t + curve * t ** 2
        continuum = flux * (1.0 + 0.3 * t) * (1 - 0.6 * np.exp(-((yy - 180.0) / 2.0) ** 2))
        image += continuum / (np.sqrt(2 * np.pi) * sigma) * \
            np.exp(-((xx - position) ** 2) / (2 * sigma ** 2))
    gain, rdnoise = 2.0, 5.0
    electrons = rng.poisson(np.clip(image * gain, 0, None)).astype(float)
    image = (electrons + rng.normal(0, rdnoise, image.shape)) / gain
    for y, x, value in ((60, 35, 4000.0), (61, 35, 2500.0), (150, 70, 3000.0),
                        (210, 33, 5000.0), (90, 10, 2000.0), (275, 69, 1800.0)):
        image[y - 1, x - 1] += value
    return image.astype(np.float32)


def arc_spectrum(flip=False):
    """A 1-D thorium arc for identify: 1024 pixels over about 4010-4310 A
    with a gently curved dispersion, a few hundred lines of
    tests/data/arc_lines.dat at random strengths, ten lines that are not in
    the list, a sloping continuum and noise. `flip` reverses it, as a
    spectrograph with the red end on the left gives."""
    from pathlib import Path

    rng = np.random.default_rng(71)
    text = (Path(__file__).parent / "data" / "arc_lines.dat").read_text()
    lines = np.array([float(row.split()[0]) for row in text.splitlines()
                      if row.strip() and not row.startswith("#")])
    pix = np.arange(1, 1025, dtype=float)
    t = (pix - 512.5) / 512.0
    wave = 4160.0 + 147.0 * t + 2.2 * t ** 2 - 0.6 * t ** 3
    image = 40.0 + 10.0 * t
    for w in lines:
        if wave[0] - 2 < w < wave[-1] + 2:
            p = np.interp(w, wave, pix)
            image += 10 ** rng.uniform(1.6, 3.8) * np.exp(-((pix - p) / 1.4) ** 2 / 2)
    for p in rng.uniform(30, 1000, 10):
        image += 10 ** rng.uniform(2.0, 3.0) * np.exp(-((pix - p) / 1.4) ** 2 / 2)
    image += rng.normal(0, 3.0, pix.size)
    image = image.astype(np.float32)
    return image[::-1].copy() if flip else image


def star_spectrum(flip=False):
    """A 1-D stellar spectrum on arc_spectrum()'s pixels: a sloping
    continuum, absorption lines of several widths, one emission line,
    noise."""
    rng = np.random.default_rng(83)
    pix = np.arange(1, 1025, dtype=float)
    image = 2000.0 + 0.8 * (pix - 512)
    for p, depth, width in ((120.4, 900, 2.0), (300.7, 400, 1.2), (512.2, 1200, 3.5),
                            (700.9, 300, 1.0), (888.3, 700, 2.4), (1000.6, 500, 1.6)):
        image -= depth * np.exp(-((pix - p) / width) ** 2 / 2)
    image += 1500 * np.exp(-((pix - 640.5) / 1.3) ** 2 / 2)
    image += rng.normal(0, 15.0, pix.size)
    image = image.astype(np.float32)
    return image[::-1].copy() if flip else image


#: A made-up flux standard for the calibration chain: band centre, AB-like
#: magnitude and band width, every 20 A over the synthetic spectra's range.
FAKE_STANDARD = [(4040.0 + 20 * k, 10.0 + 0.3 * np.sin(k / 3.0) + 0.002 * k, 16.0)
                 for k in range(13)]


def standard_spectra():
    """Two exposures of FAKE_STANDARD and one object, on the grid IRAF's
    dispcor gives the synthetic arc (4014.2985 A + 0.28769 A/px, 1024 px):
    counts = flux x a smooth instrument response x extinction x time x dw.
    Returns {name: (counts, airmass, exptime)}."""
    from pathlib import Path

    w1, dw, n = 4014.29852458081, 0.28769335609626, 1024
    wave = w1 + dw * np.arange(n)
    ext = np.loadtxt(Path(__file__).parent / "data" / "kpnoextinct.dat", comments="#")
    k = np.interp(wave, ext[:, 0], ext[:, 1])
    bands = np.array(FAKE_STANDARD)
    mag = np.interp(wave, bands[:, 0], bands[:, 1])
    flam = 3.68e-20 * 10 ** (-0.4 * mag) * 2.997925e18 / wave ** 2
    response = 3e15 * (1 + 0.4 * np.sin((wave - 4000) / 90.0)) * (wave / 4100) ** 2
    rng = np.random.default_rng(97)
    out = {}
    for name, airmass, exptime, scale in (("std1", 1.15, 300.0, 1.0),
                                          ("std2", 1.85, 600.0, 1.0),
                                          ("obj", 1.40, 900.0, 0.3)):
        counts = scale * flam * response * 10 ** (-0.4 * k * airmass) * exptime * dw
        counts = counts + rng.normal(0, 0.003 * counts.mean(), n)
        out[name] = (counts.astype(np.float32), airmass, exptime)
    return out


def standard_cases():
    """The spectra IRAF's standard was run on for tests/data/iraf_standard:
    the two standard exposures, the same extinction-corrected (EX-FLAG 0),
    without AIRMASS (RA, DEC and HA or ST instead), without anything,
    beam-switched object and sky pairs (one sky on another dispersion), an
    infrared one for the J blackbody, and three apertures of one frame with
    their own dispersions. Returns {name: (apertures, header)} with
    apertures a list of (number, w1, dw, flux, title)."""
    from pathlib import Path

    w1, dw, n = 4014.29852458081, 0.28769335609626, 1024
    spectra = standard_spectra()
    ext = np.loadtxt(Path(__file__).parent / "data" / "kpnoextinct.dat", comments="#")
    wave = w1 + dw * np.arange(n)
    k = np.interp(wave, ext[:, 0], ext[:, 1])
    rng = np.random.default_rng(5)
    out = {}

    def one(name, data, start=w1, step=dw, **header):
        header.setdefault("OBJECT", "fakestar")
        out[name] = ([(1, start, step, np.asarray(data, np.float32), header["OBJECT"])],
                     header)

    c1, am1, t1 = spectra["std1"]
    c2, am2, t2 = spectra["std2"]
    one("std1", c1, AIRMASS=am1, EXPTIME=t1)
    one("std2", c2, AIRMASS=am2, EXPTIME=t2)
    one("ex1", (c1 * 10 ** (0.4 * k * am1)).astype(np.float32), AIRMASS=am1, EXPTIME=t1,
        **{"EX-FLAG": 0})
    one("noam", c1, EXPTIME=t1, RA="12:34:56.7", DEC="31:12:45", HA="-01:22:10")
    one("noamst", c1, EXPTIME=t1, RA="12:34:56.7", DEC="31:12:45", ST="13:58:01")
    one("ask", c2)
    sky = (2000.0 + 0.5 * np.arange(n) + rng.normal(0, 5, n)).astype(np.float32)
    sky2 = (2100.0 + 0.4 * np.arange(n) + rng.normal(0, 5, n)).astype(np.float32)
    one("b1", c1 + sky, AIRMASS=am1, EXPTIME=t1, OFLAG=1)
    one("b2", sky, AIRMASS=am1, EXPTIME=t1, OFLAG=0)
    one("b3", sky2, AIRMASS=am2, EXPTIME=t2, OFLAG=0)
    one("b4", c2 + sky2, AIRMASS=am2, EXPTIME=t2, OFLAG=1)
    one("b5", c2 + sky2, AIRMASS=am2, EXPTIME=t2, OFLAG=1)
    one("b6", c1 + sky, AIRMASS=am1, EXPTIME=t1, OFLAG=1)
    wsky = w1 - 1.7 + dw * 1.03 * np.arange(n)
    one("b7", 2000.0 + 0.002 * (wsky - 4000) ** 2 + 30 * np.sin(wsky / 9.0), w1 - 1.7,
        dw * 1.03, AIRMASS=am1, EXPTIME=t1, OFLAG=0)
    one("b8", c1 + (2000.0 + 0.002 * (wave - 4000) ** 2 + 30 * np.sin(wave / 9.0))
        .astype(np.float32), AIRMASS=am1, EXPTIME=t1, OFLAG=1)
    wir = 11000.0 + 2.0 * np.arange(n)
    one("ir1", 5e4 * (1 + 0.1 * np.sin(wir / 300.0)), 11000.0, 2.0, AIRMASS=1.2,
        EXPTIME=100.0)
    apertures = []
    bands = np.array(FAKE_STANDARD)
    for ap, (start, step, scale) in {1: (w1, dw, 1.0), 2: (w1 + 3.1, dw * 0.98, 0.7),
                                     3: (w1 - 2.0, dw * 1.01, 1.3)}.items():
        wv = start + step * np.arange(n)
        kk = np.interp(wv, ext[:, 0], ext[:, 1])
        mag = np.interp(wv, bands[:, 0], bands[:, 1])
        flam = 3.68e-20 * 10 ** (-0.4 * mag) * 2.997925e18 / wv ** 2
        response = scale * 3e15 * (1 + 0.3 * np.cos((wv - 4000) / 70.0 + ap)) * \
            (wv / 4100) ** 2
        counts = flam * response * 10 ** (-0.4 * kk * 1.3) * 400.0 * step
        counts = counts + rng.normal(0, 0.003 * counts.mean(), n)
        apertures.append((ap, start, step, counts.astype(np.float32), f"fake aperture {ap}"))
    out["ms3"] = (apertures, {"OBJECT": "fakestar", "AIRMASS": 1.3, "EXPTIME": 400.0})
    return out
