"""Cosmic rays: finding them, and knowing whether you found them.

Two kinds of check. The detector is measured against injected tracks, where
the answer is known exactly. And `verify` -- which measures the detector
against a set of exposures -- is itself measured in closed loop, because a
diagnostic that has not been checked is worse than none: it looks like
evidence.
"""

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from harwood.core import cosmic

GAIN, RDNOISE, SKY = 2.0, 6.0, 220.0


def scene(seed=0, ny=300, nx=300, nstars=90, sigma=(0.9, 1.8)):
    """A star field with no cosmic rays, and a mask of where the stars are."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ny, 0:nx]
    image = np.full((ny, nx), SKY)
    stars = np.zeros((ny, nx), dtype=bool)
    for _ in range(nstars):
        cy, cx = rng.uniform(6, ny - 6), rng.uniform(6, nx - 6)
        peak = 10 ** rng.uniform(2.0, 4.1)
        width = rng.uniform(*sigma)
        light = peak * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / width ** 2)
        image += light
        stars |= light > 0.2 * peak
    return image, stars


def strike(image, rng, ntracks=80):
    """Add cosmic rays, and say exactly which pixels they hit."""
    ny, nx = image.shape
    hit = np.zeros(image.shape, dtype=bool)
    struck = image.copy()
    for _ in range(ntracks):
        cy, cx = int(rng.integers(4, ny - 4)), int(rng.integers(4, nx - 4))
        step = int(rng.integers(-1, 2))
        for offset in range(int(rng.integers(1, 4))):
            row, column = cy + offset * step, cx + offset
            if 0 <= row < ny and 0 <= column < nx:
                struck[row, column] += 10 ** rng.uniform(3.0, 4.3)
                hit[row, column] = True
    return struck, hit


def observe(image, rng):
    """Photon noise and read noise on top of a clean scene."""
    return (rng.poisson(np.clip(image * GAIN, 0, None)) / GAIN
            + rng.normal(0, RDNOISE / GAIN, image.shape))


@pytest.fixture
def struck_field():
    rng = np.random.default_rng(4)
    clean, stars = scene(4)
    struck, hit = strike(clean, rng)
    return observe(struck, rng), hit, stars


def _score(mask, hit, stars):
    near = binary_dilation(hit, np.ones((3, 3), bool))
    return {
        "completeness": (mask & hit).sum() / hit.sum(),
        "false": int((mask & ~near).sum()),
        "on stars": int((mask & ~near & stars).sum()),
    }


# ------------------------------------------------------------- the detector


def test_it_finds_the_tracks_that_were_put_in(struck_field):
    data, hit, stars = struck_field
    result = cosmic.lacosmic(data, gain=GAIN, rdnoise=RDNOISE)

    score = _score(result.mask, hit, stars)
    assert score["completeness"] > 0.95, score
    assert result.count > 0 and result.passes >= 1


def test_a_frame_with_no_cosmic_rays_gives_almost_nothing():
    """The other half of the claim. A detector that finds tracks in a clean
    frame has learned to find noise."""
    rng = np.random.default_rng(11)
    clean, _ = scene(11)
    result = cosmic.lacosmic(observe(clean, rng), gain=GAIN, rdnoise=RDNOISE)

    assert result.fraction < 0.0005, f"{result.count} pixels on a clean frame"


def test_strict_growth_protects_stars_at_a_measured_cost():
    """The documented departure from the paper, with the numbers it is
    claimed on. Growing into a neighbour normally asks only that the
    neighbour be significant; keeping the fine-structure test as well is what
    stops a detection walking along a star.
    """
    strict = {"completeness": 0.0, "on stars": 0}
    loose = {"completeness": 0.0, "on stars": 0}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        clean, stars = scene(seed, sigma=(0.7, 1.5))     # undersampled
        data, hit = strike(clean, rng)
        data = observe(data, rng)
        for target, flag in ((strict, True), (loose, False)):
            score = _score(cosmic.lacosmic(data, gain=GAIN, rdnoise=RDNOISE,
                                           strict_growth=flag).mask, hit, stars)
            target["completeness"] += score["completeness"] / 3
            target["on stars"] += score["on stars"]

    assert strict["on stars"] < 0.3 * max(loose["on stars"], 1), (
        f"strict flagged {strict['on stars']} star pixels against "
        f"{loose['on stars']}: the protection is the point of the option")
    assert strict["completeness"] > loose["completeness"] - 0.02, \
        "and it must not cost more than a per cent or two of completeness"


def test_it_agrees_with_the_reference_implementation():
    """Reimplementing a published algorithm is how subtle errors get in, so
    this is checked against astroscrappy rather than against belief.

    Configured as published -- `strict_growth=False` -- the two must find
    substantially the same pixels. They did not, once: the growth here relaxed
    the threshold on the first ring as well as the second, which let a
    detection walk along any coherent edge.
    """
    astroscrappy = pytest.importorskip("astroscrappy")

    rng = np.random.default_rng(2)
    clean, _ = scene(2)
    data, _ = strike(clean, rng)
    data = observe(data, rng)

    mine = cosmic.lacosmic(data, gain=GAIN, rdnoise=RDNOISE,
                           strict_growth=False).mask
    theirs, _ = astroscrappy.detect_cosmics(data, gain=GAIN, readnoise=RDNOISE,
                                            sepmed=False, cleantype="medmask",
                                            niter=4)
    overlap = (mine & theirs).sum() / max(theirs.sum(), 1)
    assert overlap > 0.8, f"only {100 * overlap:.0f}% of astroscrappy's pixels"
    assert 0.7 < mine.sum() / theirs.sum() < 1.4, \
        f"{mine.sum()} against {theirs.sum()} in total"


def test_the_fast_median_costs_little(struck_field):
    """It is an approximation and says so; what it costs has to be measured
    rather than waved at."""
    data, hit, stars = struck_field
    exact = _score(cosmic.lacosmic(data, gain=GAIN, rdnoise=RDNOISE).mask, hit, stars)
    quick = _score(cosmic.lacosmic(data, gain=GAIN, rdnoise=RDNOISE,
                                   separable=True).mask, hit, stars)

    assert abs(quick["completeness"] - exact["completeness"]) < 0.03
    assert quick["on stars"] <= exact["on stars"] + 2


def test_saturated_pixels_are_left_alone():
    """A flat-topped star has a large Laplacian all round its edge, and
    replacing its core with a median would invent a measurement."""
    rng = np.random.default_rng(7)
    clean, _ = scene(7)
    clean[150:154, 150:154] = 70000.0
    result = cosmic.lacosmic(observe(clean, rng), gain=GAIN, rdnoise=RDNOISE,
                             satlevel=65535.0)

    assert not result.mask[150:154, 150:154].any()
    assert result.saturated >= 16


def test_cleaning_uses_neighbours_that_are_not_themselves_flagged():
    """A track is several pixels wide; a median that includes its neighbours
    swaps one bad value for another."""
    data = np.full((40, 40), 100.0)
    mask = np.zeros((40, 40), dtype=bool)
    data[20, 18:23] = 9000.0
    mask[20, 18:23] = True

    cleaned = cosmic._clean(data, mask)
    assert np.allclose(cleaned[20, 18:23], 100.0, atol=1.0)
    assert np.isfinite(cleaned).all()


def test_the_iraf_algorithm_still_runs_and_is_worse_on_a_sharp_star():
    """Kept so an old reduction can be reproduced -- and the reason it is not
    the default, measured rather than asserted."""
    rng = np.random.default_rng(5)
    clean, stars = scene(5, sigma=(0.6, 1.0))            # badly undersampled
    data, hit = strike(clean, rng)
    data = observe(data, rng)

    old = _score(cosmic.iraf_cosmicrays(data, threshold=25.0, fluxratio=2.0).mask,
                 hit, stars)
    new = _score(cosmic.lacosmic(data, gain=GAIN, rdnoise=RDNOISE).mask, hit, stars)
    assert new["completeness"] > old["completeness"]


# --------------------------------------------------------------- verify


def _stack(nframes=5, seed=0, ntracks=60, scale=None):
    """A set of exposures of one field, with the cosmic rays known per frame."""
    rng = np.random.default_rng(seed)
    clean, _ = scene(seed)
    frames, truth = [], []
    for index in range(nframes):
        level = clean * (scale[index] if scale else 1.0)
        struck, hit = strike(level, rng, ntracks)
        frames.append(observe(struck, rng))
        truth.append(hit)
    return frames, np.array(truth)


def test_verify_recovers_a_completeness_it_can_be_checked_against():
    """The diagnostic, in closed loop. Tracks are injected frame by frame, so
    what the set *should* find is known -- and a diagnostic that has not been
    checked is worse than none, because it looks like evidence."""
    frames, truth = _stack(seed=3)

    def detector(data):
        return cosmic.lacosmic(data, gain=GAIN, rdnoise=RDNOISE)

    score = cosmic.verify(frames, detector, gain=GAIN, rdnoise=RDNOISE)

    assert score["static"], "nothing moved between these frames"
    assert score["completeness"] > 0.75, score["completeness"]
    found = sum(row["frames agree"] for row in score["rows"])
    assert 0.5 * truth.sum() < found < 1.5 * truth.sum(), (
        f"the set found {found} where {truth.sum()} were injected")


def test_verify_scales_the_frames_before_comparing_them():
    """Transparency changes between exposures. Comparing unscaled frames
    counts that real change as a cosmic ray in every bright pixel -- which is
    what it did, on a real night, until this was put in."""
    frames, _ = _stack(seed=6, scale=[0.8, 0.9, 1.0, 1.1, 1.2])

    def detector(data):
        return cosmic.lacosmic(data, gain=GAIN, rdnoise=RDNOISE)

    score = cosmic.verify(frames, detector, gain=GAIN, rdnoise=RDNOISE)
    assert min(score["scaled_by"]) < 1.0 < max(score["scaled_by"])
    assert score["static"], "scaling should make a dimmed frame comparable"


def test_verify_notices_when_the_scene_is_not_static():
    """A cosmic ray does not care how bright the pixel it lands on is. When
    the disagreement follows the light, the frames differ for another reason
    and the score measures that instead of the detector."""
    rng = np.random.default_rng(9)
    frames = []
    for index in range(4):
        # The source varies in shape, not just in level: the seeing changed.
        moved, _ = scene(9, sigma=(0.9 + 0.25 * index, 1.8 + 0.25 * index))
        frames.append(observe(moved, rng))

    def detector(data):
        return cosmic.lacosmic(data, gain=GAIN, rdnoise=RDNOISE)

    score = cosmic.verify(frames, detector, gain=GAIN, rdnoise=RDNOISE)
    assert not score["static"]
    assert score["rate_bright"] > 3 * score["rate_faint"]


def test_verify_refuses_two_frames():
    """With two, a pixel that disagrees gives no way to say which frame is
    the odd one out."""
    frames, _ = _stack(nframes=2)
    with pytest.raises(ValueError) as raised:
        cosmic.verify(frames, lambda data: cosmic.lacosmic(data))
    assert "three" in str(raised.value) and "odd one out" in raised.value.fix


def test_verify_refuses_frames_that_are_not_aligned():
    frames, _ = _stack(nframes=3)
    with pytest.raises(ValueError) as raised:
        cosmic.verify(frames, lambda data: cosmic.lacosmic(data), aligned=False)
    assert "imalign" in raised.value.fix


# ----------------------------------------------------------------- the task


@pytest.fixture
def frames_on_disk(tmp_path, monkeypatch):
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    frames, _ = _stack(nframes=4, seed=8)
    for index, frame in enumerate(frames, start=1):
        fits.PrimaryHDU(frame.astype(np.float32), fits.Header(
            {"OBJECT": "field", "EXPTIME": 600.0})).writeto(
                tmp_path / f"f{index}.fits")
    return tmp_path


def test_the_task_writes_a_cleaned_frame_and_a_mask(frames_on_disk):
    from astropy.io import fits

    from harwood.core.registry import get_task

    result = get_task("cosmicrays").run(input=["f1.fits"], output="out/",
                                        gain=GAIN, rdnoise=RDNOISE)
    assert (frames_on_disk / "out" / "f1.fits").exists()
    assert (frames_on_disk / "out" / "f1.mask.fits").exists()

    mask = fits.getdata(frames_on_disk / "out" / "f1.mask.fits")
    header = fits.getheader(frames_on_disk / "out" / "f1.mask.fits")
    assert mask.sum() == header["HRWDCRM"] == result.data["rows"][0]["found"]
    # The cleaned frame points at the mask, so nothing downstream has to guess.
    assert fits.getheader(frames_on_disk / "out" / "f1.fits")["CRMASK"]


def test_the_task_scores_itself_when_asked(frames_on_disk):
    from harwood.core.registry import get_task

    result = get_task("cosmicrays").run(
        input=["f*.fits"], output="out/", gain=GAIN, rdnoise=RDNOISE,
        verify=True)

    step, = [item for item in result.steps if item.action == "verify"]
    assert "completeness" in step.measured
    assert "scaled by" in step.measured


def test_an_implausible_fraction_is_called_out(frames_on_disk):
    """Flagging one pixel in a hundred is the noise model being wrong, not the
    sky being busy -- and the gain is the usual culprit.

    Too *high* a gain is the dangerous direction: the expected noise goes as
    the square root of counts over gain, so overstating the gain understates
    the noise and makes ordinary pixels look significant. Understating it is
    harmless by comparison -- the detector simply finds nothing.
    """
    from harwood.core.registry import get_task

    result = get_task("cosmicrays").run(input=["f1.fits"], output="out/",
                                        gain=500.0, rdnoise=0.1, clean=False)
    assert any("--gain" in warning for warning in result.warnings)
    assert any(warning.level == "problem" for warning in result.warnings)


def test_an_unknown_algorithm_names_the_ones_that_exist(frames_on_disk):
    from harwood.core.registry import get_task

    with pytest.raises(ValueError) as raised:
        get_task("cosmicrays").run(input=["f1.fits"], algorithm="deepcr")
    assert "lacosmic" in raised.value.fix and "iraf" in raised.value.fix


# --------------------------------------------------- the long-slit weakness

#: The spectroscopic frames use the 1992 detector's numbers. The gain matters
#: to this comparison more than it looks: it sets how significant a marginal
#: pixel is, and the whole flank problem is marginal pixels.
SGAIN, SRDNOISE = 5.4, 5.0


def longslit(seed=0, ny=1024, nx=101, fwhm=4.0, peak=1200.0, nrays=200):
    """A long-slit spectrum: a bright ridge, with lines and a drifting trace.

    The object here is a *line*, which is what breaks the usual
    fine-structure model -- that one was built to recognise a point.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ny, 0:nx]
    sigma = fwhm / 2.3548
    trace = 54.0 + 1.4 * np.arange(ny) / ny

    along = peak * (1 + 0.3 * np.sin(y[:, 0] / 90.0))
    for line in (150, 400, 700, 880):
        along = along * (1 - 0.6 * np.exp(-0.5 * ((y[:, 0] - line) / 3.0) ** 2))
    image = 60.0 + along[:, None] * np.exp(
        -0.5 * ((x - trace[:, None]) / sigma) ** 2)

    hit = np.zeros(image.shape, dtype=bool)
    for _ in range(nrays):
        cy, cx = int(rng.integers(4, ny - 4)), int(rng.integers(2, nx - 2))
        step = int(rng.integers(-1, 2))
        for offset in range(int(rng.integers(1, 4))):
            row, column = cy + offset * step, cx + offset
            if 0 <= row < ny and 0 <= column < nx:
                image[row, column] += 10 ** rng.uniform(2.8, 4.2)
                hit[row, column] = True

    observed = (rng.poisson(np.clip(image * SGAIN, 0, None)) / SGAIN
                + rng.normal(0, SRDNOISE / SGAIN, image.shape))
    return observed, hit, trace


def _flank(shape, trace, low=1.5, high=4.5):
    offset = np.abs(np.arange(shape[1])[None, :] - trace[:, None])
    return (offset >= low) & (offset <= high)


def _slit_score(mask, hit, trace, shape):
    near = binary_dilation(hit, np.ones((3, 3), bool))
    false = mask & ~near
    flank = _flank(shape, trace)
    return {
        "completeness": (mask & hit).sum() / hit.sum(),
        "false": int(false.sum()),
        "on flank": int((false & flank).sum()),
        "flank area": int(flank.sum()),
    }


def test_the_false_detections_gather_on_the_spectrum_flanks():
    """The weakness, measured before anything claims to fix it.

    On the smooth side of a spectrum both of van Dokkum's medians see the same
    ramp, the fine-structure image is nothing, and the test that is supposed
    to protect real structure protects none.
    """
    total = flank = area = pixels = 0
    for seed in range(3):
        data, hit, trace = longslit(seed)
        score = _slit_score(cosmic.lacosmic(data, gain=SGAIN,
                                            rdnoise=SRDNOISE).mask,
                            hit, trace, data.shape)
        total += score["false"]
        flank += score["on flank"]
        area += score["flank area"]
        pixels += data.size

    assert total > 20, "there has to be something to be concentrated"
    concentration = (flank / area) / (total / pixels)
    assert concentration > 5.0, (
        f"{concentration:.1f}x -- if this has stopped being true, the fix "
        "below is no longer being tested against anything")


def test_knowing_the_dispersion_axis_removes_them():
    """The fix. Model the object as a line as well as a point, and take
    whichever model finds more real structure -- but only where the pixel
    agrees with that model, or a cosmic ray sitting on the spectrum is
    protected along with it.
    """
    plain = {"false": 0, "on flank": 0, "completeness": 0.0}
    fixed = {"false": 0, "on flank": 0, "completeness": 0.0}
    for seed in range(3):
        data, hit, trace = longslit(seed)
        for target, axis in ((plain, None), (fixed, 2)):
            score = _slit_score(
                cosmic.lacosmic(data, gain=SGAIN, rdnoise=SRDNOISE,
                                dispaxis=axis).mask, hit, trace, data.shape)
            target["false"] += score["false"]
            target["on flank"] += score["on flank"]
            target["completeness"] += score["completeness"] / 3

    assert fixed["on flank"] < 0.1 * plain["on flank"], (
        f"{fixed['on flank']} against {plain['on flank']} on the flanks")
    assert fixed["false"] < 0.3 * plain["false"]
    # And it must not be bought with completeness, which is what the
    # agreement gate is for.
    assert fixed["completeness"] > plain["completeness"] - 0.01, (
        f"{fixed['completeness']:.3f} against {plain['completeness']:.3f}")


def test_the_line_model_finds_the_flank_and_ignores_blank_sky():
    """Directly, rather than through the detection. Three things have to hold
    at once, and the middle one is what a seven-pixel window got wrong."""
    data, hit, trace = longslit(0)
    noise = cosmic._noise(data, SGAIN, SRDNOISE)
    flank = _flank(data.shape, trace)
    blank = np.abs(np.arange(data.shape[1])[None, :] - trace[:, None]) > 20

    point = cosmic._fine_structure(data) / noise
    line = cosmic._line_structure(data, dispaxis=2) / noise

    assert np.median(line[flank]) > 1.0 > np.median(point[flank]), \
        "the line model has to find the structure the point model misses"
    assert abs(np.median(line[blank])) < 0.2, \
        "and none where there is none, or a ray on empty sky is protected"
    assert np.median(line[hit]) < 1.0, \
        "and none at a cosmic ray, which the model was built without"


def test_a_narrow_cross_slit_window_protects_nothing():
    """Why the width is 25 and not 7. At seven pixels, on a four-pixel
    profile, the window centred on a flank is still inside the spectrum: the
    subtraction gives nothing and the model is decoration."""
    data, _, trace = longslit(0)
    noise = cosmic._noise(data, SGAIN, SRDNOISE)
    flank = _flank(data.shape, trace)

    narrow = cosmic._line_structure(data, 2, across=7) / noise
    wide = cosmic._line_structure(data, 2, across=25) / noise
    assert np.median(narrow[flank]) < 0.1 < np.median(wide[flank])


def test_the_task_reads_the_dispersion_axis_from_the_header(tmp_path, monkeypatch):
    from astropy.io import fits

    from harwood.core.registry import get_task

    monkeypatch.chdir(tmp_path)
    data, _, _ = longslit(1, ny=400)
    fits.PrimaryHDU(data.astype(np.float32),
                    fits.Header({"DISPAXIS": 2})).writeto(tmp_path / "s.fits")

    told = get_task("cosmicrays").run(input=["s.fits"], output="a/",
                                      gain=SGAIN, rdnoise=SRDNOISE, dispaxis=2)
    read = get_task("cosmicrays").run(input=["s.fits"], output="b/",
                                      gain=SGAIN, rdnoise=SRDNOISE)
    assert read.data["rows"][0]["found"] == told.data["rows"][0]["found"]
