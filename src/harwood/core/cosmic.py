"""Finding cosmic rays, and knowing whether you found them.

A cosmic ray deposits its energy in one pass through the detector, so it makes
a mark that is *sharper than the telescope can focus*. That is the whole
handle: the optics impose a point spread function, and anything with edges
steeper than the PSF did not come through the optics.

**Why not IRAF's algorithm.** `cosmicrays` in IRAF dates from 1989 and works
from a flux ratio: the peak against the mean of its neighbours, with a
threshold you train by eye. It cannot tell a cosmic ray from the core of an
undersampled star, because on a flux ratio they look the same -- which is why
the task is interactive and why its help page tells you to check the results.
On well-sampled data it is serviceable; on anything near Nyquist it eats
stars.

**L.A.Cosmic** (van Dokkum 2001) replaced it, and this is an implementation of
that. The image is subsampled by two, convolved with a Laplacian, and resampled
back: the subsampling matters, because it makes the Laplacian respond to an
edge that is sharp *at the pixel scale* rather than to the pixel grid itself.
The result is divided by the local noise to give a significance, and then --
the part that distinguishes it from everything before -- compared against a
**fine-structure image**, a model of how much unresolved structure a real
object would have there. A cosmic ray is sharp and has no fine structure
around it; a star is sharp and sits on a wing. The ratio separates them.

Both algorithms are here. IRAF's is kept because a reduction that has to be
reproduced needs the method it was done with, not the method we would use now.

**What harwood adds.** The algorithm is thirty years old and well tested; what
is not usually available is any way to know whether it worked *on your data*.
`verify()` does that: given several exposures of the same field, a cosmic ray
falls in one frame and not the others, which is ground truth. The single-frame
detector can then be scored against it -- completeness and false positives,
measured, on the frames in front of you, rather than quoted from the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The Laplacian used to find edges, from van Dokkum (2001).
LAPLACIAN = np.array([[0.0, -1.0, 0.0],
                      [-1.0, 4.0, -1.0],
                      [0.0, -1.0, 0.0]])

#: A fine-structure image is divided by, so it is clipped away from zero. The
#: value is astroscrappy's, and it sets how a flat region is treated.
FLOOR = 0.01


@dataclass
class Detection:
    """Which pixels are cosmic rays, and what the detector saw on the way."""

    mask: np.ndarray                      #: True where a cosmic ray was found
    cleaned: np.ndarray                   #: the frame with them replaced
    passes: int = 0
    found_per_pass: list = field(default_factory=list)
    significance: np.ndarray | None = None   #: the Laplacian S/N image
    structure: np.ndarray | None = None      #: the fine-structure ratio
    saturated: int = 0

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    @property
    def fraction(self) -> float:
        return float(self.mask.mean())


def _subsampled_laplacian(data: np.ndarray) -> np.ndarray:
    """Convolve at twice the sampling, then come back.

    Doing it at the native scale makes the Laplacian respond to the pixel grid;
    at twice the scale it responds to the *edge*, which is the thing being
    measured. This is the step that makes the method work at all.
    """
    from scipy.ndimage import convolve

    big = np.repeat(np.repeat(data, 2, axis=0), 2, axis=1)
    convolved = convolve(big, LAPLACIAN, mode="nearest")
    np.clip(convolved, 0, None, out=convolved)
    # Back to the original grid by averaging each 2x2 block.
    ny, nx = data.shape
    return convolved.reshape(ny, 2, nx, 2).mean(axis=(1, 3))


def _median(data: np.ndarray, size: int, separable: bool = False) -> np.ndarray:
    """A median over a square window, or a fast approximation to one.

    The approximation -- median along the rows, then along the columns -- is
    not a two-dimensional median and does not claim to be. It costs O(n)
    per pixel instead of O(n^2) and is what makes this tractable on a large
    frame; it is astroscrappy's default for the same reason. What it costs in
    accuracy is measured in `tests/test_cosmic.py` rather than assumed.
    """
    from scipy.ndimage import median_filter

    if not separable:
        return median_filter(data, size=size, mode="nearest")
    rows = median_filter(data, size=(1, size), mode="nearest")
    return median_filter(rows, size=(size, 1), mode="nearest")


def _noise(data: np.ndarray, gain: float, rdnoise: float,
           separable: bool = False) -> np.ndarray:
    """Expected noise per pixel, from a smoothed version of the data.

    Smoothed, because the noise has to be estimated from what the sky and the
    object really are -- using the pixel's own value would let a cosmic ray
    inflate the noise that is supposed to detect it.
    """
    smooth = _median(data, 5, separable)
    np.clip(smooth, 0.00001, None, out=smooth)
    return np.sqrt(smooth * gain + rdnoise ** 2) / gain


def _fine_structure(data: np.ndarray, separable: bool = False) -> np.ndarray:
    """How much unresolved structure a real object would have here.

    A 3x3 median keeps a star's core and removes single-pixel spikes; taking a
    7x7 median of *that* removes the core too. The difference is what the
    object contributes on scales the optics can produce -- which a cosmic ray,
    by definition, does not.
    """
    three = _median(data, 3, separable)
    return three - _median(three, 7, separable)


def _clean(data: np.ndarray, mask: np.ndarray, size: int = 5) -> np.ndarray:
    """Replace flagged pixels with the median of their unflagged neighbours.

    Unflagged matters: a cosmic ray is often several pixels, and a median that
    includes its neighbours replaces one bad value with another.
    """
    if not mask.any():
        return data.copy()

    work = data.copy()
    work[mask] = np.nan

    # Only the flagged pixels need a new value, and they are a fraction of a
    # per cent of the frame. Filtering the whole image to fill them would
    # build a window stack over every pixel -- at 2048 square with a 5x5
    # window that is 840 MB of temporaries to use 0.2% of the result.
    #
    # The window widens each round, for a pixel in the middle of a track whose
    # whole neighbourhood was flagged and whose median is therefore undefined.
    for width in (size, size + 4, size + 10):
        rows, columns = np.nonzero(np.isnan(work))
        if rows.size == 0:
            break
        work[rows, columns] = _neighbour_median(work, rows, columns, width)

    left = np.isnan(work)
    if left.any():
        work[left] = float(np.median(data[~mask])) if (~mask).any() else 0.0
    return work


def _neighbour_median(values: np.ndarray, rows, columns, size: int) -> np.ndarray:
    """The median of each listed pixel's neighbourhood, ignoring NaN.

    Gathers only the windows that are wanted, which is why this is cheap: the
    cost is the number of flagged pixels, not the size of the frame.
    """
    half = size // 2
    padded = np.pad(values, half, mode="edge")
    offsets = np.arange(-half, half + 1)
    window = (padded[(rows[:, None, None] + half + offsets[None, :, None]),
                     (columns[:, None, None] + half + offsets[None, None, :])])
    # A pixel in the middle of a wide track can have a neighbourhood that is
    # entirely flagged; the caller widens the window and tries again, so an
    # all-NaN window here is expected rather than exceptional.
    import warnings

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(window.reshape(rows.size, -1), axis=1)


def lacosmic(data, gain: float = 1.0, rdnoise: float = 0.0, sigclip: float = 4.5,
             sigfrac: float = 0.3, objlim: float = 5.0, niter: int = 4,
             satlevel: float = 65535.0, strict_growth: bool = True,
             separable: bool = False, mask=None) -> Detection:
    """L.A.Cosmic: Laplacian edge detection (van Dokkum 2001).

    `sigclip` is how many sigma above the noise a Laplacian edge must be.
    `objlim` is the ratio of that edge to the fine structure around it -- the
    criterion that separates a cosmic ray from a star, and the one worth
    turning down first when real objects are being eaten. `sigfrac` sets a
    lower threshold for pixels *touching* something already flagged, so the
    tail of a track is caught without lowering the bar everywhere.

    **`strict_growth` is a deliberate departure from the paper**, and it is
    off by default there and in every implementation that follows it. Growing
    into a neighbour normally asks only that the neighbour be significant; it
    does not re-apply the fine-structure test. That is what lets a detection
    which happens to touch a star spread along the star, and it is the
    mechanism behind the one failure everybody knows this method for.

    Keeping the test during growth costs a little completeness and removes
    that failure. Measured over five synthetic fields of 110 tracks each,
    undersampled so the failure has room to happen:

                            completeness   false   on a star
        strict growth              99.1%      29           1
        as published               99.4%     137          53
        astroscrappy               99.4%     192          78

    astroscrappy is the reference implementation of the paper, on the same
    frames; the middle row is this function with `strict_growth=False`, and it
    is there to show the implementation is faithful before the departure is
    argued for. Use `strict_growth=False` when reproducing someone else's
    reduction, and the default otherwise.
    """
    from scipy.ndimage import binary_dilation

    data = np.asarray(data, dtype=float)
    working = data.copy()
    found = np.zeros(data.shape, dtype=bool)
    if mask is not None:
        found |= np.asarray(mask, dtype=bool)

    # Saturated pixels and their surroundings are not cosmic rays, and the
    # Laplacian of a flat-topped star is large everywhere along its edge.
    saturated = data >= satlevel
    protect = binary_dilation(saturated, iterations=2) if saturated.any() else saturated

    per_pass, significance, structure = [], None, None
    for _step in range(max(1, niter)):
        noise = _noise(working, gain, rdnoise, separable)
        laplacian = _subsampled_laplacian(working)
        # Two, because the subsampling doubled the signal of a real edge.
        s = laplacian / (2.0 * noise)
        # Take out what the sampling itself contributes, so that a bright
        # gradient does not read as an edge.
        s = s - _median(s, 5, separable)

        fine = _fine_structure(working, separable) / noise
        np.clip(fine, FLOOR, None, out=fine)
        ratio = laplacian / (2.0 * noise * fine)

        candidates = (s > sigclip) & (ratio > objlim) & ~protect & ~found

        # Grow into neighbours, in two rounds with *different* thresholds. A
        # track is bright in the middle and marginal at its ends, so the bar
        # is lowered -- but only for the second ring out. Relaxing it on the
        # first ring as well lets a detection walk along any coherent edge:
        # on a long-slit spectrum that produced two ridges down the flanks of
        # the spectrum itself, which is how this was caught.
        square = np.ones((3, 3), bool)
        first = binary_dilation(candidates, square) & (s > sigclip) & ~protect
        second = (binary_dilation(first, square)
                  & (s > sigfrac * sigclip) & ~protect)
        grown = first | second
        if strict_growth:
            grown &= ratio > objlim
        new = (candidates | grown) & ~found

        per_pass.append(int(new.sum()))
        if significance is None:
            significance, structure = s, ratio
        if not new.any():
            break
        found |= new
        working = _clean(data, found)

    return Detection(mask=found, cleaned=working, passes=len(per_pass),
                     found_per_pass=per_pass, significance=significance,
                     structure=structure, saturated=int(saturated.sum()))


def iraf_cosmicrays(data, threshold: float = 25.0, fluxratio: float = 2.0,
                    npasses: int = 5, window: int = 5) -> Detection:
    """IRAF's original algorithm, kept so old reductions can be reproduced.

    A pixel is a candidate when it stands `threshold` above the mean of its
    neighbours; it is called a cosmic ray when the ratio of that mean to the
    peak is below `fluxratio` per cent -- that is, when the neighbours are
    dark, so the spike is alone. It has no way to tell a lone spike from the
    core of an undersampled star, which is why IRAF's task is interactive.
    """
    from scipy.ndimage import uniform_filter

    data = np.asarray(data, dtype=float)
    found = np.zeros(data.shape, dtype=bool)
    working = data.copy()
    per_pass = []

    half = max(1, window // 2)
    for _ in range(max(1, npasses)):
        background = _median(working, 2 * half + 1)
        residual = working - background

        # The mean of the neighbours, excluding the pixel itself.
        area = (2 * half + 1) ** 2
        neighbours = (uniform_filter(residual, size=2 * half + 1, mode="nearest")
                      * area - residual) / (area - 1)

        peak = residual
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = 100.0 * np.where(peak > 0, neighbours / peak, np.inf)
        new = (peak > threshold) & (ratio < fluxratio) & ~found
        per_pass.append(int(new.sum()))
        if not new.any():
            break
        found |= new
        working = _clean(data, found)

    return Detection(mask=found, cleaned=working, passes=len(per_pass),
                     found_per_pass=per_pass)


def verify(frames, detector, gain: float = 1.0, rdnoise: float = 0.0,
           threshold: float = 6.0, aligned: bool = True) -> dict:
    """Score a single-frame detector against what several frames agree on.

    This is the part no cosmic-ray task normally offers. A cosmic ray falls on
    one exposure and not on the next, so a set of exposures of the same field
    *knows* where they are, and the single-frame detector can be measured
    rather than trusted -- on your own data, at your own threshold, instead of
    from the numbers in a paper.

    Two things have to be right or the answer is worthless, and both were
    found the hard way on a real night:

    **The scatter cannot be measured from the frames.** With three exposures,
    the median absolute deviation about their median is one number away from
    zero, so the threshold collapses and everything looks discrepant -- the
    same trap as any iterative rejection. The noise here therefore comes from
    the detector model, `gain` and `rdnoise`, applied to the *other* frames'
    estimate of what the pixel should be.

    **The frames are not the same brightness.** Transparency and airmass change
    between exposures; on the night this was written against, the spectrum
    dropped by a quarter over an hour. Comparing unscaled frames counts that
    real change as a cosmic ray in every bright pixel. Each frame is therefore
    scaled to the set's common level first, which is what `imcombine` does for
    the same reason.

    Returns completeness -- what share of the set's cosmic rays the
    single-frame method found -- and how many of its detections the set does
    not support.
    """
    stack = np.asarray([np.asarray(frame, dtype=float) for frame in frames])
    if stack.shape[0] < 3:
        from . import advice

        advice.fail(
            f"verifying against a set of frames needs at least three, and got "
            f"{stack.shape[0]}.",
            "with two, a pixel that disagrees gives no way to say which frame "
            "is the odd one out. Give the whole set of exposures.")
    if not aligned:
        from . import advice

        advice.fail(
            "the frames must be on the same pixel grid.",
            "run imalign first. On unaligned frames every star edge looks like "
            "a cosmic ray, and the score would be meaningless.")

    # Scale out transparency, so a night that changed does not read as a frame
    # full of cosmic rays. The level is taken from the whole frame, which is
    # robust when the object occupies a small part of it.
    levels = np.array([np.median(frame) for frame in stack])
    levels = np.where(levels > 0, levels, 1.0)
    scaled = stack * (float(np.median(levels)) / levels)[:, None, None]

    truth = np.zeros(stack.shape, dtype=bool)
    level = np.zeros(stack.shape, dtype=float)
    for index in range(stack.shape[0]):
        others = np.delete(scaled, index, axis=0)
        expected = np.median(others, axis=0)
        # The noise the detector would have given at that level -- not the
        # scatter of three numbers, which is not an estimate of anything.
        noise = np.sqrt(np.clip(expected, 0, None) * gain + rdnoise ** 2) / gain
        noise = np.where(noise > 0, noise, 1.0)
        truth[index] = (scaled[index] - expected) / noise > threshold
        level[index] = expected

    # Is this a static scene? A cosmic ray does not care how bright the pixel
    # it lands on is, so the rate on bright pixels and faint ones should be
    # the same. When it is not, the frames differ for a reason other than
    # cosmic rays -- a source that varied, a profile that changed with the
    # seeing, a trace that drifted -- and what is being measured is that,
    # not the detector. Saying so is the difference between a diagnostic and
    # a number that looks like one.
    split = float(np.median(level))
    bright, faint = level > split, level <= split
    rate = (float(truth[bright].mean()) if bright.any() else 0.0,
            float(truth[faint].mean()) if faint.any() else 0.0)
    static = rate[0] <= 3.0 * max(rate[1], 1e-9)

    rows = []
    for index, frame in enumerate(stack):
        detection = detector(frame)
        real = truth[index]
        hit = int((detection.mask & real).sum())
        rows.append({
            "frame": index + 1,
            "frames agree": int(real.sum()),
            "found": detection.count,
            "agreed": hit,
            "missed": int(real.sum()) - hit,
            "extra": detection.count - hit,
        })

    total_real = sum(row["frames agree"] for row in rows)
    total_found = sum(row["found"] for row in rows)
    total_hit = sum(row["agreed"] for row in rows)
    return {
        "rows": rows,
        "completeness": total_hit / total_real if total_real else float("nan"),
        "false_positive": ((total_found - total_hit) / total_found
                           if total_found else float("nan")),
        "scaled_by": [round(float(np.median(levels) / value), 4)
                      for value in levels],
        "static": bool(static),
        "rate_bright": rate[0],
        "rate_faint": rate[1],
    }
