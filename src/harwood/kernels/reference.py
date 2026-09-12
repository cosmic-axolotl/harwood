"""Reference kernels: pure NumPy, written to be read.

Section 7.1 fixes the rule that holds the whole compiled layer to the
transparency the project is for: **every compiled kernel has a readable Python
implementation, tested against the fast one.** This module is that
implementation. It is the specification; the numba and Rust versions are
optimisations that must agree with it, including on the discrete decisions --
which pixels were rejected, not only what the mean came to.

Everything here takes NumPy arrays and numbers and returns arrays. No Pydantic
models, no astropy objects, no dictionaries: that is the contract at the line
the compiled backends have to cross.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sigclip_mean", "ccdclip_mean"]


def sigclip_mean(
    stack: np.ndarray,
    lsigma: float = 3.0,
    hsigma: float = 3.0,
    maxiter: int = 5,
    mclip: bool = True,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of a stack with iterative sigma clipping, pixel by pixel.

    The threshold comes from the *spread of the stack itself*, which is what
    makes this the wrong choice when few images go in: two cosmic rays landing
    on the same pixel inflate the spread enough to hide both. That is the
    failure `ccdclip_mean` exists to avoid, and it is documented as a test.

    Parameters
    ----------
    stack
        Shape (n, ...) -- n images of identical shape.
    lsigma, hsigma
        Rejection thresholds below and above the running mean, in sigma.
    maxiter
        Maximum clipping passes. Iteration also stops early when a pass
        rejects nothing.
    mclip
        Judge deviations against the median rather than the mean, as IRAF's
        `imcombine` does by default. It matters more than it looks: the mean
        is dragged towards the very outlier being hunted, so with a strong
        deviant the whole stack can end up outside the window and everything
        gets rejected at once. The median does not move.
    mask
        Optional boolean array shaped like `stack`; True marks a pixel that is
        excluded from the start (a bad-pixel mask).

    Returns
    -------
    mean
        The clipped mean, shaped like one image.
    nrejected
        How many values were rejected at each pixel, as int32.
    """
    data = np.asarray(stack, dtype=np.float64)
    if data.ndim < 2:
        raise ValueError("sigclip_mean needs a stack of at least two dimensions")
    keep = np.ones(data.shape, dtype=bool)
    if mask is not None:
        keep &= ~np.asarray(mask, dtype=bool)

    for _ in range(max(1, int(maxiter))):
        count = keep.sum(axis=0)
        total = np.where(keep, data, 0.0).sum(axis=0)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        centre = _centre(data, keep, mean) if mclip else mean

        deviation = data - centre
        variance = np.divide(
            np.where(keep, deviation**2, 0.0).sum(axis=0),
            np.maximum(count - 1, 1),
            out=np.zeros_like(total),
            where=count > 1,
        )
        sigma = np.sqrt(variance)

        # A pixel with one value left is never clipped: there is no spread to
        # judge it against, and rejecting it would leave nothing behind.
        rejectable = count > 2
        too_low = deviation < -lsigma * sigma
        too_high = deviation > hsigma * sigma
        drop = keep & rejectable & (too_low | too_high)
        if not drop.any():
            break
        keep &= ~drop

    count = keep.sum(axis=0)
    total = np.where(keep, data, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    nrejected = (data.shape[0] - count).astype(np.int32)
    return mean, nrejected


def ccdclip_mean(
    stack: np.ndarray,
    gain: float,
    rdnoise: float,
    snoise: float = 0.0,
    lsigma: float = 3.0,
    hsigma: float = 3.0,
    maxiter: int = 5,
    mclip: bool = True,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of a stack, clipped against the CCD noise model rather than the spread.

    The expected sigma of a pixel of mean level ``m`` (in ADU) is

        sigma = sqrt( rdnoise^2 / gain^2  +  m / gain  +  (snoise * m)^2 )

    -- read noise, photon noise, and a sensitivity term for flat-field error.
    Because the threshold does not come from the stack, a second deviant value
    cannot hide the first. This is IRAF's `ccdclip`, and the reason IRAF offers
    it alongside `sigclip`.

    Parameters
    ----------
    gain
        Detector gain in electrons per ADU.
    rdnoise
        Read noise in electrons.
    snoise
        Sensitivity noise as a fraction, typically 0 to 0.01.
    mclip
        Judge deviations against the median rather than the mean. See
        `sigclip_mean`; with the noise model it matters even more, because a
        mean dragged upwards also inflates the predicted sigma.
    """
    if gain <= 0:
        raise ValueError("gain must be positive, in electrons per ADU")
    data = np.asarray(stack, dtype=np.float64)
    keep = np.ones(data.shape, dtype=bool)
    if mask is not None:
        keep &= ~np.asarray(mask, dtype=bool)

    for _ in range(max(1, int(maxiter))):
        count = keep.sum(axis=0)
        total = np.where(keep, data, 0.0).sum(axis=0)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)

        centre = _centre(data, keep, mean) if mclip else mean
        level = np.maximum(centre, 0.0)
        variance = (rdnoise / gain) ** 2 + level / gain + (snoise * level) ** 2
        sigma = np.sqrt(variance)

        deviation = data - centre
        rejectable = count > 2
        drop = keep & rejectable & (
            (deviation < -lsigma * sigma) | (deviation > hsigma * sigma)
        )
        if not drop.any():
            break
        keep &= ~drop

    count = keep.sum(axis=0)
    total = np.where(keep, data, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    nrejected = (data.shape[0] - count).astype(np.int32)
    return mean, nrejected


def _centre(data: np.ndarray, keep: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Median of the surviving values at each pixel.

    Masked-out values are set to NaN so `nanmedian` skips them; a pixel with
    nothing left falls back to the mean, which is zero there anyway.
    """
    masked = np.where(keep, data, np.nan)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(masked, axis=0)
    return np.where(np.isnan(median), fallback, median)
