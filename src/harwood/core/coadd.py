"""Proper coaddition: Zackay & Ofek (2017, ApJ 836, 187). What IRAF has is imcombine: a
weighted mean of the pixels, which has one PSF only if every frame has, and a good night's
sharp frames pay for a bad night's by being weighted as a whole. Here each frame counts at each
spatial frequency for what it knows there, its sharpness and depth together, and the result is
a sufficient statistic: nothing the frames say about a source is left out of it.

With M_j the frames, background subtracted and on the same pixels, P_j their PSFs, s_j their
noises and F_j their zero points, in the paper's terms (its equations 7 to 11):

    R^ = sum (F_j / s_j^2) conj(P_j^) M_j^ / sqrt(sum (F_j^2 / s_j^2) |P_j^|^2)
    F_R = sqrt(sum F_j^2 / s_j^2)
    P_R^ = sqrt(sum (F_j^2 / s_j^2) |P_j^|^2) / F_R

R has white noise of unit variance, and is divided here by F_R, so that it reads as one
exposure with a zero point of 1 and a noise of 1 / F_R. A single frame with a PSF whose
transform is real and positive comes back as itself. S is the same sum before the root, the
filter matched to a point source, whose noise and flux follow from the same denominator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import advice
from .zogy import _centred


@dataclass
class Coadd:
    image: np.ndarray      #: the coadd, white, of noise `sigma`, in units of a zero point of 1
    psf: np.ndarray        #: its PSF, centred, summing to one
    score: np.ndarray      #: the match to a point source, in sigma of the background
    flux: np.ndarray       #: the flux of a point source at each pixel
    flux_err: float
    sigma: float
    filled: int = 0        #: bad pixels that took their value from the other frames
    unfilled: int = 0      #: those bad in all of them


def fill(images, bad, fluxes) -> tuple[list[np.ndarray], int, int]:
    """Transforms have no gaps. A frame's bad pixel takes the median of the others there, at
    its zero point, and zero, the sky, if none is good."""
    stack = np.array([np.asarray(m, float) / f for m, f in zip(images, fluxes, strict=True)])
    flags = np.array(bad, bool)
    if not flags.any():
        return [np.asarray(m, float) for m in images], 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        typical = np.nanmedian(np.where(flags, np.nan, stack), axis=0)
    nothing = ~np.isfinite(typical)
    typical = np.where(nothing, 0.0, typical)
    out = [np.where(b, typical * f, np.asarray(m, float))
           for m, b, f in zip(images, flags, fluxes, strict=True)]
    return out, int((flags & ~nothing).sum()), int(nothing.sum())


def coadd(images, psfs, sigmas, fluxes=None, bad=None) -> Coadd:
    """`images`, background subtracted and on the same pixels; a PSF and a noise for each. The
    zero points are 1 when left out, and only their ratios matter to the result's shape."""
    images = [np.nan_to_num(np.asarray(m, dtype=np.float64)) for m in images]
    if not images:
        advice.fail("there is nothing to coadd", "give the frames")
    shape = images[0].shape
    if any(m.shape != shape for m in images):
        advice.fail("the frames are not all the same shape",
                    "put them on the same pixels first, with imalign")
    fluxes = [1.0] * len(images) if fluxes is None else [float(f) for f in fluxes]
    sigmas = [float(s) for s in sigmas]
    if len(psfs) != len(images) or len(sigmas) != len(images) or len(fluxes) != len(images):
        advice.fail(f"{len(images)} frames, {len(psfs)} PSFs, {len(sigmas)} noises and "
                    f"{len(fluxes)} zero points", "give one of each for each frame")
    if min(sigmas) <= 0 or min(fluxes) <= 0:
        advice.fail("a frame's noise and zero point have to be above zero",
                    "give its background noise, and its zero point as a factor")
    filled, unfilled = 0, 0
    if bad is not None:
        images, filled, unfilled = fill(images, bad, fluxes)
    upper = np.zeros(shape, complex)
    power = np.zeros(shape)
    for image, psf, sigma, flux in zip(images, psfs, sigmas, fluxes, strict=True):
        transform = np.fft.fft2(_centred(psf, shape))
        upper += flux / sigma ** 2 * np.conj(transform) * np.fft.fft2(image)
        power += flux ** 2 / sigma ** 2 * np.abs(transform) ** 2
    root = np.sqrt(np.maximum(power, 1e-300))
    F_R = float(np.sqrt(sum(f ** 2 / s ** 2 for f, s in zip(fluxes, sigmas, strict=True))))
    image = np.fft.ifft2(upper / root).real / F_R
    psf = np.fft.fftshift(np.fft.ifft2(root / F_R).real)
    S = np.fft.ifft2(upper).real
    # What S's noise is, and what a source of unit flux gives it at its own pixel
    denominator = float(power.sum()) / image.size
    return Coadd(image, psf / psf.sum(), S / np.sqrt(denominator), S / denominator,
                 float(1.0 / np.sqrt(denominator)), 1.0 / F_R, filled, unfilled)
