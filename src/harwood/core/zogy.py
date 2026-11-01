"""Image subtraction by ZOGY: Zackay, Ofek & Gal-Yam (2016, ApJ 830, 27), the statistic behind
ZTF's and most of today's searches for what has changed. IRAF's way is psfmatch and imarith:
blur the sharper image to the other, losing what it had, and subtract. Here neither is blurred
to match: the difference is worked out in Fourier space from both PSFs and both noises, its
noise is white and flat, and it is proved optimal for a point source.

With N and R the new image and the reference, background subtracted and on the same pixels,
P_n and P_r their PSFs, s_n and s_r their noises and F_n and F_r their zero points, in the
paper's terms (its equations 13 to 17 and 25 to 28):

    D^ = (F_r P_r^ N^ - F_n P_n^ R^) / sqrt(s_n^2 F_r^2 |P_r^|^2 + s_r^2 F_n^2 |P_n^|^2)
    F_D = F_r F_n / sqrt(s_n^2 F_r^2 + s_r^2 F_n^2)
    P_D^ = F_r F_n P_r^ P_n^ / (F_D sqrt(...))
    S^ = F_D D^ conj(P_D^)

S is the score for a point source, and it is the one to threshold: corrected by the noise of
the sources themselves (V_S from k_n, k_r) and by the astrometric noise, it is in sigma at
each pixel (Scorr). alpha = S / F_S is the flux of a point source there in the new image's
units, with error sqrt(V_S) / F_S.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import advice


@dataclass
class Subtraction:
    D: np.ndarray            #: the difference, white, of unit variance
    S: np.ndarray            #: the match to a point source
    Scorr: np.ndarray        #: S over its noise, in sigma at each pixel
    alpha: np.ndarray        #: the flux of a point source at each pixel
    alpha_err: np.ndarray
    P_D: np.ndarray          #: the difference's PSF
    F_D: float
    F_S: float


def _centred(psf, shape) -> np.ndarray:
    """The PSF on the image's grid, with its centre at the first pixel, for the transforms."""
    psf = np.asarray(psf, dtype=np.float64)
    out = np.zeros(shape)
    ny, nx = psf.shape
    cy, cx = shape[0] // 2, shape[1] // 2
    y1, x1 = cy - ny // 2, cx - nx // 2
    if y1 < 0 or x1 < 0:
        advice.fail(f"a PSF of {nx}x{ny} on an image of {shape[1]}x{shape[0]}",
                    "the PSF has to be smaller than the image")
    out[y1:y1 + ny, x1:x1 + nx] = psf
    out /= out.sum()
    return np.fft.ifftshift(out)


def shifted(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Moved by a fraction of a pixel in Fourier space, which leaves the noise white: an
    interpolant would smooth it, and differently from pixel to pixel."""
    ny, nx = image.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    return np.fft.ifft2(np.fft.fft2(image) * np.exp(-2j * np.pi * (fx * dx + fy * dy))).real


def centre_offsets(new, reference, stars, fwhm: float = 3.0, saturated=None):
    """Where each star is in the new image against the reference, by a centroid under a
    Gaussian window, gone round until it settles: an array of (dx, dy) of those it measured
    in both. `stars` are 1-based (x, y)."""
    sigma = fwhm / 2.3548200450309493
    half = max(5, int(4 * sigma))
    rows, columns = new.shape
    jj, ii = np.mgrid[0:rows, 0:columns]
    out = []

    def centroid(image, x0, y0):
        x, y = x0, y0
        for _ in range(20):
            ix, iy = int(round(x)), int(round(y))
            if ix - half < 0 or iy - half < 0 or ix + half >= columns or iy + half >= rows:
                return None
            cut = (slice(iy - half, iy + half + 1), slice(ix - half, ix + half + 1))
            if saturated is not None and saturated[cut].any():
                return None
            box = np.maximum(image[cut], 0.0)
            window = np.exp(-((ii[cut] - x) ** 2 + (jj[cut] - y) ** 2) / (2 * sigma * sigma))
            total = (box * window).sum()
            if total <= 0:
                return None
            # The window narrows the centroid's reach: twice the step is SExtractor's XWIN.
            nx = x + 2 * ((box * window * (ii[cut] - x)).sum() / total)
            ny = y + 2 * ((box * window * (jj[cut] - y)).sum() / total)
            moved = abs(nx - x) + abs(ny - y)
            x, y = nx, ny
            if moved < 0.002:
                return x, y, total
        return None

    for x, y in stars:
        a = centroid(new, float(x) - 1, float(y) - 1)
        b = centroid(reference, float(x) - 1, float(y) - 1)
        if a is not None and b is not None and abs(a[0] - b[0]) < 3 and abs(a[1] - b[1]) < 3:
            out.append((a[0] - b[0], a[1] - b[1], min(a[2], b[2])))
    return np.array(out) if out else np.zeros((0, 3))


def gaussian(fwhm: float, size: int | None = None) -> np.ndarray:
    size = size or int(fwhm * 6) | 1
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    sigma = fwhm / 2.3548200450309493
    psf = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return psf / psf.sum()


def subtract(new, reference, psf_new, psf_reference, sigma_new: float, sigma_reference: float,
             flux_new: float = 1.0, flux_reference: float = 1.0, variance_new=None,
             variance_reference=None, astrometric=(0.0, 0.0)) -> Subtraction:
    """`new` and `reference`, background subtracted and on the same pixels. The variances are
    images with everything in them, the sources' noise too; left out, the sigmas' flat one."""
    N = np.nan_to_num(np.asarray(new, dtype=np.float64))
    R = np.nan_to_num(np.asarray(reference, dtype=np.float64))
    if N.shape != R.shape:
        advice.fail(f"the images are {N.shape} and {R.shape}",
                    "put them on the same pixels first, with imalign")
    if sigma_new <= 0 or sigma_reference <= 0:
        advice.fail("the noise of an image has to be above zero",
                    "give its background noise, in its own units")
    Pn = np.fft.fft2(_centred(psf_new, N.shape))
    Pr = np.fft.fft2(_centred(psf_reference, N.shape))
    Nf, Rf = np.fft.fft2(N), np.fft.fft2(R)
    sn2, sr2 = sigma_new ** 2, sigma_reference ** 2
    Fn, Fr = flux_new, flux_reference
    An, Ar = np.abs(Pn) ** 2, np.abs(Pr) ** 2
    denominator = sn2 * Fr ** 2 * Ar + sr2 * Fn ** 2 * An
    denominator = np.maximum(denominator, 1e-300)
    root = np.sqrt(denominator)
    Df = (Fr * Pr * Nf - Fn * Pn * Rf) / root
    F_D = Fr * Fn / np.sqrt(sn2 * Fr ** 2 + sr2 * Fn ** 2)
    PDf = Fr * Fn * Pr * Pn / (F_D * root)
    Sf = F_D * Df * np.conj(PDf)
    D = np.fft.ifft2(Df).real
    S = np.fft.ifft2(Sf).real
    P_D = np.fft.fftshift(np.fft.ifft2(PDf).real)
    F_S = F_D ** 2 * float((np.abs(PDf) ** 2).sum()) / N.size

    kn = np.conj(Pn) * Ar * Fn * Fr ** 2 / denominator
    kr = np.conj(Pr) * An * Fr * Fn ** 2 / denominator
    vn = np.full(N.shape, sn2) if variance_new is None else np.asarray(variance_new, float)
    vr = (np.full(N.shape, sr2) if variance_reference is None
          else np.asarray(variance_reference, float))
    Kn2 = np.fft.fft2(np.fft.ifft2(kn).real ** 2)
    Kr2 = np.fft.fft2(np.fft.ifft2(kr).real ** 2)
    Vs = (np.fft.ifft2(np.fft.fft2(vn) * Kn2).real + np.fft.ifft2(np.fft.fft2(vr) * Kr2).real)
    Vs = np.maximum(Vs, 0.0)
    if any(astrometric):
        # The paper's equations 30 and 31: how much each image's own score moves with a shift,
        # at each pixel; a star's flanks, where a misaligned one leaves its dipole.
        for part in (np.fft.ifft2(kn * Nf).real, np.fft.ifft2(kr * Rf).real):
            gy, gx = np.gradient(part)
            Vs = Vs + (astrometric[0] * gx) ** 2 + (astrometric[1] * gy) ** 2
    scorr = S / np.sqrt(np.where(Vs > 0, Vs, np.inf))
    return Subtraction(D, S, scorr, S / F_S, np.sqrt(Vs) / F_S, P_D, float(F_D), float(F_S))
