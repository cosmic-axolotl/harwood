"""The same kernels as loops, compiled on first call.

Rules this module follows, from section 7.2:

* ``cache=True`` so only the first process pays compilation; the cache lives
  under ``~/.cache/harwood`` (set in `harwood.kernels.__init__`).
* **never** ``fastmath=True``: it lets the compiler reorder operations and
  ignore NaN, which changes results and breaks reproducibility. A kernel that
  disagrees with the reference implementation is worthless however fast it is.
* nothing but arrays and numbers crosses into here.

The loops are written the way the reference implementation reads, so the two
can be compared line by line.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = ["sigclip_mean", "ccdclip_mean", "version"]


def version() -> str:
    import numba

    return numba.__version__


@njit(cache=True, parallel=True)
def _clip(data, keep, lsigma, hsigma, maxiter, mclip, use_model, gain, rdnoise, snoise):
    """One loop serving both rejection rules.

    `use_model` picks where sigma comes from: the spread of the surviving
    values (sigclip) or the CCD noise model (ccdclip).
    """
    nimages, npix = data.shape
    mean_out = np.zeros(npix, dtype=np.float64)
    nrej_out = np.zeros(npix, dtype=np.int32)

    for pixel in prange(npix):
        survivors = np.empty(nimages, dtype=np.float64)

        for _ in range(maxiter):
            count = 0
            total = 0.0
            for image in range(nimages):
                if keep[image, pixel]:
                    survivors[count] = data[image, pixel]
                    total += data[image, pixel]
                    count += 1
            if count == 0:
                break
            mean = total / count

            if mclip and count > 0:
                block = np.sort(survivors[:count])
                half = count // 2
                centre = block[half] if count % 2 == 1 else \
                    0.5 * (block[half - 1] + block[half])
            else:
                centre = mean

            if use_model:
                level = centre if centre > 0.0 else 0.0
                sigma = np.sqrt((rdnoise / gain) ** 2 + level / gain
                                + (snoise * level) ** 2)
            else:
                if count > 1:
                    spread = 0.0
                    for index in range(count):
                        difference = survivors[index] - centre
                        spread += difference * difference
                    sigma = np.sqrt(spread / (count - 1))
                else:
                    sigma = 0.0

            if count <= 2:
                break

            dropped = 0
            low = centre - lsigma * sigma
            high = centre + hsigma * sigma
            for image in range(nimages):
                if keep[image, pixel]:
                    value = data[image, pixel]
                    if value < low or value > high:
                        keep[image, pixel] = False
                        dropped += 1
            if dropped == 0:
                break

        count = 0
        total = 0.0
        for image in range(nimages):
            if keep[image, pixel]:
                total += data[image, pixel]
                count += 1
        mean_out[pixel] = total / count if count > 0 else 0.0
        nrej_out[pixel] = nimages - count

    return mean_out, nrej_out


def _prepare(stack, mask):
    data = np.ascontiguousarray(np.asarray(stack, dtype=np.float64))
    shape = data.shape[1:]
    flat = data.reshape(data.shape[0], -1)
    keep = np.ones(flat.shape, dtype=np.bool_)
    if mask is not None:
        keep &= ~np.asarray(mask, dtype=np.bool_).reshape(flat.shape)
    return flat, keep, shape


def sigclip_mean(stack, lsigma=3.0, hsigma=3.0, maxiter=5, mclip=True, mask=None):
    flat, keep, shape = _prepare(stack, mask)
    mean, nrej = _clip(flat, keep, float(lsigma), float(hsigma), max(1, int(maxiter)),
                       bool(mclip), False, 1.0, 0.0, 0.0)
    return mean.reshape(shape), nrej.reshape(shape)


def ccdclip_mean(stack, gain, rdnoise, snoise=0.0, lsigma=3.0, hsigma=3.0,
                 maxiter=5, mclip=True, mask=None):
    if gain <= 0:
        raise ValueError("gain must be positive, in electrons per ADU")
    flat, keep, shape = _prepare(stack, mask)
    mean, nrej = _clip(flat, keep, float(lsigma), float(hsigma), max(1, int(maxiter)),
                       bool(mclip), True, float(gain), float(rdnoise), float(snoise))
    return mean.reshape(shape), nrej.reshape(shape)
