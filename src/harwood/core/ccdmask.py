# ruff: noqa: E741
# A line-by-line port keeps IRAF's names (l, j, k).
"""ccdmask: bad pixels from a flat field, by IRAF's algorithm.

Against a moving median, with a sigma from percentiles in blocks, a pixel that
deviates is bad; and so is a run along a column whose sum deviates, at scales from
two pixels to the whole column, which is what finds a charge trap too weak to see
pixel by pixel. All in real, as t_ccdmask.x has it; the blocks of columns it goes
through are not only a matter of memory, since the sigmas start over with each.

From iraf-community 2.18: noao/imred/ccdred/src/t_ccdmask.x. See
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import numpy as np

from .jit import jit

MAXBUF = 500000
PLSIG = 30.9
PHSIG = 69.1


@jit
def _median(work, n):
    """amedr: the first of two; of more, the (n + 1) / 2 th smallest."""
    if n <= 2:
        return work[0]
    return np.sort(work[:n])[(n + 1) // 2 - 1]


@jit
def cm_mask(data, bp, nc, nl, nc1, nc2, ncmed, nlmed, ncsig, nlsig, lsig, hsig, ngood,
            plsig, phsig):
    """data and bp are [line, column], the block's; nc1 and nc2 1-based."""
    nsig = nl // nlsig
    bkg = np.zeros(nl + 1, dtype=np.float32)
    sig = np.zeros(nsig + 1, dtype=np.float32)
    work = np.zeros(max(ncsig * nlsig, ncmed * nlmed), dtype=np.float32)
    bp1 = np.zeros(nl + 1, dtype=np.int16)
    for i in range(nc1, nc2 + 1):
        l = min(nc, i + ncmed // 2)
        l = max(1, l - ncmed + 1)
        for j in range(1, nl + 1):
            k = min(nl, j + nlmed // 2)
            k = max(1, k - nlmed + 1)
            at = 0
            for m in range(k, k + nlmed):
                for c in range(ncmed):
                    work[at] = data[m - 1, l - 1 + c]
                    at += 1
            bkg[j] = _median(work, ncmed * nlmed)

        if (i - nc1) % ncsig == 0 and i < nc - ncsig + 1:
            for j in range(1, nl - nlsig + 2, nlsig):
                at = 0
                for k in range(j, j + nlsig):
                    for c in range(ncsig):
                        work[at] = data[k - 1, i - 1 + c]
                        at += 1
                ordered = np.sort(work[:ncsig * nlsig])
                sigma = np.float32(ordered[phsig] - ordered[plsig])
                sig[(j + nlsig - 1) // nlsig] = np.float32(sigma * sigma)

        k = 0
        for j in range(1, nl + 1):
            if bp[j - 1, i - 1] == 1:
                k += 1
            else:
                jsig = min((j + nlsig - 1) // nlsig, nsig)
                back = bkg[j]
                sigma = np.float32(np.sqrt(sig[jsig]))
                low = np.float32(back - np.float32(lsig * sigma))
                high = np.float32(back + np.float32(hsig * sigma))
                if data[j - 1, i - 1] < low or data[j - 1, i - 1] > high:
                    bp[j - 1, i - 1] = 1
                    k += 1
        if k == nl:
            continue

        l = 2
        while l <= nl:
            for j in range(1, nl + 1):
                bp1[j] = bp[j - 1, i - 1]
            sum1 = np.float32(0.0)
            sum2 = np.float32(0.0)
            nsum = 0
            k = 1
            for j in range(k, l):
                if bp[j - 1, i - 1] == 1:
                    continue
                jsig = min((j + nlsig - 1) // nlsig, nsig)
                sum1 = np.float32(np.float32(sum1 + data[j - 1, i - 1]) - bkg[j])
                sum2 = np.float32(sum2 + sig[jsig])
                nsum += 1
            for j in range(l, nl + 1):
                if bp[j - 1, i - 1] == 0:
                    jsig = min((j + nlsig - 1) // nlsig, nsig)
                    sum1 = np.float32(np.float32(sum1 + data[j - 1, i - 1]) - bkg[j])
                    sum2 = np.float32(sum2 + sig[jsig])
                    nsum += 1
                if nsum > 0:
                    sigma = np.float32(np.sqrt(sum2))
                    low = np.float32(-lsig * sigma)
                    high = np.float32(hsig * sigma)
                    if sum1 < low or sum1 > high:
                        for m in range(k, j + 1):
                            bp[m - 1, i - 1] = 1
                if bp1[k] == 0:
                    jsig = min((k + nlsig - 1) // nlsig, nsig)
                    sum1 = np.float32(np.float32(sum1 - data[k - 1, i - 1]) + bkg[k])
                    sum2 = np.float32(sum2 - sig[jsig])
                    nsum -= 1
                k += 1
            if l == nl:
                break
            if l < 10:
                l += 1
            else:
                l = min(l * 2, nl)

        if ngood > 1:
            k = 1
            while k <= nl and bp[k - 1, i - 1] != 0:
                k += 1
            while k < nl:
                l = k + 1
                while l <= nl and bp[l - 1, i - 1] == 0:
                    l += 1
                if l - k < ngood:
                    for j in range(k, l):
                        bp[j - 1, i - 1] = 1
                k = l + 1
                while k <= nl and bp[k - 1, i - 1] != 0:
                    k += 1


@jit
def cm_interp(bp, nc, nl, nc1, nc2, ncimage, linterp, cinterp, eqinterp):
    """The value of each bad pixel from the lengths of its run along column and line."""
    for i in range(nc1, nc2 + 1):
        k = 1
        while k <= nl and bp[k - 1, i - 1] == 0:
            k += 1
        while k <= nl:
            l = k + 1
            while l <= nl and bp[l - 1, i - 1] != 0:
                l += 1
            m = l - k
            for j in range(k, l):
                bp[j - 1, i - 1] = m
            k = l + 1
            while k <= nl and bp[k - 1, i - 1] == 0:
                k += 1
    if nc == ncimage:
        for j in range(1, nl + 1):
            k = 1
            while k <= nc and bp[j - 1, k - 1] == 0:
                k += 1
            while k <= nc:
                l = k + 1
                while l <= nc and bp[j - 1, l - 1] != 0:
                    l += 1
                m = l - k
                for i in range(k, l):
                    n = bp[j - 1, i - 1]
                    if n > m or n == nl:
                        bp[j - 1, i - 1] = linterp
                    elif n < m:
                        bp[j - 1, i - 1] = cinterp
                    else:
                        bp[j - 1, i - 1] = eqinterp
                k = l + 1
                while k <= nc and bp[j - 1, k - 1] == 0:
                    k += 1


@jit
def cm_interp1(line, out, nc, nl, linterp, cinterp, eqinterp):
    """The same along a whole line, when the image went through in blocks. It
    never writes the pixel right after a run, nor a run in the last column: in
    IRAF those are what the buffer had, mostly 0 and now and then not. Here 0."""
    j = 1
    while j <= nc and line[j - 1] == 0:
        out[j - 1] = 0
        j += 1
    while j < nc:
        l = j + 1
        while l <= nc and line[l - 1] != 0:
            l += 1
        m = l - j
        for i in range(j, l):
            n = line[i - 1]
            if n > m or n == nl:
                out[i - 1] = linterp
            elif n < m:
                out[i - 1] = cinterp
            else:
                out[i - 1] = eqinterp
        j = l + 1
        while j <= nc and line[j - 1] == 0:
            out[j - 1] = 0
            j += 1


def _nint(x: float) -> int:
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(0.5 - x))


def ccdmask(data, ncmed=7, nlmed=7, ncsig=15, nlsig=15, lsigma=6.0, hsigma=6.0, ngood=5,
            linterp=2, cinterp=3, eqinterp=2) -> np.ndarray:
    """t_ccdmask, over the blocks of columns. data is [line, column]; the mask's
    values are int16."""
    real = np.float32
    pixels = np.asarray(data, dtype=real)
    nl, nc = pixels.shape
    count = real(nlsig * ncsig)
    plsig = _nint(float(real(real(real(PLSIG) * count) / real(100.0)) - real(1.0)))
    phsig = _nint(float(real(real(real(PHSIG) * count) / real(100.0)) - real(1.0)))
    out = np.zeros((nl, nc), dtype=np.int16)
    ncstep = max(1, MAXBUF // nl - ncmed)
    nc1 = nc
    for i in range(1, nc + 1, ncstep):
        c1 = i
        c2 = min(nc, i + ncstep - 1)
        c3 = max(1, c1 - ncmed // 2)
        c4 = min(nc, c2 + ncmed // 2)
        nc1 = c4 - c3 + 1
        block = np.zeros((nl, nc1), dtype=np.int16)
        cm_mask(pixels[:, c3 - 1:c4], block, nc1, nl, c1 - c3 + 1, c2 - c3 + 1, ncmed, nlmed,
                ncsig, nlsig, real(lsigma), real(hsigma), ngood, plsig, phsig)
        cm_interp(block, nc1, nl, c1 - c3 + 1, c2 - c3 + 1, nc, linterp, cinterp, eqinterp)
        out[:, c1 - 1:c2] = block[:, c1 - c3:c2 - c3 + 1]
    if nc1 != nc:
        for j in range(nl):
            line = out[j].copy()
            out[j] = 0
            cm_interp1(line, out[j], nc, nl, linterp, cinterp, eqinterp)
    return out
