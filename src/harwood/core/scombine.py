# ruff: noqa: E741, SIM102, SIM108
# A line-by-line port keeps IRAF's names (l, j, nl, nh) and its nesting, so that it
# can be read beside the .x it came from.
"""scombine's combining and rejection, as IRAF's onedspec does them.

From noao/onedspec/scombine (icombine.x, icgdata.x, icscale.x, icstat.x, icsum.x
and generic/*.x): the icombine scombine was built with, older than the imcombine
core/combine.py ports, and different where it matters -- values added back after a
sigma clip are looked for up to n2 rather than nh, minmax keeps no track of which
spectrum each value came from unless it must, grow is a whole number of pixels
along the spectrum, and the median of the statistics is the n/2-th value, below the
middle for an odd number. Each output pixel carries its values in slots (`d`),
which spectrum each came from (`m`) and how many are in use (`n`), and every
algorithm moves them as IRAF's moves them; all of it in real.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

f32 = np.float32
MAX_REAL = f32(3.4028235e38)
TOL = f32(0.001)
D_ALL, D_NONE, D_MIX = 0, 1, 2
NMIN, ZRANGE, ZSTEP, ZBIN = 10, 0.8, 0.01, 0.1


@dataclass
class State:
    """icombine.com: what every routine reads and some set."""
    combine: str = "average"
    reject: str = "none"
    blank: float = 0.0
    lsigma: float = 3.0
    hsigma: float = 3.0
    pclip: float = -0.5
    flow: float = 0.0
    fhigh: float = 0.0
    nkeep: int = 1
    grow: int = 0
    mclip: bool = True
    sigscale: float = 0.1
    dothresh: bool = False
    lthresh: float = -3.4028235e38
    hthresh: float = 3.4028235e38
    doscale: bool = False
    doscale1: bool = False
    dowts: bool = False
    keepids: bool = False
    docombine: bool = True
    dflag: int = D_ALL


class Stack:
    """The output pixel's values: d[j][i] for slot j (1-based), m the ids, n."""

    def __init__(self, nimages: int, npts: int):
        self.d = [np.zeros(npts, dtype=f32) for _ in range(nimages + 1)]
        self.m = [np.zeros(npts, dtype=np.int64) for _ in range(nimages + 1)]
        self.n = np.zeros(npts, dtype=np.int64)


def gdata(st: State, data, masks, lflag, scales, zeros, nimages: int, npts: int) -> Stack:
    """ic_gdatar: thresholds, scaling, the good values compacted into the slots, and
    sorted when mclip is to use them."""
    data = [np.asarray(d, dtype=f32).copy() for d in data]
    masks = [np.asarray(m, dtype=f32).copy() for m in masks]
    lflag = list(lflag)
    if st.dothresh:
        for i in range(nimages):
            dp = data[i]
            if lflag[i] == D_ALL:
                for j in range(npts):
                    if dp[j] < st.lthresh or dp[j] > st.hthresh:
                        masks[i][j] = 1
                        lflag[i] = D_MIX
                        st.dflag = D_MIX
            elif lflag[i] == D_MIX:
                for j in range(npts):
                    if masks[i][j] == 0 and (dp[j] < st.lthresh or dp[j] > st.hthresh):
                        masks[i][j] = 1
                        st.dflag = D_MIX
            if lflag[i] == D_MIX:
                lflag[i] = D_NONE if not (masks[i] == 0).any() else D_MIX
    if st.doscale:
        for i in range(nimages):
            a, b = f32(scales[i]), f32(-zeros[i])
            if st.dflag == D_ALL or lflag[i] == D_ALL:
                data[i] = (data[i] / a + b).astype(f32)
            elif st.dflag == D_MIX and lflag[i] == D_MIX:
                good = masks[i] == 0
                data[i][good] = (data[i][good] / a + b).astype(f32)
    stack = Stack(nimages, npts)
    if st.dflag == D_ALL:
        for i in range(nimages):
            stack.d[i + 1][:] = data[i]
            stack.m[i + 1][:] = i + 1
        stack.n[:] = nimages
    else:
        used = [i for i in range(nimages) if lflag[i] != D_NONE]
        if not used:
            st.dflag = D_NONE
        for slot, i in enumerate(used, 1):
            stack.d[slot][:] = data[i]
            stack.m[slot][:] = i + 1
        for slot, i in enumerate(used, 1):
            for j in range(npts):
                if masks[i][j] == 0:
                    stack.n[j] += 1
                    k = stack.n[j]
                    if k < slot:
                        stack.d[k][j] = stack.d[slot][j]
                        stack.m[k][j] = i + 1
        if st.grow > 0:
            for j in range(npts):
                for slot in range(int(stack.n[j]) + 1, nimages + 1):
                    stack.m[slot][j] = 0
    if st.mclip:
        for j in range(npts):
            sort(stack, j, int(stack.n[j]), st.keepids)
    return stack


def sort(stack: Stack, l: int, npix: int, ids: bool) -> None:
    """ic_sortr and ic_2sortr on one pixel: its values in increasing order, and with
    them the ids -- but for IRAF's slip with three."""
    if npix <= 1:
        return
    b = [f32(stack.d[i][l]) for i in range(1, npix + 1)]
    c = [int(stack.m[i][l]) for i in range(1, npix + 1)]
    order = sorted(range(npix), key=lambda i: (b[i], i))
    if npix <= 3:
        order = _three(b) if npix == 3 else ([1, 0] if b[1] < b[0] else [0, 1])
    else:
        order = _quick(b)
    for i, o in enumerate(order, 1):
        stack.d[i][l] = b[o]
        if ids:
            stack.m[i][l] = c[o]
    if ids and npix == 3 and order == [1, 2, 0]:
        # The case IRAF calls bca moves the values but only half the ids: the
        # second takes the third's, the others stay where they were.
        stack.m[1][l], stack.m[2][l], stack.m[3][l] = c[0], c[2], c[2]


def _three(b):
    """The special case of three: the same exchanges, as positions."""
    p, t, t3 = b
    if t < p:
        if t < t3:
            return [1, 0, 2] if p < t3 else [1, 2, 0]
        return [2, 1, 0]
    if t3 < t:
        return [0, 2, 1] if p < t3 else [2, 0, 1]
    return [0, 1, 2]


def _quick(values):
    """IRAF's quicksort, the pivot from the middle; positions carried along."""
    b = [(v, i) for i, v in enumerate(values)]
    lv, uv, p = [0, 0], [0, len(b) - 1], 1
    while p > 0:
        if lv[p] >= uv[p]:
            p -= 1
            continue
        i, j = lv[p] - 1, uv[p]
        k = (lv[p] + uv[p]) // 2
        b[j], b[k] = b[k], b[j]
        pivot = b[j][0]
        while i < j:
            i += 1
            while b[i][0] < pivot:
                i += 1
            j -= 1
            while j > i and not b[j][0] <= pivot:
                j -= 1
            if i < j:
                b[i], b[j] = b[j], b[i]
        j = uv[p]
        b[i], b[j] = b[j], b[i]
        if len(lv) <= p + 1:
            lv.append(0)
            uv.append(0)
        if i - lv[p] < uv[p] - i:
            lv[p + 1], uv[p + 1], lv[p] = lv[p], i - 1, i + 1
        else:
            lv[p + 1], uv[p + 1], uv[p] = i + 1, uv[p], i - 1
        p += 1
    return [i for _, i in b]


def _maxkeep(nkeep: int, n: int) -> int:
    return max(0, n + nkeep) if nkeep < 0 else min(n, nkeep)


def mm(st: State, s: Stack, npts: int) -> None:
    """ic_mmr: the nlow lowest and nhigh highest of each pixel."""
    if st.dflag == D_NONE:
        return
    d, m, n = s.d, s.m, s.n
    if st.dflag == D_ALL:
        n1 = int(n[0])
        nlow = int(st.flow * n1 + 0.001)
        nhigh = int(st.fhigh * n1 + 0.001)
        npairs = min(nlow, nhigh)
        nlow, nhigh = nlow - npairs, nhigh - npairs
    for i in range(npts):
        n1 = int(n[i])
        if st.dflag == D_MIX:
            nlow = int(st.flow * n1 + 0.001)
            nhigh = int(st.fhigh * n1 + 0.001)
            npairs = min(nlow, nhigh)
            nlow, nhigh = nlow - npairs, nhigh - npairs
        for _ in range(npairs):
            d1 = d[1][i]
            dmax = dmin = d1
            jmax = jmin = 1
            d2 = d1
            for j in range(2, n1 + 1):
                d2 = d1
                d1 = d[j][i]
                if d1 > dmax:
                    dmax, jmax = d1, j
                elif d1 < dmin:
                    dmin, jmin = d1, j
            j = n1 - 1
            if jmax < j:
                if jmin != j:
                    d[jmax][i] = d2
                    if st.keepids:
                        m[jmax][i] = m[j][i]
                else:
                    d[jmax][i] = d1
                    if st.keepids:
                        m[jmax][i] = m[n1][i]
            if jmin < j:
                if jmax != n1:
                    d[jmin][i] = d1
                    if st.keepids:
                        m[jmin][i] = m[n1][i]
                else:
                    d[jmin][i] = d2
                    if st.keepids:
                        m[jmin][i] = m[j][i]
            n1 -= 2
        for _ in range(nlow):
            d1 = d[1][i]
            dmin, jmin = d1, 1
            for j in range(2, n1 + 1):
                d1 = d[j][i]
                if d1 < dmin:
                    dmin, jmin = d1, j
            if jmin < n1:
                d[jmin][i] = d1
                if st.keepids:
                    m[jmin][i] = m[n1][i]
            n1 -= 1
        for _ in range(nhigh):
            d1 = d[1][i]
            dmax, jmax = d1, 1
            for j in range(2, n1 + 1):
                d1 = d[j][i]
                if d1 > dmax:
                    dmax, jmax = d1, j
            if jmax < n1:
                d[jmax][i] = d1
                if st.keepids:
                    m[jmax][i] = m[n1][i]
            n1 -= 1
        n[i] = n1
    if st.dflag == D_ALL and npairs + nlow + nhigh > 0:
        st.dflag = D_MIX


def asigclip(st: State, s: Stack, scales, zeros, nimages: int, npts: int,
             average: np.ndarray) -> None:
    """ic_asigclipr: about the average, the highest and lowest left out of it."""
    d, m, n = s.d, s.m, s.n
    maxkeep = _maxkeep(st.nkeep, nimages)
    if nimages < max(3, maxkeep + 1) or st.dflag == D_NONE:
        st.docombine = True
        return
    st.docombine = st.dowts or st.combine != "average"
    one = f32(1)
    resid = [f32(0)] * (nimages + 2)
    w = [f32(0)] * (nimages + 1)
    nin = int(n[0])
    lsig, hsig = f32(st.lsigma), f32(st.hsigma)
    for i in range(npts):
        n1 = int(n[i])
        maxkeep = _maxkeep(st.nkeep, n1)
        if n1 < max(3, maxkeep):
            if not st.docombine:
                if n1 == 0:
                    average[i] = st.blank
                else:
                    total = d[1][i]
                    for j in range(2, n1 + 1):
                        total = f32(total + d[j][i])
                    average[i] = f32(total / f32(n1))
            continue
        a, total = _low_high_average(d, i, n1)
        while True:
            n2 = n1
            if st.doscale1:
                sg = f32(0)
                for j in range(1, n1 + 1):
                    l = int(m[j][i])
                    r = f32(math.sqrt(max(one, f32(f32(a + f32(zeros[l - 1]))
                                                  / f32(scales[l - 1])))))
                    sg = f32(sg + f32(f32(f32(d[j][i] - a) / r) ** 2))
                    w[j] = r
                sg = f32(math.sqrt(f32(sg / f32(n1 - 1))))
                if sg > 0:
                    j = 1
                    while j <= n1:
                        d1 = d[j][i]
                        r = f32(f32(d1 - a) / f32(sg * w[j]))
                        if r < -lsig or r > hsig:
                            resid[n1] = abs(r)
                            if j < n1:
                                d[j][i] = d[n1][i]
                                d[n1][i] = d1
                                w[j] = w[n1]
                                l = m[j][i]
                                m[j][i] = m[n1][i]
                                m[n1][i] = l
                                j -= 1
                            total = f32(total - d1)
                            n1 -= 1
                        j += 1
            else:
                sg = f32(0)
                for j in range(1, n1 + 1):
                    sg = f32(sg + f32(f32(d[j][i] - a) ** 2))
                sg = f32(math.sqrt(f32(sg / f32(n1 - 1))))
                if sg > 0:
                    j = 1
                    while j <= n1:
                        d1 = d[j][i]
                        r = f32(f32(d1 - a) / sg)
                        if r < -lsig or r > hsig:
                            resid[n1] = abs(r)
                            if j < n1:
                                d[j][i] = d[n1][i]
                                d[n1][i] = d1
                                if st.keepids:
                                    l = m[j][i]
                                    m[j][i] = m[n1][i]
                                    m[n1][i] = l
                                j -= 1
                            total = f32(total - d1)
                            n1 -= 1
                        j += 1
            if n1 > 1:
                a = f32(total / f32(n1))
            if n1 == n2 or n1 <= max(2, maxkeep):
                break
        if n1 < maxkeep:
            n1, total, _ = _add_back(st, d, m, i, resid, n1, n2, maxkeep, total)
            if n1 > 1:
                a = f32(total / f32(n1))
        n[i] = n1
        if not st.docombine:
            average[i] = a if n1 > 0 else f32(st.blank)
    if st.dflag == D_ALL and any(int(n[i]) != nin for i in range(npts)):
        st.dflag = D_MIX


def msigclip(st: State, s: Stack, scales, zeros, nimages: int, npts: int,
             median: np.ndarray) -> None:
    """ic_msigclipr: about the median, the values already in order."""
    d, m, n = s.d, s.m, s.n
    maxkeep = _maxkeep(st.nkeep, nimages)
    if nimages < max(3, maxkeep + 1) or st.dflag == D_NONE:
        st.docombine = True
        return
    one = f32(1)
    resid = [f32(0)] * (nimages + 2)
    w = [f32(0)] * (nimages + 1)
    nin = int(n[0])
    lsig, hsig = f32(st.lsigma), f32(st.hsigma)
    for i in range(npts):
        n1 = int(n[i])
        maxkeep = _maxkeep(st.nkeep, n1)
        nl, nh = 1, n1
        while True:
            n2 = n1
            n3 = nl + n1 // 2
            if n1 == 0:
                med = f32(st.blank)
            elif n1 % 2 == 0:
                med = f32(f32(d[n3 - 1][i] + d[n3][i]) / f32(2))
            else:
                med = d[n3][i]
            if n1 >= max(3, maxkeep + 1):
                sg = f32(0)
                if st.doscale1:
                    for j in range(nl, nh + 1):
                        l = int(m[j][i])
                        r = f32(math.sqrt(max(one, f32(f32(med + f32(zeros[l - 1]))
                                                      / f32(scales[l - 1])))))
                        sg = f32(sg + f32(f32(f32(d[j][i] - med) / r) ** 2))
                        w[j] = r
                else:
                    for j in range(nl, nh + 1):
                        sg = f32(sg + f32(f32(d[j][i] - med) ** 2))
                sg = f32(math.sqrt(f32(sg / f32(n1 - 1))))
                if sg > 0:
                    while nl <= n2:
                        scale = f32(sg * w[nl]) if st.doscale1 else sg
                        r = f32(f32(med - d[nl][i]) / scale)
                        if r <= lsig:
                            break
                        resid[nl] = r
                        n1 -= 1
                        nl += 1
                    while nh >= nl:
                        scale = f32(sg * w[nh]) if st.doscale1 else sg
                        r = f32(f32(d[nh][i] - med) / scale)
                        if r <= hsig:
                            break
                        resid[nh] = r
                        n1 -= 1
                        nh -= 1
            if n1 == n2 or n1 < max(3, maxkeep + 1):
                break
        nl, nh, n1, _ = _restore(resid, nl, nh, n1, n2, int(n[i]), maxkeep)
        n[i] = n1
        _compact(st, d, m, i, n1, nl)
        if st.combine == "median":
            median[i] = med
    if st.dflag == D_ALL and any(int(n[i]) != nin for i in range(npts)):
        st.dflag = D_MIX
    st.docombine = st.combine != "median"


def grow(st: State, s: Stack, nimages: int, npts: int, average: np.ndarray) -> None:
    """ic_growr: a value rejected takes with it its spectrum's neighbours, grow
    pixels either side."""
    if st.dflag == D_NONE:
        return
    d, m, n = s.d, s.m, s.n
    for i1 in range(npts):
        lo, hi = max(0, i1 - st.grow), min(npts - 1, i1 + st.grow)
        for j1 in range(int(n[i1]) + 1, nimages + 1):
            l = int(m[j1][i1])
            if l == 0:
                continue
            if st.combine == "median":
                st.docombine = True
            for i2 in range(lo, hi + 1):
                if i2 == i1:
                    continue
                n2 = int(n[i2])
                if n2 <= _maxkeep(st.nkeep, n2):
                    continue
                for j2 in range(1, n2 + 1):
                    if int(m[j2][i2]) == l:
                        if not st.docombine and n2 > 1:
                            average[i2] = f32(f32(f32(f32(n2) * average[i2])
                                                  - d[j2][i2]) / f32(n2 - 1))
                        if j2 < n2:
                            d[j2][i2] = d[n2][i2]
                            m[j2][i2] = m[n2][i2]
                        m[n2][i2] = 0
                        n[i2] = n2 - 1
                        break


def average(st: State, s: Stack, wts, npts: int) -> np.ndarray:
    """ic_averager."""
    d, m, n = s.d, s.m, s.n
    out = np.empty(npts, dtype=f32)
    for i in range(npts):
        n1 = int(n[i])
        if st.dflag == D_NONE or (st.dflag == D_MIX and n1 == 0):
            out[i] = st.blank
            continue
        if st.dowts:
            wt = f32(wts[int(m[1][i]) - 1])
            total = f32(d[1][i] * wt)
            sumwt = wt
            for j in range(2, n1 + 1):
                wt = f32(wts[int(m[j][i]) - 1])
                total = f32(total + f32(d[j][i] * wt))
                sumwt = f32(sumwt + wt)
            out[i] = total if st.dflag == D_ALL else f32(total / sumwt)
        else:
            total = d[1][i]
            for j in range(2, n1 + 1):
                total = f32(total + d[j][i])
            out[i] = f32(total / f32(n1))
    return out


def median(st: State, s: Stack, npts: int) -> np.ndarray:
    """ic_medianr: of values in order, or by exchanging the extremes."""
    d, n = s.d, s.n
    out = np.empty(npts, dtype=f32)
    if st.dflag == D_NONE:
        out[:] = st.blank
        return out
    two = f32(2)
    for i in range(npts):
        n1 = int(n[i])
        if st.mclip:
            if n1 == 0:
                out[i] = st.blank
            elif n1 % 2 == 0:
                out[i] = f32(f32(d[n1 // 2 + 1][i] + d[n1 // 2][i]) / two)
            else:
                out[i] = d[n1 // 2 + 1][i]
            continue
        while n1 > 3:
            j1 = j2 = 1
            val1 = val2 = d[1][i]
            val3 = val1
            for j3 in range(2, n1 + 1):
                val3 = d[j3][i]
                if val3 > val1:
                    j1, val1 = j3, val3
                elif val3 < val2:
                    j2, val2 = j3, val3
            j3 = n1 - 1
            if j1 < j3 and j2 < j3:
                d[j1][i] = val3
                d[j2][i] = d[j3][i]
                d[j3][i] = val1
                d[n1][i] = val2
            elif j1 < j3:
                if j2 == j3:
                    d[j1][i] = val3
                    d[n1][i] = val1
                else:
                    d[j1][i] = d[j3][i]
                    d[j3][i] = val1
            elif j2 < j3:
                if j1 == j3:
                    d[j2][i] = val3
                    d[n1][i] = val2
                else:
                    d[j2][i] = d[j3][i]
                    d[j3][i] = val2
            n1 -= 2
        if n1 == 3:
            v1, v2, v3 = d[1][i], d[2][i], d[3][i]
            if v1 < v2:
                out[i] = v2 if v2 < v3 else (v3 if v1 < v3 else v1)
            else:
                out[i] = v2 if v2 > v3 else (v3 if v1 > v3 else v1)
        elif n1 == 2:
            out[i] = f32(f32(d[1][i] + d[2][i]) / two)
        elif n1 == 1:
            out[i] = d[1][i]
        else:
            out[i] = st.blank
    return out


def total(st: State, s: Stack, npts: int) -> np.ndarray:
    """ic_sumr."""
    d, n = s.d, s.n
    out = np.empty(npts, dtype=f32)
    for i in range(npts):
        n1 = int(n[i])
        if st.dflag == D_NONE or (st.dflag == D_MIX and n1 == 0):
            out[i] = st.blank
            continue
        value = d[1][i]
        for j in range(2, n1 + 1):
            value = f32(value + d[j][i])
        out[i] = value
    return out


def _pooled(st, d, m, n, scales, zeros, npts, centre_of):
    """The sigma pooled over the spectrum, about each pixel's centre (the average
    without its extremes, or the median): s, and the centres."""
    one = f32(1)
    s, n2 = f32(0), 0
    centres = np.zeros(npts, dtype=f32)
    for i in range(npts):
        n1 = int(n[i])
        c = centre_of(i, n1)
        if c is None:
            continue
        centres[i] = c[0]
        if st.doscale1:
            for j in range(1, n1 + 1):
                l = int(m[j][i])
                s1 = max(one, f32(f32(c[0] + f32(zeros[l - 1])) / f32(scales[l - 1])))
                s = f32(s + f32(f32(f32(d[j][i] - c[0]) ** 2) / s1))
        else:
            s1 = max(one, c[0])
            for j in range(1, n1 + 1):
                s = f32(s + f32(f32(f32(d[j][i] - c[0]) ** 2) / s1))
        n2 += n1
    return s, n2, centres


def aavsigclip(st: State, s_: Stack, scales, zeros, nimages: int, npts: int,
               average: np.ndarray) -> None:
    """ic_aavsigclipr: about the average, the sigma pooled over the spectrum. Adding
    values back reuses the pooled sigma's variable, and so changes it for the pixels
    after, as IRAF's does."""
    d, m, n = s_.d, s_.m, s_.n
    maxkeep = _maxkeep(st.nkeep, nimages)
    if nimages < max(3, maxkeep + 1) or st.dflag == D_NONE:
        st.docombine = True
        return
    st.docombine = st.dowts or st.combine != "average"
    one = f32(1)
    sums = np.zeros(npts, dtype=f32)
    resid = [f32(0)] * (nimages + 2)
    nin = int(n[0])

    def centre(i, n1):
        if n1 < 3:
            return None
        a, sums[i] = _low_high_average(d, i, n1)
        average[i] = a
        return (a,)

    s, n2, _ = _pooled(st, d, m, n, scales, zeros, npts, centre)
    if n2 > 1:
        s = f32(math.sqrt(f32(s / f32(n2 - 1))))
    lsig, hsig = f32(st.lsigma), f32(st.hsigma)
    for i in range(npts):
        n1 = int(n[i])
        maxkeep = _maxkeep(st.nkeep, n1)
        if n1 <= max(2, maxkeep):
            if not st.docombine:
                if n1 == 0:
                    average[i] = st.blank
                else:
                    total = d[1][i]
                    for j in range(2, n1 + 1):
                        total = f32(total + d[j][i])
                    average[i] = f32(total / f32(n1))
            continue
        a, total = average[i], sums[i]
        while True:
            n2 = n1
            if s > 0:
                j = 1
                if not st.doscale1:
                    s1 = f32(s * f32(math.sqrt(max(one, a))))
                while j <= n1:
                    d1 = d[j][i]
                    if st.doscale1:
                        l = int(m[j][i])
                        s1 = f32(s * f32(math.sqrt(max(one, f32(
                            f32(a + f32(zeros[l - 1])) / f32(scales[l - 1]))))))
                    r = f32(f32(d1 - a) / s1)
                    if r < -lsig or r > hsig:
                        resid[n1] = abs(r)
                        if j < n1:
                            d[j][i] = d[n1][i]
                            d[n1][i] = d1
                            if st.doscale1 or st.keepids:
                                t = m[j][i]
                                m[j][i] = m[n1][i]
                                m[n1][i] = t
                            j -= 1
                        total = f32(total - d1)
                        n1 -= 1
                    j += 1
            if n1 > 1:
                a = f32(total / f32(n1))
            if n1 == n2 or n1 <= max(2, maxkeep):
                break
        if n1 < maxkeep:
            n1, total, last = _add_back(st, d, m, i, resid, n1, n2, maxkeep, total)
            if last is not None:
                s = last                        # the pooled sigma's variable
            if n1 > 1:
                a = f32(total / f32(n1))
        n[i] = n1
        if not st.docombine:
            average[i] = a if n1 > 0 else f32(st.blank)
    if st.dflag == D_ALL and any(int(n[i]) != nin for i in range(npts)):
        st.dflag = D_MIX


def _median_of(d, i, nl, n1, blank):
    if n1 <= 0:
        return f32(blank)
    n3 = nl + n1 // 2
    if n1 % 2 == 0:
        return f32(f32(d[n3 - 1][i] + d[n3][i]) / f32(2))
    return d[n3][i]


def mavsigclip(st: State, s_: Stack, scales, zeros, nimages: int, npts: int,
               median: np.ndarray) -> None:
    """ic_mavsigclipr: about the median, the sigma pooled over the spectrum; its
    adding back reuses the pooled sigma's variable too."""
    d, m, n = s_.d, s_.m, s_.n
    maxkeep = _maxkeep(st.nkeep, nimages)
    if nimages < max(3, maxkeep + 1) or st.dflag == D_NONE:
        st.docombine = True
        return
    one = f32(1)
    resid = [f32(0)] * (nimages + 2)
    nin = int(n[0])

    def centre(i, n1):
        if n1 < 3:
            if n1 == 0:
                median[i] = st.blank
            elif n1 == 1:
                median[i] = d[1][i]
            else:
                median[i] = f32(f32(d[1][i] + d[2][i]) / f32(2))
            return None
        med = _median_of(d, i, 1, n1, st.blank)
        median[i] = med
        return (med,)

    s, n2, _ = _pooled(st, d, m, n, scales, zeros, npts, centre)
    if n2 > 1:
        s = f32(math.sqrt(f32(s / f32(n2 - 1))))
    else:
        return
    lsig, hsig = f32(st.lsigma), f32(st.hsigma)
    for i in range(npts):
        n1 = int(n[i])
        maxkeep = _maxkeep(st.nkeep, n1)
        if n1 < max(3, maxkeep + 1):
            continue
        nl, nh = 1, n1
        med = median[i]
        while True:
            n2 = n1
            if n1 >= max(3, maxkeep + 1) and s > 0:
                def s1_at(j, i=i, s=s, med=med):
                    if st.doscale1:
                        l = int(m[j][i])
                        return f32(s * f32(math.sqrt(max(one, f32(
                            f32(med + f32(zeros[l - 1])) / f32(scales[l - 1]))))))
                    return f32(s * f32(math.sqrt(max(one, med))))
                while nl <= n2:
                    r = f32(f32(med - d[nl][i]) / s1_at(nl))
                    if r <= lsig:
                        break
                    resid[nl] = r
                    n1 -= 1
                    nl += 1
                while nh >= nl:
                    r = f32(f32(d[nh][i] - med) / s1_at(nh))
                    if r <= hsig:
                        break
                    resid[nh] = r
                    n1 -= 1
                    nh -= 1
                if n1 < n2:
                    med = _median_of(d, i, nl, n1, st.blank)
            if n1 == n2 or n1 < max(3, maxkeep + 1):
                break
        nall = int(n[i])
        if n1 < maxkeep:
            box = [med]

            def again(nl, n1, i=i, box=box, n2=n2):
                if n1 < n2:
                    box[0] = _median_of(d, i, nl, n1, st.blank)
            nl, nh, n1, last = _restore(resid, nl, nh, n1, n2, nall, maxkeep, again)
            med = box[0]
            if last is not None:
                s = last                        # the pooled sigma's variable
        n[i] = n1
        _compact(st, d, m, i, n1, nl)
        if st.combine == "median":
            median[i] = med
    if st.dflag == D_ALL and any(int(n[i]) != nin for i in range(npts)):
        st.dflag = D_MIX
    st.docombine = st.combine != "median"


def _noise(nm, l: int, value) -> np.float32:
    """The CCD noise model's sigma at a value, for the l'th image (1-based):
    sqrt (rdnoise**2/gain**2 + value/gain + (value*snoise)**2)."""
    r = f32(value)
    q = f32(f32(nm[l - 1][0] + f32(r / nm[l - 1][1])) + f32(f32(r * nm[l - 1][2]) ** 2))
    return f32(math.sqrt(q))


def _low_high_average(d, i, n1):
    """The average with the lowest and highest left out, and the sum of them all."""
    low, high = d[1][i], d[2][i]
    if low > high:
        low, high = high, low
    total = f32(0)
    for j in range(3, n1 + 1):
        d1 = d[j][i]
        if d1 < low:
            total = f32(total + low)
            low = d1
        elif d1 > high:
            total = f32(total + high)
            high = d1
        else:
            total = f32(total + d1)
    return f32(total / f32(n1 - 2)), f32(f32(total + low) + high)


def accdclip(st: State, s_: Stack, scales, zeros, nm, nimages: int, npts: int,
             average: np.ndarray) -> None:
    """ic_accdclipr: about the average, the sigma from the CCD noise model. The
    average is taken anew, its extremes left out, at each pass."""
    d, m, n = s_.d, s_.m, s_.n
    zero = f32(0)
    maxkeep = _maxkeep(st.nkeep, nimages)
    if nimages < max(2, maxkeep + 1) or st.dflag == D_NONE:
        st.docombine = True
        return
    st.docombine = st.dowts or st.combine != "average"
    resid = [f32(0)] * (nimages + 2)
    nin = int(n[0])
    lsig, hsig = f32(st.lsigma), f32(st.hsigma)
    for i in range(npts):
        n1 = int(n[i])
        maxkeep = _maxkeep(st.nkeep, n1)
        if n1 <= max(1, maxkeep):
            if not st.docombine:
                if n1 == 0:
                    average[i] = st.blank
                else:
                    total = d[1][i]
                    for j in range(2, n1 + 1):
                        total = f32(total + d[j][i])
                    average[i] = f32(total / f32(n1))
            continue
        while True:
            if n1 == 2:
                total = f32(d[1][i] + d[2][i])
                a = f32(total / f32(2))
            else:
                a, total = _low_high_average(d, i, n1)
            n2 = n1
            if not st.doscale1 and not st.keepids:
                s = _noise(nm, 1, max(zero, a))
            j = 1
            while j <= n1:
                if st.doscale1:
                    l = int(m[j][i])
                    sc = f32(scales[l - 1])
                    s = f32(_noise(nm, l, max(zero, f32(sc * f32(a + f32(zeros[l - 1])))))
                            / sc)
                elif st.keepids:
                    s = _noise(nm, int(m[j][i]), max(zero, a))
                d1 = d[j][i]
                r = f32(f32(d1 - a) / s)
                if r < -lsig or r > hsig:
                    resid[n1] = abs(r)
                    if j < n1:
                        d[j][i] = d[n1][i]
                        d[n1][i] = d1
                        if st.doscale1 or st.keepids:
                            t = m[j][i]
                            m[j][i] = m[n1][i]
                            m[n1][i] = t
                        j -= 1
                    total = f32(total - d1)
                    n1 -= 1
                j += 1
            if n1 == n2 or n1 < max(2, maxkeep + 1):
                break
        if n1 < maxkeep:
            n1, total, _ = _add_back(st, d, m, i, resid, n1, n2, maxkeep, total)
        n[i] = n1
        if not st.docombine:
            average[i] = f32(total / f32(n1)) if n1 > 0 else f32(st.blank)
    if st.dflag == D_ALL and any(int(n[i]) != nin for i in range(npts)):
        st.dflag = D_MIX


def _add_back(st, d, m, i, resid, n1, n2, maxkeep, total):
    """Too many rejected about an average: the least deviant put back, those with
    equal residuals together. Returns n1, the sum, and what was last left in the
    variable IRAF names s (a residual, or with scaling an image's id), None if
    nothing was."""
    s = None
    nk = maxkeep
    j = n1 + 1
    while j <= nk:
        r = resid[j]
        jj = 0
        for l in range(j + 1, n2 + 1):
            s = resid[l]
            if s < r + TOL:
                if s > r - TOL:
                    jj += 1
                else:
                    jj = 0
                    resid[l] = r
                    r = s
                    d1 = d[j][i]
                    d[j][i] = d[l][i]
                    d[l][i] = d1
                    if st.doscale1 or st.keepids:
                        s = f32(m[j][i])
                        m[j][i] = m[l][i]
                        m[l][i] = int(s)
        total = f32(total + d[j][i])
        n1 += 1
        nk = max(nk, j + jj)
        j += 1
    return n1, total, s


def _compact(st, d, m, i, n1, nl):
    """After clipping about the median: the values kept moved down to the start."""
    if n1 > 0 and nl > 1 and (st.combine != "median" or st.grow > 0):
        j = max(nl, n1 + 1)
        for l in range(1, min(n1, nl - 1) + 1):
            d[l][i] = d[j][i]
            if st.keepids:
                if st.grow > 0:
                    m[l][i], m[j][i] = m[j][i], m[l][i]
                else:
                    m[l][i] = m[j][i]
            j += 1


def _restore(resid, nl, nh, n1, n2, nall, maxkeep, tracks=None):
    """Too many rejected about a median: put back from whichever end has the smaller
    residual, equal ones together. Returns nl, nh, n1 and the last value of IRAF's s;
    `tracks` is called after each step (the median taken again)."""
    s = None
    while n1 < maxkeep:
        if nl == 1:
            nh += 1
        elif nh == nall:
            nl -= 1
        else:
            r, s = resid[nl - 1], resid[nh + 1]
            if r < s:
                nl -= 1
                r = f32(r + TOL)
                if s <= r:
                    nh += 1
                if nl > 1 and resid[nl - 1] <= r:
                    nl -= 1
            else:
                nh += 1
                s = f32(s + TOL)
                if r <= s:
                    nl -= 1
                if nh < n2 and resid[nh + 1] <= s:
                    nh += 1
        n1 = nh - nl + 1
        if tracks is not None:
            tracks(nl, n1)
    return nl, nh, n1, s


def mccdclip(st: State, s_: Stack, scales, zeros, nm, nimages: int, npts: int,
             median: np.ndarray) -> None:
    """ic_mccdclipr: about the median, the sigma from the CCD noise model. The median
    is taken at the start of each pass, and not again after values are put back."""
    d, m, n = s_.d, s_.m, s_.n
    zero = f32(0)
    maxkeep = _maxkeep(st.nkeep, nimages)
    if nimages < max(2, maxkeep + 1) or st.dflag == D_NONE:
        st.docombine = True
        return
    resid = [f32(0)] * (nimages + 2)
    nin = int(n[0])
    lsig, hsig = f32(st.lsigma), f32(st.hsigma)
    for i in range(npts):
        n1 = int(n[i])
        maxkeep = _maxkeep(st.nkeep, n1)
        nl, nh = 1, n1
        while True:
            n2 = n1
            med = _median_of(d, i, nl, n1, st.blank)
            if n1 >= max(2, maxkeep + 1):
                def sigma(j, i=i, med=med):
                    if st.doscale1:
                        l = int(m[j][i])
                        sc = f32(scales[l - 1])
                        r = max(zero, f32(sc * f32(med + f32(zeros[l - 1]))))
                        return f32(_noise(nm, l, r) / sc)
                    return _noise(nm, int(m[j][i]) if st.keepids else 1, max(zero, med))
                while nl <= n2:
                    r = f32(f32(med - d[nl][i]) / sigma(nl))
                    if r <= lsig:
                        break
                    resid[nl] = r
                    n1 -= 1
                    nl += 1
                while nh >= nl:
                    r = f32(f32(d[nh][i] - med) / sigma(nh))
                    if r <= hsig:
                        break
                    resid[nh] = r
                    n1 -= 1
                    nh -= 1
            if n1 == n2 or n1 < max(2, maxkeep + 1):
                break
        nl, nh, n1, _ = _restore(resid, nl, nh, n1, n2, int(n[i]), maxkeep)
        n[i] = n1
        _compact(st, d, m, i, n1, nl)
        if st.combine == "median":
            median[i] = med
    if st.dflag == D_ALL and any(int(n[i]) != nin for i in range(npts)):
        st.dflag = D_MIX
    st.docombine = st.combine != "median"


def _nint(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def pclip(st: State, s_: Stack, nimages: int, npts: int, median: np.ndarray) -> None:
    """ic_pclipr: the sigma is the distance from the median to the value pclip places
    from it. Its putting back stops at the median's place where the others stop at
    the last value, and the values kept are moved down counting the first number of
    them, as IRAF's do."""
    d, m, n = s_.d, s_.m, s_.n
    maxkeep = _maxkeep(st.nkeep, nimages)
    if nimages < max(3, maxkeep + 1) or st.dflag == D_NONE:
        st.docombine = True
        return
    resid = [f32(0)] * (nimages + 2)
    p = float(st.pclip)
    t = f32(-1) if p < 0 else f32(1)
    lsig, hsig = f32(st.lsigma), f32(st.hsigma)

    def places(n1):
        n2 = 1 + n1 // 2
        even = n1 % 2 == 0
        if p < 0:
            n3 = max(1, _nint(n2 - 1 + p)) if even else max(1, _nint(n2 + p))
        else:
            n3 = min(n1, _nint(n2 + p))
        return n2, even, n3

    if st.dflag == D_ALL:
        n1 = int(n[0])
        maxkeep = _maxkeep(st.nkeep, n1)
        n2, even, n3 = places(n1)
        nin = n1
    for i in range(npts):
        if st.dflag == D_MIX:
            n1 = int(n[i])
            maxkeep = _maxkeep(st.nkeep, n1)
            if n1 == 0:
                if st.combine == "median":
                    median[i] = st.blank
                continue
            n2, even, n3 = places(n1)
        if even:
            med = f32(f32(d[n2 - 1][i] + d[n2][i]) / f32(2))
        else:
            med = d[n2][i]
        if n1 < max(3, maxkeep + 1):
            if st.combine == "median":
                median[i] = med
            continue
        sigma = f32(t * f32(d[n3][i] - med))
        if sigma == 0:
            if st.combine == "median":
                median[i] = med
            continue
        nl = 1
        while nl <= n1:
            r = f32(f32(med - d[nl][i]) / sigma)
            if r < lsig:
                break
            resid[nl] = r
            nl += 1
        nh = n1
        while nh >= 1:
            r = f32(f32(d[nh][i] - med) / sigma)
            if r < hsig:
                break
            resid[nh] = r
            nh -= 1
        n4 = nh - nl + 1
        nl, nh, n4, _ = _restore(resid, nl, nh, n4, n2, int(n[i]), maxkeep)
        if nl > 1 or nh < n1:
            med = _median_of(d, i, nl, n4, st.blank)
            n[i] = n4
        if st.combine == "median":
            median[i] = med
        if nl > 1 and (st.combine != "median" or st.grow > 0):
            k = max(nl, n4 + 1)
            for l in range(1, min(n1, nl - 1) + 1):
                if k >= len(d):                 # IRAF reads past its slots here
                    break
                d[l][i] = d[k][i]
                if st.keepids:
                    if st.grow > 0:
                        m[l][i], m[k][i] = m[k][i], m[l][i]
                    else:
                        m[l][i] = m[k][i]
                k += 1
    if st.dflag == D_ALL and any(int(n[i]) != nin for i in range(npts)):
        st.dflag = D_MIX
    st.docombine = st.combine != "median"


# ----------------------------------------------------------------- the driver

EPSILONR = f32(1.192e-7)
INDEF_TEXT = "INDEF"
REJECTS = ("none", "ccdclip", "crreject", "minmax", "pclip", "sigclip", "avsigclip")
STYPES = ("none", "mode", "median", "mean", "exposure")
ZTYPES = ("none", "mode", "median", "mean")


@dataclass
class Spectrum:
    """One input spectrum as t_scombine keeps it: its band 1, and the header values
    it may need once its image is closed."""
    sh: object                      #: sarith.Sh
    name: str                       #: IMNAME
    dc: int                         #: its dispersion type, -1 for none
    aperture: object                #: the harwood aperture it came from
    exptime: float | None = None    #: IT: EXPOSURE, ITIME, EXPTIME, the last there
    keys: dict = None               #: scale, zero, weight, rdnoise, gain, snoise


def _fp_normr(x):
    tol = f32(EPSILONR * f32(10))
    absx, expon = abs(f32(x)), 0
    if absx > 0:
        while absx < f32(1 - tol):
            absx = f32(absx * f32(10))
            expon -= 1
            if absx == 0:
                return f32(0), 0
        while absx >= f32(10 + tol):
            absx = f32(absx / f32(10))
            expon += 1
    return (-absx if x < 0 else absx), expon


def fp_equalr(x, y) -> bool:
    x, y = f32(x), f32(y)
    if x == y:
        return True
    if x == 0 or y == 0:
        return False
    nx, ex = _fp_normr(x)
    ny, ey = _fp_normr(y)
    if ex != ey:
        return False
    return f32(f32(1) + abs(f32(nx - ny))) <= f32(f32(1) + f32(EPSILONR * f32(32)))


def mode(a) -> np.float32:
    """ic_moder: the densest bin a tenth of the central 80% of the values wide, on
    values sorted; fewer than ten, the n/2-th."""
    n = len(a)
    if n < NMIN:
        return a[n // 2 - 1]
    i = int(f32(1) + f32(f32(n) * f32(1 - ZRANGE)) / f32(2))
    j = int(f32(1) + f32(f32(n) * f32(1 + ZRANGE)) / f32(2))
    z1, z2 = a[i - 1], a[j - 1]
    if fp_equalr(z1, z2):
        return z1
    zstep = f32(f32(ZSTEP) * f32(z2 - z1))
    zbin = f32(f32(ZBIN) * f32(z2 - z1))
    z1 = f32(z1 - zstep)
    k, nmax, found = i, 0, z1
    while True:
        z1 = f32(z1 + zstep)
        z2 = f32(z1 + zbin)
        while i < j and a[i - 1] < z1:
            i += 1
        while k < j and a[k - 1] < z2:
            k += 1
        if k - i > nmax:
            nmax = k - i
            found = a[(i + k) // 2 - 1]
        if k >= j:
            break
    return found


def asum(values) -> np.float32:
    """asumr: in order, in real."""
    values = np.asarray(values, dtype=f32)
    return np.add.accumulate(values, dtype=f32)[-1] if values.size else f32(0)


def wranges(text: str):
    """ic_wranges: wavelength ranges w1:w2, separated by commas or blanks, or in an
    @file; None for none."""
    from pathlib import Path

    from . import advice, iraffmt

    found = []

    def add(line):
        for word in line.replace(",", " ").split():
            if word.startswith("@"):
                advice.fail("Cannot nest @files", "write the ranges in the one file.")
            parts = word.replace(":", " ").split()
            values = [iraffmt.ctod(p) for p in parts[:2]]
            if len(values) < 2 or None in values:
                advice.fail("Range syntax error", "write a range as w1:w2.")
            w1, w2 = f32(values[0]), f32(values[1])
            found.append((min(w1, w2), max(w1, w2)))

    for word in str(text or "").replace(",", " ").split():
        if word.startswith("@"):
            for line in Path(word[1:]).read_text(encoding="utf-8").splitlines():
                add(line)
        else:
            add(word)
    return found or None


def in_ranges(ranges, w) -> bool:
    if ranges is None:
        return True
    return any(lo <= w <= hi for lo, hi in ranges)


def statistics(st: State, spec: Spectrum, mask, lflag: int, ranges, domode: bool,
               domedian: bool, domean: bool):
    """ic_statr: the mode, median and mean of the spectrum on the output's pixels,
    those masked or outside the thresholds or the sample left out; None if none
    is left."""
    sh = spec.sh
    values = []
    if lflag == D_ALL and ranges is None:
        values = [v for v in sh.data if not st.dothresh or st.lthresh <= v <= st.hthresh]
    elif lflag == D_MIX or (ranges is not None and lflag != D_NONE):
        for i, v in enumerate(sh.data, 1):
            if mask[i - 1] != 0:
                continue
            if st.dothresh and not (st.lthresh <= v <= st.hthresh):
                continue
            if in_ranges(ranges, f32(sh.lw(float(i)))):
                values.append(v)
    if not values:
        return None, None, None
    a = np.asarray(values, dtype=f32)
    md = me = mn = None
    if domode or domedian:
        a = np.sort(a)
        md = mode(a)
        me = a[len(a) // 2 - 1]
    if domean:
        mn = f32(asum(a) / f32(len(a)))
    return md, me, mn


def _leading_number(text: str):
    """ctor: the number a string starts with, or None."""
    import re

    found = re.match(r"\s*([+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?)", text or "")
    return float(found.group(1).replace("d", "e").replace("D", "e")) if found else None


def gscale(param: str, text: str, dic, specs, exptime):
    """ic_gscale: the type and the values; text is what was given, made whole from
    an abbreviation."""
    from pathlib import Path

    from . import advice, iraffmt

    name = "".join(str(text).split())
    n = len(specs)
    values = [None] * n
    if not name:
        return "none", name, values
    if name.startswith("@"):
        found, said = [], []
        for line in Path(name[1:]).read_text(encoding="utf-8").splitlines():
            words = line.split()
            value = iraffmt.ctod(words[0]) if words else None
            if value is None:
                continue
            if len(found) == n:
                said.append(f"Warning: Ignoring additional {param} values in {name[1:]}")
                break
            found.append(f32(value))
        if len(found) < n:
            advice.fail(f"Insufficient {param} values in {name[1:]}",
                        f"write one {param} for each spectrum, one per line.")
        return "file", name, found
    if name.startswith("!"):
        return "keyword", name, [s.keys[param] for s in specs]
    matches = [d for d in dic if d.startswith(name)]
    if name in dic:
        matches = [name]
    if len(matches) != 1:
        advice.fail("Unknown scale, zero, or weight type",
                    f"give one of {', '.join(dic)}, @file or !keyword.")
    kind = matches[0]
    if kind == "exposure":
        for i, s in enumerate(specs):
            if s.exptime is None:
                advice.fail("Exposure time not defined",
                            "the header needs EXPTIME, ITIME or EXPOSURE.")
            exptime[i] = f32(s.exptime)
            values[i] = max(f32(0.001), exptime[i])
    return kind, kind, values


@dataclass
class Scaling:
    scales: np.ndarray
    zeros: np.ndarray
    wts: np.ndarray
    log: list
    said: list
    ncombine: int
    expname: str
    exposure: float | None


def scale(st: State, specs, lflags, masks, combine: str, texts: dict, sample: str,
          logged: dict) -> Scaling:
    """ic_scale: the scales, zero offsets and weights, relative; the flags that say
    which are used; NCOMBINE and the exposure; and the log."""
    n = len(specs)
    ncombine = [1] * n
    exptime = [f32(0)] * n
    modes, medians, means = [None] * n, [None] * n, [None] * n
    scales = [f32(1)] * n
    zeros = [f32(0)] * n
    wts = [f32(1)] * n
    said = []
    if combine == "sum":
        stype = ztype = wtype = "none"
        sname = zname = wname = ""
        exptime = [None if s.exptime is None else f32(s.exptime) for s in specs]
    else:
        stype, sname, v = gscale("scale", texts["scale"], STYPES, specs, exptime)
        scales = [f32(1) if x is None else f32(x) for x in v]
        ztype, zname, v = gscale("zero", texts["zero"], ZTYPES, specs, exptime)
        zeros = [f32(0) if x is None else f32(x) for x in v]
        wtype, wname, v = gscale("weight", texts["weight"], STYPES, specs, exptime)
        wts = [f32(1) if x is None else f32(x) for x in v]
    expname = ""
    if combine == "sum" or stype == "exposure" or wtype == "exposure":
        expname = "exptime"
        if any(e is None for e in exptime):
            expname = ""
    kinds = (stype, ztype, wtype)
    domode, domedian, domean = ("mode" in kinds), ("median" in kinds), ("mean" in kinds)
    if domode or domedian or domean:
        ranges = wranges(sample)
        for i, s in enumerate(specs):
            md, me, mn = statistics(st, s, masks[i], lflags[i], ranges, domode,
                                    domedian, domean)
            for kind, value, store in (("mode", md, modes), ("median", me, medians),
                                       ("mean", mn, means)):
                if not {"mode": domode, "median": domedian, "mean": domean}[kind]:
                    continue
                store[i] = value
                fill = f32(1.6e38) if value is None else value      # INDEF
                if stype == kind:
                    scales[i] = fill
                if ztype == kind:
                    zeros[i] = fill
                if wtype == kind:
                    wts[i] = fill
    if any(x <= 0 for x in scales):
        said.append("WARNING: Negative scale factors -- ignoring scaling")
        scales = [f32(1)] * n
    scales = np.asarray(scales, dtype=f32)
    zeros = np.asarray(zeros, dtype=f32)
    wts = np.asarray(wts, dtype=f32)
    mean = f32(asum(scales) / f32(n))
    scales = (scales / mean).astype(f32)
    zeros = (zeros / scales).astype(f32)
    zmean = f32(asum(zeros) / f32(n))
    if wtype != "none":
        for i in range(n):
            if wts[i] <= 0:
                said.append("WARNING: Negative weights -- using only NCOMBINE weights")
                wts[:] = np.asarray(ncombine, dtype=f32)
                break
            if ztype == "none":
                wts[i] = f32(f32(ncombine[i]) * wts[i])
            else:
                if zeros[i] <= 0:
                    said.append("WARNING: Negative zero offsets -- ignoring zero weight "
                                "adjustments")
                    for j in range(n):
                        wts[j] = f32(f32(ncombine[j]) * wts[j])
                    break
                wts[i] = f32(f32(f32(f32(ncombine[i]) * wts[i]) * zmean) / zeros[i])
    zeros = (zeros - zmean).astype(f32)
    total = asum(wts)
    wts = (wts / total).astype(f32)
    if all(zeros[i] == zeros[0] for i in range(1, n)):
        zeros[:] = 0
    st.doscale = any(scales[i] != scales[0] for i in range(1, n))
    dozero = any(zeros[i] != zeros[0] for i in range(1, n))
    st.dowts = any(wts[i] != wts[0] for i in range(1, n))
    st.doscale1 = False
    if st.doscale and st.sigscale != 0:
        sig = f32(st.sigscale)
        if any(abs(f32(scales[i] - f32(1))) > sig for i in range(n)):
            st.doscale1 = True
        if not st.doscale1 and zmean > 0:
            if any(abs(f32(zeros[i] / zmean)) > sig for i in range(n)):
                st.doscale1 = True
    nout = sum(ncombine)
    exposure = None
    if expname:
        exposure = f32(0)
        for i in range(n):
            if combine == "sum":
                exposure = f32(exposure + exptime[i])
            else:
                exposure = f32(exposure + f32(f32(wts[i] * exptime[i]) / scales[i]))
    log = _log(st, specs, ncombine, exptime, sname, zname, wname, modes, medians,
               means, scales, zeros, wts, dozero, nout, expname, exposure, combine,
               logged)
    st.doscale = st.doscale or dozero
    return Scaling(scales, zeros, wts, log, said, nout, expname,
                   None if exposure is None else float(exposure))


def _r(value, fmt: str, decpl=None, width=None) -> str:
    """A real as fmtio prints it, INDEF for none."""
    from . import iraffmt

    if value is None:
        text = INDEF_TEXT
    else:
        text = iraffmt.printf(float(f32(value)), fmt, decpl, width=width)
    return text.rjust(width) if width else text


def _log(st, specs, ncombine, exptime, sname, zname, wname, modes, medians, means,
         scales, zeros, wts, dozero, nout, expname, exposure, combine, logged):
    """ic_log: the parameters, a line for each spectrum, and the output."""
    from .iraffmt import cnvdate

    yes = lambda b: "yes" if b else "no"  # noqa: E731
    n = len(specs)
    out = ["", f"{cnvdate()}: SCOMBINE"]
    if combine == "sum":
        out.append("  combine = sum")
    else:
        out.append(f"  combine = {combine}, scale = {sname}, zero = {zname}, "
                   f"weight = {wname}")
    reject = st.reject
    g = lambda v: _r(v, "g")  # noqa: E731
    if reject == "minmax":
        out.append(f"  reject = minmax, nlow = {_nint(float(f32(st.flow) * f32(n)))}, "
                   f"nhigh = {_nint(float(f32(st.fhigh) * f32(n)))}")
    elif reject == "ccdclip":
        out.append(f"  reject = ccdclip, mclip = {yes(st.mclip)}, nkeep = {st.nkeep}")
        out.append(f"  rdnoise = {logged['rdnoise']}, gain = {logged['gain']}, "
                   f"snoise = {logged['snoise']}, sigma = {g(st.lsigma)}, "
                   f"hsigma = {g(st.hsigma)}")
    elif reject == "crreject":
        out.append(f"  reject = crreject, mclip = {yes(st.mclip)}, nkeep = {st.nkeep}")
        out.append(f"  rdnoise = {logged['rdnoise']}, gain = {logged['gain']}, "
                   f"snoise = {logged['snoise']}, hsigma = {g(st.hsigma)}")
    elif reject == "pclip":
        out.append(f"  reject = pclip, nkeep = {st.nkeep}")
        out.append(f"  pclip = {g(st.pclip)}, lsigma = {g(st.lsigma)}, "
                   f"hsigma = {g(st.hsigma)}")
    elif reject in ("sigclip", "avsigclip"):
        out.append(f"  reject = {reject}, mclip = {yes(st.mclip)}, nkeep = {st.nkeep}")
        out.append(f"  lsigma = {g(st.lsigma)}, hsigma = {g(st.hsigma)}")
    if reject != "none" and st.grow > 0:
        out.append(f"  grow = {st.grow}")
    if st.dothresh:
        if st.lthresh > -MAX_REAL and st.hthresh < MAX_REAL:
            out.append(f"  lthreshold = {g(st.lthresh)}, hthreshold = {g(st.hthresh)}")
        elif st.lthresh > -MAX_REAL:
            out.append(f"  lthreshold = {g(st.lthresh)}")
        else:
            out.append(f"  hthreshold = {g(st.hthresh)}")
    out.append(f"  blank = {g(st.blank)}")
    if logged["sample"]:
        out.append(f"  sample = {logged['sample']}")
    differs = lambda v: any(v[i] != v[0] for i in range(n))  # noqa: E731
    prncombine = differs(ncombine)
    prexptime = bool(expname) or differs(exptime)
    prscale = st.doscale or sname != "none"
    przero = dozero or zname != "none"
    prwts = st.dowts or wname != "none"
    prmode, prmedian, prmean = differs(modes), differs(medians), differs(means)
    ccd = reject in ("ccdclip", "crreject")
    prrdn = ccd and _leading_number(logged["rdnoise"]) is None
    prgain = ccd and _leading_number(logged["gain"]) is None
    prsn = ccd and _leading_number(logged["snoise"]) is None
    head = f"  {'Images':>20} "
    for show, width, label in ((prncombine, 6, "N"), (prexptime, 6, "Exp"),
                               (prmode, 7, "Mode"), (prmedian, 7, "Median"),
                               (prmean, 7, "Mean"), (prrdn, 7, "Rdnoise"),
                               (prgain, 6, "Gain"), (prsn, 6, "Snoise"),
                               (prscale, 6, "Scale"), (przero, 7, "Zero"),
                               (prwts, 6, "Weight")):
        if show:
            head += f" {label:>{width}}"
    out.append(head)
    for i, s in enumerate(specs):
        row = f"  {s.name:>16}[{s.sh.ap:3d}]"
        if prncombine:
            row += f" {ncombine[i]:6d}"
        if prexptime:
            row += " " + _r(exptime[i], "f", 1, 6)
        if prmode:
            row += " " + _r(modes[i], "g", 5, 7)
        if prmedian:
            row += " " + _r(medians[i], "g", 5, 7)
        if prmean:
            row += " " + _r(means[i], "g", 5, 7)
        if prrdn:
            row += " " + _r(s.keys["rdnoise"], "g", None, 7)
        if prgain:
            row += " " + _r(s.keys["gain"], "g", None, 6)
        if prsn:
            row += " " + _r(s.keys["snoise"], "g", None, 6)
        if prscale:
            row += " " + _r(f32(f32(1) / scales[i]), "f", 3, 6)
        if przero:
            row += " " + _r(-zeros[i], "g", 5, 7)
        if prwts:
            row += " " + _r(wts[i], "f", 3, 6)
        out.append(row)
    line = f"  Output image = {logged['output']}, ncombine = {nout}"
    if expname:
        line += f", {expname} = {g(exposure)}"
    out += ["", line]
    o = logged["out"]
    out.append(f"  w1 = {g(o.w0)}, w2 = {g(o.w1)}, dw = {g(o.wp)}, nw = {g(o.n)}, "
               f"dtype = {logged['dtype']}")
    return out


def default_grid(specs, w1a, w2a, dwa, nwa, log: bool):
    """scb_default: the output's dispersion type, first wavelength, step and length
    from the spectra and what was given (None, INDEF). The step is the smallest of
    theirs, the range all of theirs together."""
    dtype = 1 if log else 0
    if dtype == 0 and any(s.dc == -1 for s in specs):
        dtype = -1
    big = float(MAX_REAL)
    w1, w2, dw, nw = w1a, w2a, dwa, nwa
    if w1 is None:
        if dw is None or dw > 0:
            w1 = big
            for s in specs:
                aux = float(s.sh.w0 if s.sh.wp > 0 else s.sh.w1)
                w1 = min(w1, aux)
        else:
            w1 = -big
            for s in specs:
                aux = float(s.sh.w1 if s.sh.wp > 0 else s.sh.w0)
                w1 = max(w1, aux)
    if w2 is None:
        if dw is None or dw > 0:
            w2 = -big
            for s in specs:
                aux = float(s.sh.w1 if s.sh.wp > 0 else s.sh.w0)
                w2 = max(w2, aux)
        else:
            w2 = big
            for s in specs:
                aux = float(s.sh.w0 if s.sh.wp > 0 else s.sh.w1)
                w2 = min(w2, aux)
    if dw is None:
        dw = big
        for s in specs:
            dw = min(dw, abs(float(s.sh.wp)))
    if (w2 - w1) / dw < 0:
        dw = -dw
    if nw is None:
        nw = int((w2 - w1) / dw + 0.5) + 1
    if dwa is None:
        dw = (w2 - w1) / (nw - 1)
    elif w2a is None:
        w2 = w1 + (nw - 1) * dw
    elif w1a is None:
        w1 = w2 - (nw - 1) * dw
    else:
        nw = int((w2 - w1) / dw + 0.5) + 1
        w2 = w1 + (nw - 1) * dw
    return dtype, w1, dw, nw


def grid_wavelengths(dtype: int, w1: float, dw: float, nw: int):
    """The wavelengths smw_swattrs gives the output: w1 and dw are wavelengths, a
    logarithmic one's too, written to 14 figures in the world coordinates."""
    pixel = np.arange(nw, dtype=float)
    if dtype == -1:
        return None
    if dtype == 1:
        a = float(f"{math.log10(w1):.14g}")
        b = float(f"{math.log10((w1 + (nw - 1) * dw) / w1) / (nw - 1):.14g}")
        return 10.0 ** (a + b * pixel)
    a, b = float(f"{w1:.14g}"), float(f"{dw:.14g}")
    return a + b * pixel


def rebin_all(specs, out, interp: str):
    """scb_rebin: each spectrum onto the output's pixels, and its mask: the pixels it
    does not reach -- less than 1% of one of its pixels -- are bad. A spectrum that
    does not reach the output at all is left as it is, D_NONE."""
    from . import sarith

    npts = out.n
    lflags, masks = [], []
    for s in specs:
        sh = s.sh
        c = f32(out.wl(sh.lw(0.5)))
        b = f32(out.wl(sh.lw(sh.n + 0.5)))
        a = max(1, _nint(float(f32(min(b, c) + f32(0.01)))))
        b = min(npts, _nint(float(f32(max(b, c) - f32(0.01)))))
        j = b - a + 1
        if j < 1:
            lflags.append(D_NONE)
            masks.append(None)
            continue
        lflags.append(D_MIX if j < npts else D_ALL)
        sarith.rebin(sh, out, interp)
        mask = np.zeros(npts, dtype=f32)
        mask[:a - 1] = 1
        mask[b:] = 1
        masks.append(mask)
    dflag = lflags[0]
    if any(f != dflag for f in lflags[1:]):
        dflag = D_MIX
    return lflags, masks, dflag


def noise_model(specs, texts: dict):
    """The CCD noise model per spectrum: (rdnoise/gain)**2, gain, snoise; a number
    given, or the header keyword it names."""
    nm = np.zeros((len(specs), 3), dtype=f32)
    for column, key in ((0, "rdnoise"), (1, "gain"), (2, "snoise")):
        value = _leading_number(texts[key])
        for i, s in enumerate(specs):
            nm[i][column] = f32(value) if value is not None else f32(s.keys[key])
    for i in range(len(specs)):
        nm[i][0] = f32(f32(nm[i][0] / nm[i][1]) ** 2)
    return nm


def combine_group(st: State, specs, out, params: dict, logged: dict):
    """ic_combiner on one output aperture: the result, the number combined at each
    pixel, the scaling (with the log) and what was said."""
    npts = out.n
    nimages = len(specs)
    lflags, masks, st.dflag = rebin_all(specs, out, params["interp"])
    scaling = scale(st, specs, lflags, masks, st.combine, params, params["sample"],
                    logged)
    if st.combine == "average":
        st.keepids = st.dowts
    elif st.combine == "median":
        st.dowts = False
        st.keepids = False
    else:
        st.keepids = False
        st.reject = "none"
        st.grow = 0
    st.docombine = True
    nm = None
    if st.reject in ("ccdclip", "crreject"):
        nm = noise_model(specs, params)
        if not st.keepids and (st.doscale1 or st.grow > 0 or any(
                (nm[i] != nm[0]).any() for i in range(1, nimages))):
            st.keepids = True
        if st.reject == "crreject":
            st.lsigma = float(MAX_REAL)
    elif st.reject == "minmax":
        st.mclip = False
        if st.grow > 0:
            st.keepids = True
    elif st.reject == "pclip":
        st.mclip = True
        if st.grow > 0:
            st.keepids = True
    elif st.reject in ("avsigclip", "sigclip"):
        if st.doscale1 or st.grow > 0:
            st.keepids = True
    else:
        st.mclip = False
        st.grow = 0
    data = [s.sh.data if f != D_NONE else np.zeros(npts, dtype=f32)
            for s, f in zip(specs, lflags, strict=True)]
    full = [m if m is not None else np.ones(npts, dtype=f32) for m in masks]
    stack = gdata(st, data, full, lflags, scaling.scales, scaling.zeros, nimages, npts)
    result = np.zeros(npts, dtype=f32)
    r, sc, z = st.reject, scaling.scales, scaling.zeros
    if r in ("ccdclip", "crreject"):
        (mccdclip if st.mclip else accdclip)(st, stack, sc, z, nm, nimages, npts, result)
    elif r == "minmax":
        mm(st, stack, npts)
    elif r == "pclip":
        pclip(st, stack, nimages, npts, result)
    elif r == "sigclip":
        (msigclip if st.mclip else asigclip)(st, stack, sc, z, nimages, npts, result)
    elif r == "avsigclip":
        (mavsigclip if st.mclip else aavsigclip)(st, stack, sc, z, nimages, npts, result)
    if st.grow > 0:
        grow(st, stack, nimages, npts, result)
    if st.docombine:
        if st.combine == "average":
            result = average(st, stack, scaling.wts, npts)
        elif st.combine == "median":
            result = median(st, stack, npts)
        else:
            result = total(st, stack, npts)
    return result, stack.n.copy(), scaling
