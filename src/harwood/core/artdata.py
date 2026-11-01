# ruff: noqa: E741
# A line-by-line port keeps IRAF's names (l, k, nx).
"""artdata: noise and objects made to order, with IRAF's generator and templates.

The numbers come out the same as IRAF's for a seed because everything is its own:
urand, a congruential generator with m = 2**30 multiplying in 32 bits; gasdev and
poidev in their mixture of real variables and double intermediates; and the star
templates, precomputed on a grid of centres, each scaled in place to the flux last
asked of it, so that the order objects come in is in the last bits.

From iraf-community 2.18: noao/artdata/t_mknoise.x, t_mkobjects.x, mktemplates.x,
pkg/xtools/numrecipes.x, sys/etc/urand.x, sys/etc/qsort.x. See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math

import numpy as np

from .jit import jit

real = np.float32
# float() of a float32 stays a float32 under numba, and then math.log is logf.
PI = real(3.1415926535897932385)
RADIAN = real(57.295779513082320877)
NPROF = 5001
NY = 11

# urand's constants for m = 2**30: MAX_LONG / 2 + 1, with long of 32 bits.
_M = 1073741824
_A = 8 * int(_M * math.atan(1.0 / 8.0)) + 5
_C = 2 * int(_M * (0.5 - math.sqrt(3.0) / 6.0)) + 1
_MIC = (_M - _C) + _M
_SCALE = real(0.5 / _M)


def new_state(seed: int):
    """urand's seed, with its first call's max (1, seed), and gasdev's memory."""
    return np.array([max(1, seed), 0], dtype=np.int64), np.zeros(2, dtype=np.float32)


@jit
def urand(state):
    n = (state[0] * _A) & 0xFFFFFFFF
    if n >= 0x80000000:
        n -= 0x100000000
    if n > _MIC:
        n = (n - _M) - _M
    n += _C
    if n // 2 > _M:
        n = (n - _M) - _M
    if n < 0:
        n = (n + _M) + _M
    state[0] = n
    return real(real(n) * _SCALE)


@jit
def gasdev(state, memory):
    """Two values from each pair of uniform numbers, in turn."""
    if state[1] == 0:
        memory[0] = real(real(1.0) - urand(state))
        memory[1] = urand(state)
        state[1] = 1
        return real(
            math.sqrt(-2.0 * math.log(np.float64(memory[0])))
            * math.cos(np.float64(real(real(2.0 * PI) * memory[1])))
        )
    state[1] = 0
    return real(
        math.sqrt(-2.0 * math.log(np.float64(memory[0])))
        * math.sin(np.float64(real(real(2.0 * PI) * memory[1])))
    )


@jit
def gammln(xx):
    """pkg/xtools/gammln.c is lgammaf."""
    return real(math.lgamma(np.float64(xx)))


@jit
def poidev(xm, state):
    if xm < 0.0:
        return real(0.0)
    if xm < 30.0:
        x = real(0.0)
        p = real(1.0)
        e = real(math.exp(-np.float64(xm)))
        while p > e:
            r = urand(state)
            p = real(p * r)
            x = real(x + real(1.0))
        return real(x - real(1.0))
    c = real(real(0.767) - real(real(3.36) / xm))
    beta = real(np.float64(PI) / np.sqrt(np.float64(real(real(3.0) * xm))))
    alpha = real(beta * xm)
    k = real(math.log(np.float64(c)) - np.float64(xm) - math.log(np.float64(beta)))
    lhs = real(0.0)
    rhs = real(-1.0)
    while True:
        u = urand(state)
        x = real(
            (np.float64(alpha) - math.log(np.float64(real(real(real(1.0) - u) / u))))
            / np.float64(beta)
        )
        n = real(int(real(x + real(0.5))))
        if n >= 0.0:
            v = urand(state)
            y = real(alpha - real(beta * x))
            d = 1.0 + math.exp(np.float64(y))
            lhs = real(np.float64(y) + math.log(np.float64(v) / (d * d)))
            rhs = real(
                np.float64(k)
                + np.float64(n) * math.log(np.float64(xm))
                - np.float64(gammln(real(n + real(1.0))))
            )
        if lhs <= rhs:
            break
    return n


@jit
def mkrnoise(data, rdnoise, state, memory):
    """mkrnoise with no buffer: a new value for each pixel."""
    for i in range(data.size):
        data[i] = real(data[i] + real(rdnoise * gasdev(state, memory)))


@jit
def mkpnoise(source, data, b, g, state, memory):
    """mkpnoise with no buffer."""
    for i in range(data.size):
        v = real(g * real(source[i] + b))
        data[i] = real(data[i] + real(real(poidev(v, state) - v) / g))


def qsort(x: list[int], key) -> list[int]:
    """sys/etc/qsort.x, of indices by key: equal keys land where it puts them, and the
    order objects are added in is in the last bits of the sum."""
    x = [0, *x]
    nelem = len(x) - 1
    if nelem < 1:
        return []
    lv, uv = [0, 1], [0, nelem]
    p = 1

    def compare(a, b):
        return -1 if key[a] < key[b] else (1 if key[a] > key[b] else 0)

    while p > 0:
        if lv[p] >= uv[p]:
            p -= 1
            continue
        i, j = lv[p] - 1, uv[p]
        k = (lv[p] + uv[p]) // 2
        x[j], x[k] = x[k], x[j]
        pivot = x[j]
        while i < j:
            i += 1
            while compare(x[i], pivot) < 0:
                i += 1
            j -= 1
            while j > i:
                if compare(x[j], pivot) <= 0:
                    break
                j -= 1
            if i < j:
                x[i], x[j] = x[j], x[i]
        j = uv[p]
        x[i], x[j] = x[j], x[i]
        while len(lv) <= p + 1:
            lv.append(0)
            uv.append(0)
        if i - lv[p] < uv[p] - i:
            lv[p + 1], uv[p + 1] = lv[p], i - 1
            lv[p] = i + 1
        else:
            lv[p + 1], uv[p + 1] = i + 1, uv[p]
            uv[p] = i - 1
        p += 1
    return x[1:]


def nint(x) -> int:
    v = np.float64(x)
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(0.5 - v))


def binprof(prof, nprof1, radius, nsub):
    """mkt_binprof: the profile, averaged over a subpixel near the centre and
    more crudely further out."""
    nprof = prof.size
    prof1 = np.zeros(nprof1, dtype=real)
    if radius < 0.1:
        prof1[:] = 1.0
        return prof1
    _binprof(prof, prof1, nprof, nprof1, real(radius), nsub)
    return prof1


@jit
def _binprof(prof, prof1, nprof, nprof1, radius, nsub):
    scale = real(real(real(nprof) - real(1.0)) / real(real(nprof1) - real(1.0)))
    v = real(real(real(real(nprof1) - real(1.0)) / real(nsub)) / radius) / real(2.0)
    v = real(v)
    dx = int(np.floor(np.float64(v) + 0.5)) if v >= 0 else -int(np.floor(0.5 - np.float64(v)))
    dy = real(real(dx) / real(real(NY) - real(1.0)))
    for i in range(-dx, 2 * dx + 1):
        k = abs(i)
        k1 = max(1, i - dx + 1)
        k2 = i + dx + 1
        for j in range(NY):
            if j == 0:
                val = real(k)
            elif k == 0:
                val = real(dy * real(j))
            else:
                q = real(real(dy * real(j)) / real(k))
                val = real(np.float64(k) * np.sqrt(np.float64(real(real(1.0) + real(q * q)))))
            w = np.float64(real(real(scale * val) + real(1.0)))
            l = int(np.floor(w + 0.5))
            if l > nprof:
                continue
            val = real(prof[l - 1] / real(NY))
            for m in range(k1, k2 + 1):
                prof1[m - 1] = real(prof1[m - 1] + val)
    for i in range(2 * dx + 1, nprof1):
        k1 = i - dx + 1
        k2 = min(nprof1, i + dx + 1)
        w = np.float64(real(real(scale * real(i)) + real(1.0)))
        val = prof[int(np.floor(w + 0.5)) - 1]
        for k in range(k1, k2 + 1):
            prof1[k - 1] = real(prof1[k - 1] + val)


def _shape(npts, radius, ar, pa):
    r = real(real(real(npts - 1) / radius) * real(real(npts - 1) / radius))
    t0 = real(real(npts - 1) / real(ar * radius))
    t = real(t0 * t0)
    c = real(math.cos(np.float64(pa)))
    s = real(math.sin(np.float64(pa)))
    a = real(real(real(r * c) * c) + real(real(t * s) * s))
    b = real(real(real(real(2.0) * real(r - t)) * c) * s)
    c2 = real(real(real(r * s) * s) + real(real(t * c) * c))
    return a, b, c2


@jit
def _profile_plain(data, nx, ny, xc1, yc1, prof, npts, a, b, c, nxsub):
    """mkt_profile with no seeing: each pixel subsampled, more finely near the centre."""
    sum1 = real(0.0)
    for y in range(ny):
        ry = real(real(y) - yc1)
        t = real(ry * ry)
        for x in range(nx):
            rx = real(real(x) - xc1)
            i = int(np.sqrt(np.float64(real(real(rx * rx) + t))))
            dsub = real(real(1.0) / real(max(1, nxsub - i)))
            total = real(0.0)
            half = real(dsub / real(2.0))
            y1 = real(real(real(y) - real(0.5)) + half)
            top = real(real(y) + real(0.5))
            while y1 < top:
                dy = real(y1 - yc1)
                s = real(c * real(dy * dy))
                x1 = real(real(real(x) - real(0.5)) + half)
                right = real(real(x) + real(0.5))
                while x1 < right:
                    dx = real(x1 - xc1)
                    v = real(real(real(a * real(dx * dx)) + real(real(b * dx) * dy)) + s)
                    i = int(np.sqrt(np.float64(v)) + 1.5)
                    if i <= npts:
                        total = real(total + prof[i - 1])
                    x1 = real(x1 + dsub)
                y1 = real(y1 + dsub)
            total = real(total * real(dsub * dsub))
            sum1 = real(sum1 + total)
            data[y, x] = total
    return sum1


class Templates:
    """mkt_star and the rest: the seeing kernels, the star templates and the objects'
    profiles, for one run of a task."""

    def __init__(
        self,
        nxc=5,
        nyc=5,
        nxsub=10,
        nysub=10,
        nxgsub=5,
        nygsub=5,
        dynrange=100000.0,
        psfrange=10.0,
    ):
        self.nxc, self.nyc, self.nxsub, self.nysub = nxc, nyc, nxsub, nysub
        self.nxgsub, self.nygsub = nxgsub, nygsub
        self.dynrange, self.psfrange = real(dynrange), real(psfrange)
        self.star_data = None
        self.star_flux = None
        self.see_data = None
        self.objects = {}

    def star(self, name, radius, beta=2.5, ar=1.0, pa=0.0):
        """mkt_star. The templates are made once, for the first call's star."""
        if self.star_data is not None:
            return
        prof, flux, scale, seeing, nxm = self._star_profile(name, real(radius), real(beta))
        r = real(radius)
        ar = real(ar)
        pa = real(real(pa) / RADIAN)
        scaled = real(r * scale)
        seeing = real(r * seeing)
        nym = 1 + nint(real(real(scaled * real(self.nxsub)) * real(self.nysub)))
        prof1 = binprof(prof, nym, scaled, self.nxsub)
        dxc, dyc = real(real(1.0) / real(self.nxc)), real(real(1.0) / real(self.nyc))
        n = 2 * nint(seeing) + 1
        self.see_data = self._grid(n, -0.5, dxc, dyc, real(1.0), prof1, nym, scaled, ar, pa)
        n = 2 * nint(scaled) + 1
        self.star_data = self._grid(n, 0.5, dxc, dyc, flux, prof1, nym, scaled, ar, pa)
        self.star_flux = [real(1.0)] * self.star_data.shape[0]

    def _grid(self, n, start, dxc, dyc, flux, prof, npts, radius, ar, pa):
        out = []
        yc = real(real(start) + real(dyc / real(2.0)))
        while yc < start + 1.0:
            xc = real(real(start) + real(dxc / real(2.0)))
            while xc < start + 1.0:
                out.append(
                    self.profile(n, n, xc, yc, flux, prof, npts, radius, ar, pa, self.nxsub)
                )
                xc = real(xc + dxc)
            yc = real(yc + dyc)
        return np.array(out, dtype=real)

    def _star_profile(self, name, radius, beta):
        if name == "gaussian":
            r = real(np.float64(radius) / math.sqrt(math.log(np.float64(real(2.0)))))
            extent = real(math.sqrt(math.log(np.float64(self.dynrange))))
            seeing = real(math.sqrt(math.log(np.float64(self.psfrange))))
            nxm = max(
                NPROF,
                1 + nint(real(real(real(extent * r) * real(self.nxsub)) * real(self.nysub))),
            )
            dr = real(extent / real(nxm - 1))
            i = np.arange(nxm).astype(real)
            step = (i * dr).astype(real)
            prof = np.array([math.exp(-np.float64(real(v * v))) for v in step], dtype=real)
            flux = real(real(1.0) - prof[nxm - 1])
            r = real(math.sqrt(math.log(np.float64(real(2.0)))))
            return prof, flux, real(extent / r), real(seeing / r), nxm
        if name == "moffat":
            power = real(real(1.0) / beta)
            base = real(math.pow(2.0, np.float64(power)) - 1.0)
            r = real(np.float64(radius) / math.sqrt(np.float64(base)))
            extent = real(
                math.sqrt(math.pow(np.float64(self.dynrange), np.float64(power)) - 1.0)
            )
            seeing = real(
                math.sqrt(math.pow(np.float64(self.psfrange), np.float64(power)) - 1.0)
            )
            nxm = max(
                NPROF,
                1 + nint(real(real(real(extent * r) * real(self.nxsub)) * real(self.nysub))),
            )
            dr = real(extent / real(nxm - 1))
            prof = np.zeros(nxm, real)
            total = _moffat(prof, nxm, dr, beta)
            r = real(real(real(nxm - 1) * dr) + real(dr / real(2.0)))
            xc = real(real(real(2.0) * beta) - real(2.0))
            tail = np.float64(xc) * math.pow(np.float64(r), np.float64(xc))
            flux = real(np.float64(total) / (np.float64(total) + 1.0 / tail))
            r = real(math.sqrt(math.pow(2.0, np.float64(power)) - 1.0))
            return prof, flux, real(extent / r), real(seeing / r), nxm
        raise ValueError(f"star template '{name}': only gaussian and moffat are made here")

    def profile(self, nx, ny, xc, yc, flux, prof, npts, radius, ar, pa, nsub):
        """mkt_profile without the seeing."""
        data = np.zeros((ny, nx), dtype=real)
        if radius < 0.1:
            data[ny // 2, nx // 2] = flux
            return data
        a, b, c = _shape(npts, radius, ar, pa)
        xc1 = real(real(xc - real(int(real(xc + real(0.5))))) + real(nx // 2))
        yc1 = real(real(yc - real(int(real(yc + real(0.5))))) + real(ny // 2))
        total = _profile_plain(data, nx, ny, xc1, yc1, prof, npts, a, b, c, nsub)
        if total == 0.0:
            data[ny // 2, nx // 2] = flux
        else:
            data = (data * real(flux / total)).astype(real)
        return data

    def gobject(self, name, x, y, z, r, ar, pa):
        """mkt_gobject, for an object that is a formula: a raster, with the seeing if the
        star's template has more than a pixel. None when it is not known."""
        found = self.objects.get(name, False)
        if found is False:
            found = self.objects[name] = object_profile(name, self.dynrange)
        if found is None:
            return None
        prof, fraction, scale = found
        radius = real(r * scale)
        nx = max(3, 2 * nint(radius) + 1)
        ny = nx
        ntemp, nys, nxs = self.see_data.shape
        nx += nxs // 2
        ny += nys // 2
        nprof = 1 + nint(real(real(radius * real(self.nxgsub)) * real(self.nxgsub)))
        binned = binprof(prof, nprof, radius, self.nxgsub)
        flux = real(z * fraction)
        if nxs * nys <= 1:
            return self.profile(nx, ny, x, y, flux, binned, nprof, radius, ar, pa, self.nxgsub)
        data = np.zeros((ny, nx), dtype=real)
        total = real(0.0)
        if radius > 0.01:
            a, b, c = _shape(nprof, radius, ar, pa)
            xc1 = real(real(x - real(int(real(x + real(0.5))))) + real(nx // 2))
            yc1 = real(real(y - real(int(real(y + real(0.5))))) + real(ny // 2))
            _profile_seeing(
                data,
                nx,
                ny,
                xc1,
                yc1,
                binned,
                nprof,
                a,
                b,
                c,
                self.nxgsub,
                self.see_data,
                self.nxc,
                self.nyc,
            )
            total = real(np.add.accumulate(data.ravel())[-1])
        if total != 0.0:
            return (data * real(flux / total)).astype(real)
        star = self.gstar(x, y, flux)
        sy, sx = star.shape
        xs1, xs2 = max(0, sx // 2 - nx // 2), min(sx - 1, nx - nx // 2 + sx // 2 - 1)
        ys1, ys2 = max(0, sy // 2 - ny // 2), min(sy - 1, ny - ny // 2 + sy // 2 - 1)
        for y2 in range(ys1, ys2 + 1):
            for x2 in range(xs1, xs2 + 1):
                ly, lx = ny // 2 - sy // 2 + y2, nx // 2 - sx // 2 + x2
                data[ly, lx] = real(data[ly, lx] + star[y2, x2])
        return data

    def gstar(self, x, y, z):
        """mkt_gstar: the template with the nearest centre, scaled in place to z."""
        at_x, at_y = real(x + real(0.5)), real(y + real(0.5))
        i = int(real(real(at_x - real(int(at_x))) * real(self.nxc)))
        j = int(real(real(at_y - real(int(at_y))) * real(self.nyc)))
        k = j * self.nxc + i
        f = self.star_flux[k]
        if f != z:
            self.star_data[k] = (self.star_data[k] * real(z / f)).astype(real)
            self.star_flux[k] = z
        return self.star_data[k]


@jit
def _moffat(prof, nxm, dr, beta):
    flux = real(0.0)
    for i in range(nxm):
        r = real(real(i) * dr)
        prof[i] = real(
            1.0 / math.pow(np.float64(real(real(1.0) + real(r * r))), np.float64(beta))
        )
        flux = real(flux + real(r * prof[i]))
    return flux


@jit
def _profile_seeing(data, nx, ny, xc1, yc1, prof, npts, a, b, c, nxsub, see, nxc, nyc):
    """mkt_profile with seeing: each sample of the profile gets the seeing template of
    its centre, as far as it lands on the raster."""
    ntemp, nys, nxs = see.shape
    nxs2, nys2 = nxs // 2, nys // 2
    for y in range(ny):
        ry = real(real(y) - yc1)
        t = real(ry * ry)
        ys1 = max(0, nys2 - y)
        ys2 = min(nys - 1, ny - y + nys2 - 1)
        for x in range(nx):
            rx = real(real(x) - xc1)
            i = int(np.sqrt(np.float64(real(real(rx * rx) + t))))
            dsub = real(real(1.0) / real(max(1, nxsub - i)))
            dsub2 = real(dsub * dsub)
            xs1 = max(0, nxs2 - x)
            xs2 = min(nxs - 1, nx - x + nxs2 - 1)
            half = real(dsub / real(2.0))
            y1 = real(real(real(y) - real(0.5)) + half)
            top = real(real(y) + real(0.5))
            while y1 < top:
                dy = real(y1 - yc1)
                s = real(c * real(dy * dy))
                x1 = real(real(real(x) - real(0.5)) + half)
                right = real(real(x) + real(0.5))
                while x1 < right:
                    dx = real(x1 - xc1)
                    v = real(real(real(a * real(dx * dx)) + real(real(b * dx) * dy)) + s)
                    i = int(np.sqrt(np.float64(v)) + 1.5)
                    if i <= npts:
                        z = real(prof[i - 1] * dsub2)
                        w = real(x1 + real(0.5))
                        col = int(real(real(w - real(int(w))) * real(nxc)))
                        w = real(y1 + real(0.5))
                        row = int(real(real(w - real(int(w))) * real(nyc)))
                        k = row * nxc + col
                        for y2 in range(ys1, ys2 + 1):
                            for x2 in range(xs1, xs2 + 1):
                                ly, lx = y - nys2 + y2, x - nxs2 + x2
                                data[ly, lx] = real(data[ly, lx] + real(z * see[k, y2, x2]))
                    x1 = real(x1 + dsub)
                y1 = real(y1 + dsub)


def uigamma(n: int, x) -> float:
    """The upper incomplete gamma function of an integer, over its factorial."""
    numerator = math.exp(-np.float64(x))
    denominator = 1.0
    total = numerator / denominator
    for i in range(1, n):
        numerator *= np.float64(x)
        denominator *= i
        total += numerator / denominator
    return total


def object_profile(name: str, dynrange):
    """mkt_object for what is a formula: (profile, flux, radius scale), or None."""
    if name.startswith("sersic"):
        try:
            s = real(name[6:])
        except ValueError:
            return None
        n = nint(real(real(2.0) * real(s + real(0.01))))
        s = real(real(n) / real(2.0))
        if n < 1 or n > 20:
            return None
    elif name == "expdisk":
        s, n = real(1.0), 2
    elif name == "devauc":
        s, n = real(4.0), 8
    else:
        return None
    radius = real(math.pow(math.log(np.float64(dynrange)), np.float64(s)))
    dr = real(radius / real(NPROF - 1))
    power = real(real(1.0) / s)
    step = (np.arange(NPROF).astype(real) * dr).astype(real)
    r = np.array([math.pow(np.float64(v), np.float64(power)) for v in step], dtype=real)
    prof = np.array([math.exp(-np.float64(v)) for v in r], dtype=real)
    flux = real(1.0 - uigamma(n, r[-1]))
    q = real(
        real(
            real(real(n) - real(real(1.0) / real(3.0)))
            + real(real(4.0) / real(real(405.0) * s))
        )
        + real(real(46.0) / real(real(real(25515.0) * s) * s))
    )
    return prof, flux, real(np.float64(radius) / math.pow(np.float64(q), np.float64(s)))


def add_template(buffer, template, c1, l1, noise=None):
    """The object's raster into the image, what is on it, line by line."""
    nl, nc = buffer.shape
    ny, nx = template.shape
    c3, c4 = max(1, c1), min(nc, c1 + nx - 1)
    l3, l4 = max(1, l1), min(nl, l1 + ny - 1)
    for l in range(l3, l4 + 1):
        row = template[l - l1, c3 - c1 : c4 - c1 + 1]
        target = buffer[l - 1, c3 - 1 : c4]
        target[:] = (target + row).astype(real)
        if noise is not None:
            noise(row, target)
