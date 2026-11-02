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
        image = _image_model(name)
        if image is not None:
            self._star_image(image, real(radius), real(ar), real(real(pa) / RADIAN))
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

    def _star_image(self, coeff, radius, ar, pa):
        """mkt_star with an image for the PSF: the same grids of centres, through mkt_msi."""
        nym, nxm = coeff.shape[0] - 1, coeff.shape[1] - 1
        seeing = real(radius * real(0.8))
        dxc, dyc = real(real(1.0) / real(self.nxc)), real(real(1.0) / real(self.nyc))

        def grid(n, start):
            out = []
            yc = real(real(start) + real(dyc / real(2.0)))
            while yc < start + 1.0:
                xc = real(real(start) + real(dxc / real(2.0)))
                while xc < start + 1.0:
                    out.append(self.msi(n, n, xc, yc, real(1.0), coeff, nxm, nym, radius, ar,
                                        pa, self.nxsub))
                    xc = real(xc + dxc)
                yc = real(yc + dyc)
            return np.array(out, dtype=real)

        self.see_data = grid(2 * nint(seeing) + 1, -0.5)
        self.star_data = grid(2 * nint(radius) + 1, 0.5)
        self.star_flux = [real(1.0)] * self.star_data.shape[0]

    def msi(self, nx, ny, xc, yc, flux, coeff, nxm, nym, radius, ar, pa, nsub):
        """mkt_msi with no seeing."""
        data = np.zeros((ny, nx), dtype=real)
        if radius < 0.1:
            data[ny // 2, nx // 2] = flux
            return data
        terms = _msi_terms(nxm, nym, radius, ar, pa)
        xc1 = real(real(xc - real(int(real(xc + real(0.5))))) + real(nx // 2))
        yc1 = real(real(yc - real(int(real(yc + real(0.5))))) + real(ny // 2))
        total = _msi_plain(coeff, data, nx, ny, xc1, yc1, nxm, nym, *terms, nsub, pa == 0.0)
        return (data * real(flux / total)).astype(real)

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
        return self._star_from_file(name, radius)

    def _star_from_file(self, name, radius):
        """mkt_star's text file: a profile, or with a first value of 0 its integral, of as
        many points as it has, out to one radius."""
        values = _profile_values(name)
        j = len(values) - 1
        nxm = max(NPROF, 1 + nint(real(real(radius * real(self.nxsub)) * real(self.nysub))))
        prof, total = _resampled(values, nxm, j, True)
        limit = real(real(0.9) * total)
        dr = real(real(1.0) / real(nxm - 1))
        flux = real(0.0)
        i = 1
        while i < nxm and flux < limit:
            flux = real(flux + real(real(real(i) * dr) * prof[i]))
            i += 1
        seeing = real(real(real(i) - real(1.0)) * dr)
        return prof, real(1.0), real(1.0), seeing, nxm

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
        if isinstance(found, np.ndarray):
            return self._gobject_image(found, x, y, z, r, ar, pa)
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

    def _gobject_image(self, coeff, x, y, z, r, ar, pa):
        """mkt_gobject with an image for the object: its radius the half width of the image."""
        nym, nxm = coeff.shape[0] - 1, coeff.shape[1] - 1
        radius = real(r)
        nx = max(3, 2 * nint(radius) + 1)
        ny = nx
        ntemp, nys, nxs = self.see_data.shape
        nx += nxs // 2
        ny += nys // 2
        if nxs * nys <= 1:
            return self.msi(nx, ny, x, y, z, coeff, nxm, nym, radius, ar, pa, self.nxgsub)
        data = np.zeros((ny, nx), dtype=real)
        total = real(0.0)
        if radius > 0.01:
            terms = _msi_terms(nxm, nym, radius, ar, pa)
            xc1 = real(real(x - real(int(real(x + real(0.5))))) + real(nx // 2))
            yc1 = real(real(y - real(int(real(y + real(0.5))))) + real(ny // 2))
            _msi_seeing(coeff, data, nx, ny, xc1, yc1, nxm, nym, *terms, self.nxgsub,
                        pa == 0.0, self.see_data, self.nxc, self.nyc)
            total = real(np.add.accumulate(data.ravel())[-1])
        if total != 0.0:
            return (data * real(z / total)).astype(real)
        star = self.gstar(x, y, z)
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


def msi_coefficients(image) -> np.ndarray:
    """msifit for II_BILINEAR: the image, each line one further along by its slope, and one
    more line, again by the slope."""
    image = np.asarray(image, real)
    nym, nxm = image.shape
    coeff = np.zeros((nym + 1, nxm + 1), real)
    coeff[:nym, :nxm] = image
    coeff[:nym, nxm] = (real(2.0) * coeff[:nym, nxm - 1] - coeff[:nym, nxm - 2]).astype(real)
    coeff[nym] = ((coeff[nym - 1] * real(2.0)).astype(real)
                  + (coeff[nym - 2] * real(-1.0)).astype(real)).astype(real)
    return coeff


@jit
def _nearest_integral(coeff, a, b):
    """ii_1dinteg, II_NEAREST; 0-based array of IRAF's 1-based coefficients."""
    xa, xb = (b, a) if a > b else (a, b)
    half = real(0.5)
    neara, nearb = int(real(xa + half)), int(real(xb + half))
    deltaa = real(xa - real(neara))
    if nearb == neara:
        return real(real(real(xb - real(nearb)) - deltaa) * coeff[neara - 1])
    accum = real(real(half - deltaa) * coeff[neara - 1])
    for j in range(neara + 1, nearb):
        accum = real(accum + coeff[j - 1])
    deltab = real(xb - real(nearb))
    return real(accum + real(real(deltab + half) * coeff[nearb - 1]))


@jit
def _linear_integral(coeff, a, b):
    """ii_1dinteg, II_LINEAR."""
    xa, xb = (b, a) if a > b else (a, b)
    half = real(0.5)
    one = real(1.0)
    neara, nearb = int(xa), int(xb)
    deltaa = real(xa - real(neara))
    if nearb == neara:
        deltab = real(xb - real(nearb))
        first = real(real(deltab - deltaa) * coeff[neara - 1])
        slope = real(coeff[neara] - coeff[neara - 1])
        return real(first + real(real(half * slope)
                                 * real(real(deltab * deltab) - real(deltaa * deltaa))))
    slope = real(coeff[neara] - coeff[neara - 1])
    accum = real(real(real(one - deltaa) * coeff[neara - 1])
                 + real(real(half * slope) * real(one - real(deltaa * deltaa))))
    for j in range(neara + 1, nearb):
        accum = real(accum + real(half * real(coeff[j] + coeff[j - 1])))
    deltab = real(xb - real(nearb))
    if nearb < coeff.size:
        slope = real(coeff[nearb] - coeff[nearb - 1])
        accum = real(real(accum + real(coeff[nearb - 1] * deltab))
                     + real(real(real(half * slope) * deltab) * deltab))
    return accum


@jit
def msisqgrl(coeff, x1, x2, y1, y2):
    """msisqgrl: the bilinear surface over a rectangle."""
    half = real(0.5)
    nycoeff = coeff.shape[0]
    nylmin = max(1, min(nycoeff, int(real(y1 + half))))
    nylmax = min(nycoeff, max(1, int(real(y2 + half))))
    across = np.zeros(nycoeff, real)
    for i in range(nylmin, nylmax + 1):
        across[i - 1] = _linear_integral(coeff[i - 1], x1, x2)
    return _nearest_integral(across, y1, y2)


@jit
def _pyclip(xver, yver, xintr, nver, lx, ld):
    """ii_pyclip: where a line of the image crosses the polygon."""
    nintr = 0
    u1 = real(real(-lx * yver[0]) + ld)
    for i in range(2, nver + 1):
        u2 = real(real(-lx * yver[i - 1]) + ld)
        if real(u1 * u2) <= 0.0:
            if u1 != 0.0 and u2 != 0.0:
                dy = real(yver[i - 2] - yver[i - 1])
                dx = real(xver[i - 2] - xver[i - 1])
                dd = real(real(xver[i - 2] * yver[i - 1]) - real(yver[i - 2] * xver[i - 1]))
                xa = real(real(dx * ld) - real(lx * dd))
                xintr[nintr] = real(xa / real(dy * lx))
                nintr += 1
            elif u1 == 0.0 and u2 == 0.0:
                xintr[nintr] = xver[i - 2]
                xintr[nintr + 1] = xver[i - 1]
                nintr += 2
            elif u1 != 0.0:
                if i == nver:
                    dy = real(yver[1] - yver[nver - 1])
                    dd = real(dy * real(yver[nver - 2] - yver[nver - 1]))
                else:
                    dy = real(yver[i] - yver[i - 1])
                    dd = real(dy * real(yver[i - 2] - yver[i - 1]))
                if dy != 0.0:
                    xintr[nintr] = xver[i - 1]
                    nintr += 1
                if dd > 0.0:
                    xintr[nintr] = xver[i - 1]
                    nintr += 1
        u1 = u2
    return nintr


@jit
def msigrl(coeff, x, y, npts):
    """msigrl: the bilinear surface over a polygon, line by line."""
    half = real(0.5)
    nycoeff = coeff.shape[0]
    xmin, xmax = np.min(x[:npts]), np.max(x[:npts])
    ymin, ymax = np.min(y[:npts]), np.max(y[:npts])
    nylmin = max(1, min(int(real(ymin + half)), nycoeff))
    nylmax = min(nycoeff, max(1, int(real(ymax + half))))
    lx = real(xmax - xmin)
    across = np.zeros(nycoeff, real)
    xintr = np.zeros(2 * npts + 2, real)
    for i in range(nylmin, nylmax + 1):
        if ymin > i:
            ld = real(min(real(real(i) + half), ymax) * lx)
        elif ymax < i:
            ld = real(max(real(real(i) - half), ymin) * lx)
        else:
            ld = real(real(i) * lx)
        ninter = _pyclip(x, y, xintr, npts, lx, ld)
        if ninter <= 0:
            low, high = xmin, xmin
        else:
            low = min(xintr[0], xintr[1])
            high = max(xintr[0], xintr[1])
        across[i - 1] = _linear_integral(coeff[i - 1], low, high)
    return _nearest_integral(across, ymin, ymax)


@jit
def _msi_sample(coeff, nxm, nym, ac, as_, bc, bs, a, b, xcm, ycm, x1, y1, dsub, xc1, yc1,
                pa_zero):
    """One subpixel of mkt_msi: whether it lands on the model, as its loops go on to the next
    otherwise, and the surface over it."""
    xm = np.zeros(5, real)
    ym = np.zeros(5, real)
    one = real(1.0)
    asdy1 = real(y1 - yc1)
    asdy2 = real(asdy1 + dsub)
    bcdy1 = real(real(bc * asdy1) + ycm)
    bcdy2 = real(real(bc * asdy2) + ycm)
    val3 = val4 = one
    if pa_zero:
        val3 = max(one, bcdy1)
        if val3 >= nym:
            return False, real(0.0)
        val4 = min(b, bcdy2)
        if val3 >= val4:
            return False, real(0.0)
    asdy1 = real(as_ * asdy1)
    asdy2 = real(as_ * asdy2)
    bsdx1 = real(x1 - xc1)
    bsdx2 = real(bsdx1 + dsub)
    acdx1 = real(real(ac * bsdx1) + xcm)
    acdx2 = real(real(ac * bsdx2) + xcm)
    if pa_zero:
        val1 = max(one, acdx1)
        if val1 >= nxm:
            return False, real(0.0)
        val2 = min(a, acdx2)
        if val1 >= val2:
            return False, real(0.0)
        return True, msisqgrl(coeff, val1, val2, val3, val4)
    bsdx1 = real(bs * bsdx1)
    bsdx2 = real(bs * bsdx2)
    v1, v2, v3, v4 = (real(acdx1 + asdy1), real(acdx2 + asdy1), real(acdx2 + asdy2),
                      real(acdx1 + asdy2))
    if min(v1, v2, v3, v4) >= a or max(v1, v2, v3, v4) <= one:
        return False, real(0.0)
    xm[0], xm[1], xm[2], xm[3] = (max(one, min(a, v1)), max(one, min(a, v2)),
                                  max(one, min(a, v3)), max(one, min(a, v4)))
    xm[4] = xm[0]
    v1, v2, v3, v4 = (real(bcdy1 - bsdx1), real(bcdy1 - bsdx2), real(bcdy2 - bsdx2),
                      real(bcdy2 - bsdx1))
    if min(v1, v2, v3, v4) >= b or max(v1, v2, v3, v4) <= one:
        return False, real(0.0)
    ym[0], ym[1], ym[2], ym[3] = (max(one, min(b, v1)), max(one, min(b, v2)),
                                  max(one, min(b, v3)), max(one, min(b, v4)))
    ym[4] = ym[0]
    y2 = 0
    for k in range(1, 4):
        if ym[k] < ym[y2]:
            y2 = k
    if y2 == 1:
        xm[0], ym[0] = xm[1], ym[1]
        xm[1], ym[1] = xm[2], ym[2]
        xm[2], ym[2] = xm[3], ym[3]
        xm[3], ym[3] = xm[4], ym[4]
        xm[4], ym[4] = xm[0], ym[0]
    elif y2 == 2:
        xm[0], ym[0] = xm[2], ym[2]
        xm[2], ym[2] = xm[4], ym[4]
        xm[4], ym[4] = xm[1], ym[1]
        xm[1], ym[1] = xm[3], ym[3]
        xm[3], ym[3] = xm[4], ym[4]
        xm[4], ym[4] = xm[0], ym[0]
    elif y2 == 3:
        xm[4], ym[4] = xm[3], ym[3]
        xm[3], ym[3] = xm[2], ym[2]
        xm[2], ym[2] = xm[1], ym[1]
        xm[1], ym[1] = xm[0], ym[0]
        xm[0], ym[0] = xm[4], ym[4]
    return True, msigrl(coeff, xm, ym, 5)


def _msi_terms(nxm, nym, radius, ar, pa):
    half = real(2.0)
    a = real(real(real(nxm) / half) / radius)
    b = real(real(real(nym) / half) / real(ar * radius))
    c = real(math.cos(np.float64(pa)))
    s = real(math.sin(np.float64(pa)))
    return (real(a * c), real(a * s), real(b * c), real(b * s), real(nxm), real(nym),
            real(real(nxm // 2) + real(1.0)), real(real(nym // 2) + real(1.0)))


@jit
def _msi_plain(coeff, data, nx, ny, xc1, yc1, nxm, nym, ac, as_, bc, bs, a, b, xcm, ycm,
               nxsub, pa_zero):
    total = real(0.0)
    half = real(0.5)
    for y in range(ny):
        ry = real(real(y) - yc1)
        c = real(ry * ry)
        for x in range(nx):
            rx = real(real(x) - xc1)
            i = int(np.sqrt(np.float64(real(real(rx * rx) + c))))
            dsub = real(real(1.0) / real(max(1, nxsub - i)))
            part = real(dsub / real(2.0))
            value = real(0.0)
            y1 = real(real(y) - half)
            top = real(real(real(y) + half) - part)
            while y1 < top:
                x1 = real(real(x) - half)
                right = real(real(real(x) + half) - part)
                while x1 < right:
                    ok, s = _msi_sample(coeff, nxm, nym, ac, as_, bc, bs, a, b, xcm, ycm, x1,
                                        y1, dsub, xc1, yc1, pa_zero)
                    if ok:
                        value = real(value + s)
                    x1 = real(x1 + dsub)
                y1 = real(y1 + dsub)
            total = real(total + value)
            data[y, x] = value
    return total


@jit
def _msi_seeing(coeff, data, nx, ny, xc1, yc1, nxm, nym, ac, as_, bc, bs, a, b, xcm, ycm,
                nxsub, pa_zero, see, nxc, nyc):
    ntemp, nys, nxs = see.shape
    nxs2, nys2 = nxs // 2, nys // 2
    half = real(0.5)
    for y in range(ny):
        ry = real(real(y) - yc1)
        c = real(ry * ry)
        ys1 = max(0, nys2 - y)
        ys2 = min(nys - 1, ny - y + nys2 - 1)
        for x in range(nx):
            rx = real(real(x) - xc1)
            i = int(np.sqrt(np.float64(real(real(rx * rx) + c))))
            dsub = real(real(1.0) / real(max(1, nxsub - i)))
            part = real(dsub / real(2.0))
            xs1 = max(0, nxs2 - x)
            xs2 = min(nxs - 1, nx - x + nxs2 - 1)
            y1 = real(real(y) - half)
            top = real(real(real(y) + half) - part)
            while y1 < top:
                x1 = real(real(x) - half)
                right = real(real(real(x) + half) - part)
                while x1 < right:
                    ok, s = _msi_sample(coeff, nxm, nym, ac, as_, bc, bs, a, b, xcm, ycm, x1,
                                        y1, dsub, xc1, yc1, pa_zero)
                    if ok:
                        w = real(x1 + half)
                        col = int(real(real(w - real(int(w))) * real(nxc)))
                        w = real(y1 + half)
                        row = int(real(real(w - real(int(w))) * real(nyc)))
                        k = row * nxc + col
                        for y2 in range(ys1, ys2 + 1):
                            for x2 in range(xs1, xs2 + 1):
                                ly, lx = y - nys2 + y2, x - nxs2 + x2
                                data[ly, lx] = real(data[ly, lx] + real(s * see[k, y2, x2]))
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


def _image_model(name) -> np.ndarray | None:
    """The coefficients of an image used as a profile; None for a name that is no image."""
    from pathlib import Path

    path = Path(str(name).split("[")[0])
    if not path.is_file() or path.suffix.lower() not in (".fits", ".fit", ".fts", ".imh"):
        return None
    from . import images

    data, _ = images.read(name)
    if data.ndim != 2 or min(data.shape) < 2:
        return None
    return msi_coefficients(data)


def _profile_values(name) -> np.ndarray:
    """A template's text file, the first number of each line."""
    from pathlib import Path

    from . import advice, iraffmt

    path = Path(name)
    if not path.is_file():
        advice.fail(f"PSF template not found ({name})",
                    "gaussian, moffat, or a text file with the profile, a value a line")
    found = []
    for line in path.read_text(errors="replace", encoding="utf-8").splitlines():
        words = line.split()
        value = iraffmt.ctod(words[0]) if words else None
        if value is not None:
            found.append(real(value))
    if not found:
        advice.fail("PSF template not found", f"{name} has no numbers")
    return np.array(found, real)


def _resampled(values, nxm: int, j: int, sum_flux: bool):
    """mkt_star and mkt_object: the profile at nxm radii, through a spline. With a first
    value of 0 the file is the integral, and the profile its derivative over r. The total
    of r times it, for what uses it."""
    from . import asi

    fit = asi.Interpolant("spline3", values[:j])
    # The last radius is one past the points IRAF fits, into the coefficient after them,
    # which is the end of its array: a zero in what it leaves there.
    fit.coeff = np.concatenate([fit.coeff, np.zeros(4, real)])
    dr = real(real(1.0) / real(nxm - 1))
    prof = np.zeros(nxm, real)
    total = real(0.0)
    if values[0] == 0.0:
        for i in range(1, nxm):
            r = real(real(i) * dr)
            at = real(real(1.0) + real(real(j) * r))
            prof[i] = max(real(0.0), real(real(fit.derivative(float(at))) / r))
            total = real(total + real(r * prof[i]))
        prof[0] = max(real(0.0), real(real(real(2.0) * prof[1]) - prof[2]))
    else:
        for i in range(nxm):
            r = real(real(i) * dr)
            at = real(real(1.0) + real(real(j) * r))
            prof[i] = real(fit(float(at)))
            total = real(total + real(r * prof[i]))
    return prof, total


def object_profile(name: str, dynrange):
    """mkt_object: (profile, flux, radius scale), of a formula or a text file, or None."""
    from pathlib import Path

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
    elif _image_model(name) is not None:
        return _image_model(name)
    elif Path(name).is_file():
        values = _profile_values(name)
        prof, _ = _resampled(values, NPROF, len(values) - 1, False)
        return prof, real(1.0), real(1.0)
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
