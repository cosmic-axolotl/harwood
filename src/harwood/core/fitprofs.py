"""fitprofs: gaussian, lorentzian and voigt profiles fitted by least squares to a region
of each spectrum, over a linear background, with errors from Monte-Carlo samples.

From noao/onedspec/t_fitprofs.x (fp_fit, fp_background, fp_title) and splot's
deblend.x (dofit, dorefit, model, derivs, dofit1, mr_solve, mr_eval, mr_invert),
voigt.x and spdeblend.x's gasdev, operation for operation in real as f2c compiles
them: an expression with exp, log or sqrt in it is worked in double and rounded once,
the rest in real from left to right. The linear system is hfti's (core/llsq.py).

IRAF's ways kept: each profile is the mean of three subsamples across its pixel; the
positions and widths are fitted relative to the first profile's, the background first
held and then fitted, the positions first fixed, then together, then each its own;
the Levenberg-Marquardt steps go on until chi-squared has fallen by no more than 1e-4
five times running; a gaussian is taken to be 0 beyond 5 of its 1/e half widths. The
Monte-Carlo errors are the 68th percentile of the differences from the fit of fits to
the model with gaussian noise (sigma0 and invgain) added, from urand seeded with 1 for
each spectrum -- through splot's gasdev, whose second loop tests u1 where it means u2,
and which keeps its pair and its place in it from one spectrum to the next.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math

import numpy as np

from . import llsq
from .jit import jit

f32 = np.float32
GAUSS, LORENTZ, VOIGT = 1, 2, 3
PTYPES = ("gaussian", "lorentzian", "voigt")
FIXED, SINGLE, INDEP = 1, 2, 3
FITTYPES = ("fixed", "single", "all")
BKG, POS, INT, GAU, LOR = 0, 1, 2, 3, 4
NSUB = 3
#: The chi-squared of the last dofit or dorefit, which splot's r shows.
LAST_CHISQ = np.float32(0)
EPSILONR = f32(1.192e-7)

_T = np.array([.314240376, .947788391, 1.59768264, 2.27950708, 3.02063703, 3.8897249],
              f32)
_C = np.array([1.01172805, -.75197147, 1.2557727e-2, 1.00220082e-2, -2.42068135e-4,
               5.00848061e-7], f32)
_S = np.array([1.393237, .231152406, -.155351466, 6.21836624e-3, 9.19082986e-5,
               -6.27525958e-7], f32)


@jit
def voigt(xarg, yarg) -> tuple[np.float32, np.float32]:
    """voigt: the real (Voigt) and imaginary parts of exp(-z^2) erfc(-iz), Humlicek's."""
    x, y = f32(xarg), f32(abs(f32(yarg)))
    wr = wi = f32(0)
    y1 = f32(y + f32(1.5))
    y2 = f32(y1 * y1)
    if y < f32(.85) and abs(x) > f32(f32(y * f32(18.1)) + f32(1.65)):
        if abs(x) < f32(12):
            wr = f32(math.exp(np.float64(-(x * x))))
        y3 = f32(y + f32(3))
        for i in range(6):
            t, c, s = _T[i], _C[i], _S[i]
            r = f32(x - t)
            r2 = f32(r * r)
            d = f32(f32(1) / f32(r2 + y2))
            d1, d2 = f32(y1 * d), f32(r * d)
            wr = f32(wr + f32(f32(y * f32(f32(c * f32(f32(r * d2) - f32(d1 * f32(1.5))))
                                          + f32(f32(s * y3) * d2)))
                              / f32(r2 + f32(2.25))))
            r = f32(x + t)
            r2 = f32(r * r)
            d = f32(f32(1) / f32(r2 + y2))
            d3, d4 = f32(y1 * d), f32(r * d)
            wr = f32(wr + f32(f32(y * f32(f32(c * f32(f32(r * d4) - f32(d3 * f32(1.5))))
                                          - f32(f32(s * y3) * d4)))
                              / f32(r2 + f32(2.25))))
            wi = f32(f32(wi + f32(c * f32(d2 + d4))) + f32(s * f32(d1 - d3)))
        return wr, wi
    for i in range(6):
        t, c, s = _T[i], _C[i], _S[i]
        r = f32(x - t)
        d = f32(f32(1) / f32(f32(r * r) + y2))
        d1, d2 = f32(y1 * d), f32(r * d)
        r = f32(x + t)
        d = f32(f32(1) / f32(f32(r * r) + y2))
        d3, d4 = f32(y1 * d), f32(r * d)
        wr = f32(f32(wr + f32(c * f32(d1 + d3))) - f32(s * f32(d2 - d4)))
        wi = f32(f32(wi + f32(c * f32(d2 + d4))) + f32(s * f32(d1 - d3)))
    return wr, wi


@jit
def model(x, dx, nsub: int, xp, yp, gp, lp, tp) -> np.float32:
    """model: the profiles at x, the mean of nsub subsamples across a pixel dx wide."""
    dx = f32(dx)
    delta = f32(dx / f32(nsub))
    x1 = f32(f32(x) - f32(f32(dx + delta) / f32(2)))
    y = f32(0)
    for _ in range(nsub):
        x1 = f32(x1 + delta)
        for i in range(len(tp)):
            if tp[i] == GAUSS:
                arg1 = f32(abs(f32(f32(x1 - xp[i]) / gp[i])) * f32(1.66511))
                if arg1 < f32(5):
                    e = math.exp(np.float64(-(arg1 * arg1)))
                    y = f32(np.float64(y) + np.float64(yp[i]) * e)
            elif tp[i] == LORENTZ:
                arg2 = f32(abs(f32(f32(x1 - xp[i]) / f32(lp[i] / f32(2)))))
                y = f32(y + f32(yp[i] / f32(f32(arg2 * arg2) + f32(1))))
            elif tp[i] == VOIGT:
                arg1 = f32(f32(f32(x1 - xp[i]) * f32(1.66511)) / gp[i])
                arg2 = f32(f32(lp[i] * f32(.832555)) / gp[i])
                v0, u = voigt(f32(0), arg2)
                v, u = voigt(arg1, arg2)
                y = f32(y + f32(f32(yp[i] * v) / v0))
    return f32(y / f32(nsub))


SQRTOFPI = f32(1.7724538509055160273)


@jit
def derivs(x, a: np.ndarray):
    """derivs: the model and its derivatives for the parameters a (0-based here;
    IRAF's a[k] is a[k-1]). A voigt's widths are held positive in a itself."""
    na = a.size
    dx = a[0]
    nsub = np.int64(a[1])
    delta = f32(dx / f32(nsub))
    dx1 = f32(delta * f32(.1))
    x1 = f32(f32(x) - f32(f32(dx + delta) / f32(2)))
    y = f32(0)
    dyda = np.zeros(na, f32)
    for _ in range(nsub):
        x1 = f32(x1 + delta)
        y = f32(f32(y + a[2]) + f32(a[3] * x1))
        dyda[2] = f32(dyda[2] + f32(1))
        dyda[3] = f32(dyda[3] + x1)
        for i in range(8, na, 5):
            kind = np.int64(a[i + 4])
            if kind == GAUSS:
                i0 = f32(a[4] * a[i])
                wg = f32(a[6] * a[i + 2])
                arg1 = f32(f32(f32(f32(x1 - a[5]) - a[i + 1]) * f32(1.66511)) / wg)
                if abs(arg1) < f32(5):
                    di = f32(math.exp(np.float64(-(arg1 * arg1))))
                    c = f32(f32(i0 * di) * arg1)
                    y = f32(y + f32(i0 * di))
                    dyda[4] = f32(dyda[4] + f32(a[i] * di))
                    dyda[5] = f32(dyda[5] + f32(c / wg))
                    dyda[6] = f32(dyda[6] + f32(f32(c * arg1) / a[6]))
                    dyda[i] = f32(dyda[i] + f32(a[4] * di))
                    dyda[i + 1] = f32(dyda[i + 1] + f32(c / wg))
                    dyda[i + 2] = f32(dyda[i + 2] + f32(f32(c * arg1) / a[i + 2]))
            elif kind == LORENTZ:
                i0 = f32(a[4] * a[i])
                wl = f32(f32(a[7] * a[i + 3]) / f32(2))
                arg2 = f32(f32(f32(x1 - a[5]) - a[i + 1]) / wl)
                di = f32(f32(1) / f32(f32(arg2 * arg2) + f32(1)))
                c = f32(f32(f32(f32(i0 * f32(2)) * di) * di) * arg2)
                y = f32(y + f32(i0 * di))
                dyda[4] = f32(dyda[4] + f32(a[i] * di))
                dyda[5] = f32(dyda[5] + f32(c / wl))
                dyda[7] = f32(dyda[7] + f32(f32(c * arg2) / a[7]))
                dyda[i] = f32(dyda[i] + f32(a[4] * di))
                dyda[i + 1] = f32(dyda[i + 1] + f32(c / wl))
                dyda[i + 3] = f32(dyda[i + 3] + f32(f32(c * arg2) / a[i + 3]))
            elif kind == VOIGT:
                a[6] = max(dx1, f32(abs(a[6])))
                a[7] = max(dx1, f32(abs(a[7])))
                a[i + 2] = max(f32(1e-6), f32(abs(a[i + 2])))
                a[i + 3] = max(f32(1e-6), f32(abs(a[i + 3])))
                i0 = f32(a[4] * a[i])
                wg = f32(a[6] * a[i + 2])
                wl = f32(a[7] * a[i + 3])
                arg1 = f32(f32(f32(f32(x1 - a[5]) - a[i + 1]) * f32(1.66511)) / wg)
                arg2 = f32(f32(wl * f32(.832555)) / wg)
                v0, u = voigt(f32(0), arg2)
                v, u = voigt(arg1, arg2)
                v = f32(v / v0)
                u = f32(u / v0)
                di = f32(f32(f32(1) - v) / f32(v0 * SQRTOFPI))
                c = f32(f32(i0 * f32(2)) * arg2)
                y = f32(y + f32(i0 * v))
                dpos = f32(f32(f32(c * f32(2)) * f32(f32(arg1 * v) - f32(arg2 * u))) / wl)
                dgau = f32(c * f32(di + f32(arg1 * f32(f32(f32(arg1 / arg2) * v)
                                                     - f32(u * f32(2))))))
                dlor = f32(c * f32(f32(arg1 * u) - di))
                dyda[4] = f32(dyda[4] + f32(a[i] * v))
                dyda[5] = f32(dyda[5] + dpos)
                dyda[6] = f32(dyda[6] + f32(dgau / a[6]))
                dyda[7] = f32(dyda[7] + f32(dlor / a[7]))
                dyda[i] = f32(dyda[i] + f32(a[4] * v))
                dyda[i + 1] = f32(dyda[i + 1] + dpos)
                dyda[i + 2] = f32(dyda[i + 2] + f32(dgau / a[i + 2]))
                dyda[i + 3] = f32(dyda[i + 3] + f32(dlor / a[i + 3]))
    y = f32(y / f32(nsub))
    for k in range(na):
        dyda[k] = f32(dyda[k] / f32(nsub))
    return y, dyda


@jit
def _mr_eval(xs, ys, ss, flags, params):
    """mr_eval: the curvature matrix, the gradient and chi-squared, each data point
    weighted by 1/sigma^2."""
    nfit = flags.size
    a = np.zeros((nfit, nfit), f32)
    delta = np.zeros(nfit, f32)
    chisq = f32(0)
    for n in range(xs.size):
        ymod, dydp = derivs(xs[n], params)
        sig2i = f32(f32(1) / f32(ss[n] * ss[n]))
        dy = f32(ys[n] - ymod)
        for j in range(nfit):
            dydpj = f32(dydp[flags[j]] * sig2i)
            delta[j] = f32(delta[j] + f32(dy * dydpj))
            for k in range(j + 1):
                a[j, k] = f32(a[j, k] + f32(dydpj * dydp[flags[k]]))
        chisq = f32(chisq + f32(f32(dy * dy) * sig2i))
    for j in range(1, nfit):
        for k in range(j):
            a[k, j] = a[j, k]
    return a, delta, chisq


class _Marquardt:
    """mr_solve and mr_eval: one Levenberg-Marquardt step a call, its state kept
    between calls as f2c keeps an SPP procedure's locals."""

    def __init__(self, x, y, s, flags):
        self.x, self.y, self.s, self.flags = x, y, s, flags
        self.new = None

    def _eval(self, params):
        return _mr_eval(self.x, self.y, self.s, self.flags, params)

    def step(self, params, mr, chisq):
        nfit = len(self.flags)
        if mr < 0:
            self.new = params.copy()
            self.a2, self.delta2, chisq = self._eval(self.new)
            mr = f32(.001)
        a1 = self.a2.copy()
        for i in range(nfit):
            a1[i, i] = f32(self.a2[i, i] * f32(mr + f32(1)))
        flat = a1.flatten(order="F")
        delta1 = self.delta2.copy()
        llsq.hfti(flat, nfit, nfit, nfit, delta1, nfit, 1, 1e-10)
        for i in range(nfit):
            self.new[self.flags[i]] = f32(params[self.flags[i]] + delta1[i])
        a1, delta1, chisq1 = self._eval(self.new)
        if chisq1 < chisq:
            mr = max(EPSILONR, f32(mr * f32(.1)))
            chisq = chisq1
            self.a2, self.delta2 = a1, delta1
            params[:] = self.new
        else:
            mr = f32(mr * f32(10))
        return mr, chisq


def _flags(fit, nlines: int) -> list[int]:
    """dofit1's parameters to fit, 0-based."""
    out = []
    if fit[BKG] == SINGLE:
        out += [2, 3]
    if fit[INT] == SINGLE:
        out.append(4)
    elif fit[INT] == INDEP:
        out += [5 * i + 3 for i in range(1, nlines + 1)]
    for which, single, base in ((POS, 5, 4), (GAU, 6, 5), (LOR, 7, 6)):
        if fit[which] == SINGLE:
            out.append(single)
        elif fit[which] == INDEP:
            out += [5 * i + base for i in range(1, nlines + 1)]
    return out


def dofit1(fit, x, y, s, a: np.ndarray, nlines: int) -> np.float32:
    """dofit1: Marquardt steps until chi-squared stops falling five times, and one more
    undamped."""
    solver = _Marquardt(x, y, s, np.array(_flags(fit, nlines), np.int64))
    mr = f32(-1)
    chisq = f32(0)
    calm = 0
    chi2 = f32(9.9e36)
    while True:
        mr, chisq = solver.step(a, mr, chisq)
        calm = 0 if f32(chi2 - chisq) > f32(1e-4) else calm + 1
        chi2 = chisq
        if calm == 5:
            break
    _, chisq = solver.step(a, f32(0), chisq)
    return chisq


def _pack(dx, nsub, y1, dy, xp, yp, gp, lp, tp) -> np.ndarray:
    """dofit's parameters, relative to the first profile's."""
    np_ = len(tp)
    a = np.zeros(8 + 5 * np_, f32)
    a[0], a[1], a[2], a[3] = f32(dx), f32(nsub), f32(y1), f32(dy)
    a[4], a[5], a[6], a[7] = yp[0], xp[0], gp[0], lp[0]
    for i in range(1, np_ + 1):
        b = 5 * i + 3
        a[b] = f32(yp[i - 1] / a[4])
        a[b + 1] = f32(xp[i - 1] - a[5])
        if tp[i - 1] in (GAUSS, VOIGT):
            if a[6] == 0:
                a[6] = gp[i - 1]
            a[b + 2] = f32(gp[i - 1] / a[6])
        if tp[i - 1] in (LORENTZ, VOIGT):
            if a[7] == 0:
                a[7] = lp[i - 1]
            a[b + 3] = f32(lp[i - 1] / a[7])
        a[b + 4] = f32(tp[i - 1])
    return a


def _unpack(a, xp, yp, gp, lp, tp):
    for i in range(1, len(tp) + 1):
        b = 5 * i + 3
        yp[i - 1] = f32(a[b] * a[4])
        xp[i - 1] = f32(a[b + 1] + a[5])
        if tp[i - 1] in (GAUSS, VOIGT):
            gp[i - 1] = f32(abs(f32(a[b + 2] * a[6])))
        if tp[i - 1] in (LORENTZ, VOIGT):
            lp[i - 1] = f32(abs(f32(a[b + 3] * a[7])))
    return a[2], a[3]


def dofit(fit, x, y, s, dx, nsub, y1, dy, xp, yp, gp, lp, tp):
    """dofit: the background held then fitted, the positions fixed, together, each its
    own, and then each width its own. xp, yp, gp, lp change in place; (y1, dy)."""
    global LAST_CHISQ
    a = _pack(dx, nsub, y1, dy, xp, yp, gp, lp, tp)
    np_ = len(tp)
    fit1 = [0, 0, fit[INT], 0, 0]
    for i in range(1, fit[BKG] + 1):
        fit1[BKG] = i
        fit1[GAU] = min(SINGLE, fit[GAU])
        fit1[LOR] = min(SINGLE, fit[LOR])
        for j in range(FIXED, fit[POS] + 1):
            fit1[POS] = j
            if np_ > 1 or j != INDEP:
                LAST_CHISQ = dofit1(fit1, x, y, s, a, np_)
        if np_ > 1 and (fit[GAU] == INDEP or fit[LOR] == INDEP):
            fit1[GAU], fit1[LOR] = fit[GAU], fit[LOR]
            LAST_CHISQ = dofit1(fit1, x, y, s, a, np_)
    return _unpack(a, xp, yp, gp, lp, tp)


def dorefit(fit, x, y, s, dx, nsub, y1, dy, xp, yp, gp, lp, tp):
    """dorefit: one dofit1 from a solution close to the answer."""
    global LAST_CHISQ
    a = _pack(dx, nsub, y1, dy, xp, yp, gp, lp, tp)
    LAST_CHISQ = dofit1(fit, x, y, s, a, len(tp))
    return _unpack(a, xp, yp, gp, lp, tp)


class Gasdev:
    """splot's gasdev (spdeblend.x) on urand: two deviates from each pair, the pair and
    which of them is next kept from call to call. Its second loop tests u1, so u2 is
    whatever urand gives first, negative or not."""

    def __init__(self):
        self.count = 0
        self.u1 = self.u2 = f32(0)

    def __call__(self, state) -> np.float32:
        from .artdata import urand

        if self.count == 0:
            while True:
                self.u1 = f32(f32(urand(state) * f32(2)) - f32(1))
                if self.u1 > 0:
                    break
            self.u2 = f32(f32(urand(state) * f32(2)) - f32(1))
            self.count = 1
            return f32(math.sqrt(math.log(float(self.u1)) * -2)
                       * math.cos(float(f32(self.u2 * f32(6.2831853071795862)))))
        self.count = 0
        return f32(math.sqrt(math.log(float(self.u1)) * -2)
                   * math.sin(float(f32(self.u2 * f32(6.2831853071795862)))))


def flux_of(kind: int, peak, gfwhm, lfwhm) -> np.float32:
    """A profile's flux from its peak and widths."""
    if kind == GAUSS:
        return f32(f32(peak * f32(1.064467)) * gfwhm)
    if kind == LORENTZ:
        return f32(f32(peak * f32(1.570795)) * lfwhm)
    v, _ = voigt(f32(0), f32(f32(lfwhm * f32(.832555)) / gfwhm))
    return f32(f32(f32(peak * f32(1.064467)) * gfwhm) / v)


# ------------------------------------------------------------- fp_fit

INDEFR = 1.6e38
_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?"


def _ctod_at(text: str, k: int):
    """ctod from character k (0-based): (the value, where it stopped), or (None, k)."""
    import re

    from . import iraffmt

    while k < len(text) and text[k] in " \t":
        k += 1
    match = re.compile(_NUMBER).match(text, k)
    if not match:
        return None, k
    return iraffmt.ctod(match.group(0).replace("d", "e").replace("D", "e")), match.end()


def background(sh, x, y, x1, x2, text: str):
    """fp_background: the background's level at x1 and its slope, from the two words of
    `text` -- a value, avg(w1,w2[,scale]) or med(w1,w2[,scale]) of the data there -- or,
    for a word not given, the data at that end of the region."""
    from . import advice, sarith

    n = y.size
    xval = [f32(x1), f32(x2)]
    yval = [f32(0), f32(0)]
    words = text.split()
    for j in range(2):
        if j >= len(words):
            i = max(1, min(n, sarith._nint(sh.wl(float(xval[j])))))
            xval[j] = f32(sh.lw(float(i)))
            yval[j] = y[i - 1]
            continue
        word = words[j]
        value, k = _ctod_at(word, 0)
        if value is not None and k > 0:
            yval[j] = f32(value)
            continue
        name = ""
        while k < len(word) and (word[k].isalnum() or word[k] in "_$."):
            name += word[k]
            k += 1
        func = [f for f in ("avg", "med") if name and f.startswith(name)]
        if len(func) != 1 or not name[:1].isalpha():
            advice.fail("Syntax error in background specification",
                        "give a value, or avg(w1,w2) or med(w1,w2), for each end.")
        z1, k = _ctod_at(word, k + 1)
        if z1 is None:
            advice.fail("Syntax error in background specification",
                        "give a value, or avg(w1,w2) or med(w1,w2), for each end.")
        z2, k = _ctod_at(word, k + 1)
        if z2 is None:
            advice.fail("Syntax error in background specification",
                        "give a value, or avg(w1,w2) or med(w1,w2), for each end.")
        z3, k = _ctod_at(word, k + 1)
        if z3 is None:
            z3 = 1.0
        z1, z2 = sh.wl(z1), sh.wl(z2)
        i = max(1, sarith._nint(min(z1, z2)))
        m = min(n, sarith._nint(max(z1, z2))) - i + 1
        if m < 1:
            advice.fail("Syntax error in background specification",
                        "the region it names holds no pixels.")
        from .scombine import asum

        xval[j] = f32(z3 * float(asum(x[i - 1:i - 1 + m])) / m)
        if func[0] == "avg":
            yval[j] = f32(z3 * float(asum(y[i - 1:i - 1 + m])) / m)
        else:
            from .examine import _amedr

            yval[j] = f32(z3 * _amedr(np.asarray(y[i - 1:i - 1 + m], f32)))
    if xval[0] == xval[1]:
        return f32(f32(yval[0] + yval[1]) / f32(2)), f32(0)
    dy = f32(f32(yval[1] - yval[0]) / f32(xval[1] - xval[0]))
    return f32(yval[0] + f32(dy * f32(f32(x1) - xval[0]))), dy


def _g(value, decpl: int, width: int) -> str:
    """%width.decplg, as fitprofs prints it: INDEF for an eqw it has none of, and 0. for
    a NaN -- 0/0, when a profile's starting peak is 0, as one outside the region is --
    which is how IRAF's fmtio writes one (its model is NaN everywhere then too)."""
    from . import iraffmt

    if value is None:
        return "INDEF".rjust(width)
    if math.isnan(float(value)):
        value = 0.0
    return iraffmt.printf(float(value), "g", decpl, width).rjust(width)


def fit_region(sh, x, y, profiles, region, fit, nerrsample: int, sigma0, invgain,
               components, background_text: str, gasdev: Gasdev):
    """fp_fit on one spectrum: (the log's lines, the model at every pixel).

    `profiles` are (type, position, peak or None, gfwhm, lfwhm), `region` the two ends
    in world coordinates, `fit` the five constraints, `components` which profiles go
    into the log and the model. sigma0 and invgain are None for no errors."""
    from . import advice, sarith
    from .scombine import asum

    n = y.size
    ng = len(profiles)
    i1 = sarith._nint(sh.wl(float(f32(region[0]))))
    i2 = sarith._nint(sh.wl(float(f32(region[1]))))
    i = min(n, max(i1, i2))
    i1 = max(1, min(i1, i2))
    i2 = i
    nfit = i2 - i1 + 1
    if nfit < 3:
        advice.fail("Too few data points in fitting region",
                    "widen the region: it has to hold three pixels or more.")
    x1 = f32(sh.lw(float(i1)))
    x2 = f32(sh.lw(float(i2)))
    y1, dy = background(sh, x, y, x1, x2, background_text)
    scale = f32(0)
    doerr = sigma0 is not None
    xd = np.asarray(x[i1 - 1:i2], f32).copy()
    yd = np.zeros(nfit, f32)
    for k in range(nfit):
        yk = y[i1 - 1 + k]
        yd[k] = f32(yk - f32(y1 + f32(dy * f32(x[i1 - 1 + k] - x1))))
        if yk <= 0:
            doerr = False
        scale = max(scale, f32(abs(yd[k])))
    if doerr:
        s0, ig = f32(sigma0), f32(invgain)
        sd = np.array([f32(math.sqrt(float(f32(f32(s0 * s0) + f32(ig * y[i1 - 1 + k])))))
                       for k in range(nfit)], f32)
        sscale = f32(asum(sd) / f32(nfit))
    else:
        sd = np.ones(nfit, f32)
        sscale = f32(1)
    yd = (yd / scale).astype(f32)
    sd = (sd / sscale).astype(f32)
    y1 = f32(y1 / scale)
    dy = f32(dy / scale)

    tp = np.array([p[0] for p in profiles], np.int64)
    xg = np.array([p[1] for p in profiles], f32)
    sg = np.array([p[3] for p in profiles], f32)
    lg = np.array([p[4] for p in profiles], f32)
    yg = np.zeros(ng, f32)
    for k, p in enumerate(profiles):
        if p[2] is None:
            j = max(1, min(nfit, sarith._nint(sh.wl(float(p[1]))) - i1 + 1))
            yg[k] = yd[j - 1]
        else:
            yg[k] = f32(f32(p[2]) / scale)
    dx = f32(f32(x[n - 1] - x[0]) / f32(n - 1))
    z1, dz = dofit(fit, xd, yd, sd, dx, NSUB, f32(0), f32(0), xg, yg, sg, lg, tp)

    mc_n = nerrsample
    mc_sig = sarith._nint(float(f32(f32(mc_n * 68) / f32(100))))
    errors = None
    if doerr and mc_sig > 9:
        ym = np.array([model(xd[k], dx, NSUB, xg, yg, sg, lg, tp) for k in range(nfit)],
                      f32)
        state = np.array([1, 0], np.int64)
        names = ("cont", "pos", "peak", "gfwhm", "lfwhm", "flux", "eqw")
        errors = {name: np.zeros((ng, mc_n), f32) for name in names}
        factor = f32(sscale / scale)
        for i in range(mc_n):
            yd1 = np.array([f32(ym[k] + f32(f32(factor * sd[k]) * gasdev(state)))
                            for k in range(nfit)], f32)
            xg1, yg1, sg1, lg1 = xg.copy(), yg.copy(), sg.copy(), lg.copy()
            wyc1, slope1 = dorefit(fit, xd, yd1, sd, dx, NSUB, z1, dz, xg1, yg1, sg1, lg1,
                                   tp)
            for j in range(ng):
                cont = f32(f32(f32(y1 + z1) + f32(f32(dy + dz) * xg[j])) - f32(dy * x1))
                cont1 = f32(f32(f32(y1 + wyc1) + f32(f32(dy + slope1) * xg[j]))
                            - f32(dy * x1))
                flux = flux_of(tp[j], yg[j], sg[j], lg[j])
                flux1 = flux_of(tp[j], yg1[j], sg1[j], lg1[j])
                if cont > 0 and cont1 > 0:
                    eqw, eqw1 = f32(-flux / cont), f32(-flux1 / cont1)
                else:
                    eqw = eqw1 = f32(0)
                for name, a, b in (("cont", cont1, cont), ("pos", xg1[j], xg[j]),
                                   ("peak", yg1[j], yg[j]), ("gfwhm", sg1[j], sg[j]),
                                   ("lfwhm", lg1[j], lg[j]), ("flux", flux1, flux),
                                   ("eqw", eqw1, eqw)):
                    errors[name][j, i] = f32(abs(f32(a - b)))
        for name in names:
            errors[name] = np.sort(errors[name], axis=1)
        for name in ("cont", "peak", "flux"):
            errors[name] = (errors[name] * scale).astype(f32)

    yg = (yg * scale).astype(f32)
    y1 = f32(f32(f32(y1 + z1) + f32(dz * x1)) * scale)
    dy = f32(f32(dy + dz) * scale)

    names = {FIXED: "fixed", SINGLE: "single", INDEP: "all"}
    lines = [f"# Nfit={ng}, background={'yes' if fit[BKG] == SINGLE else 'no'}, "
             f"positions={names[fit[POS]]}, gfwhm={names[fit[GAU]]}, "
             f"lfwhm={names[fit[LOR]]}",
             f"# {'center':>8}{'cont':>10}{'flux':>10}{'eqw':>10}{'core':>10}"
             f"{'gfwhm':>10}{'lfwhm':>10}"]
    for i in range(ng):
        if (i + 1) not in components:
            continue
        xc = xg[i]
        cont = f32(y1 + f32(dy * f32(xc - x1)))
        flux = flux_of(tp[i], yg[i], sg[i], lg[i])
        eqw = f32(-flux / cont) if cont > 0 else None
        lines.append(" " + " ".join((_g(xc, 7, 9), _g(cont, 7, 9), _g(flux, 6, 9),
                                     _g(eqw, 4, 9), _g(yg[i], 6, 9), _g(sg[i], 4, 9),
                                     _g(lg[i], 4, 9))))
        if errors is not None:
            at = mc_sig
            values = ((errors["pos"], 7), (errors["cont"], 7), (errors["flux"], 6),
                      (errors["eqw"], 4), (errors["peak"], 6), (errors["gfwhm"], 4),
                      (errors["lfwhm"], 4))
            lines.append(" " + " ".join(f"({_g(e[i, at], d, 7)})" for e, d in values))
    mod = np.zeros(n, f32)
    for i in range(ng):
        if (i + 1) not in components:
            continue
        for j in range(n):
            mod[j] = f32(mod[j] + model(x[j], dx, NSUB, xg[i:i + 1], yg[i:i + 1],
                                        sg[i:i + 1], lg[i:i + 1], tp[i:i + 1]))
    return lines, mod
