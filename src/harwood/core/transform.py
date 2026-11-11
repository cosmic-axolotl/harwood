"""twodspec.longslit's transform without the task: noao/twodspec/longslit/lstrans
(t_transform.x and trsetup.x), as IRAF 2.18 runs them.

fitcoords leaves surfaces u(x, y) and v(x, y): the output coordinates -- the
wavelength, a position along the slit -- of each input pixel. transform inverts
them on a coarse grid of the output, every tenth pixel in each direction, by
Newton's method from the point before (the grid walked back and forth so that the
last answer is always close), and interpolates the grid bilinearly for every
output pixel; the input is then interpolated there, with the chosen interpolator,
and with flux on multiplied by the area of the input pixel the output one covers.
All of it in real, as IRAF computes it.

What follows from the code, kept:

* the inversion keeps the iterate unclamped while its working copy is held
  inside the surfaces' limits; it stops when both steps are below 0.05 pixel or
  after ten, each step shortened by 1 - 0.5 / iteration; the Jacobian kept is
  the one before the last step;
* output coordinates past the grid are not clamped to it but to the output's own
  size (nu, nv), which only matters when they reach past the grid;
* the input is read into a buffer of lines with two more at either edge, and the
  interpolator is fitted on it alone -- spline3's end conditions are those of the
  buffer, not the image;
* positions outside the input are moved to its edge (edge extension), and then,
  with a blank value, those output pixels set to it;
* with no fit of an axis, its output coordinate is the input pixel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import gsurfit, msi
from . import units as unitsmod

f32 = np.float32
STEP = 10
MAX_ITERATE = 10
ERROR = f32(0.05)
FUDGE = f32(0.5)
NBUF = 16
NEDGE = 2
LN_10 = 2.302585092994046
TYPES = msi.TYPES


class TransformError(ValueError):
    """What transform stops on, in IRAF's words."""


@dataclass
class Output:
    """transform.com's output coordinates: None for INDEF."""
    x1: float | None = None
    x2: float | None = None
    dx: float | None = None
    nx: int | None = None
    xlog: bool = False
    y1: float | None = None
    y2: float | None = None
    dy: float | None = None
    ny: int | None = None
    ylog: bool = False


def _r(value) -> f32:
    return f32(value)


def _log10(value) -> float:
    """log10 of a real, as f2c's r_lg10 computes it: in double."""
    return math.log10(float(value))


def set_output(out: Output, xmin, xmax, ymin, ymax, umin, umax, vmin, vmax) -> Output:
    """tr_setoutput: the limits, the size and the interval each given or taken from
    the surfaces' range and the input's size; an interval given overrides the size."""
    o = Output(**out.__dict__)
    u1 = _r(umin) if o.x1 is None else _r(o.x1)
    u2 = _r(umax) if o.x2 is None else _r(o.x2)
    v1 = _r(vmin) if o.y1 is None else _r(o.y1)
    v2 = _r(vmax) if o.y2 is None else _r(o.y2)
    nu = int(_r(_r(xmax) - _r(xmin)) + _r(1)) if o.nx is None else int(o.nx)
    nv = int(_r(_r(ymax) - _r(ymin)) + _r(1)) if o.ny is None else int(o.ny)

    def axis(a1, a2, d, n, log, given_n, given_1):
        if log:
            if d is None:
                d = _r((_log10(a2) - _log10(a1)) / (n - 1))
            elif given_n is None:
                n = int(round_half_away((_log10(a2) - _log10(a1)) / float(d) + 1))
            elif given_1 is None:
                a1 = _r(10.0 ** (_log10(a2) - float(_r(_r(d) * _r(n - 1)))))
            else:
                a2 = _r(10.0 ** (_log10(a1) + float(_r(_r(d) * _r(n - 1)))))
        else:
            if d is None:
                d = _r(_r(a2 - a1) / _r(n - 1))
            elif given_n is None:
                n = int(round_half_away(float(_r(_r(_r(a2 - a1) / _r(d)) + _r(1)))))
            elif given_1 is None:
                a1 = _r(a2 - _r(_r(d) * _r(n - 1)))
            else:
                a2 = _r(a1 + _r(_r(d) * _r(n - 1)))
        return a1, a2, _r(d), n

    u1, u2, du, nu = axis(u1, u2, o.dx, nu, o.xlog, o.nx, o.x1)
    v1, v2, dv, nv = axis(v1, v2, o.dy, nv, o.ylog, o.ny, o.y1)
    o.x1, o.x2, o.dx, o.nx = float(u1), float(u2), float(du), nu
    o.y1, o.y2, o.dy, o.ny = float(v1), float(v2), float(dv), nv
    return o


def round_half_away(value: float) -> float:
    """nint."""
    return math.floor(value + 0.5) if value >= 0 else -math.floor(-value + 0.5)


# ---------------------------------------------------------------- the surfaces

def _xgsder(sf, x, y, nxd, nyd) -> f32:
    return f32(gsurfit.derivative(sf, np.array([float(x)]), np.array([float(y)]),
                                  nxd, nyd)[0])


def _xgseval(sf, x, y) -> f32:
    return f32(gsurfit.evaluate(sf, float(f32(x)), float(f32(y))))


def _three(surfaces, x, y):
    """The value and the two derivatives, averaged over the surfaces of an axis."""
    a = _xgsder(surfaces[0], x, y, 0, 0)
    b = _xgsder(surfaces[0], x, y, 1, 0)
    c = _xgsder(surfaces[0], x, y, 0, 1)
    if len(surfaces) == 1:
        return a, b, c
    for sf in surfaces[1:]:
        a = f32(a + _xgsder(sf, x, y, 0, 0))
        b = f32(b + _xgsder(sf, x, y, 1, 0))
        c = f32(c + _xgsder(sf, x, y, 0, 1))
    n = f32(len(surfaces))
    return f32(a / n), f32(b / n), f32(c / n)


def tr_init(usf, vsf, x, y) -> list:
    """tr_init: the starting point, 1-based der[1..8]."""
    der = [None] + [f32(0.0)] * 8
    der[1], der[2] = f32(x), f32(y)
    if not usf:
        der[3], der[4], der[5] = der[1], f32(1.0), f32(0.0)
    else:
        der[3], der[4], der[5] = _three(usf, der[1], der[2])
    if not vsf:
        der[6], der[7], der[8] = der[2], f32(0.0), f32(1.0)
    else:
        der[6], der[7], der[8] = _three(vsf, der[1], der[2])
    return der


def tr_invert(usf, vsf, u, v, der, xmin, xmax, ymin, ymax):
    """tr_invert: x, y where the surfaces give u, v, from the last result in der."""
    u, v = f32(u), f32(v)
    x = y = f32(0.0)
    for i in range(1, MAX_ITERATE + 1):
        du = f32(u - der[3])
        dv = f32(v - der[6])
        a = f32(f32(der[8] * du) - f32(der[5] * dv))
        b = f32(f32(der[8] * der[4]) - f32(der[5] * der[7]))
        dx = (f32(-2.0) if a < 0.0 else f32(2.0)) if b == 0.0 else f32(a / b)
        a = f32(dv - f32(der[7] * dx))
        b = der[8]
        dy = (f32(-2.0) if a < 0.0 else f32(2.0)) if b == 0.0 else f32(a / b)
        fudge = f32(f32(1.0) - f32(FUDGE / f32(i)))
        x = f32(der[1] + f32(fudge * dx))
        y = f32(der[2] + f32(fudge * dy))
        der[1] = max(xmin, min(xmax, x))
        der[2] = max(ymin, min(ymax, y))
        if abs(dx) < ERROR and abs(dy) < ERROR:
            break
        if not usf:
            der[3] = der[1]
        else:
            der[3], der[4], der[5] = _three(usf, der[1], der[2])
        if not vsf:
            der[6] = der[2]
        else:
            der[6], der[7], der[8] = _three(vsf, der[1], der[2])
    return x, y


@dataclass
class Setup:
    """tr_setup's result: the output, the interpolators of x, y and the Jacobian
    over the grid, and where each output column and line falls on the grid."""
    output: Output
    xmsi: msi.Msi
    ymsi: msi.Msi
    jmsi: msi.Msi | None
    uout: np.ndarray
    vout: np.ndarray
    duout: np.ndarray
    dvout: np.ndarray
    units: list                     #: un[1], un[2]: Units or None


def surfaces(fits) -> tuple[list, list, list]:
    """tr_gsf: the u and v surfaces and the units of each axis; units that
    disagree stop it."""
    usf, vsf, un = [], [], [None, None]
    for fit in fits:
        if fit.units is not None:
            un1 = unitsmod.decode(fit.units)
            if un[fit.axis - 1] is None:
                un[fit.axis - 1] = un1
            elif not unitsmod._same(un1, un[fit.axis - 1]):
                raise TransformError("Input units disagree")
        if fit.axis == 1:
            usf.append(fit.surface)
        elif fit.axis == 2:
            vsf.append(fit.surface)
    if not usf and not vsf:
        raise TransformError("No user coordinates")
    return usf, vsf, un


def setup(usf, vsf, un, wanted: Output, flux: bool) -> Setup:
    """tr_setup."""
    lim = None
    corners = {}
    for name, group in (("u", usf), ("v", vsf)):
        lo = hi = None
        for sf in group:
            these = (f32(sf.xmin), f32(sf.xmax), f32(sf.ymin), f32(sf.ymax))
            if lim is None:
                lim = these
            elif these != lim:
                raise TransformError("tr_setup: Inconsistent coordinate fits")
            xmin, xmax, ymin, ymax = lim
            if lo is None:
                lo = hi = _xgseval(sf, xmin, ymin)
            for x, y in ((xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)):
                w = _xgseval(sf, x, y)
                lo, hi = min(w, lo), max(w, hi)
        corners[name] = (lo, hi)
    xmin, xmax, ymin, ymax = lim
    umin, umax = corners["u"] if corners["u"][0] is not None else (xmin, xmax)
    vmin, vmax = corners["v"] if corners["v"][0] is not None else (ymin, ymax)
    o = set_output(wanted, xmin, xmax, ymin, ymax, umin, umax, vmin, vmax)
    der = tr_init(usf, vsf, xmin, ymin)
    return _grid(o, flux, un, lambda u, v, d: tr_invert(usf, vsf, u, v, d, xmin, xmax,
                                                         ymin, ymax), der)


def _grid(o: Output, flux: bool, un, invert, der) -> Setup:
    """tr_setup's second half: the inverted grid, back and forth, its interpolators,
    and the output's coordinates on it."""
    u1, u2, du, nu = f32(o.x1), f32(o.x2), f32(o.dx), o.nx
    v1, v2, dv, nv = f32(o.y1), f32(o.y2), f32(o.dy), o.ny

    nu1 = max(2, nu // STEP)
    nv1 = max(2, nv // STEP)
    du1 = f32(f32(u2 - u1) / f32(nu1 - 1))
    dv1 = f32(f32(v2 - v1) / f32(nv1 - 1))
    xgrid = np.zeros((nv1, nu1), f32)
    ygrid = np.zeros((nv1, nu1), f32)
    zgrid = np.zeros((nv1, nu1), f32)

    def point(i, j):
        u = f32(u1 + f32(f32(j - 1) * du1))
        v = f32(v1 + f32(f32(i - 1) * dv1))
        x, y = invert(u, v, der)
        xgrid[i - 1, j - 1] = x
        ygrid[i - 1, j - 1] = y
        with np.errstate(divide="ignore"):
            zgrid[i - 1, j - 1] = f32(f32(1.0) / abs(f32(f32(der[4] * der[8])
                                                          - f32(der[5] * der[7]))))

    for i in range(1, nv1 + 1, 2):
        for j in range(1, nu1 + 1):
            point(i, j)
        if i == nv1:
            break
        for j in range(nu1, 0, -1):
            point(i + 1, j)

    xmsi = msi.msifit("linear", xgrid)
    ymsi = msi.msifit("linear", ygrid)
    jmsi = msi.msifit("linear", zgrid) if flux else None

    def coords(a1, d, n, log):
        k = np.arange(n)
        if log:
            dval = math.log10(float(a1))
            out = np.array([f32(10.0 ** (dval + float(f32(f32(i) * d)))) for i in k], f32)
            step = (out * f32(f32(d) * f32(LN_10))).astype(f32)   # du * LN_10 in real
        else:
            out = (a1 + (k.astype(f32) * d).astype(f32)).astype(f32)
            step = np.full(n, d, f32)
        return out, step

    uout, duout = coords(u1, du, nu, o.xlog)
    vout, dvout = coords(v1, dv, nv, o.ylog)
    o.x2 = float(uout[nu - 1])
    o.y2 = float(vout[nv - 1])
    uout = np.clip((((uout - u1).astype(f32) / du1).astype(f32) + f32(1)).astype(f32),
                   f32(1.0), f32(nu)).astype(f32)
    vout = np.clip((((vout - v1).astype(f32) / dv1).astype(f32) + f32(1)).astype(f32),
                   f32(1.0), f32(nv)).astype(f32)
    return Setup(o, xmsi, ymsi, jmsi, uout, vout, duout, dvout, un)


# --------------------------------------------------------------- the transform

def transform(data: np.ndarray, s: Setup, interp: str, flux: bool,
              blank: float | None, mask: np.ndarray | None = None,
              want_mask: bool = False):
    """tr_transform: the output image, and with want_mask the output mask: 1 where
    the input was off the image, or -- with an input mask, any value above zero bad,
    interpolated as the data are -- where the interpolated mask is above 10 percent."""
    data = np.asarray(data, dtype=f32)
    nyin, nxin = data.shape
    nu, nv = s.output.nx, s.output.ny
    out = np.zeros((nv, nu), f32)
    mout = np.zeros((nv, nu), f32) if want_mask else None
    xout = s.uout
    line1 = line2 = line3 = line4 = 0
    nlines = 0
    zmsi = mzmsi = None
    for i in range(1, nv + 1):
        y = np.full(nu, s.vout[i - 1], f32)
        xin = msi.msivector(s.xmsi, xout, y).astype(f32)
        yin = msi.msivector(s.ymsi, xout, y).astype(f32)
        xin1, yin1 = xin.copy(), yin.copy()
        a, b = f32(xin.min()), f32(xin.max())
        xofb = a < 1 or b > nxin
        if xofb:
            if a < 1:
                xin[xin < 1] = f32(1.0)
            if b > nxin:
                xin[xin > nxin] = f32(nxin)
        a, b = f32(yin.min()), f32(yin.max())
        yofb = a < 1 or b > nyin
        if yofb:
            if a < 1:
                yin[yin < 1] = f32(1.0)
                a = f32(1.0)
                b = max(a, b)
            if b > nyin:
                yin[yin > nyin] = f32(nyin)
                b = f32(nyin)
                a = min(a, b)
        if zmsi is None or b > line2 or a < line1:
            nlines = max(nlines, int(f32(f32(b - a) + f32(2 + NBUF))))
            if zmsi is None:
                if a < nyin // 2:
                    line1 = max(1, int(a))
                    line2 = min(nyin, line1 + nlines - 1)
                else:
                    line2 = min(nyin, int(f32(b + f32(1.0))))
                    line1 = max(1, line2 - nlines + 1)
            elif b > line2:
                line1 = max(1, int(a))
                line2 = min(nyin, line1 + nlines - 1)
                line1 = max(1, line2 - nlines + 1)
            else:
                line2 = min(nyin, int(f32(b + f32(1.0))))
                line1 = max(1, line2 - nlines + 1)
                line2 = min(nyin, line1 + nlines - 1)
            line3 = max(1, line1 - NEDGE)
            line4 = min(nyin, line2 + NEDGE)
            zmsi = msi.msifit(interp, data[line3 - 1:line4])
            if mask is not None:
                buf = np.asarray(mask[line3 - 1:line4], f32).copy()
                buf[buf > f32(0.1)] = f32(100.0)
                mzmsi = msi.msifit(interp, buf)
        yin = (yin - f32(line3 - 1)).astype(f32)
        row = msi.msivector(zmsi, xin, yin).astype(f32)
        if flux:
            jbuf = msi.msivector(s.jmsi, xout, y).astype(f32)
            jbuf = (s.duout * jbuf).astype(f32)
            jbuf = (jbuf * s.dvout[i - 1]).astype(f32)
            row = (row * jbuf).astype(f32)
        if blank is not None:
            if xofb:
                row[(xin1 < 1) | (xin1 > nxin)] = f32(blank)
            if yofb:
                row[(yin1 < 1) | (yin1 > nyin)] = f32(blank)
        out[i - 1] = row
        if want_mask:
            if mask is not None:
                off = (xin1 < 1) | (xin1 > nxin) | (yin1 < 1) | (yin1 > nyin)
                m = msi.msivector(mzmsi, xin, yin).astype(f32)
                m = np.where(m > 0, np.where(m > 10, f32(1), f32(0)), m).astype(f32)
                m[off] = f32(1.0)
            else:
                m = np.zeros(nu, f32)
                if xofb:
                    m[(xin1 < 1) | (xin1 > nxin)] = f32(1.0)
                if yofb:
                    m[(yin1 < 1) | (yin1 > nyin)] = f32(1.0)
            mout[i - 1] = m
    return out, mout


def wcs_cards(s: Setup, laxis: int | None) -> dict:
    """mw_saveim of the new linear world system, and -- transforming by fits, laxis
    given -- the cards for onedspec: what IRAF writes into the output's header."""
    o = s.output
    w1 = math.log10(o.x1) if o.xlog else o.x1
    w2 = math.log10(o.y1) if o.ylog else o.y1
    labels = {}
    for axis in (1, 2):
        un = s.units[axis - 1]
        if un is not None:
            labels[axis] = (un.label, un.units)
    if laxis is not None and laxis not in labels:
        labels[laxis] = ("Wavelength", "Angstroms")

    def card(value) -> float:
        return float(f"{float(f32(value)):.15g}")

    cards = {} if laxis is None else {
        "DCLOG1": "Transform", "DISPAXIS": laxis,
        "DC-FLAG": int(o.xlog if laxis == 1 else o.ylog)}
    cards |= {"WCSDIM": 2, "CTYPE1": "LINEAR", "CTYPE2": "LINEAR",
             "CRVAL1": card(w1), "CRVAL2": card(w2), "CRPIX1": 1.0, "CRPIX2": 1.0,
             "CDELT1": card(o.dx), "CDELT2": card(o.dy),
             "CD1_1": card(o.dx), "CD2_2": card(o.dy), "LTM1_1": 1.0, "LTM2_2": 1.0,
             "WAT0_001": "system=world"}
    for axis in (1, 2):
        text = "wtype=linear"
        if axis in labels:
            text += f" label={labels[axis][0]} units={labels[axis][1]}"
        cards[f"WAT{axis}_001"] = text
    return cards


#: The cards of the input's world system that the new one replaces.
OLD_WCS = ("WCSDIM", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
           "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "LTV1", "LTV2",
           "LTM1_1", "LTM1_2", "LTM2_1", "LTM2_2", "WAT0_001", "WAT1_001", "WAT2_001",
           "WAXMAP01")
#: What transforming by fits takes out besides, for onedspec.
OLD_DISPERSION = ("REFSPEC1", "REFSPEC2")


# ------------------------------------------------- by the image's world system

class ImageWcs:
    """tr_gwcs's two transformations, logical to world and back, compiled as
    mw_sctran compiles them -- in double, from the Lterm (LTM, LTV) and the Wterm
    (the header's CD and CRPIX taken to physical pixels, CRVAL) -- and applied as
    mw_c2tranr applies them: their single-precision copies, in real."""

    def __init__(self, header, nx: int, ny: int):
        from . import mwcs

        n = 2
        num = mwcs._number
        ltm = np.array([[num(header, f"LTM{i + 1}_{j + 1}", float(i == j)) for j in range(n)]
                        for i in range(n)])
        ltv = np.array([num(header, f"LTV{i + 1}", 0.0) for i in range(n)])
        cd = mwcs._matrix(header, n)
        crpix = np.array([num(header, f"CRPIX{i + 1}", 0.0) for i in range(n)])
        crval = np.array([num(header, f"CRVAL{i + 1}", 0.0) for i in range(n)])
        inverse = np.linalg.inv(ltm)
        cdp = mwcs._mmul(cd, ltm)                         # the Wterm, in physical
        rp = mwcs._vmul(inverse, crpix - ltv)
        # logical -> world: in, the logical system (CD = LTM, W = LTV); out, the world
        i_ltv = -mwcs._vmul(inverse, ltv)
        o_ltv = mwcs._vmul(cdp, -rp) + crval
        self.lw = self._single(mwcs._mmul(cdp, inverse), mwcs._vmul(cdp, i_ltv) + o_ltv)
        # world -> logical
        icd = np.linalg.inv(cdp)
        i_ltv = rp - mwcs._vmul(icd, crval)
        self.wl = self._single(mwcs._mmul(ltm, icd), mwcs._vmul(ltm, i_ltv) + ltv)
        self.nx, self.ny = nx, ny
        self.units = [_wat_units(header, axis) for axis in (1, 2)]

    @staticmethod
    def _single(matrix, vector):
        rotated = abs(matrix[0, 1]) > 100 * 2.220446049250313e-16 or \
            abs(matrix[1, 0]) > 100 * 2.220446049250313e-16
        return matrix.astype(f32), vector.astype(f32), rotated

    @staticmethod
    def _apply(ct, x, y):
        m, v, rotated = ct
        x, y = f32(x), f32(y)
        if not rotated:
            return f32(f32(m[0, 0] * x) + v[0]), f32(f32(m[1, 1] * y) + v[1])
        return (f32(f32(f32(m[0, 0] * x) + f32(m[0, 1] * y)) + v[0]),
                f32(f32(f32(m[1, 0] * x) + f32(m[1, 1] * y)) + v[1]))

    def world(self, x, y):
        return self._apply(self.lw, x, y)

    def logical(self, u, v):
        return self._apply(self.wl, u, v)


def _wat_units(header, axis: int):
    """mw_gwattrs(axis, "units"): the WAT's units attribute, un_open'ed; None
    without one."""
    import re as regex

    text = "".join(str(header.get(f"WAT{axis}_{k:03d}", "")) for k in range(1, 100)
                   if f"WAT{axis}_{k:03d}" in header)
    found = regex.search(r"units\s*=\s*(\"[^\"]*\"|\S+)", text)
    if not found:
        return None
    return unitsmod.decode(found.group(1).strip('"'))


def wcs_invert(ct: ImageWcs, u, v, der):
    """tr_invert with a CT: MWCS inverts, and the derivatives are the world's change
    over a pixel either side."""
    x, y = ct.logical(u, v)
    a1, a3 = ct.world(f32(x - f32(0.5)), y)
    a2, a4 = ct.world(f32(x + f32(0.5)), y)
    der[4] = f32(a2 - a1)
    der[7] = f32(a4 - a3)
    a1, a3 = ct.world(x, f32(y - f32(0.5)))
    a2, a4 = ct.world(x, f32(y + f32(0.5)))
    der[5] = f32(a2 - a1)
    der[8] = f32(a4 - a3)
    return x, y


def setup_wcs(ct: ImageWcs, wanted: Output, flux: bool) -> Setup:
    """tr_setup with the image's WCS: one 'surface' for each axis, the limits the
    image's size, the corners and the inversion MWCS's."""
    xmin, xmax = f32(1), f32(ct.nx)
    ymin, ymax = f32(1), f32(ct.ny)
    corners = [ct.world(x, y) for x, y in ((xmin, ymin), (xmax, ymin), (xmin, ymax),
                                           (xmax, ymax))]
    us = [c[0] for c in corners]
    vs = [c[1] for c in corners]
    umin = umax = us[0]
    for u in us:
        umin, umax = min(u, umin), max(u, umax)
    vmin = vmax = vs[0]
    for v in vs:
        vmin, vmax = min(v, vmin), max(v, vmax)
    return _grid(set_output(wanted, xmin, xmax, ymin, ymax, umin, umax, vmin, vmax),
                 flux, ct.units, lambda u, v, der: wcs_invert(ct, u, v, der),
                 [None] + [f32(0)] * 8)
