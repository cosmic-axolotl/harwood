"""A spectrum as onedspec's tasks see it: noao/onedspec/smw (smw_openim and the
formats behind it, smw_gwattrs, the coordinate transformations) and shdr.x
(shdr_open, shdr_lw/wl, shdr_system, shdr_units), as IRAF 2.18 runs them.

An image becomes spectra in one of three ways. A 1-D image with a linear world
system, an IRAF onedspec, is EQUISPEC: one spectrum. A 2-D or 3-D image whose
lines or columns are spectra along DISPAXIS (or with no world system) is NDSPEC:
each spectrum the sum of nsum lines (or columns) about the one asked for. A
MULTISPEC image (apall's) has one spectrum per line, each with its aperture,
beam and its own dispersion function -- linear, log-linear or nonlinear. And
harwood's own spectrum files, one table per aperture, are read as the multispec
image they stand for.

What follows from the code, kept:

* the world coordinates are MWCS's, compiled in double (logical to physical
  through the Lterm, then the Wterm) and kept in real in SX;
* a 1-D image's reference pixel is moved to 1 (smw_onedspec), and the number of
  valid pixels is the larger end of the image taken to physical pixels;
* a log-linear dispersion is 10 to the power of the linear world coordinate,
  held within 10^-20 and 10^20; a spectrum's first and last pixels are those
  of its valid physical range (NP1, NP2);
* an NDSPEC spectrum of line N sums lines N - (nsum - 1) / 2 on, nsum of them,
  cut at the image's edges, in real.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import advice, iraffmt, mwcs
from . import units as unitsmod

f32 = np.float32
ND, ES, MS = 0, 1, 2
DCNO, DCLINEAR, DCLOG, DCFUNC = -1, 0, 1, 2
FCNO, FCYES = -1, 0
INDEFI = None
INDEFR = 1.6e38
CTYPES = ("LAMBDA", "FREQ", "WAVELENGTH", "VELO", "VELO-LSR", "VELO-HEL", "VELO-OBS")


class ShdrError(ValueError):
    """What smw and shdr stop on, in IRAF's words."""


# ----------------------------------------------------------------- headers

def wat(header, axis: int) -> str:
    """An axis's WAT cards joined as IRAF joins them, 68 characters each."""
    text = ""
    for number in range(1, 10000):
        key = f"WAT{axis}_{number:03d}"
        if key not in header:
            break
        text += str(header[key]).ljust(68)
    return text


def attributes(header, axis: int) -> dict[str, str]:
    """mw_gwattrs: the attributes of an axis (0: the system's)."""
    return {name: quoted if plain == "" else plain
            for name, quoted, plain in
            re.findall(r'(\w+)\s*=\s*(?:"([^"]*)"|(\S+))', wat(header, axis))}


def _num(header, key, default):
    return mwcs._number(header, key, default)


def _int(header, key, default):
    if key not in header:
        return default
    try:
        return int(header[key])
    except (TypeError, ValueError):
        return default


def _real(header, key, default):
    if key not in header:
        return default
    try:
        return float(f32(float(header[key])))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------ the transforms

@dataclass
class Linear:
    """A compiled linear transformation of one axis, in double: out = m in + v; a
    log-linear axis taken to 10^ (forward) or log10 (inverse) of the world side."""
    m: float
    v: float
    log: bool = False
    inverse: bool = False

    def __call__(self, value: float) -> float:
        if self.inverse:
            if self.log:
                value = math.log10(value) if value > 0 else -1e300
            return self.m * value + self.v
        w = self.m * value + self.v
        if self.log:
            w = 10.0 ** max(-20.0, min(20.0, w))
        return w


def _compile(ltm, ltv, r, w, cd, axis: int, world: bool):
    """mw_sctran logical -> world (world True) or logical -> physical, and back,
    for one axis of a diagonal system: the row of the composed matrix and vector."""
    n = len(ltv)
    ltm, ltv = np.asarray(ltm, float), np.asarray(ltv, float)
    inverse = np.linalg.inv(ltm)
    i_ltv = -mwcs._vmul(inverse, ltv)
    if world:
        cd, r, w = np.asarray(cd, float), np.asarray(r, float), np.asarray(w, float)
        o_ltv = mwcs._vmul(cd, -r) + w
        fwd_m = mwcs._mmul(cd, inverse)
        fwd_v = mwcs._vmul(cd, i_ltv) + o_ltv
        icd = np.linalg.inv(cd)
        j_ltv = r - mwcs._vmul(icd, w)
        inv_m = mwcs._mmul(ltm, icd)
        inv_v = mwcs._vmul(ltm, j_ltv) + ltv
    else:
        fwd_m, fwd_v = inverse, i_ltv
        inv_m, inv_v = ltm, ltv
    a = axis - 1
    del n
    return (float(fwd_m[a, a]), float(fwd_v[a])), (float(inv_m[a, a]), float(inv_v[a]))


def _single(m: float, v: float):
    """The real copy of a compiled transformation, applied as mw_c1tranr does."""
    mr, vr = f32(m), f32(v)
    return lambda x: float(f32(f32(mr * f32(x)) + vr))


# ------------------------------------------------------- multispec functions

CHEBYSHEV, LEGENDRE, SPLINE3, SPLINE1, PIXEL, SAMPLE = 1, 2, 3, 4, 5, 6


def msp_coeff(text: str, xmin: float, xmax: float) -> list[float] | None:
    """wf_msp_coeff: the nonlinear coefficients, 1-based in a list, the ranges in
    their efficient form and the function's limits."""
    values = []
    for word in text.split():
        v = iraffmt.ctod(word)
        if v is None:
            break
        values.append(v)
    if not values:
        return None
    ncoeff = 5 + len(values)             # IRAF's count includes its five
    coeff = [0.0] + [0.0] * 5 + values   # 1-based; the functions from coeff[6] on
    coeff[1] = float(ncoeff)
    i = 6
    while i < ncoeff:
        kind = round(coeff[i + 2])
        order = round(coeff[i + 3])
        if kind in (CHEBYSHEV, LEGENDRE):
            dval = 2 / (coeff[i + 5] - coeff[i + 4])
            coeff[i + 4] = (coeff[i + 5] + coeff[i + 4]) / 2
            coeff[i + 5] = dval
            i += 6 + order
        elif kind == SPLINE3:
            coeff[i + 5] = round(coeff[i + 3]) / (coeff[i + 5] - coeff[i + 4])
            i += 9 + order
        elif kind == SPLINE1:
            coeff[i + 5] = round(coeff[i + 3]) / (coeff[i + 5] - coeff[i + 4])
            i += 7 + order
        elif kind == PIXEL:
            i += 4 + order
        elif kind == SAMPLE:
            coeff[i + 4] = i + 5
            i += 5 + order
        else:
            break
    coeff[2], coeff[3] = xmin, xmax
    a, b = msp_eval(coeff, xmin), msp_eval(coeff, xmax)
    coeff[4], coeff[5] = min(a, b), max(a, b)
    return coeff


def msp_eval(coeff: list[float], xin: float) -> float:
    """wf_msp_eval: the weighted sum of the functions at a physical pixel."""
    ncoeff = round(coeff[1])
    xval = max(coeff[2], min(coeff[3], xin))
    ysum = wsum = 0.0
    j = 6
    while j < ncoeff:
        kind = round(coeff[j + 2])
        order = round(coeff[j + 3])
        y = coeff[j + 1]
        w = coeff[j]
        if kind in (CHEBYSHEV, LEGENDRE):
            x = (xval - coeff[j + 4]) * coeff[j + 5]
            y = y + coeff[j + 6]
            if order > 1:
                y = y + coeff[j + 7] * x
            if order > 2:
                k, a, b = j + 8, 1.0, x
                for i in range(3, order + 1):
                    c = 2 * x * b - a if kind == CHEBYSHEV else \
                        ((2 * i - 3) * x * b - (i - 2) * a) / (i - 1)
                    y = y + coeff[k] * c
                    a, b = b, c
                    k += 1
            j += 6 + order
        elif kind == SPLINE3:
            x = (xval - coeff[j + 4]) * coeff[j + 5]
            i = max(0, min(int(x), order - 1))
            k = j + 6 + i
            b = x - i
            a = 1 - b
            y = y + a * a * a * coeff[k]
            y = y + (1 + 3 * a * (1 + a * b)) * coeff[k + 1]
            y = y + (1 + 3 * b * (1 + a * b)) * coeff[k + 2]
            y = y + b * b * b * coeff[k + 3]
            j += 9 + order
        elif kind == SPLINE1:
            x = (xval - coeff[j + 4]) * coeff[j + 5]
            i = max(0, min(int(x), order - 1))
            k = j + 6 + i
            b = x - i
            y = y + (1 - b) * coeff[k] + b * coeff[k + 1]
            j += 7 + order
        elif kind == PIXEL:
            i = max(1, min(int(xval), order - 1))
            x = xval - i
            y = y + (1 - x) * coeff[j + 3 + i] + x * coeff[j + 4 + i]
            j += 4 + order
        elif kind == SAMPLE:
            i = round(coeff[j + 4])
            k = j + 2 + order
            while i < k and xval > coeff[i + 2]:
                i += 2
            k = j + 5
            while i > k and xval < coeff[i - 2]:
                i -= 2
            coeff[j + 4] = i
            x = (xval - coeff[i]) / (coeff[i + 2] - coeff[i])
            y = y + (1 - x) * coeff[i + 1] + x * coeff[i + 3]
            j += 5 + order
        else:
            break
        ysum = ysum + w * y
        wsum = wsum + w
    return ysum / wsum


def msp_evali(coeff, y: float, state: list[float]) -> float:
    """wf_msp_evali: the physical pixel of a world coordinate, by Newton's method
    from the last one (state = [x, dydx])."""
    x, dydx = state
    yval = max(coeff[4], min(coeff[5], y))
    dx = dy = 0.0
    converged = False
    for _ in range(10):
        y1 = msp_eval(coeff, x)
        if dx > 1.0:
            dy = msp_eval(coeff, x + 1.0) - y1 if x + 1 < coeff[3] else \
                y1 - msp_eval(coeff, x - 1.0)
        elif dx < -1.0:
            dy = y1 - msp_eval(coeff, x - 1.0) if x - 1 > coeff[2] else \
                msp_eval(coeff, x + 1.0) - y1
        if dy != 0.0:
            dydx = dy
        dx = (yval - y1) / dydx
        x = max(coeff[2], min(coeff[3], x + dx))
        if abs(dx) < 0.0001:
            converged = True
            break
    if not converged:
        xval = (coeff[2] + coeff[3]) / 2.0
        yval = abs(msp_eval(coeff, xval) - y)
        dx = (coeff[3] - coeff[2]) / 18.0
        while dx > 0.0001:
            x = max(coeff[2], xval - 9 * dx)
            top = min(coeff[3], xval + 9 * dx)
            while x <= top:
                d = abs(msp_eval(coeff, x) - y)
                if d < yval:
                    xval, yval = x, d
                x = x + dx
            dx = dx / 10.0
        x = xval
        dy = msp_eval(coeff, x + 1.0) - msp_eval(coeff, x) if x + 1 < coeff[3] else \
            msp_eval(coeff, x) - msp_eval(coeff, x - 1.0)
        if dy != 0.0:
            dydx = dy
    state[0], state[1] = x, dydx
    whole = float(int(x))
    return whole + round_half_away((x - whole) / 0.0001) * 0.0001


def round_half_away(value: float) -> float:
    return math.floor(value + 0.5) if value >= 0 else -math.floor(-value + 0.5)


@dataclass
class Spec:
    """A multispec specN attribute, as smw_gwattrs and wf_msp_init read it."""
    ap: int
    beam: int
    dtype: int
    w1: float
    dw: float
    nw: int
    z: float
    aplow: float
    aphigh: float
    coeff: str = ""
    function: list | None = None          #: wf_msp_coeff of the nonlinear terms
    state: list = field(default_factory=lambda: [0.0, 0.0])


def parse_spec(text: str) -> Spec:
    words = text.split()

    def ctoi(i):
        try:
            return int(words[i])
        except (IndexError, ValueError):
            return 0

    def ctod(i):
        v = iraffmt.ctod(words[i]) if i < len(words) else None
        return 0.0 if v is None else v

    spec = Spec(ctoi(0), ctoi(1), ctoi(2), ctod(3), ctod(4), ctoi(5), ctod(6),
                float(f32(ctod(7))), float(f32(ctod(8))), " ".join(words[9:]))
    if spec.dtype == DCFUNC:
        spec.function = msp_coeff(spec.coeff, 0.5, spec.nw + 0.5)
        if spec.function is not None:
            spec.state = [float(spec.nw), msp_eval(spec.function, float(spec.nw))
                          - msp_eval(spec.function, float(spec.nw) - 1)]
    return spec


# ------------------------------------------------------------------- SMW

@dataclass
class Smw:
    """smw.h's structure, for what shdr needs."""
    format: int
    header: object
    shape: tuple                      #: the image's lengths, axis 1 first
    title: str
    pdim: int
    ldim: int
    paxis: list
    laxis: list
    llen: list
    nsum: list
    nspec: int = 1
    nbands: int = 1
    dtype: int | None = None
    w1: float = 0.0
    dw: float = 0.0
    nw: int = 0
    z: float = 0.0
    aps: list = field(default_factory=list)
    beams: list = field(default_factory=list)
    aplows: list = field(default_factory=list)
    aphighs: list = field(default_factory=list)
    apids: dict = field(default_factory=dict)
    specs: dict = field(default_factory=dict)  #: multispec: physical line -> Spec
    lterm: tuple = ()                 #: (ltm matrix, ltv)
    wterm: tuple = ()                 #: (r, w, cd)
    label: str = ""
    units: str = ""
    units_display: str | None = None
    system: str = "world"
    trans: bool = False


def _lterm(header, n: int):
    ltm = np.eye(n)
    for i in range(n):
        for j in range(n):
            ltm[i, j] = _num(header, f"LTM{i + 1}_{j + 1}", ltm[i, j])
    ltv = np.array([_num(header, f"LTV{i + 1}", 0.0) for i in range(n)])
    return ltm, ltv


def _wterm(header, n: int, ltm, ltv):
    """The Wterm as MWCS holds it, in physical pixels."""
    cd = mwcs._matrix(header, n)
    crpix = np.array([_num(header, f"CRPIX{i + 1}", 0.0) for i in range(n)])
    crval = np.array([_num(header, f"CRVAL{i + 1}", 0.0) for i in range(n)])
    inverse = np.linalg.inv(ltm)
    return mwcs._vmul(inverse, crpix - ltv), crval, mwcs._mmul(cd, ltm)


def open_smw(header, shape: tuple, dispaxis: int = 1, nsum: tuple = (1, 1),
             title: str | None = None) -> Smw:
    """smw_openim on a header and the image's lengths (axis 1 first)."""
    shape = tuple(int(s) for s in shape)
    ldim = len(shape)
    shape3 = shape + (1,) * (3 - ldim)
    wcsdim = _int(header, "WCSDIM", ldim)
    pdim = max(wcsdim, ldim)
    system = attributes(header, 0).get("system", "world")
    title = str(header.get("OBJECT", "")).strip() if title is None else title
    ltm, ltv = _lterm(header, pdim)
    smw = Smw(ND, header, shape3, title, pdim, ldim, [1, 2, 3], [1, 2, 3],
              [shape3[0], 1, 1], list(nsum), lterm=(ltm, ltv))
    if system == "equispec":
        _equispec(smw)
    elif system == "multispec":
        _multispec(smw)
    elif "APFORMAT" in header:
        advice.fail(f"{title}: IRAF's old multispec format (APFORMAT) is not read",
                    "convert it with IRAF's imcopy first, or extract the spectra "
                    "again with harwood")
    elif wcsdim == 1:
        _onedspec(smw)
    else:
        _nd(smw, dispaxis)
    return smw


def _axis_attrs(smw: Smw, axis: int) -> None:
    attrs = attributes(smw.header, axis)
    smw.label = attrs.get("label", "")
    smw.units = attrs.get("units", "")
    smw.units_display = attrs.get("units_display")


def _onedspec(smw: Smw) -> None:
    """smw_onedspec: the reference moved to pixel 1, the dispersion type."""
    h = smw.header
    ltm, ltv = smw.lterm
    r, w, cd = _wterm(h, smw.pdim, ltm, ltv)
    dw = cd[0, 0]
    w0 = w[0] - (r[0] - 1) * dw
    r, w = r.copy(), w.copy()
    r[0], w[0] = 1.0, w0
    smw.wterm = (r, w, cd)
    phys = _single(*_compile(ltm, ltv, r, w, cd, 1, False)[0])
    nw = int(max(phys(1.0), phys(float(smw.shape[0]))))
    dtype = _int(h, "DC-FLAG", None)
    if dtype is None:
        dtype = DCNO if (iraffmt_fp_equal(1.0, w0) or iraffmt_fp_equal(1.0, dw)) \
            else DCLINEAR
    w1 = w0
    if dtype == DCLOG:
        if abs(w1) > 20 or abs(w1 + (nw - 1) * dw) > 20:
            dtype = DCLINEAR
        else:
            w1 = 10.0 ** w1
            dw = w1 * (10.0 ** ((nw - 1) * dw) - 1) / (nw - 1)
    _axis_attrs(smw, 1)
    if dtype != DCNO and "label" not in attributes(h, 1) and \
            "units" not in attributes(h, 1):
        smw.units, smw.label = "angstroms", "Wavelength"
    smw.format = ES
    _saxes(smw)
    beam = _int(h, "BEAM-NUM", 1)
    ap = _int(h, "APNUM", beam)
    smw.aps, smw.beams = [ap], [beam]
    smw.aplows = [(_real(h, "APLOW", None), None)]
    smw.aphighs = [(_real(h, "APHIGH", None), None)]
    smw.dtype, smw.w1, smw.dw, smw.nw = dtype, w1, dw, nw
    smw.z = _num(h, "DOPCOR", 0.0)


def iraffmt_fp_equal(a: float, b: float) -> bool:
    """fp_equald."""
    if a == b:
        return True
    if a == 0.0 or b == 0.0:
        return False
    na, ea = math.frexp(a)
    nb, eb = math.frexp(b)
    return ea == eb and abs(na - nb) <= 32 * 2.220446049250313e-16


def _equispec(smw: Smw) -> None:
    """smw_equispec: the dispersion of axis 1, the apertures from APNUMn."""
    h = smw.header
    ltm, ltv = smw.lterm
    r, w, cd = _wterm(h, smw.pdim, ltm, ltv)
    smw.wterm = (r, w, cd)
    w1, dw = w[0], cd[0, 0]
    phys = _single(*_compile(ltm, ltv, r, w, cd, 1, False)[0])
    nw = int(max(phys(1.0), phys(float(smw.shape[0]))))
    dtype = _int(h, "DC-FLAG", DCNO)
    if dtype == DCLOG:
        if abs(w1) > 20 or abs(w1 + (nw - 1) * dw) > 20:
            dtype = DCLINEAR
        else:
            w1 = 10.0 ** w1
            dw = w1 * (10.0 ** ((nw - 1) * dw) - 1) / (nw - 1)
    _axis_attrs(smw, 1)
    smw.format = ES
    _saxes(smw)
    smw.dtype, smw.w1, smw.dw, smw.nw, smw.z = dtype, w1, dw, nw, 0.0
    for i in range(1, smw.nspec + 1):
        j = _physical_line(smw, i)
        key = f"APNUM{j}" if j < 1000 else f"AP{j}"
        words = str(h.get(key, "")).split()
        nums = []
        for word in words:
            try:
                nums.append(float(word))
            except ValueError:
                break
        ap = int(nums[0]) if nums else 0
        beam = int(nums[1]) if len(nums) > 1 else 0
        lim = [float(f32(v)) for v in nums[2:6]] + [None] * 4
        smw.aps.append(ap)
        smw.beams.append(beam)
        smw.aplows.append((lim[0], lim[2]))
        smw.aphighs.append((lim[1], lim[3]))
        if f"APID{j}" in h:
            smw.apids[i] = str(h[f"APID{j}"])


def _multispec(smw: Smw) -> None:
    """smw_multispec: each line's specN attribute."""
    h = smw.header
    _axis_attrs(smw, 1)
    smw.format = MS
    _saxes(smw)
    attrs = attributes(h, 2)
    for i in range(1, smw.nspec + 1):
        j = _physical_line(smw, i)
        text = attrs.get(f"spec{j}")
        if text is None:
            raise ShdrError(f"Attribute not found (spec{j})")
        spec = parse_spec(text)
        smw.specs[j] = spec
        smw.aps.append(spec.ap)
        smw.beams.append(spec.beam)
        if f"APID{j}" in h:
            smw.apids[i] = str(h[f"APID{j}"])


def _physical_line(smw: Smw, line: int) -> int:
    """smw_mw's physical line of a logical one (the Lterm of axis 2)."""
    if smw.pdim < 2:
        return line
    ltm, ltv = smw.lterm
    phys = _single(*_compile(ltm, ltv, None, None, None, 2, False)[0])
    return int(round_half_away(phys(float(line))))


def _nd(smw: Smw, dispaxis: int) -> None:
    """smw_nd: smw_daxis and smw_saxes; a single spectrum along the first axis
    is then EQUISPEC (smw_ndes)."""
    h = smw.header
    paxis = None
    if "DISPAXIS" in h:
        paxis = _int(h, "DISPAXIS", dispaxis)
    else:
        paxis = dispaxis
        for i in range(1, 8):
            key = f"CTYPE{i}"
            if key not in h:
                break
            value = str(h[key]).replace(" ", "").upper()
            if value and any(c.startswith(value) or value == c for c in CTYPES):
                paxis = i
                break
    if paxis < 1 or paxis > 7:
        paxis = dispaxis
    smw.paxis[0] = paxis
    _saxes(smw)


def _saxes(smw: Smw) -> None:
    """smw_saxes."""
    h = smw.header
    pdim, ldim = smw.pdim, smw.ldim
    ltm, ltv = smw.lterm
    if smw.format == ND:
        smw.trans = False
        if pdim == 2:
            if ltm[0, 0] == 0 and ltm[1, 1] == 0:
                ltm = np.array([[ltm[0, 1], 0.0], [0.0, ltm[1, 0]]])
                smw.paxis[0] = 2 if smw.paxis[0] == 1 else 1
                smw.trans = True
            elif ltm[0, 1] != 0 or ltm[1, 0] != 0:
                ltm, ltv = np.eye(2), np.zeros(2)
            smw.lterm = (ltm, ltv)
        paxis = smw.paxis[0]
        laxis = max(1, min(pdim, paxis))
        if smw.shape[laxis - 1] == 1:
            for laxis in range(1, ldim + 1):
                if smw.shape[laxis - 1] != 1:
                    break
        nspec = 1
        for i in range(1, ldim + 1):
            if i != laxis:
                nspec *= smw.shape[i - 1]
        smw.nspec, smw.nbands = nspec, 1
        paxis = laxis
        r, w, cd = _wterm(h, pdim, ltm, ltv)
        if smw.trans:
            cd = np.array([[cd[0, 1], 0.0], [0.0, cd[1, 0]]])
        if pdim == 2 and (cd[0, 1] != 0 or cd[1, 0] != 0):
            if _int(h, "DC-FLAG", DCNO) != DCNO:
                raise ShdrError("Rotated, dispersion calibrated spectra are not allowed")
            r, w, cd = np.zeros(2), np.zeros(2), np.eye(2)
        for i in range(pdim):
            if cd[i, i] == 0:
                cd[i, i] = 1.0
        smw.wterm = (r, w, cd)
        dw = cd[paxis - 1, paxis - 1]
        w1 = w[paxis - 1] - (r[paxis - 1] - 1) * dw
        nw = smw.shape[laxis - 1]
        phys = _single(*_compile(ltm, ltv, None, None, None, paxis, False)[0])
        nw = int(max(phys(0.5), phys(nw + 0.5)))
        dtype = _int(h, "DC-FLAG", None)
        attrs = attributes(h, paxis)
        if dtype is None:
            if attrs.get("axtype") in ("ra", "dec") or iraffmt_fp_equal(1.0, w1) or \
                    iraffmt_fp_equal(1.0, dw):
                dtype = DCNO
            else:
                dtype = DCLINEAR
        if dtype == DCLOG:
            if abs(w1) > 20 or abs(w1 + (nw - 1) * dw) > 20:
                dtype = DCLINEAR
            else:
                w1 = 10.0 ** w1
                dw = w1 * (10.0 ** ((nw - 1) * dw) - 1) / (nw - 1)
        smw.label = attrs.get("label", "")
        smw.units = attrs.get("units", "")
        smw.units_display = attrs.get("units_display")
        if dtype != DCNO and "label" not in attrs and "units" not in attrs:
            smw.units, smw.label = "angstroms", "Wavelength"
        smw.dtype, smw.w1, smw.dw, smw.nw, smw.z = dtype, w1, dw, nw, 0.0
    else:
        paxis = 1
        laxis = 1
        smw.nspec = smw.shape[1] if ldim > 1 else 1
        smw.nbands = smw.shape[2] if ldim > 2 else 1
    smw.paxis[0] = paxis
    smw.laxis[0] = laxis
    smw.llen = [smw.shape[laxis - 1], 1, 1]
    i = 2
    for ax in range(1, ldim + 1):
        if ax != laxis:
            smw.paxis[i - 1] = ax
            smw.laxis[i - 1] = ax
            smw.llen[i - 1] = smw.shape[ax - 1]
            i += 1


def nd_limits(smw: Smw, index: int) -> tuple[int, int]:
    """smw_gwattrs's NDSPEC aperture: the nsum lines about one, cut at the edges."""
    low = index - (smw.nsum[0] - 1) // 2
    high = int(round_half_away(low)) + smw.nsum[0] - 1
    return max(1, low), min(smw.llen[1], high)


# ------------------------------------------------------------------ SHDR

@dataclass
class Shdr:
    """shdr's structure: one spectrum, its coordinates and its data in real."""
    smw: Smw
    image: str
    imsec: str
    title: str
    label: str
    units: str
    flabel: str
    funits: str
    ap: int
    beam: int
    aplow: list
    aphigh: list
    lindex: list
    apindex: int
    np1: int
    np2: int
    dc: int
    fc: int
    ec: int
    it: float | None
    un: unitsmod.Units
    mwun: unitsmod.Units
    system: str = "world"
    x: np.ndarray | None = None          #: SX, real
    y: np.ndarray | None = None          #: SY, real
    w0: float = 0.0
    w1: float = 0.0
    wp: float = 0.0
    spec: Spec | None = None

    @property
    def sn(self) -> int:
        return self.np2 - self.np1 + 1

    # --------------------------------------------------------- coordinates

    def _world(self, l0: float) -> float:
        """The dispersion axis's logical -> world (or physical) transformation."""
        smw = self.smw
        if smw.format == MS:
            ltm, ltv = smw.lterm
            to_phys = _compile(ltm, ltv, None, None, None, 1, False)[0]
            p = to_phys[0] * l0 + to_phys[1]
            if self.system == "physical" or self.spec is None:
                return p
            s = self.spec
            if s.dtype == DCFUNC and s.function is not None:
                w = msp_eval(s.function, p)
            else:
                din = max(0.5, min(s.nw + 0.5, p))
                w = s.w1 + s.dw * (din - 1)
                if s.dtype == DCLOG:
                    w = 10.0 ** w
            return w / (s.z + 1)
        ltm, ltv = smw.lterm
        r, w, cd = smw.wterm
        if self.system == "physical":
            m, v = _compile(ltm, ltv, None, None, None, smw.paxis[0], False)[0]
            return m * l0 + v
        m, v = _compile(ltm, ltv, r, w, cd, smw.paxis[0], True)[0]
        return Linear(m, v, smw.dtype == DCLOG)(l0)

    def _logical(self, w: float) -> float:
        smw = self.smw
        if smw.format == MS:
            ltm, ltv = smw.lterm
            to_log = _compile(ltm, ltv, None, None, None, 1, False)[1]
            if self.system == "physical" or self.spec is None:
                p = w
            else:
                s = self.spec
                din = w * (s.z + 1)
                if s.dtype == DCFUNC and s.function is not None:
                    p = msp_evali(s.function, din, s.state)
                else:
                    if s.dtype == DCLOG:
                        din = math.log10(din)
                    p = (din - s.w1) / s.dw + 1
                    p = max(0.5, min(s.nw + 0.5, p))
            return to_log[0] * p + to_log[1]
        ltm, ltv = smw.lterm
        r, wv, cd = smw.wterm
        if self.system == "physical":
            m, v = _compile(ltm, ltv, None, None, None, smw.paxis[0], False)[1]
            return m * w + v
        m, v = _compile(ltm, ltv, r, wv, cd, smw.paxis[0], True)[1]
        return Linear(m, v, smw.dtype == DCLOG, inverse=True)(w)

    def lw(self, pixel: float) -> float:
        """shdr_lw: a logical pixel's coordinate, in the spectrum's units."""
        w = self._world(pixel + self.np1 - 1)
        return float(unitsmod.convert([w], self.mwun, self.un)[0]) \
            if _convertible(self.mwun, self.un) else w

    def wl(self, w: float) -> float:
        """shdr_wl: a coordinate's logical pixel."""
        w1 = float(unitsmod.convert([w], self.un, self.mwun)[0]) \
            if _convertible(self.un, self.mwun) else w
        return self._logical(w1) - self.np1 + 1

    def set_coordinates(self) -> None:
        sn = self.sn
        self.w0 = float(f32(self.lw(1.0)))
        self.w1 = float(f32(self.lw(float(sn))))
        self.wp = float(f32(f32(f32(self.w1) - f32(self.w0)) / f32(sn - 1))) if sn > 1 \
            else 0.0
        self.x = np.array([self.lw(float(i)) for i in range(1, sn + 1)], f32)

    def set_system(self, system: str) -> None:
        """shdr_system: world or physical coordinates, the labels with them."""
        self.system = system
        if system == "physical":
            self.label, self.units = "Pixel", ""
            self.mwun = unitsmod.decode("")
        else:
            self.mwun = unitsmod.decode(self.smw.units)
            self.label, self.units = self.un.label, self.un.units
        self.set_coordinates()

    def set_units(self, units: str) -> None:
        """shdr_units: the coordinates in other units; 'display' and 'default' the
        world system's own."""
        if units == "display":
            un = unitsmod.decode(self.smw.units_display) if self.smw.units_display \
                else self.mwun
        elif units == "default":
            un = self.mwun
        else:
            un = unitsmod.decode(units)
        if un.kind == 0 or self.mwun.kind == 0:
            raise ShdrError("Cannot convert to specified units")
        self.un = un
        self.label, self.units = un.label, un.units
        self.set_coordinates()


def _convertible(a, b) -> bool:
    return a.kind != 0 and b.kind != 0 and not unitsmod._same(a, b)


def shdr_open(smw: Smw, data: np.ndarray, image: str, index1: int, index2: int,
              ap: int | None) -> Shdr:
    """shdr_open(..., SHDATA): the spectrum, its coordinates and header values.
    `data` is the image as numpy has it (axis 1 last), in real."""
    h = smw.header
    dc = _int(h, "DC-FLAG", DCNO)
    fc = _int(h, "CA-FLAG", FCNO)
    ec = _int(h, "EX-FLAG", -1)
    it = None
    for key in ("EXPOSURE", "ITIME", "EXPTIME"):
        if key in h:
            it = _real(h, key, it)
    units = smw.units
    if not units:
        lab = smw.label.lower()
        units = {"lambda": "angstroms", "freq": "hertz", "waveleng": "angstroms"}.get(
            lab, "m/s" if lab.startswith("velo") else "")
        if lab.startswith("velo"):
            units += " 21 centimeters"
    if not units and dc != DCNO:
        units = "Angstroms"
    smw.units = units
    mwun = unitsmod.decode(units) if units else unitsmod.Units("")
    bunit = str(h.get("BUNIT", ""))
    flabel, funits, fclass = _fun(bunit)
    if fclass != 0:
        fc = FCYES
    label = smw.label if smw.label not in ("equispec", "multispe") else ""
    sh = Shdr(smw, image, "", smw.title, label, units, flabel, funits, 0, 0,
              [None, None], [None, None], [1, 1], 1, 1, 1, dc, fc, ec, it,
              unitsmod.decode(units) if units else unitsmod.Units(""), mwun)
    if smw.format == ND:
        i, j = (max(1, min(smw.nspec, ap)), 1) if ap is not None else \
            (max(1, index1), max(1, index2))
        x = (i - 1) % smw.llen[1] + 1
        y = (i - 1) // smw.llen[1] + j
        sh.lindex = [max(1, min(smw.llen[1], x)), max(1, min(smw.llen[2], y))]
        sh.apindex = sh.lindex[0]
        sh.ap, sh.beam, sh.dc = i, 0, smw.dtype
        lo1 = hi1 = lo2 = hi2 = 1
        if smw.ldim > 1:
            lo1, hi1 = nd_limits(smw, x)
        if smw.ldim > 2:
            lo2 = y - (smw.nsum[1] - 1) // 2
            hi2 = min(smw.llen[2], lo2 + smw.nsum[1] - 1)
            lo2 = max(1, lo2)
        sh.aplow, sh.aphigh = [lo1, lo2], [hi1, hi2]
        sh.title = smw.title
        if smw.ldim == 2:
            fmt = "[*,{}]" if smw.laxis[0] == 1 else "[{},*]"
            sh.imsec = fmt.format(lo1) if lo1 == hi1 else \
                fmt.format(f"{lo1}:{hi1}")
        elif smw.ldim == 3:
            sh.imsec = _section3(smw.laxis[0], lo1, hi1, lo2, hi2)
        np1, np2 = 1, smw.nw
    else:
        if smw.format == ES or smw.pdim < 2 or smw.paxis[1] != 3:
            pin1 = max(1, min(smw.llen[1], index1))
            pin2 = max(1, min(smw.llen[2], index2))
            sh.lindex = [pin1, pin2]
            sh.apindex = pin1
        else:
            sh.lindex = [max(1, min(smw.llen[1], index2)), 1]
            sh.apindex = 1
        if ap is not None:
            for k in range(1, smw.nspec + 1):
                if smw.aps[k - 1] == ap:
                    sh.lindex[0] = sh.apindex = k
                    break
        k = sh.apindex
        sh.ap, sh.beam = smw.aps[k - 1], smw.beams[k - 1]
        if smw.format == MS:
            sh.spec = smw.specs[_physical_line(smw, k)]
            sh.dc = sh.spec.dtype
            sh.aplow = [sh.spec.aplow, None]
            sh.aphigh = [sh.spec.aphigh, None]
            nw = sh.spec.nw
        else:
            sh.dc = smw.dtype
            sh.aplow, sh.aphigh = list(smw.aplows[k - 1]), list(smw.aphighs[k - 1])
            nw = smw.nw
        sh.title = smw.apids.get(k, smw.title)
        if smw.ldim == 2:
            sh.imsec = f"[*,{sh.apindex}]"
        elif smw.ldim == 3:
            sh.imsec = f"[*,{sh.apindex},{sh.lindex[1]}]"
        np1, np2 = 1, nw
    # NP1 and NP2 in logical pixels
    ltm, ltv = smw.lterm
    to_log = _single(*_compile(ltm, ltv, None, None, None, smw.paxis[0], False)[1])
    a = max(1, min(int(to_log(float(np1))), smw.llen[0]))
    b = max(1, min(int(to_log(float(np2))), smw.llen[0]))
    sh.np1, sh.np2 = min(a, b), max(a, b)
    sh.set_coordinates()
    sh.y = _data(smw, data, sh)
    if sh.fc == FCYES and fclass == 0:
        mid = sh.y[sh.sn // 2]
        guess = "erg/cm2/s/Hz" if mid < 1e-18 else ("erg/cm2/s/A" if mid < 1e-5 else "")
        sh.flabel, sh.funits, _ = _fun(guess)
    return sh


def _section3(laxis, lo1, hi1, lo2, hi2) -> str:
    a = str(lo1) if lo1 == hi1 else f"{lo1}:{hi1}"
    b = str(lo2) if lo2 == hi2 else f"{lo2}:{hi2}"
    return {1: f"[*,{a},{b}]", 2: f"[{a},*,{b}]", 3: f"[{a},{b},*]"}[laxis]


def _data(smw: Smw, data: np.ndarray, sh: Shdr) -> np.ndarray:
    """The spectrum's pixels: an NDSPEC's sum in real, a line of the others."""
    arr = np.asarray(data, f32)
    while arr.ndim < 3:
        arr = arr[None]
    np1, np2 = sh.np1, sh.np2
    if smw.format != ND:
        return arr[sh.lindex[1] - 1, sh.lindex[0] - 1, np1 - 1:np2].astype(f32).copy()
    out = np.zeros(np2 - np1 + 1, f32)
    lo1, hi1 = sh.aplow[0], sh.aphigh[0]
    lo2, hi2 = sh.aplow[1], sh.aphigh[1]
    la = smw.laxis[0]
    if la == 1:
        for j in range(lo2, hi2 + 1):
            for i in range(lo1, hi1 + 1):
                out = (out + arr[j - 1, i - 1, np1 - 1:np2]).astype(f32)
    elif la == 2:
        for j in range(lo2, hi2 + 1):
            for i in range(np1, np2 + 1):
                s = f32(0)
                for v in arr[j - 1, i - 1, lo1 - 1:hi1]:
                    s = f32(s + v)
                out[i - np1] = f32(out[i - np1] + s)
    else:
        for i in range(np1, np2 + 1):
            for j in range(lo2, hi2 + 1):
                s = f32(0)
                for v in arr[i - 1, j - 1, lo1 - 1:hi1]:
                    s = f32(s + v)
                out[i - np1] = f32(out[i - np1] + s)
    return out


def _fun(text: str) -> tuple[str, str, int]:
    """fun_decode: the flux label, units and class (0 unknown)."""
    names = ("jansky", "fu", "erg/cm2/s/hz", "erg/cm2/s/a")
    classes = (1, 1, 1, 2)
    words = text.lower().split()
    mod = None
    kind = 0
    for word in words[:2]:
        found = [i for i, n in enumerate(names + ("log", "mag"), 1) if n.startswith(word)]
        if len(found) != 1:
            break
        j = found[0]
        if j > 4:
            if mod is not None:
                break
            mod = j - 4
        else:
            kind = j
            break
    if kind == 0:
        return "", text, 0
    label = {1: "Log ", 2: "Mag "}.get(mod, "") + "Flux"
    return label, names[kind - 1], classes[kind - 1]


# ------------------------------------------------------- harwood's spectra

def from_harwood(path, header=None) -> tuple[Smw, np.ndarray]:
    """harwood's spectrum file -- a table per aperture -- as the multispec image it
    stands for: one line per aperture, its wavelengths as a linear, log-linear or
    (a pixel table) nonlinear dispersion, APZ its redshift."""
    from astropy.io import fits

    from . import spectrum

    apertures = spectrum.read(path)
    if header is None:
        with fits.open(str(path), memmap=False) as hdul:
            header = hdul[0].header.copy()
    npix = max(len(a.flux) for a in apertures)
    data = np.zeros((len(apertures), npix), f32)
    smw = Smw(MS, header, (npix, len(apertures), 1),
              str(header.get("OBJECT", "")).strip(), 2, 2, [1, 2, 3], [1, 2, 3],
              [npix, len(apertures), 1], [1, 1], lterm=(np.eye(2), np.zeros(2)))
    smw.nspec = len(apertures)
    units = ""
    for k, ap in enumerate(apertures, start=1):
        flux = np.asarray(ap.flux, f32)
        data[k - 1, :flux.size] = flux
        n = flux.size
        z = float(ap.redshift or 0.0)
        if ap.wavelength is None:
            spec = Spec(ap.number, ap.beam or ap.number, DCNO, 1.0, 1.0, n, 0.0,
                        float(f32(ap.centre + ap.low)), float(f32(ap.centre + ap.high)))
        else:
            wave = np.asarray(ap.wavelength, float) * (1 + z)
            dc = spectrum.dispersion_type(ap.wavelength)
            if dc == 0:
                spec = Spec(ap.number, ap.beam or ap.number, DCLINEAR, float(wave[0]),
                            float((wave[-1] - wave[0]) / (n - 1)), n, z,
                            float(f32(ap.centre + ap.low)), float(f32(ap.centre + ap.high)))
            elif dc == 1:
                lw = np.log10(wave)
                spec = Spec(ap.number, ap.beam or ap.number, DCLOG, float(lw[0]),
                            float((lw[-1] - lw[0]) / (n - 1)), n, z,
                            float(f32(ap.centre + ap.low)), float(f32(ap.centre + ap.high)))
            else:
                table = " ".join(repr(float(w)) for w in wave)
                spec = Spec(ap.number, ap.beam or ap.number, DCFUNC, float(wave[0]),
                            float((wave[-1] - wave[0]) / (n - 1)), n, z,
                            float(f32(ap.centre + ap.low)), float(f32(ap.centre + ap.high)),
                            f"1. 0. {PIXEL} {n} {table}")
                spec.function = msp_coeff(spec.coeff, 0.5, n + 0.5)
                spec.state = [float(n), msp_eval(spec.function, float(n))
                              - msp_eval(spec.function, float(n) - 1)]
            units = ap.units or "angstroms"
        smw.specs[k] = spec
        smw.aps.append(spec.ap)
        smw.beams.append(spec.beam)
        if ap.title:
            smw.apids[k] = ap.title
    smw.units = units
    smw.label = "Wavelength" if units else ""
    return smw, data
