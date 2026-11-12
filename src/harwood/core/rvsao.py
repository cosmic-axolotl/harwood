"""What RVSAO's tasks share: the spectrum as getspec reads it, its header values as
imgdpar and imgdate read them, the time of the observation (juldate), the
barycentric and heliocentric velocity of the observer (compbcv and bcv.f, Stumpff's
BARVEL), and the wavelength-to-pixel conversions of wlcon.x.

RVSAO 2.8.5 (Smithsonian Astrophysical Observatory: Doug Mink, Michael Kurtz,
Jessica Mink) is an IRAF external package; harwood keeps its arithmetic and its
defects, as for IRAF's own tasks, and runs it as the gabarito. Defects kept:

* wcs_w2p clamps the wavelength to the spectrum's range and then converts the
  wavelength it was given: the clamp does nothing.
* getimage writes RA "00:00:00.000" and DEC "00:00:00.00" into a header without
  them, so a spectrum with no position is at 0h 0d to every later step.
* The fourth word of an APIDn card sets the catalogue number and is then
  overwritten with -10, so the name is always the card's first word.
* compbcv returns a REAL: the velocity correction is rounded to single precision.

Translated from RVSAO, Copyright (c) 1991-2022 Smithsonian Astrophysical
Observatory, whose notice is in LICENSES/RVSAO-COPYRIGHT.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import evexpr, iraffmt

f32 = np.float32
INDEFD = 1.6e308
C0 = 299792.5               # RVSAO's speed of light, km/s
VERSION = "2.8.5"
MIN_PIXEL_VALUE = -1000000
LINES = Path(__file__).resolve().parent.parent / "data" / "rvsao"

#: vel_corr's dictionary (HC_VTYPES) and its codes.
VEL_CORR = ("none", "file", "heliocentric", "barycentric", "hfile")
NONE, FBCV, HCV, BCV, FHCV = 1, 2, 3, 4, 5


class RvsaoError(RuntimeError):
    """What stops an RVSAO task: its message."""


# ------------------------------------------------------------- reading text


class Scan:
    """sscan with gargd, gargi and gargwrd: a field at a time, nscan counting up to
    the first that cannot be read."""

    def __init__(self, line: str):
        self.line, self.ip, self.n, self.stop = line, 0, 0, False

    def _skip(self) -> None:
        while self.ip < len(self.line) and self.line[self.ip] in " \t":
            self.ip += 1

    def d(self) -> float | None:
        if self.stop:
            return None
        self._skip()
        kind, length = evexpr.lexnum(self.line, self.ip)
        if kind == evexpr.NON or length == 0:
            self.stop = True
            return None
        text = self.line[self.ip:self.ip + length]
        self.ip += length
        self.n += 1
        if text.startswith("INDEF"):
            return INDEFD
        if kind == evexpr.OCT:
            return float(int(text.rstrip("bB"), 8))
        if kind == evexpr.HEX:
            return float(int(text.rstrip("xX"), 16))
        if kind == evexpr.DEC:
            return float(int(text))
        value = iraffmt.ctod(text)
        return INDEFD if value is None else value

    def i(self) -> int | None:
        value = self.d()
        return None if value is None else int(value)

    def wrd(self, size: int = 1023) -> str | None:
        if self.stop:
            return None
        self._skip()
        rest = self.line[self.ip:]
        if not rest.strip():
            self.stop = True
            return None
        word = rest.split()[0]
        self.ip += len(word)
        self.n += 1
        return word[:size]


def line_file(name: str, linedir: str) -> Path:
    """eminit's path: a name with / or $ is a path, else in linedir (rvsao$lib/,
    which is harwood's copy of RVSAO's line lists)."""
    if "/" in name or "$" in name:
        if name.startswith("rvsao$lib/"):
            return LINES / name[len("rvsao$lib/"):]
        return Path(name)
    if linedir in ("rvsao$lib/", "rvsao$lib", ""):
        return LINES / name if linedir else Path(name)
    return Path(linedir.rstrip("/")) / name


# ------------------------------------------------------------- header values


class Header:
    """The image header as RVSAO reads and edits it: imgdpar and imgspar on the
    text of the cards, imaddX writing to the copy in memory only."""

    def __init__(self, header):
        self.h = header

    def has(self, key: str) -> bool:
        return key.strip() in self.h

    def string(self, key: str) -> str | None:
        key = key.strip()
        if key not in self.h:
            return None
        value = self.h[key]
        if isinstance(value, bool):
            return "T" if value else "F"
        if isinstance(value, str):
            return value
        card = self.h.cards[key]
        image = card.image
        if not getattr(card, "_modified", True) and len(image) > 10:
            return image[10:].split("/", 1)[0].strip()
        return repr(value) if isinstance(value, float) else str(value)

    def d(self, key: str, default: float) -> float:
        """imgdpar: ctod of the card's text, the default when there is none."""
        text = self.string(key)
        if text is None:
            return default
        value = iraffmt.ctod(text.strip())
        if value is None:
            m = re.match(r"\s*[+-]?[\d.:]+(?:[eEdD][+-]?\d+)?", text)
            value = iraffmt.ctod(m.group(0)) if m else None
        return default if value is None else value

    def i(self, key: str, default: int) -> int:
        text = self.string(key)
        if text is None:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default

    def b(self, key: str, default: bool) -> bool:
        text = self.string(key)
        if text is None:
            return default
        return text.strip()[:1] in ("T", "t", "Y", "y")

    def put(self, key: str, value) -> None:
        self.h[key] = value


def imgdate(head: Header, key: str):
    """imgdate: (mm, dd, yyyy) from yyyy-mm-dd[Thh:mm:ss], dd-mm-yy or dd/mm/yy; None
    for each part it cannot read."""
    text = head.string(key)
    mm = dd = yyyy = None
    if text is None or len(text) <= 7:
        return mm, dd, yyyy

    def ctoi(s: str) -> int | None:
        m = re.match(r"\s*([+-]?\d+)", s)
        return int(m.group(1)) if m else None

    if "-" in text:
        ix1 = text.index("-")
        dd = 0
        j = ctoi(text[:ix1])
        if j is not None:
            if j > 31:
                yyyy = j
            else:
                dd = j
        else:
            yyyy = 0
        ix2 = text.rindex("-")
        j = ctoi(text[ix1 + 1:ix2])
        mm = j if j is not None else 0
        tee = text.rfind("T")
        rest = text[ix2 + 1:tee] if tee >= 0 else text[ix2 + 1:]
        j = ctoi(rest)
        if j is not None:
            if dd > 0:
                yyyy = j + 1900 if j < 120 else j
            else:
                dd = j
        else:
            dd = 0
    elif "/" in text:
        ix1 = text.index("/")
        j = ctoi(text[:ix1])
        dd = j if j is not None else 0
        ix2 = text.rindex("/")
        j = ctoi(text[ix1 + 1:ix2])
        mm = j if j is not None else 0
        j = ctoi(text[ix2 + 1:ix2 + 5])
        yyyy = (2000 + j if j < 20 else 1900 + j if j < 120 else j) if j is not None else 0
    return mm, dd, yyyy


def juldate(head: Header):
    """juldate: the mid-exposure UT in hours, the Julian date and the heliocentric
    Julian date; 0 where they cannot be found."""
    jd = hjd = 0.0
    mm = dd = yyyy = None
    if head.has("DATE-OBS"):
        mm, dd, yyyy = imgdate(head, "DATE-OBS")
    elif head.has("DATE"):
        mm, dd, yyyy = imgdate(head, "DATE")
    ut = head.d("UTMID", INDEFD)
    if ut == INDEFD:
        ut1 = head.d("UTOPEN", INDEFD)
        if ut1 == INDEFD:
            ut1 = head.d("UTSTART", INDEFD)
        if ut1 == INDEFD:
            ut1 = head.d("UT", INDEFD)
        ut2 = head.d("UTEND", INDEFD)
        exp = head.d("EXPOSURE", 0.0)
        if exp == 0.0:
            exp = head.d("EXPTIME", exp)
        if ut1 == INDEFD:
            ut = ut2 - (exp * 0.5 / 3600.0) if ut2 != INDEFD else 0.0
        elif ut2 != INDEFD:
            ut = (ut1 + ut2) * 0.5
        else:
            ut = ut1 + (exp * 0.5 / 3600.0)
    hjd = head.d("HJDN", 0.0)
    if hjd == 0.0:
        hjd = head.d("HJD", hjd)
    if hjd == 0.0:
        hjd = head.d("BJD", hjd)
    jd = head.d("GJDN", 0.0)
    if jd == 0.0:
        jd = head.d("GJD", jd)
    if jd == 0.0:
        jd = head.d("MJD-OBS", 0.0)
        if jd == 0.0:
            jd = head.d("MJD", jd)
        if jd != 0.0:
            jd = jd + 2400000.5
    if jd == 0.0:
        if yyyy is None or mm is None or dd is None or yyyy < 0:
            return ut, jd, hjd, (mm, dd, yyyy)

        def idiv(a: int, b: int) -> int:
            q = abs(a) // abs(b)
            return q if (a >= 0) == (b >= 0) else -q
        jday = (idiv(1461 * (yyyy + 4800 + idiv(mm - 14, 12)), 4)
                + idiv(367 * (mm - 2 - 12 * idiv(mm - 14, 12)), 12)
                - idiv(3 * idiv(yyyy + 4900 + idiv(mm - 14, 12), 100), 4)
                + dd - 32075)
        jd = float(jday) - 0.5 + (ut / 24.0)
    if hjd != 0.0:
        return ut, jd, hjd, (mm, dd, yyyy)
    ra = head.d("RA", INDEFD)
    dec = head.d("DEC", INDEFD)
    hjd = jd2hjd(ra, dec, jd)[1] if ra != INDEFD and dec != INDEFD else jd
    return ut, jd, hjd, (mm, dd, yyyy)


def jd2hjd(ra: float, dec: float, jd: float):
    """jd2hjd: the light time to the Sun (days) and the heliocentric Julian date."""
    t = (jd - 2415020.0) / 36525.0
    manom = 358.47583 + t * (35999.04975 - t * (0.000150 + t * 0.000003))
    lperi = 101.22083 + t * (1.7191733 + t * (0.000453 + t * 0.000003))
    oblq = 23.452294 - t * (0.0130125 + t * (0.00000164 - t * 0.000000503))
    eccen = 0.01675104 - t * (0.00004180 + t * 0.000000126)
    manom = math.fmod(manom, 360.0)
    lperi = math.fmod(lperi, 360.0)
    rad = math.pi / 180.0
    r, d = ra * 15 * rad, dec * rad
    manom, lperi, oblq = manom * rad, lperi * rad, oblq * rad
    tanom = (manom + (2 * eccen - 0.25 * eccen ** 3) * math.sin(manom)
             + 1.25 * eccen ** 2 * math.sin(2 * manom)
             + 13.0 / 12.0 * eccen ** 3 * math.sin(3 * manom))
    slong = lperi + tanom + math.pi
    lon, b = concrd(0.0, 0.0, -math.pi / 2, math.pi / 2 - oblq, r, d)
    rsun = (1.0 - eccen ** 2) / (1.0 + eccen * math.cos(tanom))
    lt = -0.005770 * rsun * math.cos(b) * math.cos(lon - slong)
    return lt, jd + lt


def concrd(ao, bo, ap, bp, a1, b1):
    x = math.cos(a1) * math.cos(b1)
    y = math.sin(a1) * math.cos(b1)
    z = math.sin(b1)
    xp = math.cos(ap) * math.cos(bp)
    yp = math.sin(ap) * math.cos(bp)
    zp = math.sin(bp)
    sao, cao, sbo, cbo = math.sin(ao), math.cos(ao), math.sin(bo), math.cos(bo)
    xp, yp = xp * cao + yp * sao, -xp * sao + yp * cao
    x, y = x * cao + y * sao, -x * sao + y * cao
    xp, zp = xp * cbo + zp * sbo, -xp * sbo + zp * cbo
    x, z = x * cbo + z * sbo, -x * sbo + z * cbo
    sbp, cbp = zp, yp
    y, z = y * sbp - z * cbp, y * cbp + z * sbp
    return math.atan2(y, x), math.asin(z)


# ------------------------------------------------------------- bcv.f


def _dmod(a: float, b: float) -> float:
    return math.fmod(a, b)


def sidtim(djd: float, dlong: float) -> float:
    dtpi = 2.0 * 3.141592653589793
    df = 1.00273790934
    djd0 = int(djd) + 0.5
    if djd0 > djd:
        djd0 -= 1.0
    dut = (djd - djd0) * dtpi
    dt = (djd0 - 2415020.0) / 36525.0
    dst0 = 1.739935934667999 + 6.283319509909095e02 * dt + 6.755878646261384e-06 * dt * dt
    dst0 = _dmod(dst0, dtpi)
    dst = df * dut + dst0 - dlong
    return _dmod(dst + 2.0 * dtpi, dtpi)


def pre(deq1: float, deq2: float):
    """Newcomb's precession matrix, from equinox deq1 to deq2 (Fortran order [k][j])."""
    dcsar = 4.848136812e-6
    dt0 = (deq1 - 1900.0) * 0.01
    dt = (deq2 - deq1) * 0.01
    dts = dt * dt
    dtc = dts * dt
    dzeta = ((2304.25 + 1.396 * dt0) * dt + 0.302 * dts + 0.018 * dtc) * dcsar
    dzett = dzeta + 0.791 * dts * dcsar
    dthet = ((2004.683 + -0.853 * dt0) * dt + -0.426 * dts + -0.042 * dtc) * dcsar
    sz, cz = math.sin(dzeta), math.cos(dzeta)
    szt, czt = math.sin(dzett), math.cos(dzett)
    st, ct = math.sin(dthet), math.cos(dthet)
    da, db, dc, dd = sz * szt, cz * szt, sz * czt, cz * czt
    return [[dd * ct - da, -1.0 * dc * ct - db, -1.0 * st * czt],
            [db * ct + dc, -1.0 * da * ct + dd, -1.0 * st * szt],
            [cz * st, -1.0 * sz * st, ct]]


def geovel(dphi, dh, dec, dha) -> float:
    da, dfl, dw = 6378.140, 0.00335281, 7.2921158554e-05
    de2 = dfl * (2.0 - dfl)
    d1 = 1.0 - de2 * (2.0 - de2) * math.sin(dphi) ** 2
    d2 = 1.0 - de2 * math.sin(dphi) ** 2
    dr0 = da * math.sqrt(d1 / d2)
    d1 = de2 * math.sin(2.0 * dphi)
    d2 = 2.0 * d2
    dphig = dphi - math.atan(d1 / d2)
    drh = dr0 * math.cos(dphig) + dh / 1.0e3 * math.cos(dphi)
    return dw * drh * math.cos(dec) * math.sin(dha)


_DCFEL = ((1.7400353e+00, 6.2833195099091e+02, 5.2796e-06),
          (6.2565836e+00, 6.2830194572674e+02, -2.6180e-06),
          (4.7199666e+00, 8.3997091449254e+03, -1.9780e-05),
          (1.9636505e-01, 8.4334662911720e+03, -5.6044e-05),
          (4.1547339e+00, 5.2993466764997e+01, 5.8845e-06),
          (4.6524223e+00, 2.1354275911213e+01, 5.6797e-06),
          (4.2620486e+00, 7.5025342197656e+00, 5.5317e-06),
          (1.4740694e+00, 3.8377331909193e+00, 5.6093e-06))
_DCEPS = (4.093198e-01, -2.271110e-04, -2.860401e-08)
_CCSEL = ((1.675104e-02, -4.179579e-05, -1.260516e-07),
          (2.220221e-01, 2.809917e-02, 1.852532e-05),
          (1.589963e+00, 3.418075e-02, 1.430200e-05),
          (2.994089e+00, 2.590824e-02, 4.155840e-06),
          (8.155457e-01, 2.486352e-02, 6.836840e-06),
          (1.735614e+00, 1.763719e-02, 6.370440e-06),
          (1.968564e+00, 1.524020e-02, -2.517152e-06),
          (1.282417e+00, 8.703393e-03, 2.289292e-05),
          (2.280820e+00, 1.918010e-02, 4.484520e-06),
          (4.833473e-02, 1.641773e-04, -4.654200e-07),
          (5.589232e-02, -3.455092e-04, -7.388560e-07),
          (4.634443e-02, -2.658234e-05, 7.757000e-08),
          (8.997041e-03, 6.329728e-06, -1.939256e-09),
          (2.284178e-02, -9.941590e-05, 6.787400e-08),
          (4.350267e-02, -6.839749e-05, -2.714956e-07),
          (1.348204e-02, 1.091504e-05, 6.903760e-07),
          (3.106570e-02, -1.665665e-04, -1.590188e-07))
_DCARGS = ((5.0974222e+00, -7.8604195454652e+02), (3.9584962e+00, -5.7533848094674e+02),
           (1.6338070e+00, -1.1506769618935e+03), (2.5487111e+00, -3.9302097727326e+02),
           (4.9255514e+00, -5.8849265665348e+02), (1.3363463e+00, -5.5076098609303e+02),
           (1.6072053e+00, -5.2237501616674e+02), (1.3629480e+00, -1.1790629318198e+03),
           (5.5657014e+00, -1.0977134971135e+03), (5.0708205e+00, -1.5774000881978e+02),
           (3.9318944e+00, 5.2963464780000e+01), (4.8989497e+00, 3.9809289073258e+01),
           (1.3097446e+00, 7.7540959633708e+01), (3.5147141e+00, 7.9618578146517e+01),
           (3.5413158e+00, -5.4868336758022e+02))
_CCAMPS = ((-2.279594e-5, 1.407414e-5, 8.273188e-6, 1.340565e-5, -2.490817e-7),
           (-3.494537e-5, 2.860401e-7, 1.289448e-7, 1.627237e-5, -1.823138e-7),
           (6.593466e-7, 1.322572e-5, 9.258695e-6, -4.674248e-7, -3.646275e-7),
           (1.140767e-5, -2.049792e-5, -4.747930e-6, -2.638763e-6, -1.245408e-7),
           (9.516893e-6, -2.748894e-6, -1.319381e-6, -4.549908e-6, -1.864821e-7),
           (7.310990e-6, -1.924710e-6, -8.772849e-7, -3.334143e-6, -1.745256e-7),
           (-2.603449e-6, 7.359472e-6, 3.168357e-6, 1.119056e-6, -1.655307e-7),
           (-3.228859e-6, 1.308997e-7, 1.013137e-7, 2.403899e-6, -3.736225e-7),
           (3.442177e-7, 2.671323e-6, 1.832858e-6, -2.394688e-7, -3.478444e-7),
           (8.702406e-6, -8.421214e-6, -1.372341e-6, -1.455234e-6, -4.998479e-8),
           (-1.488378e-6, -1.251789e-5, 5.226868e-7, -2.049301e-7, 0.0),
           (-8.043059e-6, -2.991300e-6, 1.473654e-7, -3.154542e-7, 0.0),
           (3.699128e-6, -3.316126e-6, 2.901257e-7, 3.407826e-7, 0.0),
           (2.550120e-6, -1.241123e-6, 9.901116e-8, 2.210482e-7, 0.0),
           (-6.351059e-7, 2.341650e-6, 1.061492e-6, 2.878231e-7, 0.0))
_CCSEC = ((1.289600e-06, 5.550147e-01, 2.076942e+00),
          (3.102810e-05, 4.035027e+00, 3.525565e-01),
          (9.124190e-06, 9.990265e-01, 2.622706e+00),
          (9.793240e-07, 5.508259e+00, 1.559103e+01))
_DCARGM = ((5.1679830e+00, 8.3286911095275e+03), (5.4913150e+00, -7.2140632838100e+03),
           (5.9598530e+00, 1.5542754389685e+04))
_CCAMPM = ((1.097594e-01, 2.896773e-07, 5.450474e-02, 1.438491e-07),
           (-2.223581e-02, 5.083103e-08, 1.002548e-02, -2.291823e-08),
           (1.148966e-02, 5.658888e-08, 8.249439e-03, 4.063015e-08))
_CCPAMV = (8.326827e-11, 1.843484e-11, 1.988712e-12, 1.881276e-12)


def barvel(dje: float, deq: float):
    """Stumpff's BARVEL: the Earth's heliocentric and barycentric velocity (AU/s)
    in equatorial coordinates of equinox deq (0: of the date)."""
    dc2pi, cc2pi = 6.2831853071796, 6.283185
    ccsec3, dcsld, ccsgd = -7.757020e-08, 1.990987e-07, 1.990969e-07
    cckm, ccmld, ccfdi, dc1mme = 3.122140e-05, 2.661699e-06, 2.399485e-07, 0.99999696
    ideq = int(deq)
    dt = (dje - 2415020.0) / 36525.0
    t, dtsq = dt, dt * dt
    tsq = dtsq
    forbel = [0.0] * 7
    dml = 0.0
    for k in range(8):
        dlocal = _dmod(_DCFEL[k][0] + dt * _DCFEL[k][1] + dtsq * _DCFEL[k][2], dc2pi)
        if k == 0:
            dml = dlocal
        else:
            forbel[k - 1] = dlocal
    deps = _dmod(_DCEPS[0] + dt * _DCEPS[1] + dtsq * _DCEPS[2], dc2pi)
    sorbel = [_dmod(c[0] + t * c[1] + tsq * c[2], dc2pi) for c in _CCSEL]
    e, g = sorbel[0], forbel[0]
    sn = [math.sin(_dmod(c[1] + t * c[2], dc2pi)) for c in _CCSEC]
    pertl = (_CCSEC[0][0] * sn[0] + _CCSEC[1][0] * sn[1]
             + (_CCSEC[2][0] + t * ccsec3) * sn[2] + _CCSEC[3][0] * sn[3])
    pertld = pertr = pertrd = 0.0
    for k in range(15):
        a = _dmod(_DCARGS[k][0] + dt * _DCARGS[k][1], dc2pi)
        cosa, sina = math.cos(a), math.sin(a)
        c = _CCAMPS[k]
        pertl = pertl + c[0] * cosa + c[1] * sina
        pertr = pertr + c[2] * cosa + c[3] * sina
        if k >= 10:
            continue
        pertld = pertld + (c[1] * cosa - c[0] * sina) * c[4]
        pertrd = pertrd + (c[3] * cosa - c[2] * sina) * c[4]
    esq = e * e
    dparam = 1.0 - esq
    param = dparam
    twoe, twog = e + e, g + g
    phi = twoe * ((1.0 - esq * (1.0 / 8.0)) * math.sin(g) + e * (5.0 / 8.0) * math.sin(twog)
                  + esq * 0.5416667 * math.sin(g + twog))
    f = g + phi
    sinf, cosf = math.sin(f), math.cos(f)
    dpsi = dparam / (1.0 + e * cosf)
    phid = twoe * ccsgd * ((1.0 + esq * 1.5) * cosf + e * (1.25 - sinf * sinf * 0.5))
    psid = ccsgd * e * sinf / math.sqrt(param)
    d1pdro = 1.0 + pertr
    drd = d1pdro * (psid + dpsi * pertrd)
    drld = d1pdro * dpsi * (dcsld + phid + pertld)
    dtl = _dmod(dml + phi + pertl, dc2pi)
    dsinls, dcosls = math.sin(dtl), math.cos(dtl)
    dxhd = drd * dcosls - drld * dsinls
    dyhd = drd * dsinls + drld * dcosls
    pertl = pertld = pertp = pertpd = 0.0
    for k in range(3):
        a = _dmod(_DCARGM[k][0] + dt * _DCARGM[k][1], dc2pi)
        sina, cosa = math.sin(a), math.cos(a)
        c = _CCAMPM[k]
        pertl = pertl + c[0] * sina
        pertld = pertld + c[1] * cosa
        pertp = pertp + c[2] * cosa
        pertpd = pertpd - c[3] * sina
    tl = forbel[1] + pertl
    sinlm, coslm = math.sin(tl), math.cos(tl)
    sigma = cckm / (1.0 + pertp)
    a = sigma * (ccmld + pertld)
    b = sigma * pertpd
    dxhd = dxhd + a * sinlm + b * coslm
    dyhd = dyhd - a * coslm + b * sinlm
    dzhd = -sigma * ccfdi * math.cos(forbel[2])
    dxbd, dybd, dzbd = dxhd * dc1mme, dyhd * dc1mme, dzhd * dc1mme
    for k in range(4):
        plon = forbel[k + 3]
        pomg = sorbel[k + 1]
        pecc = sorbel[k + 9]
        tl = _dmod(plon + 2.0 * pecc * math.sin(plon - pomg), cc2pi)
        dxbd = dxbd + _CCPAMV[k] * (math.sin(tl) + pecc * math.sin(pomg))
        dybd = dybd - _CCPAMV[k] * (math.cos(tl) + pecc * math.cos(pomg))
        dzbd = dzbd - _CCPAMV[k] * sorbel[k + 13] * math.cos(plon - sorbel[k + 5])
    dcosep, dsinep = math.cos(deps), math.sin(deps)
    dyahd = dcosep * dyhd - dsinep * dzhd
    dzahd = dsinep * dyhd + dcosep * dzhd
    dyabd = dcosep * dybd - dsinep * dzbd
    dzabd = dsinep * dybd + dcosep * dzbd
    if ideq == 0:
        return [dxhd, dyahd, dzahd], [dxbd, dyabd, dzabd]
    deqdat = (dje - 2415020.0 - 0.313) / 365.24219572 + 1900.0
    p = pre(deqdat, deq)
    velh = [dxhd * p[n][0] + dyahd * p[n][1] + dzahd * p[n][2] for n in range(3)]
    velb = [dxbd * p[n][0] + dyabd * p[n][1] + dzabd * p[n][2] for n in range(3)]
    return velh, velb


def bcv(djd, dlong, dlat, dalt, dra, dec, deq):
    """bcv.f's BCV: the barycentric, heliocentric and geocentric (diurnal) velocity
    corrections, km/s, toward (dra hours, dec degrees, equinox deq)."""
    dpi = 3.1415926535897932
    daukm = 1.4959787e08
    dtr = dpi / 180.0
    dlongs, dlats = dlong * dtr, dlat * dtr
    dras, decs = dra * 15.0 * dtr, dec * dtr
    dst = sidtim(djd, dlongs)
    deqt = (djd - 2415020.0 - 0.313) / 365.24219572 + 1900.0
    dc = (math.cos(dras) * math.cos(decs), math.sin(dras) * math.cos(decs), math.sin(decs))
    p = pre(deq, deqt)
    dcc = [dc[0] * p[k][0] + dc[1] * p[k][1] + dc[2] * p[k][2] for k in range(3)]
    if dcc[0] != 0.0:
        dra2 = math.atan(dcc[1] / dcc[0])
        if dcc[0] < 0.0:
            dra2 += dpi
        elif dcc[1] < 0.0:
            dra2 += 2.0 * dpi
    else:
        dra2 = dpi / 2.0 if dcc[1] > 0.0 else 1.5 * dpi
    dec2 = math.asin(dcc[2])
    dha = dst - dra2
    gcvel = geovel(dlats, dalt, dec2, -dha)
    velh, velb = barvel(djd, deqt)
    bcvel = hcvel = 0.0
    for k in range(3):
        bcvel = bcvel + velb[k] * dcc[k] * daukm
        hcvel = hcvel + velh[k] * dcc[k] * daukm
    return bcvel, hcvel, gcvel


def compbcv(head: Header, barycentric: bool) -> float:
    """compbcv: the velocity correction of the spectrum, from its header (RA, DEC,
    EPOCH or EQUINOX, SITELONG/SITELAT/SITEELEV or OBS-*, or OBSERVAT's), 0 when any
    is missing; in REAL, as compbcv returns it."""
    from . import observatory as obsmod

    ut, dj, hdj, _ = juldate(head)
    if dj == 0.0:
        dj = hdj
    ra = head.d("RA", INDEFD)
    if ra != INDEFD:
        text = head.string("RA") or ""
        if ":" not in text:
            ra = ra / 15.0
    dec = head.d("DEC", INDEFD)
    epoch = head.d("EPOCH", 1950.0)
    epoch = head.d("EQUINOX", epoch)
    dlong = head.d("SITELONG", INDEFD)
    if dlong == INDEFD:
        dlong = head.d("OBS-LONG", INDEFD)
    dlat = head.d("SITELAT", INDEFD)
    if dlat == INDEFD:
        dlat = head.d("OBS-LAT", INDEFD)
    dalt = head.d("SITEELEV", INDEFD)
    if dalt == INDEFD:
        dalt = head.d("OBS-ELEV", INDEFD)
    if dlong == INDEFD:
        name = (head.string("OBSERVAT") or "").strip()
        if name:
            site = obsmod.find(name)        # obsvopen(name, NO): no site, no correction
            if site is not None:
                dlat, dlong, dalt = site.latitude, site.longitude, site.altitude
    dbcv = dhcv = dgcv = 0.0
    if ra != INDEFD and dec != INDEFD and dj != 0.0 and INDEFD not in (dlong, dlat, dalt):
        dbcv, dhcv, dgcv = bcv(dj, dlong, dlat, dalt, ra, dec, epoch)
    return float(f32(dbcv + dgcv)) if barycentric else float(f32(dhcv + dgcv))


# ------------------------------------------------------------- the spectrum


@dataclass
class Wcs:
    """wlcon.x's conversions on one spectrum: shdr_lw and shdr_wl, the pixels
    clamped to the spectrum (the wavelengths not, see wcs_w2p)."""

    sh: object
    pxshift: float = 0.0
    px1: float = 0.0
    px2: float = 0.0
    wl1: float = 0.0
    wl2: float = 0.0

    def __post_init__(self) -> None:
        self.px1 = float(self.sh.np1) - 0.5
        self.px2 = float(self.sh.np2) + 0.5
        self.wl1, self.wl2 = self.sh.lw(self.px1), self.sh.lw(self.px2)
        if self.wl1 > self.wl2:
            self.wl1, self.wl2 = self.wl2, self.wl1

    def w2p(self, wl: float) -> float:
        return self.sh.wl(wl) + self.pxshift

    def p2w(self, px: float) -> float:
        tpx = min(max(px, self.px1), self.px2) - self.pxshift
        return self.sh.lw(tpx)

    def p2l(self, px: float) -> float:
        return math.log10(self.p2w(px))


@dataclass
class Spectrum:
    """getspec's result: the pixels (real; the ends below MIN_PIXEL_VALUE zeroed and
    the gaps between filled), the header as RVSAO left it, the name, the WCS."""

    name: str              #: specid: the file's name, with [ap] for a multispec
    object: str            #: specname: the title, or the APID's first word
    data: np.ndarray
    head: Header
    sh: object
    wcs: Wcs
    mspec: int = 0
    hcv: float = 0.0       #: spechcv
    vb: bool = False       #: specvb: the correction is barycentric
    messages: list[str] = field(default_factory=list)


def _imgspar(head: Header, key: str) -> str:
    text = head.string(key)
    return "" if text is None else text.strip()


def getspec(path: str, mspec: int, band: int, svcor: int,
            state: dict | None = None) -> Spectrum:
    """getspec and getimage: the spectrum, its header defaults, and the velocity
    correction vel_corr asks for. `state` carries getimage's static buffer from one
    spectrum to the next: it ends each call holding "APIDn", so after the first
    spectrum of a run a header without RA, DEC, HA or ST is not given them."""
    if state is None:
        state = {}
    from . import shdr, splot

    image = splot.open_image(path, 1, (1, 1))
    head = Header(image.header.copy() if hasattr(image.header, "copy") else image.header)
    smw = image.smw
    smw.header = head.h
    if smw.format == shdr.ND:
        nline, nband, nap = (mspec if mspec > 0 else 1), (band if band > 0 else 1), None
        if smw.llen[1] <= 1:
            nline = 0
        if smw.llen[2] <= 1:
            nband = 0
    else:
        nline = mspec if mspec > 0 else 1
        nband = band if band > 0 else (1 if smw.nbands > 1 else 0)
        nap = None
        if smw.nspec <= 1:
            nline = 0
    for key, value in (("RA", "00:00:00.000"), ("DEC", "00:00:00.00"),
                       ("HA", "00:00:00.000"), ("ST", "00:00:00.000")):
        if head.has(key):
            state["temp"] = (head.string(key) or "")[:31]
        if len(state.get("temp", "")) < 1:
            head.put(key, value)
    sh = shdr.shdr_open(smw, image.data, path, nline, nband, nap)
    data = np.asarray(sh.y, dtype=f32).copy()
    name = (sh.title or "").strip() or str(head.h.get("OBJECT", "")).strip() \
        or re.split(r"[$/]", path)[-1]
    if mspec > 0:
        name += f"_{mspec}"
    state["temp"] = f"APID{sh.ap}"
    apid = head.string(f"APID{sh.ap}")
    if apid is not None:
        words = apid.split()
        tname = apid.split(" ")[0] if apid[:1] != " " else ""
        if len(words) > 1:
            head.put("RA", words[1])
        if len(words) > 2:
            head.put("DEC", words[2])
        name = tname
    specid = re.split(r"/", path)[-1]
    if mspec > 0:
        specid += f"[{mspec} ap{sh.ap}]" if sh.ap != mspec else f"[{mspec}]"
    wcs = Wcs(sh)
    wcs.pxshift = head.d("PIXSHIFT", 0.0)
    spec = Spectrum(specid, name, data, head, sh, wcs, mspec)
    if wcs.p2w(float(sh.np1)) <= 0.0 or wcs.p2w(float(sh.np2)) <= 0.0:
        raise RvsaoError("GETSPEC: Illegal starting wavelength")
    n = data.size
    ipix1, ipix2 = 1, sh.np2
    for i in range(sh.np1, min(sh.np1 + 300, n) + 1):
        if data[i - 1] < MIN_PIXEL_VALUE:
            data[i - 1] = 0.0
            ipix1 = i + 1
        else:
            break
    for i in range(sh.np2, max(sh.np2 - 300, 1) - 1, -1):
        if data[i - 1] < MIN_PIXEL_VALUE:
            data[i - 1] = 0.0
            ipix2 = i - 1
        else:
            break
    ip1 = ip2 = 0
    nzpix = 0
    for i in range(ipix1, ipix2 + 1):
        if data[i - 1] < MIN_PIXEL_VALUE:
            if ip1 == 0:
                ip1 = i
            ip2 = i
        elif ip1 > 0:
            fillpix(data, ip1, ip2)
            ip1 = ip2 = 0
        if data[i - 1] != 0.0:
            nzpix += 1
    if nzpix == 0:
        raise RvsaoError("GETSPEC: Entire spectrum is zeroes")
    sbcv = head.d("BCV", INDEFD)
    shcv = head.d("HCV", INDEFD)
    hcv = INDEFD
    if svcor == NONE:
        hcv = 0.0
    elif svcor == HCV:
        hcv = compbcv(head, False)
    elif svcor == BCV:
        hcv, spec.vb = compbcv(head, True), True
    elif svcor == FHCV:
        if shcv != INDEFD:
            hcv, spec.vb = shcv, False
        elif sbcv != INDEFD:
            hcv, spec.vb = sbcv, True
        else:
            hcv, spec.vb = compbcv(head, False), False
    elif svcor == FBCV:
        if sbcv != INDEFD:
            hcv, spec.vb = sbcv, True
        elif shcv != INDEFD:
            hcv, spec.vb = shcv, False
        else:
            hcv, spec.vb = compbcv(head, True), True
    spec.hcv = 0.0 if hcv == INDEFD else hcv
    return spec


def fillpix(pix: np.ndarray, ip1: int, ip2: int) -> None:
    """fillpix: a run of bad pixels as a line between its neighbours."""
    npts = pix.size
    if ip2 < ip1:
        ip1, ip2 = ip2, ip1
    if ip1 < 2 and ip2 < npts:
        p1, slope = pix[ip2], f32(0)
    elif ip2 > npts - 1 and ip1 > 1:
        p1, slope = pix[ip1 - 2], f32(0)
    elif ip1 > 1 and ip2 < npts:
        p1, p2 = pix[ip1 - 2], pix[ip2]
        slope = f32(f32(p2 - p1) / f32(ip2 - ip1 + 2))
    else:
        return
    for i in range(ip1, ip2 + 1):
        pix[i - 1] = f32(p1 + f32(slope * f32(i - ip1 + 1)))


def smooth(fdata: np.ndarray, npoints: int, nsmooth: int) -> None:
    """smooth.x: nsmooth passes of 1-2-1 / 4, in real, in place, the ends repeated."""
    if nsmooth <= 0:
        return
    for _ in range(nsmooth):
        val0 = val1 = fdata[0]
        for ip in range(npoints - 1):
            val2 = fdata[ip + 1]
            fdata[ip] = f32(f32(f32(f32(val0 + val1) + val1) + val2) / f32(4.0))
            val0, val1 = val1, val2
        val2 = fdata[npoints - 1]
        fdata[npoints - 1] = f32(f32(f32(f32(val0 + val1) + val1) + val2) / f32(4.0))


def loghead(taskname: str) -> str:
    """loghead: "<who> <task> <version> user@host  dd-Mon-yyyy hh:mm" -- RVSAO writes
    IRAF where harwood writes harwood, the program that ran."""
    import getpass
    import socket
    import time

    now = time.localtime(time.time() + 30)
    month = "JanFebMarAprMayJunJulAugSepOctNovDec"[3 * (now.tm_mon - 1):3 * now.tm_mon]
    stamp = f" {now.tm_mday:02d}-{month}-{now.tm_year:04d} {now.tm_hour:02d}:{now.tm_min:02d}"
    host = socket.gethostname()
    return f"harwood {taskname} {VERSION} {getpass.getuser()}@{host} {stamp}"
