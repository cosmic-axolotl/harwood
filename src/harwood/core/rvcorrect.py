"""rvcorrect -- the components of an observer's radial velocity, as IRAF's
noao.astutil.rvcorrect computes them: the Earth's rotation (diurnal), the motion of
the Earth about the Earth-Moon barycentre (lunar), the Earth-Moon barycentre about
the Sun (annual), and the Sun's motion toward the solar apex (vsun 20 km/s toward
18h, +30, 1900 by default). The heliocentric velocity is the observed one plus the
first three; the LSR adds the fourth.

From noao/astutil/t_rvcorrect.x and asttools (astvr, astvorbit, astvbary,
astvrotate, asthjd, astprecess, asttimes) of IRAF 2.18; dates, precession and
sidereal time are core/asttools.py, the same port setjd and setairmass use.
IRAF's are low-precision theories, good to about 0.01 km/s; harwood keeps them,
and offers astropy's ephemeris beside them (core/rvcorrect.precise).

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research
in Astronomy Inc.; see LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import asttools, iraffmt

#: How f2c compiles SPP's REAL constants (math.h's PI, TWOPI, HALFPI have no d): one
#: standing alone in a double expression is a float literal, single precision; an
#: expression of constants only (TWOPI / 365.2564d0, 13./12.) is folded in double
#: from the decimal text. Found against IRAF 2.18's own values, to 5e-10 km/s.
PI = float(np.float32(math.pi))
HALFPI = float(np.float32(math.pi / 2))
TWOPI = 2.0 * math.pi                         # folded: TWOPI / 365.2564d0 ...
TWOPI_REAL = float(np.float32(2.0 * math.pi))   # vrotate: TWOPI * (r / 1000.)
THIRTEEN_TWELFTHS = 13.0 / 12.0


def _rad(x: float) -> float:
    return asttools._rad(x)


def vr(ra1, dec1, v1, ra2, dec2) -> float:
    """ast_vr: the projection of a velocity (v1 toward ra1, dec1) on a direction."""
    vx = v1 * math.cos(_rad(15. * ra1)) * math.cos(_rad(dec1))
    vy = v1 * math.sin(_rad(15. * ra1)) * math.cos(_rad(dec1))
    vz = v1 * math.sin(_rad(dec1))
    cc = math.cos(_rad(dec2)) * math.cos(_rad(15. * ra2))
    cs = math.cos(_rad(dec2)) * math.sin(_rad(15. * ra2))
    s = math.sin(_rad(dec2))
    return vx * cc + vy * cs + vz * s


def _ecliptic(ra, dec, oblq):
    r, d = _rad(ra * 15), _rad(dec)
    return asttools.coord(0.0, 0.0, -HALFPI, HALFPI - oblq, r, d)


def vorbit(ra, dec, epoch) -> float:
    """ast_vorbit: the Earth-Moon barycentre's orbital velocity toward (ra, dec)."""
    t = (asttools.julday(epoch) - 2415020.0) / 36525.
    manom = 358.47583 + t * (35999.04975 - t * (0.000150 + t * 0.000003))
    lperi = 101.22083 + t * (1.7191733 + t * (0.000453 + t * 0.000003))
    oblq = 23.452294 - t * (0.0130125 + t * (0.00000164 - t * 0.000000503))
    eccen = 0.01675104 - t * (0.00004180 + t * 0.000000126)
    manom = math.fmod(manom, 360.0)
    lperi = math.fmod(lperi, 360.0)
    manom, lperi, oblq = _rad(manom), _rad(lperi), _rad(oblq)
    tanom = (manom + (2 * eccen - 0.25 * (eccen * eccen * eccen)) * math.sin(manom)
             + 1.25 * (eccen * eccen) * math.sin(2 * manom)
             + THIRTEEN_TWELFTHS * (eccen * eccen * eccen) * math.sin(3 * manom))
    slong = lperi + tanom + PI
    lon, b = _ecliptic(ra, dec, oblq)
    vorb = ((TWOPI / 365.2564) * 149598500.0 / math.sqrt(1. - (eccen * eccen))) / 86400.0
    return vorb * math.cos(b) * (math.sin(slong - lon) - eccen * math.sin(lperi - lon))


def vbary(ra, dec, epoch) -> float:
    """ast_vbary: the Earth's velocity about the Earth-Moon barycentre (VLUNAR)."""
    t = (asttools.julday(epoch) - 2415020) / 36525.
    oblq = 23.452294 - t * (0.0130125 + t * (0.00000164 - t * 0.000000503))
    omega = 259.183275 - t * (1934.142008 + t * (0.002078 + t * 0.000002))
    llong = 270.434164 + t * (481267.88315 + t * (-0.001133 + t * 0.0000019)) - omega
    lperi = 334.329556 + t * (4069.034029 - t * (0.010325 + t * 0.000012)) - omega
    em = 0.054900489
    inclin = 5.1453964
    lperi, llong = _rad(lperi), _rad(llong)
    anom = llong - lperi
    anom = (anom + (2. * em - 0.25 * (em * em * em)) * math.sin(anom)
            + 1.25 * (em * em) * math.sin(2. * anom)
            + THIRTEEN_TWELFTHS * (em * em * em) * math.sin(3. * anom))
    llong = anom + lperi
    omega, oblq, inclin = _rad(omega), _rad(oblq), _rad(inclin)
    lon, b = _ecliptic(ra, dec, oblq)
    lm, bm = asttools.coord(omega, 0.0, omega - HALFPI, HALFPI - inclin, lon, b)
    vmoon = (TWOPI / 27.321661) * 384403.12040 / math.sqrt(1. - (em * em)) / 86400.
    v = vmoon * math.cos(bm) * (math.sin(llong - lm) - em * math.sin(lperi - lm))
    return v / float(np.float32(81.53))      # a REAL constant


def vrotate(ra, dec, epoch, latitude, longitude, altitude) -> float:
    """ast_vrotate: the Earth's rotation toward (ra, dec), from a site (longitude
    positive west, as IRAF's)."""
    lat = _rad(latitude)
    dlat = (-(11. * 60. + 32.743000) * math.sin(2 * lat) + 1.163300 * math.sin(4 * lat)
            - 0.002600 * math.sin(6 * lat))
    lat = lat + _rad(dlat / 3600.)
    r = 6378160.0 * (0.998327073 + 0.00167643800 * math.cos(2 * lat)
                     - 0.00000351 * math.cos(4 * lat) + 0.000000008 * math.cos(6 * lat)) \
        + altitude
    vc = TWOPI_REAL * (r / 1000.) / (23.934469591229 * 3600.)
    lmst = asttools.mst(asttools.julday(epoch), -longitude)
    return vc * math.cos(lat) * math.cos(_rad(dec)) * math.sin(_rad((ra - lmst) * 15.))


@dataclass
class Solar:
    """The solar motion relative to the standard of rest: speed and apex."""
    vs: float = 20.0
    ras: float = 18.0
    decs: float = 30.0
    eps: float = 1900.0


def rvcorrect(ra, dec, ep, year, month, day, ut, site, solar: Solar):
    """rvcorrect: (hjd, vrot, vbary, vorb, vsol) for an observation."""
    epoch = asttools.date_to_epoch(year, month, day, ut)
    ra_obs, dec_obs = asttools.precess(ra, dec, ep, epoch)
    ra_vsun, dec_vsun = asttools.precess(solar.ras, solar.decs, solar.eps, epoch)
    hjd = asttools.jd_to_hjd(ra_obs, dec_obs, asttools.julday(epoch))[0]
    vsol = vr(ra_vsun, dec_vsun, solar.vs, ra_obs, dec_obs)
    vorb = vorbit(ra_obs, dec_obs, epoch)
    vbar = vbary(ra_obs, dec_obs, epoch)
    vrot = vrotate(ra_obs, dec_obs, epoch, site.latitude, site.longitude, site.altitude)
    return hjd, vrot, vbar, vorb, vsol


def precise(ra, dec, ep, year, month, day, ut, site, kind: str = "barycentric") -> float:
    """astropy's correction (its ephemeris, ERFA's precession and nutation, the
    site's rotation): what to add to an observed velocity to put it at the solar
    system's barycentre or the Sun's centre, km/s. To about a cm/s, where IRAF's
    theories are good to about 10 m/s."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time

    location = EarthLocation.from_geodetic(site.east * u.deg, site.latitude * u.deg,
                                           site.altitude * u.m)
    when = Time(asttools.date_to_julday(year, month, day, ut), format="jd", scale="utc")
    equinox = Time(ep, format="jyear") if ep else when
    star = SkyCoord(ra * 15 * u.deg, dec * u.deg, frame="fk5", equinox=equinox)
    return float(star.radial_velocity_correction(kind=kind, obstime=when,
                                                 location=location).to(u.km / u.s).value)


# ------------------------------------------------------------- the output


def raw_fields(code: str) -> tuple[str, dict[str, str]] | None:
    """An observatory's entry as obsdb.dat writes it, for obslog: its name and the
    text of each parameter."""
    from . import observatory as obsmod

    paths = [Path(obsmod.PROJECT_OBSDB), obsmod.OBSDB]
    key = code.strip().lower()
    for path in paths:
        if not path.exists():
            continue
        entry: dict[str, str] | None = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r'\s*(\w+)\s*=\s*"?([^"]*?)"?\s*$', line)
            if not m:
                continue
            name, value = m.group(1).lower(), m.group(2)
            if name == "observatory":
                if entry is not None and entry.get("observatory", "").lower() == key:
                    return entry.get("name", value), entry
                entry = {"observatory": value}
            elif entry is not None:
                entry[name] = value
        if entry is not None and entry.get("observatory", "").lower() == key:
            return entry.get("name", ""), entry
    return None


def obslog(code: str, task: str = "RVCORRECT") -> str:
    found = raw_fields(code)
    if found is None:
        return ""
    name, fields = found
    out = f"# {task}: Observatory parameters for {name}\n"
    for param in ("latitude", "longitude", "altitude"):
        if param in fields:
            out += f"#\t{param} = {fields[param]}\n"
    return out


def header_lines(input_: bool) -> str:
    out = ""
    if input_:
        out += f"{'##YR':>4s} {'MO':>2s} {'DY':>2s} {'   UT   ':>8s} {'   RA   ':>8s} " \
               f"{'   DEC   ':>9s} {'   VOBS ':>8s}\n"
    out += (f"{'##   HJD     ':>13s} {'VOBS':>8s} {'VHELIO':>8s} {'VLSR':>8s}   "
            f"{'VDIURNAL':>8s} {'VLUNAR':>8s} {'VANNUAL':>8s} {'VSOLAR':>8s}\n")
    return out


def _f(value: float, width: int, decpl: int) -> str:
    return iraffmt.printf(value, "f", decpl, width).rjust(width)


def output_line(year, month, day, ut, ra, dec, hjd, vobs, vrot, vbar, vorb, vsol,
                input_: bool) -> str:
    out = ""
    if input_:
        h = iraffmt.Format
        out += (f"{year:4d} {month:2d} {day:2d} {h('%8.0h')(ut)} {h('%8.0h')(ra)} "
                f"{h('%9.0h')(dec)} {_f(vobs, 8, 1)}\n")
    out += (f"{_f(hjd, 13, 5)} {_f(vobs, 8, 2)} {_f(vobs + vrot + vbar + vorb, 8, 2)} "
            f"{_f(vobs + vrot + vbar + vorb + vsol, 8, 2)}   {_f(vrot, 8, 3)} "
            f"{_f(vbar, 8, 3)} {_f(vorb, 8, 3)} {_f(vsol, 8, 3)}\n")
    return out


# ------------------------------------------------------------- the task


def dtm_decode(text: str):
    """dtm_decode: (year, month, day, hours) of yyyy-mm-dd[Thh:mm:ss] or dd/mm/yy
    (hours None without a time), or None when the text is neither."""
    s = str(text)

    def ctoi(ip: int):
        m = re.match(r"[ \t]*([+-]?\d+)", s[ip:])
        return (int(m.group(1)), ip + m.end(), len(m.group(0).strip())) if m else (None, ip, 0)

    ival, ip, n = ctoi(0)
    if n == 2:
        old, day, year = True, ival, None
    elif n == 4:
        old, year, day = False, ival, None
    else:
        return None
    sep = "/" if old else "-"
    if s[ip:ip + 1] != sep:
        return None
    month, ip, n = ctoi(ip + 1)
    if n != 2 or s[ip:ip + 1] != sep:
        return None
    ival, ip, n = ctoi(ip + 1)
    if n != 2:
        return None
    if old:
        year = 1900 + ival
    else:
        day = ival
    if s[ip:ip + 1] != "T" or old:
        return year, month, day, None
    rest = s[ip + 1:]
    m = re.match(r"[\d:.]+", rest)
    if m is None or len(m.group(0)) < 8 or rest[m.end():].strip():
        return None
    return year, month, day, iraffmt.ctod(m.group(0))


class Rvcorrect:
    """t_rvcorrect: from text files, from images' headers, or from the parameters;
    the table it prints, and what it records (the parameters' hjd, vhelio and
    vlsr, or the images' HJD, VHELIO, VLSR and VSUN)."""

    def __init__(self, params: dict):
        self.p = params
        self.solar = Solar(float(params.get("vsun", 20.0)), float(params.get("ra_vsun", 18.0)),
                           float(params.get("dec_vsun", 30.0)),
                           float(params.get("epoch_vsun", 1900.0)))
        self.header = bool(params.get("header", True))
        self.input = bool(params.get("input", False))
        self.out = ""
        self.err = ""
        self.last: tuple[float, float, float] | None = None

    def _site(self, code: str):
        from . import observatory as obsmod

        site = obsmod.find(code)
        if site is None:
            raise ValueError(f"observatory '{code}' not found")
        self.out += obslog(site.code)
        return site

    def _emit(self, year, month, day, ut, ra, dec, ep, vobs, site) -> tuple[float, ...]:
        hjd, vrot, vbar, vorb, vsol = rvcorrect(ra, dec, ep, year, month, day, ut, site,
                                                self.solar)
        if self.header:
            self.out += header_lines(self.input)
            self.header = False
        self.out += output_line(year, month, day, ut, ra, dec, hjd, vobs, vrot, vbar,
                                vorb, vsol, self.input)
        vhelio = vobs + vrot + vbar + vorb
        self.last = (hjd, vhelio, vhelio + vsol)
        return hjd, vhelio, vhelio + vsol

    def files(self, names: list[str]) -> None:
        from .rvsao import Scan

        site = self._site(str(self.p.get("observatory", "kpno")))
        for name in names:
            path = Path(name)
            if not path.exists():
                self.err += f"Warning: Cannot open file ({name})\n"
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                s = Scan(line)
                year, month, day = s.i(), s.i(), s.i()
                ut, ra, dec = s.d(), s.d(), s.d()
                if s.n != 6:
                    continue
                ep = s.d()
                if s.n != 7:
                    ep = None
                vobs = s.d()
                if s.n != 8:
                    vobs = 0.0
                self._emit(year, month, day, ut, ra, dec, ep, vobs, site)

    def parameters(self) -> None:
        p = self.p
        site = self._site(str(p.get("observatory", "kpno")))
        ep = p.get("epoch")
        self._emit(int(p["year"]), int(p["month"]), int(p["day"]), float(p["ut"]),
                   float(p["ra"]), float(p["dec"]), None if ep is None else float(ep),
                   float(p.get("vobs", 0.0)), site)

    def images(self, names: list[str]) -> list[dict]:
        """rvc_images: each image's keywords (keywpars' names), its observatory
        (OBSERVAT or the parameter), and with imupdate the cards to write."""
        from . import images as imgs
        from . import observatory as obsmod
        from .rvsao import Header

        k = {"date_obs": "DATE-OBS", "ut": "UT", "ra": "RA", "dec": "DEC",
             "epoch": "EPOCH", "vobs": "VOBS", **dict(self.p.get("keywords", {}))}
        updates = []
        code = None
        for name in names:
            try:
                _data, header = imgs.read(name)
            except (OSError, ValueError):
                self.err += f"Warning: Cannot open image ({name})\n"
                continue
            head = Header(header)
            try:
                obs = (head.string("OBSERVAT") or str(self.p.get("observatory", "kpno")))
                obs = obs.strip().lower()
                if obs != code:
                    site = obsmod.find(obs)
                    if site is None:
                        raise ValueError(f"observatory '{obs}' not found")
                    self.out += obslog(site.code)
                    code = obs
                utext = head.string(k["ut"])
                if utext is None:
                    raise ValueError("Error parsing UT keyword")
                got = dtm_decode(utext)
                if got is not None and got[3] is not None:
                    ut = got[3]
                else:
                    ut = head.d(k["ut"], None)
                    if ut is None:
                        raise ValueError("Error parsing UT keyword")
                dtext = head.string(k["date_obs"])
                got = dtm_decode(dtext) if dtext is not None else None
                if got is None:
                    raise ValueError("Error parsing DATE-OBS keyword")
                year, month, day, t = got
                if t is not None:
                    ut = t
                got = {key: head.d(k[key], None) for key in ("ra", "dec", "epoch")}
                for key, value in got.items():
                    if value is None:
                        raise ValueError(f"Image header parameter not found ({k[key]})")
                ra, dec, ep = got["ra"], got["dec"], got["epoch"]
                vobs = head.d(k["vobs"], 0.0)
                hjd, vhelio, vlsr = self._emit(year, month, day, ut, ra, dec, ep, vobs, site)
                if self.p.get("imupdate"):
                    s = self.solar
                    updates.append({"image": name, "HJD": hjd, "VHELIO": vhelio,
                                    "VLSR": vlsr,
                                    "VSUN": " ".join(iraffmt.g(v, double=True).rjust(6)
                                                     for v in (s.vs, s.ras, s.decs, s.eps))})
            except ValueError as error:
                self.err += f"Warning: {error}\n"
        return updates
