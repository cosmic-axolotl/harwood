"""IRAF's astronomical arithmetic, ported: dates, precession, sidereal time
and the airmass.

From noao/astutil/asttools (asttimes.x, astprecess.x) and noao/astutil
(airmass.x) of iraf-community 2.18, line for line, so that setairmass and
setjd give IRAF's numbers and not merely close ones. Nothing here reaches for
a table of the Earth's rotation or for the internet: IRAF takes UT as UT1, and
the difference is under a second.
"""

from __future__ import annotations

import math

J2000 = 2000.0
JD2000 = 2451545.0
JYEAR = 365.25

#: Solar to sidereal time, as t_setairmass.x's SOLTOSID.
SOLTOSID = 1.00273790935


def _int(value: float) -> int:
    """SPP's int(): truncation towards zero."""
    return int(value)


def date_to_julday(year: int, month: int, day: int, t: float) -> float:
    """ast_date_to_julday: Julian date of a Gregorian date and time (hours).
    A two-digit year is 19xx, as IRAF has it."""
    y = 1900 + year if year < 100 else year
    if month > 2:
        m = month + 1
    else:
        m = month + 13
        y = y - 1
    jd = float(_int(JYEAR * y) + _int(30.6001 * m) + day + 1720995)
    if day + 31 * (m + 12 * y) >= 588829:
        d = _int(y / 100)
        m = _int(y / 400)
        jd = jd + 2 - d + m
    return jd - 0.5 + _int(t * 360000.0 + 0.5) / 360000.0 / 24.0


def julday_to_date(j: float) -> tuple[int, int, int, float]:
    """ast_julday_to_date (ERFA's eraJd2cal): year, month, day and hours."""
    dj2 = 0.0
    d1, d2 = (j, dj2) if j >= dj2 else (dj2, j)
    d2 = d2 - 0.5
    f1 = math.fmod(d1, 1.0)
    f2 = math.fmod(d2, 1.0)
    f = math.fmod(f1 + f2, 1.0)
    if f < 0.0:
        f = f + 1.0
    d = _nint(d1 - f1) + _nint(d2 - f2) + _nint(f1 + f2 - f)
    jd = int(_nint(d)) + 1
    l = jd + 68569                                      # noqa: E741
    n = (4 * l) // 146097
    l = l - (146097 * n + 3) // 4                       # noqa: E741
    i = (4000 * (l + 1)) // 1461001
    l = l - ((1461 * i) // 4 - 31)                      # noqa: E741
    k = (80 * l) // 2447
    day = int(l - (2447 * k) // 80)
    l = k // 11                                         # noqa: E741
    month = int(k + 2 - 12 * l)
    year = int(100 * (n - 49) + i + l)
    return year, month, day, f * 24.0


def _nint(value: float) -> float:
    """SPP's nint(): to the nearest integer, halves away from zero."""
    return float(math.floor(value + 0.5)) if value >= 0 else -float(math.floor(-value + 0.5))


def date_to_epoch(year: int, month: int, day: int, ut: float) -> float:
    """ast_date_to_epoch: the Julian epoch of a date and UT."""
    return J2000 + (date_to_julday(year, month, day, ut) - JD2000) / JYEAR


def julday(epoch: float) -> float:
    """ast_julday: the Julian date of a Julian epoch."""
    return JD2000 + (epoch - J2000) * JYEAR


def rotmatrix(epoch: float) -> list[list[float]]:
    """ast_rotmatrix: precession from J2000.0 to `epoch`, p[i][j] as IRAF's
    p[i+1, j+1]."""
    t = (julday(epoch) - 2451545.0) / 36525.0
    a = t * (0.6406161 + t * (0.0000839 + t * 0.0000050))
    b = t * (0.6406161 + t * (0.0003041 + t * 0.0000051))
    c = t * (0.5567530 - t * (0.0001185 + t * 0.0000116))
    ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
    cb, sb = math.cos(math.radians(b)), math.sin(math.radians(b))
    cc, sc = math.cos(math.radians(c)), math.sin(math.radians(c))
    p = [[0.0] * 3 for _ in range(3)]
    p[0][0] = ca * cb * cc - sa * sb
    p[1][0] = -sa * cb * cc - ca * sb
    p[2][0] = -cb * sc
    p[0][1] = ca * sb * cc + sa * cb
    p[1][1] = -sa * sb * cc + ca * cb
    p[2][1] = -sb * sc
    p[0][2] = ca * sc
    p[1][2] = -sa * sc
    p[2][2] = cc
    return p


def precess(ra1: float, dec1: float, epoch1: float | None,
            epoch2: float) -> tuple[float, float]:
    """ast_precess: (hours, degrees) at epoch1 to epoch2, through J2000.0. An
    epoch of 0 or none means the coordinates are already of epoch2."""
    if not epoch1 or epoch1 == epoch2:
        return ra1, dec1
    ra2 = math.radians(ra1 * 15.0)
    dec2 = math.radians(dec1)
    r0 = [math.cos(ra2) * math.cos(dec2), math.sin(ra2) * math.cos(dec2),
          math.sin(dec2)]
    if epoch1 != 2000.0:
        p = rotmatrix(epoch1)
        # The inverse of p, which is its transpose.
        r0 = [p[0][0] * r0[0] + p[0][1] * r0[1] + p[0][2] * r0[2],
              p[1][0] * r0[0] + p[1][1] * r0[1] + p[1][2] * r0[2],
              p[2][0] * r0[0] + p[2][1] * r0[1] + p[2][2] * r0[2]]
    if epoch2 != 2000.0:
        p = rotmatrix(epoch2)
        r0 = [p[0][0] * r0[0] + p[1][0] * r0[1] + p[2][0] * r0[2],
              p[0][1] * r0[0] + p[1][1] * r0[1] + p[2][1] * r0[2],
              p[0][2] * r0[0] + p[1][2] * r0[1] + p[2][2] * r0[2]]
    ra = math.degrees(math.atan2(r0[1], r0[0])) / 15.0
    dec = math.degrees(math.asin(r0[2]))
    if ra < 0.0:
        ra = ra + 24.0
    return ra, dec


def mst(jd: float, east_longitude: float) -> float:
    """ast_mst: local mean sidereal time, hours, from the Julian date of the
    UT (taken as UT1, as IRAF does) and the longitude, positive east."""
    ut = (jd - int(jd) - 0.5) * 24.0
    t = (jd - 2451545.0) / 36525.0
    st = 24110.54841 + t * (8640184.812866 + t * (0.093104 - t * 6.2e-6))
    st = math.fmod(st / 3600.0 + ut + east_longitude / 15.0, 24.0)
    return st + 24.0 if st < 0 else st


def airmassx(ha: float, dec: float, lat: float, scale: float = 750.0) -> float:
    """airmassx (airmass.x): Allen's airmass for an hour angle (hours), a
    declination and a latitude (degrees), with the atmosphere's scale height
    -- the plane-parallel secant, rounded off near the horizon by a spherical
    atmosphere `scale` times thinner than the Earth is wide."""
    cos_zd = (math.sin(math.radians(lat)) * math.sin(math.radians(dec))
              + math.cos(math.radians(lat)) * math.cos(math.radians(dec))
              * math.cos(math.radians(ha * 15.0)))
    x = scale * cos_zd
    return math.sqrt(x ** 2 + 2.0 * scale + 1.0) - x


def elevation_airmass(elevation: float, scale: float = 750.0) -> float:
    """t_airmass: the airmass at an elevation above the horizon, degrees."""
    x = scale * math.sin(math.radians(elevation))
    return math.sqrt(x ** 2 + 2 * scale + 1) - x


def dtm_encode(year: int, month: int, day: int, hours: float, prec: int = 2) -> str:
    """dtm_encode with a time: '1992-01-25T01:22:50.00', as setairmass writes
    UTMIDDLE."""
    seconds = round(hours * 3600.0, prec)
    h, rest = divmod(seconds, 3600.0)
    m, s = divmod(rest, 60.0)
    width = 3 + prec if prec else 2
    return (f"{year:04d}-{month:02d}-{day:02d}T{int(h):02d}:{int(m):02d}:"
            f"{s:0{width}.{prec}f}")


def coord(ao: float, bo: float, ap: float, bp: float, a1: float,
          b1: float) -> tuple[float, float]:
    """ast_coord (astcoord.x): spherical coordinates in a system with origin
    (ao, bo) and pole (ap, bp), all radians."""
    x, y, z = math.cos(a1) * math.cos(b1), math.sin(a1) * math.cos(b1), math.sin(b1)
    xp, yp, zp = math.cos(ap) * math.cos(bp), math.sin(ap) * math.cos(bp), math.sin(bp)
    sao, cao, sbo, cbo = math.sin(ao), math.cos(ao), math.sin(bo), math.cos(bo)
    xp, yp = xp * cao + yp * sao, -xp * sao + yp * cao
    x, y = x * cao + y * sao, -x * sao + y * cao
    xp, zp = xp * cbo + zp * sbo, -xp * sbo + zp * cbo
    x, z = x * cbo + z * sbo, -x * sbo + z * cbo
    sbp, cbp = zp, yp
    y, z = y * sbp - z * cbp, y * cbp + z * sbp
    return math.atan2(y, x), math.asin(z)


def jd_to_hjd(ra: float, dec: float, jd: float) -> tuple[float, float]:
    """ast_jd_to_hjd (asthjd.x): the heliocentric Julian date of a geocentric
    one, for a star at (ra hours, dec degrees) of the date. Returns (hjd, the
    light-time correction in days). IRAF's approximate Sun, good to about a
    second."""
    t = (jd - 2415020.0) / 36525.0
    manom = 358.47583 + t * (35999.04975 - t * (0.000150 + t * 0.000003))
    lperi = 101.22083 + t * (1.7191733 + t * (0.000453 + t * 0.000003))
    oblq = 23.452294 - t * (0.0130125 + t * (0.00000164 - t * 0.000000503))
    eccen = 0.01675104 - t * (0.00004180 + t * 0.000000126)
    manom = math.fmod(manom, 360.0)
    lperi = math.fmod(lperi, 360.0)
    r, d = math.radians(ra * 15), math.radians(dec)
    manom, lperi, oblq = math.radians(manom), math.radians(lperi), math.radians(oblq)
    tanom = (manom + (2 * eccen - 0.25 * eccen ** 3) * math.sin(manom)
             + 1.25 * eccen ** 2 * math.sin(2 * manom)
             + 13.0 / 12.0 * eccen ** 3 * math.sin(3 * manom))
    slong = lperi + tanom + math.pi
    l, b = coord(0.0, 0.0, -math.pi / 2, math.pi / 2 - oblq, r, d)  # noqa: E741
    rsun = (1.0 - eccen ** 2) / (1.0 + eccen * math.cos(tanom))
    lt = -0.005770 * rsun * math.cos(b) * math.cos(l - slong)
    return jd + lt, lt


DAYS = ("SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT")


def day_of_week(jd: float) -> str:
    """ast_day_of_week: the day a Julian date falls on."""
    return DAYS[int(math.fmod(jd + 1.5, 7.0))]


def times(year: int, month: int, day: int, zt: float, zone: float,
          west_longitude: float) -> dict:
    """ast_times (t_asttimes.x): a zone time at a site, as UT, epoch, Julian
    date and local mean sidereal time, with the day of the week."""
    if zt < 0.0:
        zt, day = zt + 24, day - 1
    if zt >= 24.0:
        zt, day = zt - 24, day + 1
    jd = date_to_julday(year, month, day, zt)
    year, month, day, zt = julday_to_date(jd)
    dow = day_of_week(jd)
    ut = zone
    while ut < -12.0:
        ut += 24.0
    while ut >= 12.0:
        ut -= 24.0
    ut = zt + ut
    d = day
    if ut < 0.0:
        ut, d = ut + 24, d - 1
    if ut >= 24.0:
        ut, d = ut - 24, d + 1
    jd = date_to_julday(year, month, d, ut)
    epoch = date_to_epoch(year, month, d, ut)
    lmst = mst(julday(epoch), -west_longitude)
    return {"year": year, "month": month, "day": day, "dow": dow, "zt": zt, "ut": ut,
            "epoch": epoch, "jd": jd, "lmst": lmst}
