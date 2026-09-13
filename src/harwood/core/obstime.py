"""When an exposure happened, which is less obvious than it sounds.

A header says the exposure started at UT 02:08:22 and lasted 1200 seconds. Most
things that care -- the airmass, which arc to use, the heliocentric correction
-- want the *middle* of it, because that is the moment the light average
belongs to. IRAF's `setairmass` computes that and writes it back as `UTMIDDLE`,
and `refspectra` sorts on it.

Two details that bite:

**Midnight.** A night's exposures run from 23:50 to 05:00, and sorted as plain
hours the evening ones come last. IRAF handles this with a wrap point -- an
hour before which the clock is taken to belong to the next day -- and so does
this. Seventeen hundred is the usual choice: nobody observes at five in the
afternoon.

**Two-digit years.** These headers say `25/01/92`, and a naive reading makes
that 1925 or 2092 depending on the library. The rule here is the one the FITS
standard settled on: a two-digit year is 1900-something, because the files that
use the format are old ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Hour before which a time is taken to belong to the following day. An
#: observing night crosses midnight and nothing is taken at five in the
#: afternoon.
TIMEWRAP = 17.0


def hours(text: Any) -> float | None:
    """A time written as hours, `hh:mm:ss`, or a number, as decimal hours."""
    if text is None or text == "":
        return None
    if isinstance(text, (int, float)):
        return float(text)
    parts = str(text).strip().split(":")
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    total = 0.0
    for index, value in enumerate(values[:3]):
        total += abs(value) / (60 ** index)
    return -total if str(text).strip().startswith("-") else total


def date(header: Any, keyword: str = "DATE-OBS") -> tuple[int, int, int] | None:
    """The observing date as (year, month, day), from either FITS convention."""
    raw = header.get(keyword)
    if not raw:
        return None
    text = str(raw).strip()

    modern = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if modern:
        return tuple(int(part) for part in modern.groups())

    old = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2})$", text)
    if old:
        day, month, year = (int(part) for part in old.groups())
        # A two-digit year is 1900-something: the files that use this format
        # are old ones, and the FITS standard says so.
        return 1900 + year, month, day
    return None


def middle(header: Any, ut_key: str = "UT", exposure_key: str = "EXPTIME",
           key: str = "UTMIDDLE") -> float | None:
    """UT at the middle of the exposure, in decimal hours.

    Taken from the header when it is already there -- `setairmass` may have
    written it, or IRAF may have -- and computed from the start and the
    duration when it is not.
    """
    existing = hours(header.get(key))
    if existing is not None:
        return existing

    start = hours(header.get(ut_key))
    if start is None:
        return None
    exposure = float(header.get(exposure_key, 0.0) or 0.0)
    return (start + exposure / 7200.0) % 24.0


def sortable(value: float | None, timewrap: float = TIMEWRAP) -> float | None:
    """A time that sorts in observing order across midnight.

    Anything earlier than the wrap point belongs to the following day, so it
    sorts after the evening rather than before it.
    """
    if value is None:
        return None
    return value + 24.0 if value < timewrap else value


def julian(header: Any, ut_key: str = "UT", exposure_key: str = "EXPTIME") -> float | None:
    """Julian date at the middle of the exposure."""
    parts = date(header)
    hour = middle(header, ut_key, exposure_key)
    if parts is None or hour is None:
        return None

    from astropy.time import Time, TimeDelta

    year, month, day = parts
    stamp = Time(f"{year:04d}-{month:02d}-{day:02d}T00:00:00", scale="utc")
    return float((stamp + TimeDelta(hour * 3600.0, format="sec")).jd)


def sky_position(header: Any):
    """Where the telescope was pointing, or None when the header does not say."""
    right = header.get("RA")
    declination = header.get("DEC")
    if not right or not declination:
        return None

    from astropy.coordinates import SkyCoord

    try:
        return SkyCoord(str(right), str(declination), unit=("hourangle", "deg"),
                        frame="icrs")
    except Exception:
        return None


#: Coefficients of the refraction correction to sec z, from Allen's
#: *Astrophysical Quantities*. Worth about half a per cent at two airmasses --
#: small, but it is the difference between matching the number the observer
#: wrote down and not.
REFRACTION = (0.0018167, 0.002875, 0.0008083)


def refracted(secant: float) -> float:
    """sec z corrected for refraction: the airmass light actually travelled."""
    excess = secant - 1.0
    return secant - sum(coefficient * excess ** power
                        for power, coefficient in enumerate(REFRACTION, start=1))


@dataclass
class Airmass:
    """What the star did during the exposure, not just where it started.

    IRAF's `setairmass` reports the *effective* airmass -- the mean over the
    exposure, weighted by Simpson's rule over its start, middle and end. For a
    snapshot the three agree and the distinction is pedantry. For twenty
    minutes at two airmasses they do not: the header written at the shutter
    opening understates what the light went through, and a flux calibration
    that uses it takes the extinction out with the wrong coefficient.
    """

    begin: float
    middle: float
    end: float

    @property
    def effective(self) -> float:
        return (self.begin + 4.0 * self.middle + self.end) / 6.0

    @property
    def drift(self) -> float:
        return self.end - self.begin


def secant(header: Any, latitude: float, longitude: float, height: float,
           hour: float) -> float | None:
    """Refraction-corrected airmass at one instant, as decimal hours UT.

    `longitude` is positive east, which is what every library wants and the
    opposite of what IRAF stores; `core.observatory.Site.east` does the flip.
    """
    parts = date(header)
    position = sky_position(header)
    if parts is None or position is None:
        return None

    import numpy as np
    from astropy.coordinates import AltAz, EarthLocation
    from astropy.time import Time, TimeDelta

    year, month, day = parts
    when = Time(f"{year:04d}-{month:02d}-{day:02d}T00:00:00", scale="utc") \
        + TimeDelta(hour * 3600.0, format="sec")
    where = EarthLocation(lat=latitude, lon=longitude, height=height)
    altitude = position.transform_to(AltAz(obstime=when, location=where)).alt.deg
    if altitude <= 3.0:
        # Below three degrees the plane-parallel picture has stopped being a
        # picture of anything, and no polynomial rescues it.
        return None
    return refracted(float(1.0 / np.cos(np.radians(90.0 - altitude))))


def airmass(header: Any, latitude: float, longitude: float, height: float = 2000.0,
            ut_key: str = "UT", exposure_key: str = "EXPTIME") -> Airmass | None:
    """The airmass at the start, middle and end of the exposure."""
    start = hours(header.get(ut_key))
    if start is None:
        return None
    span = float(header.get(exposure_key, 0.0) or 0.0) / 3600.0

    values = [secant(header, latitude, longitude, height, start + fraction * span)
              for fraction in (0.0, 0.5, 1.0)]
    if any(value is None for value in values):
        return None
    return Airmass(*values)


def sidereal(header: Any, longitude: float, ut_key: str = "UT",
             exposure_key: str = "EXPTIME") -> float | None:
    """Local mean sidereal time at mid-exposure, in decimal hours.

    `longitude` is positive east. Sidereal time is what says which part of the
    sky is overhead, so it is the quantity behind the hour angle and behind
    the airmass -- and the one an observer reads off the wall to decide what
    to point at next.
    """
    jd = julian(header, ut_key, exposure_key)
    if jd is None:
        return None

    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    when = Time(jd, format="jd", scale="utc",
                location=EarthLocation(lat=0.0 * u.deg, lon=longitude * u.deg))
    return float(when.sidereal_time("mean").hour)


def heliocentric(header: Any, latitude: float, longitude: float,
                 height: float = 2000.0, ut_key: str = "UT",
                 exposure_key: str = "EXPTIME",
                 kind: str = "heliocentric") -> tuple[float, float] | None:
    """Heliocentric (or barycentric) Julian date, and the correction applied.

    The Earth's orbit is 300 light-seconds across, so a photon arriving today
    and the same photon arriving in six months differ by up to 16.6 minutes of
    travel time. For anything periodic -- an eclipsing binary, a Cepheid, a
    transit -- that is the difference between a coherent light curve and a
    smeared one, and it is why a period is quoted against HJD rather than JD.

    Returns (corrected JD, correction in seconds), or None when the header
    does not say enough.
    """
    jd = julian(header, ut_key, exposure_key)
    position = sky_position(header)
    if jd is None or position is None:
        return None

    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    where = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg,
                          height=height * u.m)
    when = Time(jd, format="jd", scale="utc", location=where)
    delay = when.light_travel_time(position, kind)
    return float((when.utc + delay).jd), float(delay.sec)
