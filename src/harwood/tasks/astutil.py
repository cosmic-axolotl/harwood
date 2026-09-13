"""astutil -- the arithmetic of when and where.

One task, and it looks like bookkeeping until you check it against real data.
The 1992 headers in the course exercises carry `AIRMASS = 1.52` for a
twenty-minute exposure of DH Cep. That number is correct and it is also not the
airmass of the exposure: it is the airmass at the moment the shutter opened.
By the time it closed the star had sunk to 1.61. The light that landed on the
detector came through a mean of 1.57, and a flux calibration that takes the
extinction out with 1.52 removes too little of it, by about three per cent at
that airmass in the blue.

That is what `setairmass` is for, and it is also why `refspectra` needs it:
both want a time that means the middle of the exposure rather than its edge.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "site": "An airmass is the sky as seen from somewhere, so the site has to "
            "be known before anything can be computed. Headers rarely carry "
            "coordinates -- at most a name -- so the name is looked up in a "
            "table of observatories, the same one IRAF keeps in obsdb.dat.",
    "when": "The exposure has a start and a duration; what everything "
            "downstream wants is the middle, because that is the moment the "
            "light average belongs to. For a snapshot the distinction is "
            "pedantry. For twenty minutes it is the difference between the "
            "airmass the observer wrote down and the airmass the photons "
            "actually travelled through.",
    "where": "Right ascension, declination and the sidereal time at the site "
             "give the zenith distance. Airmass is sec of that, corrected for "
             "refraction -- the atmosphere bends the light, so the star is a "
             "little higher than geometry says and the path a little shorter.",
    "effective": "The airmass is not constant during the exposure, and the "
                 "mean that matters is weighted over it. Simpson's rule over "
                 "the start, the middle and the end gets that to far better "
                 "than the accuracy of the extinction coefficient it will be "
                 "used with.",
    "write": "AIRMASS, UTMIDDLE and JD go into the header, where "
             "setairmass, calibrate and anything computing a light curve look "
             "for them. The value that was there before is kept in HISTORY: "
             "overwriting a number without recording what it was makes the "
             "file unreproducible.",
}


class SetAirmassParams(TaskParams):
    images: list[str] = Field(
        ..., description="Frames to stamp. Globs and @lists expand.",
        json_schema_extra={"positional": True},
    )
    observatory: str = Field(
        "",
        description="Where the telescope is -- 'kpno', 'ctio', 'lna'. Read "
                    "from the OBSERVAT keyword when left out. "
                    "'harwood observatory' lists them.",
    )
    latitude: float | None = Field(
        None, ge=-90, le=90,
        description="Site latitude in degrees, for an observatory not in the "
                    "table. Positive north.",
    )
    longitude: float | None = Field(
        None,
        description="Site longitude in degrees, positive WEST as IRAF stores "
                    "it. Kitt Peak is +111.6, not -111.6.",
    )
    altitude: float = Field(
        2000.0, description="Site height above sea level in metres."
    )
    ut: str = Field("UT", description="Keyword holding the UT at the start.")
    date: str = Field("DATE-OBS", description="Keyword holding the date.")
    exposure: str = Field("EXPTIME", description="Keyword holding the duration.")
    update: bool = Field(
        True, description="Write the results back. Off, it only reports them."
    )
    override: bool = Field(
        True,
        description="Replace an AIRMASS that is already there. The old value "
                    "goes to HISTORY either way.",
    )


@register
class SetAirmass(Task[SetAirmassParams]):
    name = "setairmass"
    package = "astutil"
    description = "Compute the effective airmass and the mid-exposure time."
    Params = SetAirmassParams

    def plan(self, params: SetAirmassParams) -> list[Step]:
        return [
            Step("site", params.observatory or "from the OBSERVAT keyword",
                 why=WHY["site"]),
            Step("when", f"{params.ut} + {params.exposure}/2", why=WHY["when"]),
            Step("where", "zenith distance from RA, DEC and sidereal time",
                 why=WHY["where"]),
            Step("effective", "Simpson's rule over the exposure",
                 why=WHY["effective"]),
            Step("write", "AIRMASS, UTMIDDLE, JD" if params.update
                 else "nothing -- --no-update was given", why=WHY["write"],
                 skipped=not params.update,
                 reason="" if params.update else "reporting only"),
        ]

    def execute(self, params: SetAirmassParams) -> TaskResult:
        from astropy.io import fits

        from ..core import images, observatory, obstime

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}

        site = None
        if params.latitude is not None and params.longitude is not None:
            site = observatory.Site("given", "coordinates given on the command line",
                                    params.longitude, params.latitude,
                                    params.altitude)
        elif params.observatory:
            site = observatory.find(params.observatory)
            if site is None:
                raise ValueError(
                    f"no observatory called '{params.observatory}'. Known ones "
                    f"are {', '.join(observatory.listing())}; for anywhere else "
                    "give --latitude and --longitude, the longitude positive "
                    "west."
                )

        paths = [spec.path for spec in images.expand(params.images)]
        if not paths:
            raise ValueError(f"nothing matched {', '.join(params.images)}.")

        rows, warnings, changed = [], [], []
        for path in paths:
            with fits.open(path, mode="update" if params.update else "readonly") as hdus:
                header = hdus[0].header
                here = site or observatory.from_header(header)
                if here is None:
                    warnings.append(
                        f"{Path(path).name}: the header does not name an "
                        "observatory and none was given, so it was left alone.")
                    continue

                measured = obstime.airmass(header, here.latitude, here.east,
                                           here.altitude, params.ut, params.exposure)
                middle = obstime.middle(header, params.ut, params.exposure,
                                        key="__none__")
                jd = obstime.julian(header, params.ut, params.exposure)
                if measured is None or middle is None:
                    warnings.append(
                        f"{Path(path).name}: not enough in the header to compute "
                        "an airmass -- it needs RA, DEC, "
                        f"{params.date} and {params.ut}.")
                    continue

                before = header.get("AIRMASS")
                rows.append({
                    "image": Path(path).name,
                    "utmiddle": _clock(middle),
                    "airmass": round(measured.effective, 4),
                    "begin": round(measured.begin, 4),
                    "end": round(measured.end, 4),
                    "was": before,
                })
                if not params.update:
                    continue
                if before is not None and not params.override:
                    continue

                header["UTMIDDLE"] = (_clock(middle), "UT at mid-exposure")
                if jd is not None:
                    header["JD"] = (round(jd, 6), "julian date at mid-exposure")
                header["AIRMASS"] = (round(measured.effective, 4),
                                     "effective airmass over the exposure")
                header["AMSTART"] = (round(measured.begin, 4),
                                     "airmass at the start")
                header["AMEND"] = (round(measured.end, 4), "airmass at the end")
                # Two lines rather than one: a HISTORY card wraps at 72
                # columns, and a number split across the wrap is a number
                # nobody will find again with grep.
                header.add_history(
                    f"setairmass: {here.code} airmass {measured.effective:.4f} "
                    f"({measured.begin:.4f} to {measured.end:.4f})")
                if before is not None:
                    header.add_history(
                        f"setairmass: replacing AIRMASS {before}")
                changed.append(Path(path))

        if not rows:
            raise RuntimeError(
                "nothing could be computed: " + (warnings[0] if warnings else "")
            )

        site_used = site or observatory.from_header(fits.getheader(paths[0]))
        by_action["site"].measured = {"site": site_used.describe()}
        drifts = [row["end"] - row["begin"] for row in rows]
        by_action["effective"].measured = {
            "frames": len(rows),
            "airmass range": f"{min(row['airmass'] for row in rows):.3f} to "
                             f"{max(row['airmass'] for row in rows):.3f}",
            "largest drift within one exposure": f"{max(drifts):.3f}",
        }

        # The point of the task, said out loud: the number that was in the
        # header is not the number the exposure lived through.
        moved = [row for row in rows if row["was"] is not None
                 and abs(float(row["was"]) - row["airmass"]) > 0.01]
        if moved:
            worst = max(moved, key=lambda row: abs(float(row["was"]) - row["airmass"]))
            warnings.append(
                f"{len(moved)} of {len(rows)} frames had an airmass in the "
                "header that differs from the effective one by more than 0.01 "
                f"-- {worst['image']} had {worst['was']} and travelled through "
                f"{worst['airmass']:.3f}. The old value was almost certainly "
                "the airmass when the shutter opened; over a long exposure "
                "that understates the extinction."
            )

        return TaskResult(
            outputs=changed,
            steps=steps,
            warnings=warnings,
            data={"rows": rows},
            history=[],
        )


def _clock(hours: float) -> str:
    """Decimal hours as `hh:mm:ss.s`, which is how a header writes a time."""
    total = round(hours % 24.0 * 3600.0, 1)
    hour, rest = divmod(total, 3600.0)
    minute, second = divmod(rest, 60.0)
    return f"{int(hour):02d}:{int(minute):02d}:{second:04.1f}"


JDWHY = {
    "when": "The middle of the exposure, on a continuous count of days rather "
            "than a calendar. A Julian date has no months, no leap years and "
            "no midnight: subtracting two of them gives an interval, which "
            "subtracting two calendar dates does not.",
    "helio": "The Earth's orbit is 300 light-seconds across, so the same "
             "photon arriving in January and in July differs by up to 16.6 "
             "minutes of travel time. For anything periodic -- an eclipsing "
             "binary, a Cepheid, a transit -- that is the difference between a "
             "coherent light curve and a smeared one, and it is why a period "
             "is quoted against HJD and not JD.",
    "write": "JD and HJD go into the header, beside the UTMIDDLE they are "
             "computed from. Anything folding a light curve reads them from "
             "there rather than recomputing, so the correction is applied once "
             "and recorded.",
}


class SetJDParams(TaskParams):
    images: list[str] = Field(
        ..., description="Frames to stamp. Globs and @lists expand.",
        json_schema_extra={"positional": True},
    )
    observatory: str = Field(
        "", description="Where the telescope is. Read from OBSERVAT when left out."
    )
    jd: str = Field("JD", description="Keyword for the Julian date.")
    hjd: str = Field("HJD", description="Keyword for the heliocentric Julian date.")
    ljd: str = Field(
        "LJD", description="Keyword for the local Julian date -- the local "
                           "civil day the night belongs to."
    )
    barycentric: bool = Field(
        False,
        description="Correct to the solar system barycentre rather than the "
                    "Sun's centre. Better by up to 4 seconds, and what pulsar "
                    "timing uses.",
    )
    ut: str = Field("UT", description="Keyword holding the UT at the start.")
    exposure: str = Field("EXPTIME", description="Keyword holding the duration.")
    update: bool = Field(True, description="Write the results back.")


@register
class SetJD(Task[SetJDParams]):
    name = "setjd"
    package = "astutil"
    description = "Put Julian and heliocentric Julian dates in the headers."
    Params = SetJDParams

    def plan(self, params: SetJDParams) -> list[Step]:
        return [
            Step("when", "mid-exposure, as a Julian date", why=JDWHY["when"]),
            Step("helio",
                 "to the barycentre" if params.barycentric else "to the Sun",
                 why=JDWHY["helio"]),
            Step("write", f"{params.jd}, {params.hjd}, {params.ljd}"
                 if params.update else "nothing -- --no-update was given",
                 why=JDWHY["write"], skipped=not params.update,
                 reason="" if params.update else "reporting only"),
        ]

    def execute(self, params: SetJDParams) -> TaskResult:
        from astropy.io import fits

        from ..core import images, observatory, obstime

        site = (observatory.find(params.observatory) if params.observatory
                else None)
        if params.observatory and site is None:
            raise ValueError(
                f"no observatory called '{params.observatory}'. Known ones are "
                f"{', '.join(observatory.listing())}.")

        paths = [spec.path for spec in images.expand(params.images)]
        if not paths:
            raise ValueError(f"nothing matched {', '.join(params.images)}.")

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        kind = "barycentric" if params.barycentric else "heliocentric"

        rows, warnings, changed = [], [], []
        for path in paths:
            mode = "update" if params.update else "readonly"
            with fits.open(path, mode=mode) as hdus:
                header = hdus[0].header
                here = site or observatory.from_header(header)
                jd = obstime.julian(header, params.ut, params.exposure)
                if jd is None:
                    warnings.append(
                        f"{Path(path).name}: no Julian date could be computed "
                        f"-- it needs a date and {params.ut}.")
                    continue

                corrected = None
                if here is not None:
                    corrected = obstime.heliocentric(
                        header, here.latitude, here.east, here.altitude,
                        params.ut, params.exposure, kind)
                if corrected is None:
                    warnings.append(
                        f"{Path(path).name}: JD only. The "
                        f"{kind} correction needs RA, DEC and a site, and one "
                        "of them is missing -- for a lamp or a bias frame that "
                        "is expected.")

                # The local Julian day: which civil night this belongs to. The
                # half-day offset is what makes a whole night carry one number
                # instead of changing at midnight, which is the point.
                local = float(int(jd + 0.5 - (here.timezone / 24.0 if here else 0.0)))
                rows.append({
                    "image": Path(path).name,
                    "jd": round(jd, 6),
                    "hjd": round(corrected[0], 6) if corrected else None,
                    "correction s": round(corrected[1], 1) if corrected else None,
                    "ljd": local,
                })
                if not params.update:
                    continue
                header[params.jd] = (round(jd, 6), "julian date at mid-exposure")
                header[params.ljd] = (local, "local julian day of this night")
                if corrected:
                    header[params.hjd] = (
                        round(corrected[0], 6),
                        f"{kind} julian date at mid-exposure")
                    header.add_history(
                        f"setjd: JD {jd:.6f}, {kind} correction "
                        f"{corrected[1]:+.1f} s")
                else:
                    header.add_history(f"setjd: JD {jd:.6f}, no sky correction")
                changed.append(Path(path))

        if not rows:
            raise RuntimeError(
                "nothing could be computed: " + (warnings[0] if warnings else ""))

        by_action["when"].measured = {
            "frames": len(rows),
            "span": f"{max(row['jd'] for row in rows) - min(row['jd'] for row in rows):.5f} d",
        }
        corrections = [row["correction s"] for row in rows
                       if row["correction s"] is not None]
        if corrections:
            by_action["helio"].measured = {
                "kind": kind,
                "correction": (f"{corrections[0]:+.1f} s"
                               if min(corrections) == max(corrections) else
                               f"{min(corrections):+.1f} to {max(corrections):+.1f} s"),
                "of a maximum": "+-499 s, the light-crossing time of the orbit",
            }
        else:
            by_action["helio"].skipped = True
            by_action["helio"].reason = "no frame had both coordinates and a site"

        return TaskResult(outputs=changed, steps=steps, warnings=warnings,
                          data={"rows": rows}, history=[])


class AirmassParams(TaskParams):
    image: str = Field(
        "", description="Read the coordinates and time from this frame's header.",
        json_schema_extra={"positional": True},
    )
    ra: str = Field("", description="Right ascension, hh:mm:ss or decimal hours.")
    dec: str = Field("", description="Declination, dd:mm:ss or decimal degrees.")
    date: str = Field("", description="Observing date, yyyy-mm-dd or dd/mm/yy.")
    ut: str = Field("", description="UT at the start of the exposure.")
    exptime: float = Field(
        0.0, ge=0, description="Exposure length in seconds, for the effective airmass."
    )
    observatory: str = Field("", description="Where the telescope is.")
    latitude: float | None = Field(None, ge=-90, le=90, description="Degrees, positive north.")
    longitude: float | None = Field(
        None, description="Degrees, positive WEST as IRAF stores it."
    )
    altitude: float = Field(2000.0, description="Metres above sea level.")


@register
class Airmass(Task[AirmassParams]):
    name = "airmass"
    package = "astutil"
    description = "Work out the airmass for a position, a time and a site."
    Params = AirmassParams

    def plan(self, params: AirmassParams) -> list[Step]:
        return [
            Step("where", params.image or f"{params.ra} {params.dec}",
                 why="Airmass is a property of where the object was in the sky "
                     "as seen from one place at one moment, so all three have "
                     "to be known before there is anything to compute."),
            Step("compute", "sec z, corrected for refraction",
                 why=WHY["where"]),
        ]

    def execute(self, params: AirmassParams) -> TaskResult:
        from astropy.io import fits

        from ..core import observatory, obstime

        if params.image:
            header = fits.getheader(params.image)
        else:
            missing = [name for name, value in
                       (("--ra", params.ra), ("--dec", params.dec),
                        ("--date", params.date), ("--ut", params.ut))
                       if not value]
            if missing:
                raise ValueError(
                    "give an image to read from, or all of "
                    f"{', '.join(missing)}. There is no useful default for "
                    "where a telescope was pointing.")
            header = fits.Header({
                "RA": params.ra, "DEC": params.dec, "DATE-OBS": params.date,
                "UT": params.ut, "EXPTIME": params.exptime})

        site = _site(params, observatory, header)
        measured = obstime.airmass(header, site.latitude, site.east, site.altitude)
        if measured is None:
            raise ValueError(
                "not enough to compute an airmass, or the object was below "
                "three degrees -- where the plane-parallel atmosphere has "
                "stopped being a picture of anything.")

        lst = obstime.sidereal(header, site.east)
        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        by_action["where"].measured = {
            "site": site.describe(),
            "position": f"{header.get('RA')} {header.get('DEC')}",
            "mid-exposure UT": _clock(obstime.middle(header)),
        }
        by_action["compute"].measured = {
            "airmass": f"{measured.effective:.4f}",
            "start to end": f"{measured.begin:.4f} to {measured.end:.4f}",
            "sidereal time": "unknown" if lst is None else _clock(lst),
        }

        warnings = []
        if measured.effective > 2.0:
            warnings.append(
                f"{measured.effective:.2f} airmasses is low in the sky: "
                "differential refraction spreads the image into a short "
                "spectrum along the vertical, and unless the slit was turned "
                "to the parallactic angle some colours miss it.")

        return TaskResult(
            outputs=[], steps=steps, warnings=warnings,
            data={"airmass": round(measured.effective, 4),
                  "begin": round(measured.begin, 4),
                  "end": round(measured.end, 4),
                  "sidereal": None if lst is None else _clock(lst)},
            history=[])


class AstTimesParams(TaskParams):
    image: str = Field(
        "", description="Read the date and time from this frame's header.",
        json_schema_extra={"positional": True},
    )
    date: str = Field("", description="yyyy-mm-dd or dd/mm/yy.")
    ut: str = Field("", description="Universal time, hh:mm:ss.")
    observatory: str = Field("", description="Where the telescope is.")
    longitude: float | None = Field(
        None, description="Degrees, positive WEST as IRAF stores it."
    )
    latitude: float | None = Field(None, ge=-90, le=90, description="Degrees, positive north.")
    altitude: float = Field(2000.0, description="Metres above sea level.")


@register
class AstTimes(Task[AstTimesParams]):
    name = "asttimes"
    package = "astutil"
    description = "Convert between UT, Julian date, epoch and sidereal time."
    Params = AstTimesParams

    def plan(self, params: AstTimesParams) -> list[Step]:
        return [
            Step("read", params.image or f"{params.date} {params.ut}",
                 why="One instant, named four ways. Which name is right "
                     "depends on the question: a calendar date for a log, a "
                     "Julian date for an interval, an epoch for a coordinate "
                     "system, sidereal time for what is overhead."),
            Step("convert", "to JD, epoch and local sidereal time",
                 why="Sidereal time runs about four minutes a day fast on "
                     "solar time, because a day is one rotation *plus* the "
                     "bit the Earth moved around the Sun. That is why the "
                     "same stars rise four minutes earlier each night, and "
                     "why an observing plan is made in sidereal time."),
        ]

    def execute(self, params: AstTimesParams) -> TaskResult:
        from astropy.io import fits
        from astropy.time import Time

        from ..core import observatory, obstime

        if params.image:
            header = fits.getheader(params.image)
        else:
            if not params.date or not params.ut:
                raise ValueError(
                    "give an image to read from, or both --date and --ut.")
            header = fits.Header({"DATE-OBS": params.date, "UT": params.ut,
                                  "EXPTIME": 0.0})

        site = _site(params, observatory, header)
        jd = obstime.julian(header)
        if jd is None:
            raise ValueError(
                "no date and time could be read. It needs DATE-OBS and UT.")
        lst = obstime.sidereal(header, site.east)
        stamp = Time(jd, format="jd", scale="utc")
        parts = obstime.date(header)

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        by_action["read"].measured = {
            "site": site.describe(),
            "date": f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}",
            "ut": _clock(obstime.hours(header.get("UT")) or 0.0),
        }
        by_action["convert"].measured = {
            "julian date": f"{jd:.6f}",
            "modified julian date": f"{jd - 2400000.5:.6f}",
            "epoch": f"{stamp.jyear:.5f}",
            "local sidereal time": "unknown" if lst is None else _clock(lst),
        }

        return TaskResult(
            outputs=[], steps=steps, warnings=[],
            data={"jd": round(jd, 6), "mjd": round(jd - 2400000.5, 6),
                  "epoch": round(float(stamp.jyear), 5),
                  "sidereal": None if lst is None else _clock(lst)},
            history=[])


class ObservatoryParams(TaskParams):
    name: str = Field(
        "", description="A site to look up. Left out, every one is listed.",
        json_schema_extra={"positional": True},
    )


@register
class Observatory(Task[ObservatoryParams]):
    name = "observatory"
    package = "astutil"
    description = "List the observatories, or look one up."
    Params = ObservatoryParams

    def plan(self, params: ObservatoryParams) -> list[Step]:
        return [
            Step("look", params.name or "every site",
                 why="Longitude here is positive WEST, as IRAF's obsdb stores "
                     "it and the opposite of every modern library. A sign "
                     "error does not fail -- it puts the object on the other "
                     "side of the meridian and returns a perfectly plausible "
                     "wrong airmass -- so it is worth seeing the value with "
                     "its hemisphere spelled out."),
        ]

    def execute(self, params: ObservatoryParams) -> TaskResult:
        from ..core import observatory

        if params.name:
            site = observatory.find(params.name)
            if site is None:
                raise ValueError(
                    f"no observatory called '{params.name}'. Known ones are "
                    f"{', '.join(observatory.listing())}. For anywhere else, "
                    "setairmass and airmass take --latitude and --longitude.")
            sites = [site]
        else:
            sites = [observatory.SITES[code] for code in observatory.listing()]

        rows = [{"code": site.code, "name": site.name,
                 "latitude": round(site.latitude, 4),
                 "longitude W": round(site.longitude, 4),
                 "height m": round(site.altitude),
                 "timezone": site.timezone}
                for site in sites]
        steps = self.plan(params)
        steps[0].measured = {"sites": len(rows)}
        return TaskResult(outputs=[], steps=steps, warnings=[],
                          data={"rows": rows}, history=[])


def _site(params, observatory, header):
    """The site from the parameters, the header, or an error saying so."""
    if params.latitude is not None and params.longitude is not None:
        return observatory.Site("given", "coordinates given on the command line",
                                params.longitude, params.latitude,
                                params.altitude)
    if params.observatory:
        site = observatory.find(params.observatory)
        if site is None:
            raise ValueError(
                f"no observatory called '{params.observatory}'. Known ones are "
                f"{', '.join(observatory.listing())}; for anywhere else give "
                "--latitude and --longitude, the longitude positive west.")
        return site
    site = observatory.from_header(header)
    if site is None:
        raise ValueError(
            "no observatory: the header does not name one and none was given. "
            "Use --observatory, or --latitude and --longitude. "
            "'harwood observatory' lists the sites.")
    return site
