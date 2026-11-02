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
from typing import Literal

from pydantic import Field

from ..core import advice
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
    "where": "The coordinates are precessed to the date, and the hour angle is "
             "the sidereal time -- the header's ST, as IRAF reads it, or worked "
             "out from the UT and the site's longitude when there is none -- "
             "minus the right ascension. With the declination and the "
             "latitude it gives the zenith distance, and the airmass is "
             "Allen's: the secant, rounded off near the horizon by an "
             "atmosphere `scale` times thinner than the Earth's radius.",
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


TIMES = Literal["beginning", "middle", "end"]


class SetAirmassParams(TaskParams):
    images: list[str] = Field(
        ..., description="Frames to stamp. Globs and @lists expand.",
        json_schema_extra={"positional": True},
    )
    observatory: str = Field(
        "",
        description="Where the telescope is -- 'kpno', 'ctio', 'lna'. Read "
                    "from the OBSERVAT keyword when left out, then from this "
                    "reduction's default ('harwood observatory set').",
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
    intype: TIMES = Field(
        "beginning", description="What the header's time marks: the beginning "
                                 "of the exposure, its middle, or its end.")
    outtype: Literal["beginning", "middle", "end", "effective"] = Field(
        "effective", description="Which airmass goes into the AIRMASS keyword.")
    ra: str = Field("RA", description="Keyword holding the right ascension, hours.")
    dec: str = Field("DEC", description="Keyword holding the declination, degrees.")
    equinox: str = Field("EPOCH", description="Keyword holding the coordinates' "
                                              "equinox, years.")
    st: str = Field("ST", description="Keyword holding the local sidereal time. "
                                      "Without one, it is worked out from the UT "
                                      "and the site's longitude.")
    ut: str = Field("UT", description="Keyword holding the UT.")
    date: str = Field("DATE-OBS", description="Keyword holding the date.")
    exposure: str = Field("EXPTIME", description="Keyword holding the duration, s.")
    airmass: str = Field("AIRMASS", description="Keyword the airmass is written to.")
    utmiddle: str = Field("UTMIDDLE", description="Keyword the mid-exposure UT is "
                                                  "written to, as IRAF writes it: "
                                                  "1992-01-25T02:18:22.00.")
    scale: float = Field(750.0, gt=0, description="The atmosphere's scale height, in "
                                                  "units of the Earth's radius.")
    update: bool = Field(
        True, description="Write the results back. Off, it only reports them."
    )
    override: bool = Field(
        True,
        description="Replace an airmass and a mid-UT already there. The old "
                    "airmass goes to HISTORY either way.",
    )


@register
class SetAirmass(Task[SetAirmassParams]):
    """IRAF's setairmass (t_setairmass.x), ported and checked against IRAF's
    output: the airmass at the beginning, middle and end of each exposure,
    combined by Simpson's rule into the effective one, and the UT of its
    middle. As in IRAF, the hour angle is ST minus RA after precessing the
    coordinates to the date, and the airmass is airmassx's; unlike IRAF, a
    header with no ST has it worked out from the UT and the site's longitude
    (IRAF's ast_mst) rather than being refused."""

    name = "setairmass"
    package = "astutil"
    description = "Compute the effective airmass and the mid-exposure time."
    Params = SetAirmassParams

    def plan(self, params: SetAirmassParams) -> list[Step]:
        return [
            Step("site", params.observatory or "from the OBSERVAT keyword",
                 why=WHY["site"]),
            Step("when", f"{params.ut} marks the {params.intype}; the middle from "
                         f"{params.exposure}", why=WHY["when"]),
            Step("where", f"hour angle {params.st} - {params.ra}, precessed from "
                          f"{params.equinox}; scale height {params.scale:g}",
                 why=WHY["where"]),
            Step("effective", "Simpson's rule over the exposure",
                 why=WHY["effective"]),
            Step("write", f"{params.airmass} ({params.outtype}), {params.utmiddle}"
                 if params.update else "nothing -- --no-update was given",
                 why=WHY["write"], skipped=not params.update,
                 reason="" if params.update else "reporting only"),
        ]

    def execute(self, params: SetAirmassParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import asttools, images, observatory

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
                advice.fail(
                    f"no observatory called '{params.observatory}'.",
                    f"use one of {', '.join(observatory.listing())}; for "
                    "anywhere else give --latitude and --longitude, the "
                    "longitude positive west.")

        paths = [spec.path for spec in images.expand(params.images)]
        if not paths:
            advice.fail(f"nothing matched {', '.join(params.images)}.",
                        "check the path and the pattern; 'harwood sections' "
                        "shows what a template expands to.")

        rows, warnings, changed, computed, unprecessed = [], [], [], set(), set()
        for path in paths:
            with fits.open(path, mode="update" if params.update else "readonly") as hdus:
                header = hdus[0].header
                here = site or observatory.from_header(header) or observatory.default()
                if here is None:
                    warnings.append(advice.warn(
                        f"{Path(path).name}: the header does not name an "
                        "observatory and none was given, so it was left alone.",
                        "give --observatory, or --latitude and --longitude. "
                        "'harwood observatory' lists the sites it knows."))
                    continue
                try:
                    found = _read_for_airmass(header, params, here, asttools)
                except ValueError as error:
                    warnings.append(advice.warn(
                        f"{Path(path).name}: {error}",
                        f"look with 'harwood imheader {Path(path).name} --long'. "
                        "A bias or a lamp often has no coordinates, and for "
                        "those this is expected."))
                    continue
                ha, dec, year, month, day, ut, exptime, st_given, precessed = found
                if not st_given:
                    computed.add(Path(path).name)
                if not precessed:
                    unprecessed.add(Path(path).name)
                sid = asttools.SOLTOSID * exptime
                beg, mid, end = {"beginning": (ha, ha + sid / 2, ha + sid),
                                 "middle": (ha - sid / 2, ha, ha + sid / 2),
                                 "end": (ha - sid, ha - sid / 2, ha)}[params.intype]
                shift = {"beginning": exptime / 2, "middle": 0.0,
                         "end": -exptime / 2}[params.intype]
                ut_mid = ut + shift
                # A middle past midnight belongs to the next day.
                jd = asttools.date_to_julday(year, month, day, ut_mid)
                year, month, day, ut_mid = asttools.julday_to_date(jd)
                airmasses = {"beginning": asttools.airmassx(beg, dec, here.latitude,
                                                            params.scale),
                             "middle": asttools.airmassx(mid, dec, here.latitude,
                                                         params.scale),
                             "end": asttools.airmassx(end, dec, here.latitude,
                                                      params.scale)}
                airmasses["effective"] = (airmasses["beginning"]
                                          + 4.0 * airmasses["middle"]
                                          + airmasses["end"]) / 6.0
                stamp = asttools.dtm_encode(year, month, day, ut_mid, 2)

                before = header.get(params.airmass)
                rows.append({
                    "image": Path(path).name,
                    "utmiddle": _clock(ut_mid),
                    "airmass": round(airmasses["effective"], 4),
                    "begin": round(airmasses["beginning"], 4),
                    "middle": round(airmasses["middle"], 4),
                    "end": round(airmasses["end"], 4),
                    "was": before,
                })
                if not params.update:
                    continue
                # IRAF writes the airmass as a real: single precision.
                value = float(np.float32(airmasses[params.outtype]))
                if before is None or params.override:
                    header[params.airmass] = (value, f"{params.outtype} airmass")
                    header.add_history(
                        f"setairmass: {here.code} airmass {airmasses['effective']:.4f} "
                        f"({airmasses['beginning']:.4f} to {airmasses['end']:.4f})")
                    if before is not None:
                        header.add_history(f"setairmass: replacing {params.airmass} "
                                           f"{before}")
                    # harwood's own, beside IRAF's: the two ends of the range.
                    header["AMSTART"] = (float(np.float32(airmasses["beginning"])),
                                         "airmass at the start")
                    header["AMEND"] = (float(np.float32(airmasses["end"])),
                                       "airmass at the end")
                if params.utmiddle not in header or params.override:
                    header[params.utmiddle] = (stamp, "UT at mid-exposure")
                header["JD"] = (round(jd, 6), "julian date at mid-exposure")
                changed.append(Path(path))

        if not rows:
            advice.fail(
                "nothing could be computed for any frame.",
                str(warnings[0].fix) if warnings else
                "check the headers carry RA, DEC, a date and a time.",
                RuntimeError)

        if computed:
            warnings.append(advice.note(
                f"{_count(computed)} had no {params.st} keyword, so the sidereal "
                "time was worked out from the UT and the site's longitude "
                "(IRAF's ast_mst). IRAF's setairmass would have refused "
                f"{'it' if len(computed) == 1 else 'them'}.",
                f"nothing, if {params.ut} is right; --st names another keyword"))
        if unprecessed:
            warnings.append(advice.note(
                f"{_count(unprecessed)} had no {params.equinox}, so the "
                "coordinates were taken as being of the date, unprecessed -- as "
                "IRAF does, and says.",
                "--equinox names the keyword, if the header has it under "
                "another name"))

        site_used = (site or observatory.from_header(fits.getheader(paths[0]))
                     or observatory.default())
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
            warnings.append(advice.note(
                f"{len(moved)} of {len(rows)} frames had an airmass in the "
                "header that differs from the effective one by more than 0.01 "
                f"-- {worst['image']} had {worst['was']} and travelled through "
                f"{worst['airmass']:.3f}. The old value was almost certainly "
                "the airmass when the shutter opened; over a long exposure "
                "that understates the extinction.",
                "nothing -- the new value is the better one. Use --no-override "
                "to keep what the header already had."))

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
        "", description="Where the telescope is, for its time zone. Read from "
                        "OBSERVAT when left out, then from this reduction's default.")
    date: str = Field("DATE-OBS", description="Keyword holding the date (and the "
                                              "time, when it is an ISO date).")
    time: str = Field("UT", description="Keyword holding the time, when the date "
                                        "does not.")
    exposure: str = Field("EXPTIME", description="Keyword holding the exposure, "
                                                 "seconds; the times are moved to "
                                                 "its middle. '-EXPTIME' moves "
                                                 "them back instead; empty, not "
                                                 "at all.")
    ra: str = Field("RA", description="Keyword holding the right ascension, hours.")
    dec: str = Field("DEC", description="Keyword holding the declination, degrees.")
    epoch: str = Field("EPOCH", description="Keyword holding the coordinates' epoch "
                                            "(B or J in front is read).")
    jd: str = Field("JD", description="Keyword for the Julian date; empty, none.")
    hjd: str = Field("HJD", description="Keyword for the heliocentric Julian date; "
                                        "empty, none.")
    ljd: str = Field(
        "LJD", description="Keyword for the local Julian day -- one number for a "
                           "whole night, changing at local noon; empty, none."
    )
    utdate: bool = Field(True, description="Is the date UT (rather than local)?")
    uttime: bool = Field(True, description="Is the time UT (rather than local)?")
    barycentric: bool = Field(
        False,
        description="harwood's own: correct to the solar system barycentre rather "
                    "than the Sun's centre, with astropy's ephemeris -- better by "
                    "up to 4 seconds, and what pulsar timing uses.",
    )
    listonly: bool = Field(False, description="List the dates without writing them.")


@register
class SetJD(Task[SetJDParams]):
    """IRAF's setjd (t_setjd.x), ported and checked against IRAF's output: the
    geocentric Julian date of mid-exposure, the heliocentric one by IRAF's
    ast_jd_to_hjd (an approximate Sun, good to about a second), and the local
    Julian day of the night, written in double precision as IRAF writes
    them. --barycentric uses astropy's ephemeris instead of IRAF's Sun."""

    name = "setjd"
    package = "astutil"
    description = "Put Julian and heliocentric Julian dates in the headers."
    Params = SetJDParams

    def plan(self, params: SetJDParams) -> list[Step]:
        return [
            Step("when", f"mid-exposure, from {params.date} and {params.time}",
                 why=JDWHY["when"]),
            Step("helio",
                 "to the barycentre" if params.barycentric else "to the Sun",
                 why=JDWHY["helio"], skipped=not params.hjd,
                 reason="" if params.hjd else "no hjd keyword"),
            Step("write", ", ".join(k for k in (params.jd, params.hjd, params.ljd) if k)
                 if not params.listonly else "nothing -- --listonly was given",
                 why=JDWHY["write"], skipped=params.listonly,
                 reason="reporting only" if params.listonly else ""),
        ]

    def execute(self, params: SetJDParams) -> TaskResult:
        from astropy.io import fits

        from ..core import asttools, images, observatory, obstime

        site = (observatory.find(params.observatory) if params.observatory
                else None)
        if params.observatory and site is None:
            advice.fail(
                f"no observatory called '{params.observatory}'.",
                f"use one of {', '.join(observatory.listing())}.")

        paths = [spec.path for spec in images.expand(params.images)]
        if not paths:
            advice.fail(f"nothing matched {', '.join(params.images)}.",
                        "check the path and the pattern; 'harwood sections' "
                        "shows what a template expands to.")

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        kind = "barycentric" if params.barycentric else "heliocentric"

        rows, warnings, changed = [], [], []
        for path in paths:
            mode = "readonly" if params.listonly else "update"
            with fits.open(path, mode=mode) as hdus:
                header = hdus[0].header
                here = site or observatory.from_header(header) or observatory.default()
                zone = here.timezone if here is not None else 0.0
                try:
                    found = _read_for_jd(header, params, zone, asttools, obstime)
                except ValueError as error:
                    warnings.append(advice.warn(
                        f"{Path(path).name}: {error}",
                        f"check {params.date} and {params.time} are in the header, "
                        "or name the keywords this camera uses with --date and "
                        "--time."))
                    continue
                ujd, ljd, epoch = found

                hjd = correction = None
                if params.hjd:
                    try:
                        ra = obstime.hours(header.get(params.ra))
                        dec = obstime.hours(header.get(params.dec))
                    except (TypeError, ValueError):
                        ra = dec = None
                    if ra is None or dec is None:
                        warnings.append(advice.note(
                            f"{Path(path).name}: JD only. The {kind} correction "
                            f"needs {params.ra} and {params.dec}, and the header "
                            "has not got them -- for a lamp or a bias frame that "
                            "is expected.",
                            "nothing, for a calibration frame; for an object, "
                            "name the keywords with --ra and --dec."))
                    else:
                        ep = _epoch(header.get(params.epoch)) if params.epoch else None
                        ra, dec = asttools.precess(ra, dec, ep, epoch)
                        if params.barycentric and here is not None:
                            corrected = obstime.heliocentric(
                                header, here.latitude, here.east, here.altitude,
                                params.time, params.exposure, kind)
                            if corrected:
                                hjd, correction = corrected[0], corrected[1]
                        else:
                            hjd, lt = asttools.jd_to_hjd(ra, dec, ujd)
                            correction = lt * 86400.0

                ljd = float(int(ljd))
                rows.append({
                    "image": Path(path).name,
                    "jd": round(ujd, 5),
                    "hjd": None if hjd is None else round(hjd, 5),
                    "correction s": None if correction is None else round(correction, 1),
                    "ljd": int(ljd),
                })
                if params.listonly:
                    continue
                if params.jd:
                    header[params.jd] = (ujd, "julian date at mid-exposure")
                if params.hjd and hjd is not None:
                    header[params.hjd] = (hjd, f"{kind} julian date at mid-exposure")
                if params.ljd:
                    header[params.ljd] = (ljd, "local julian day of this night")
                header.add_history(
                    f"setjd: JD {ujd:.6f}" + (f", {kind} correction {correction:+.1f} s"
                                               if correction is not None
                                               else ", no sky correction"))
                changed.append(Path(path))

        if not rows:
            advice.fail(
                "no Julian date could be computed for any frame.",
                str(warnings[0].fix) if warnings else
                "check the headers carry DATE-OBS and UT.", RuntimeError)

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
    elevation: float | None = Field(
        None, description="IRAF's airmass: the airmass at this elevation above the "
                          "horizon, degrees (radians with --radians), and nothing "
                          "else needed.")
    radians: bool = Field(False, description="The elevation is in radians.")
    scale: float = Field(750.0, gt=0, description="The atmosphere's scale height, in "
                                                  "units of the Earth's radius.")
    ra: str = Field("", description="Right ascension, hh:mm:ss or decimal hours.")
    dec: str = Field("", description="Declination, dd:mm:ss or decimal degrees.")
    epoch: float | None = Field(None, description="Equinox of --ra and --dec, years; "
                                                  "left out, they are of the date.")
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
    """The airmass of an exposure, from a frame's header or from coordinates,
    a date and a time -- by IRAF's arithmetic, the same as setairmass's: the
    hour angle from the header's ST (or worked out from the UT), the
    coordinates precessed to the date, and Allen's formula with the
    atmosphere's scale height. With --elevation it is IRAF's own airmass task:
    the airmass at an elevation, and nothing else needed."""

    name = "airmass"
    package = "astutil"
    description = "Work out the airmass for an elevation, or a position, time and site."
    Params = AirmassParams

    def plan(self, params: AirmassParams) -> list[Step]:
        if params.elevation is not None:
            return [Step("compute", f"elevation {params.elevation:g}"
                                    f"{' rad' if params.radians else ' deg'}, "
                                    f"scale {params.scale:g}", why=WHY["where"])]
        return [
            Step("where", params.image or f"{params.ra} {params.dec}",
                 why="Airmass is a property of where the object was in the sky "
                     "as seen from one place at one moment, so all three have "
                     "to be known before there is anything to compute."),
            Step("compute", f"hour angle and declination of the date; scale "
                            f"{params.scale:g}", why=WHY["where"]),
        ]

    def execute(self, params: AirmassParams) -> TaskResult:
        import math
        from types import SimpleNamespace

        from astropy.io import fits

        from ..core import asttools, observatory

        if params.elevation is not None:
            degrees = (math.degrees(params.elevation) if params.radians
                       else params.elevation)
            value = asttools.elevation_airmass(degrees, params.scale)
            steps = self.plan(params)
            steps[0].measured = {"airmass": f"{value:.5g}"}
            return TaskResult(steps=steps, data={
                "airmass": round(value, 5), "elevation": round(degrees, 5),
                "text": [f"airmass {value:.5g} at an elevation of {degrees:.5g} "
                         f"degrees ({math.radians(degrees):.5g} radians) above "
                         "horizon"]})

        if params.image:
            header = fits.getheader(params.image)
        else:
            missing = [name for name, value in
                       (("--ra", params.ra), ("--dec", params.dec),
                        ("--date", params.date), ("--ut", params.ut))
                       if not value]
            if missing:
                advice.fail(
                    f"not enough to work from: {', '.join(missing)} missing. "
                    "There is no useful default for where a telescope was "
                    "pointing.",
                    "give an image to read the header from, or all of "
                    f"{', '.join(missing)} on the command line; or --elevation "
                    "for the airmass at an elevation.")
            header = fits.Header({
                "RA": params.ra, "DEC": params.dec, "DATE-OBS": params.date,
                "UT": params.ut, "EXPTIME": params.exptime})
            if params.epoch:
                header["EPOCH"] = params.epoch

        site = _site(params, observatory, header)
        keys = SimpleNamespace(ra="RA", dec="DEC", equinox="EPOCH", st="ST", ut="UT",
                               date="DATE-OBS", exposure="EXPTIME")
        try:
            ha, dec, year, month, day, ut, exptime, st_given, _ = _read_for_airmass(
                header, keys, site, asttools)
        except ValueError as error:
            advice.fail(f"not enough to compute an airmass: {error}.",
                        "check RA, DEC, the date and the time.")
        sid = asttools.SOLTOSID * exptime
        begin, middle, end = (asttools.airmassx(h, dec, site.latitude, params.scale)
                              for h in (ha, ha + sid / 2, ha + sid))
        if min(begin, middle, end) > params.scale:
            advice.fail("the object was below the horizon.",
                        "check the time, the date and the site.")
        measured = SimpleNamespace(begin=begin, middle=middle, end=end,
                                   effective=(begin + 4 * middle + end) / 6)

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        by_action["where"].measured = {
            "site": site.describe(),
            "position": f"{header.get('RA')} {header.get('DEC')}",
            "sidereal time": f"{'the header ST' if st_given else 'from the UT'}",
        }
        by_action["compute"].measured = {
            "airmass": f"{measured.effective:.4f}",
            "start to end": f"{measured.begin:.4f} to {measured.end:.4f}",
        }
        from ..core import obstime

        start = obstime.hours(header.get("ST")) if st_given else asttools.mst(
            asttools.date_to_julday(year, month, day, ut), site.east)
        lst = (start + sid / 2) % 24.0               # at mid-exposure

        warnings = []
        if measured.effective > 2.0:
            warnings.append(advice.warn(
                f"{measured.effective:.2f} airmasses is low in the sky: "
                "differential refraction spreads the image into a short "
                "spectrum along the vertical, and unless the slit was turned "
                "to the parallactic angle some colours miss it.",
                "check the observing log for the slit angle. If it was not "
                "parallactic, treat the blue end of this spectrum as "
                "unreliable for flux."))

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
    files: list[str] = Field(
        [], description="IRAF's form: files of dates, one 'year month day "
                        "zone-time' per line.")
    year: int | None = Field(None, description="IRAF's form: the year.")
    month: int | None = Field(None, ge=1, le=12, description="The month.")
    day: int | None = Field(None, ge=1, le=31, description="The day.")
    time: float | None = Field(None, description="The local zone time, hours.")
    date: str = Field("", description="yyyy-mm-dd or dd/mm/yy, with --ut.")
    ut: str = Field("", description="Universal time, hh:mm:ss.")
    jd: float | None = Field(None, description="A Julian date, to turn back into a "
                                               "date and UT.")
    epoch: float | None = Field(None, description="A Julian epoch, to turn back into "
                                                  "a date and UT.")
    lmst: str = Field("", description="With --date: when, in UT, this local mean "
                                      "sidereal time comes round that day.")
    observatory: str = Field("", description="Where the telescope is.")
    longitude: float | None = Field(
        None, description="Degrees, positive WEST as IRAF stores it."
    )
    latitude: float | None = Field(None, ge=-90, le=90, description="Degrees, positive north.")
    altitude: float = Field(2000.0, description="Metres above sea level.")


@register
class AstTimes(Task[AstTimesParams]):
    """IRAF's asttimes (t_asttimes.x), ported and checked against IRAF's
    output: a date and local zone time at a site -- given as year, month, day
    and time, or a file of them -- as UT, Julian epoch, Julian date and local
    mean sidereal time, with the day of the week. harwood adds the date and
    time read from a frame's header (or --date and --ut), and the other way
    round: a Julian date or an epoch back to a date, and the UT at which a
    sidereal time comes round on a given night."""

    name = "asttimes"
    package = "astutil"
    description = "Convert between zone time, UT, Julian date, epoch and sidereal time."
    Params = AstTimesParams

    def plan(self, params: AstTimesParams) -> list[Step]:
        return [
            Step("read", params.image or ", ".join(params.files)
                 or (f"{params.year}-{params.month}-{params.day} {params.time}"
                     if params.year else f"{params.date} {params.ut}".strip()
                     or f"jd {params.jd}" if params.jd else f"epoch {params.epoch}"),
                 why="One instant, named four ways. Which name is right "
                     "depends on the question: a calendar date for a log, a "
                     "Julian date for an interval, an epoch for a coordinate "
                     "system, sidereal time for what is overhead."),
            Step("convert", "to UT, epoch, JD and local mean sidereal time",
                 why="Sidereal time runs about four minutes a day fast on "
                     "solar time, because a day is one rotation *plus* the "
                     "bit the Earth moved around the Sun. That is why the "
                     "same stars rise four minutes earlier each night, and "
                     "why an observing plan is made in sidereal time."),
        ]

    def execute(self, params: AstTimesParams) -> TaskResult:
        import math

        from astropy.io import fits

        from ..core import asttools, observatory, obstime

        header = fits.getheader(params.image) if params.image else fits.Header()
        site = _site(params, observatory, header)
        zone, west = site.timezone, site.longitude

        moments: list[tuple[int, int, int, float]] = []   # as zone times
        if params.files:
            for name in params.files:
                for line in Path(name).read_text(encoding="utf-8").splitlines():
                    words = line.split()
                    if len(words) >= 4 and not line.lstrip().startswith("#"):
                        try:
                            moments.append((int(words[0]), int(words[1]),
                                            int(words[2]), float(words[3])))
                        except ValueError:
                            continue
        elif params.year is not None:
            if None in (params.month, params.day, params.time):
                advice.fail("IRAF's form needs --year, --month, --day and --time",
                            "the time is the local zone time, in hours")
            moments.append((params.year, params.month, params.day, params.time))
        else:
            if params.jd is not None or params.epoch is not None:
                jd = params.jd if params.jd is not None else asttools.julday(params.epoch)
                year, month, day, ut = asttools.julday_to_date(jd)
            elif params.lmst:
                if not params.date:
                    advice.fail("--lmst needs --date: a sidereal time comes round "
                                "once a day", "harwood asttimes --date 2026-09-13 "
                                "--lmst 18:00 --observatory kpno")
                parts = obstime.date(fits.Header({"DATE-OBS": params.date}))
                start = asttools.mst(asttools.date_to_julday(*parts, 0.0), site.east)
                wanted = obstime.hours(params.lmst)
                year, month, day = parts
                ut = ((wanted - start) % 24.0) / asttools.SOLTOSID
            else:
                if params.image:
                    parts, ut = obstime.date(header), obstime.middle(header)
                elif params.date and params.ut:
                    parts = obstime.date(fits.Header({"DATE-OBS": params.date}))
                    ut = obstime.hours(params.ut)
                else:
                    advice.fail("no date and time given.",
                                "give an image, or --date and --ut, or IRAF's "
                                "--year --month --day --time (zone time), or "
                                "--jd or --epoch to turn back into a date")
                if parts is None or ut is None:
                    advice.fail("no date and time could be read.",
                                "it needs DATE-OBS and UT in the header, or --date "
                                "and --ut on the command line.")
                year, month, day = parts
            moments.append((year, month, day, ut - zone))

        rows = []
        for year, month, day, zt in moments:
            t = asttools.times(year, month, day, zt, zone, west)
            rows.append({"date": f"{t['year']:04d}-{t['month']:02d}-{t['day']:02d}",
                         "day": t["dow"], "zone time": _clock(t["zt"]),
                         "ut": _clock(t["ut"]), "epoch": f"{t['epoch']:.5f}",
                         # As IRAF prints it: halves round up.
                         "jd": f"{math.floor(t['jd'] * 1e4 + 0.5) / 1e4:.4f}",
                         "lmst": _clock(t["lmst"])})
        if not rows:
            advice.fail("no dates in the files", "one 'year month day time' a line")

        steps = self.plan(params)
        steps[0].measured = {"site": site.describe(), "dates": len(rows)}
        first = asttools.times(*moments[0], zone, west)
        jd = asttools.date_to_julday(*_ut_date(first), first["ut"])
        return TaskResult(
            outputs=[], steps=steps, warnings=[],
            data={"rows": rows, "jd": round(jd, 6), "mjd": round(jd - 2400000.5, 6),
                  "epoch": round(first["epoch"], 5), "sidereal": _clock(first["lmst"])},
            history=[])


def _ut_date(t: dict) -> tuple[int, int, int]:
    """The calendar date of the UT, which can be a day either side of the
    zone time's."""
    from ..core import asttools

    year, month, day, _ = asttools.julday_to_date(t["jd"])
    return year, month, day


class ObservatoryParams(TaskParams):
    command: str = Field(
        "list", description="list (the sites, or one), set (this reduction's "
                            "default, and a new site with --name, --latitude...), "
                            "or images (which site each image is from). A site's "
                            "name in its place is looked up, as `list` would.",
        json_schema_extra={"positional": True})
    obsid: str = Field("", description="The site to list or set.",
                       json_schema_extra={"positional": True})
    images: list[str] = Field([], description="For command images: the images.")
    name: str = Field("", description="For set: the full name of a new site.")
    longitude: float | None = Field(
        None, description="For set: longitude of a new site, degrees, positive WEST "
                          "as IRAF stores it (Kitt Peak is +111.6).")
    latitude: float | None = Field(None, ge=-90, le=90,
                                   description="For set: latitude, degrees north.")
    altitude: float = Field(0.0, description="For set: metres above sea level.")
    timezone: float = Field(0.0, description="For set: hours west of Greenwich.")


@register
class Observatory(Task[ObservatoryParams]):
    """IRAF's observatory: list the sites, set the one this reduction uses, or
    say which each image is from. The sites are IRAF's obsdb.dat, which ships
    with harwood; `set` with --name, --latitude and --longitude adds a site of
    the reduction's own, written in the same format to an obsdb.dat beside
    its harwood.yaml, where every task finds it."""

    name = "observatory"
    package = "astutil"
    description = "List the observatories, set this reduction's, or look one up."
    Params = ObservatoryParams

    def plan(self, params: ObservatoryParams) -> list[Step]:
        command, obsid = self._command(params)
        return [
            Step(command, obsid or "every site",
                 why="Longitude here is positive WEST, as IRAF's obsdb stores "
                     "it and the opposite of every modern library. A sign "
                     "error does not fail -- it puts the object on the other "
                     "side of the meridian and returns a perfectly plausible "
                     "wrong airmass -- so it is worth seeing the value with "
                     "its hemisphere spelled out."),
        ]

    @staticmethod
    def _command(params) -> tuple[str, str]:
        if params.command in ("list", "set", "images"):
            return params.command, params.obsid
        return "list", params.command        # `observatory kpno`, as before

    def execute(self, params: ObservatoryParams) -> TaskResult:
        from ..core import images as image_files
        from ..core import observatory
        from ..core import params as store

        command, obsid = self._command(params)
        outputs, warnings = [], []
        if command == "set":
            if not obsid:
                advice.fail("set needs a site", "harwood observatory set lna")
            if params.latitude is not None and params.longitude is not None:
                site = observatory.Site(obsid.lower(), params.name or obsid,
                                        params.longitude, params.latitude,
                                        params.altitude, params.timezone)
                outputs.append(observatory.add(site))
            else:
                site = self._find(observatory, obsid)
            data, path = store.load_project()
            if path is None:
                advice.fail("there is no harwood.yaml here to set it in",
                            "start the reduction with `harwood init`, then set it")
            text = path.read_text(encoding="utf-8")
            import re

            line = f"observatory: {site.code}"
            text = (re.sub(r"^observatory:.*$", line, text, flags=re.M)
                    if re.search(r"^observatory:", text, re.M) else
                    text.rstrip("\n") + f"\n\n# The site, when a header does not "
                                          f"name one.\n{line}\n")
            path.write_text(text, encoding="utf-8")
            outputs.append(path)
            sites = [site]
        elif command == "images":
            sites, rows = [], []
            for spec in image_files.expand(params.images or [obsid]):
                header = image_files.read_header(spec)
                site = observatory.from_header(header) or observatory.default()
                rows.append({"image": str(spec),
                             "OBSERVAT": header.get("OBSERVAT", "-"),
                             "site": site.code if site else "unknown",
                             "from": ("header" if observatory.from_header(header)
                                      else "this reduction's default" if site
                                      else "-")})
            return TaskResult(steps=self.plan(params), data={"rows": rows})
        elif obsid:
            sites = [self._find(observatory, obsid)]
        else:
            known = observatory.sites()
            sites = [known[code] for code in sorted(known)]

        rows = [{"code": site.code, "name": site.name,
                 "latitude": round(site.latitude, 4),
                 "longitude W": round(site.longitude, 4),
                 "height m": round(site.altitude),
                 "timezone": site.timezone}
                for site in sites]
        steps = self.plan(params)
        steps[0].measured = {"sites": len(rows)}
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings,
                          data={"rows": rows}, history=[])

    @staticmethod
    def _find(observatory, name: str):
        site = observatory.find(name)
        if site is None:
            advice.fail(
                f"no observatory called '{name}'.",
                f"use one of {', '.join(observatory.listing())}; or add it: "
                f"harwood observatory set {name} --name '...' --latitude ... "
                "--longitude ... (positive west) --altitude ...")
        return site


def _epoch(text) -> float | None:
    """An epoch keyword as setjd reads it: a number, or B1950/J2000."""
    if text is None:
        return None
    word = str(text).strip().upper()
    if word[:1] in ("B", "J"):
        word = word[1:]
    try:
        return float(word) or None
    except ValueError:
        advice.fail(f"the epoch '{text}' is not understood",
                    "a year such as 2000, J2000 or B1950")


def _read_for_jd(header, params, zone: float, asttools, obstime):
    """t_setjd.x: the UT Julian date of mid-exposure, the local one, and the
    epoch of the observation."""
    raw = header.get(params.date)
    parts = obstime.date(header, params.date)
    if raw is None or parts is None:
        raise ValueError(f"no {params.date} in the header")
    year, month, day = parts
    text = str(raw).strip()
    time = obstime.hours(text) if "T" in text else None
    if time is None:
        time = obstime.hours(header.get(params.time))
        if time is None:
            raise ValueError(f"no time: {params.date} has none and there is no "
                             f"{params.time}")
    key = params.exposure.strip()
    if key:
        sign = -1.0 if key.startswith("-") else 1.0
        exposure = header.get(key.lstrip("-"))
        if exposure is None:
            raise ValueError(f"no {key.lstrip('-')} in the header")
        time = time + sign * float(exposure) / (2 * 3600.0)
    epoch = asttools.date_to_epoch(year, month, day, time)
    ujd = asttools.julday(epoch)
    if params.utdate:
        if params.uttime:
            ljd = ujd - zone / 24.0
        else:
            ljd, ujd = ujd, ujd + zone / 24.0
    elif params.uttime:
        if time - zone < 0.0:
            ujd = ujd + 1
        if time + zone >= 24.0:
            ujd = ujd - 1
        ljd = ujd - zone / 24.0
    else:
        ljd, ujd = ujd, ujd + zone / 24.0
    return ujd, ljd, epoch


def _count(names: set[str]) -> str:
    listed = ", ".join(sorted(names)[:4]) + (" and others" if len(names) > 4 else "")
    return f"{len(names)} frame{'s' if len(names) != 1 else ''} ({listed})"


def _read_for_airmass(header, params, site, asttools):
    """sa_rheader (t_setairmass.x): hour angle, declination of the date, the
    date, the UT and the exposure in hours, from the header -- plus whether
    the ST was there and whether the coordinates were precessed."""
    from ..core import obstime

    def need(key: str):
        value = header.get(key)
        if value is None or str(value).strip() == "":
            raise ValueError(f"no {key} in the header")
        return value

    ra1 = obstime.hours(need(params.ra))
    if params.ra.upper().startswith("CRVAL"):
        ra1 = ra1 / 15.0                                 # degrees, as IRAF reads it
    dec1 = obstime.hours(need(params.dec))
    ut = obstime.hours(need(params.ut))
    parts = obstime.date(header, params.date)
    if ra1 is None or dec1 is None or ut is None or parts is None:
        raise ValueError(f"cannot read {params.ra}, {params.dec}, {params.ut} or "
                         f"{params.date}")
    year, month, day = parts
    exposure = header.get(params.exposure)
    if exposure is None:
        raise ValueError(f"no {params.exposure} in the header")
    exptime = float(exposure) / 3600.0

    try:
        epoch1 = float(header.get(params.equinox))
    except (TypeError, ValueError):
        epoch1 = None
    precessed = bool(epoch1)
    if precessed:
        epoch2 = asttools.date_to_epoch(year, month, day, ut)
        ra2, dec2 = asttools.precess(ra1, dec1, epoch1, epoch2)
    else:
        ra2, dec2 = ra1, dec1

    st = obstime.hours(header.get(params.st))
    st_given = st is not None
    if not st_given:
        st = asttools.mst(asttools.date_to_julday(year, month, day, ut), site.east)
    return st - ra2, dec2, year, month, day, ut, exptime, st_given, precessed


def _site(params, observatory, header):
    """The site from the parameters, the header, or an error saying so."""
    if params.latitude is not None and params.longitude is not None:
        return observatory.Site("given", "coordinates given on the command line",
                                params.longitude, params.latitude,
                                params.altitude)
    if params.observatory:
        site = observatory.find(params.observatory)
        if site is None:
            advice.fail(
                f"no observatory called '{params.observatory}'.",
                f"use one of {', '.join(observatory.listing())}; for anywhere "
                "else give --latitude and --longitude, the longitude positive "
                "west.")
        return site
    site = observatory.from_header(header) or observatory.default()
    if site is None:
        advice.fail(
            "no observatory: the header does not name one and none was given.",
            "give --observatory, or --latitude and --longitude, or set this "
            "reduction's own with `harwood observatory set <code>`. "
            "'harwood observatory' lists the sites it knows.")
    return site


# ------------------------------------------------------------------ precess

class PrecessParams(TaskParams):
    input: list[str] = Field(
        ..., description="Files of coordinates: ra dec, and if different the years of "
                         "that line. STDIN reads them from the terminal.",
        json_schema_extra={"positional": True},
    )
    startyear: float = Field(1950.0, description="The equinox of the coordinates.")
    endyear: float = Field(1984.0, description="The year to precess them to.")
    stdepoch: float = Field(0.0, description="A second year to give each in, 0 for none.")


def scan_doubles(line: str) -> list[float]:
    """fscan and gargd: the numbers at the start of a line, sexagesimal or not."""
    import re

    from ..core import iraffmt

    values = []
    for word in line.split():
        found = re.match(r"[-+]?(\d+\.?\d*|\.\d+)(:\d+\.?\d*)*([eEdD][-+]?\d+)?", word)
        if found is None:
            break
        values.append(iraffmt.ctod(found[0]))
        if found.end() != len(word):
            break
    return values


@register
class Precess(Task[PrecessParams]):
    """IRAF's precess: coordinates in a file from one equinox to another, printed as it
    prints them."""

    name = "precess"
    package = "astutil"
    description = "Precess a list of coordinates to another year."
    Params = PrecessParams

    def plan(self, params: PrecessParams) -> list[Step]:
        return [Step("compute", f"from {params.startyear:g} to {params.endyear:g}",
                     why="A catalogue's coordinates and the telescope's are of different "
                         "years, and the sky has turned between them.")]

    def execute(self, params: PrecessParams) -> TaskResult:
        import sys

        from ..core import asttools, images, iraffmt

        hms = iraffmt.Format("%13.2h")
        year = iraffmt.Format("%7.1f")
        second = iraffmt.Format("%18.2h")
        dec2 = iraffmt.Format("%12.2h")
        lines, warnings = [], []
        names = [word for item in params.input for word in images.split_list(item)]
        for name in names:
            if name == "STDIN":
                text = sys.stdin.read()
            else:
                file = Path(name)
                if not file.is_file():
                    advice.fail(f"Cannot open file ({name})", "check the name of the list")
                text = file.read_text(errors="replace", encoding="utf-8")
            for line in text.splitlines():
                values = scan_doubles(line)
                if len(values) < 2:
                    warnings.append(advice.warn("Bad entry in coordinate list", line[:60]))
                    continue
                ra, dec = values[0], values[1]
                y1 = values[2] if len(values) > 2 else params.startyear
                y2 = values[3] if len(values) > 3 else params.endyear
                ra2, dec2_ = asttools.precess(ra, dec, y1, y2)
                out = f"{hms(ra2)} {hms(dec2_)} {year(y2)}"
                if params.stdepoch != 0:
                    ra3, dec3 = asttools.precess(ra, dec, y1, params.stdepoch)
                    out += f"{second(ra3)} {dec2(dec3)} {year(params.stdepoch)}"
                lines.append(out)
        return TaskResult(data={"coordinates": len(lines)}, stdout=lines, warnings=warnings,
                          steps=self.plan(params))
