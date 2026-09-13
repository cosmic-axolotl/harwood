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
                    "'harwood observatories' lists them.",
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
