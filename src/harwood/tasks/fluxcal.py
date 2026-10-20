"""onedspec -- flux calibration: standard, sensfunc, calibrate.

Three tasks and one idea. Counts are a property of the telescope, the grating,
the detector, the slit and the weather; flux is a property of the star.
Getting from one to the other means observing something whose flux is already
known and measuring what this instrument did to it.

What makes it three tasks rather than one is that the middle step is shared.
`standard` measures one calibration star, and can be run again on another, and
another -- each run appends. `sensfunc` then fits all of them together, which
is the whole reason the extinction is taken out per band rather than per
exposure: bands from stars observed at different airmasses become comparable,
and the fit sees more of the wavelength range than any one star covers.

The files these read are IRAF's own, under `onedstds$`, because the standards
are somebody else's measurements and reproducing them would only introduce
errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

STDWHY = {
    "star": "The calibration star's true flux, measured by somebody else and "
            "tabulated in bands. Which table matters: the same star appears in "
            "several with different band widths and, sometimes, different "
            "fluxes, because they were calibrated against different primary "
            "standards.",
    "bands": "The observed spectrum is integrated over each band and divided by "
             "the band width and the exposure time, which gives counts per "
             "second per Angstrom -- directly comparable with a flux density. "
             "A band only partly covered by the spectrum is dropped rather "
             "than scaled up: a half-covered band looks exactly like a star "
             "half as bright, and afterwards nothing can tell the two apart.",
    "airmass": "The airmass is recorded with each band rather than applied "
               "here, so that sensfunc can remove the extinction per band and "
               "combine stars observed at different airmasses in one fit.",
    "write": "The measurements are appended, not replaced. Running standard on "
             "a second star adds to the same file, and sensfunc fits them "
             "together -- which is how the fit covers more wavelength than any "
             "single standard does.",
}


class StandardParams(TaskParams):
    input: str = Field(..., description="Wavelength-calibrated spectra of standard stars. "
                                        "Commas, globs and @lists expand.")
    output: Path = Field(
        Path("std"),
        description="Where the measurements go. Appended to if it exists.",
        json_schema_extra={"positional": True},
    )
    samestar: bool = Field(
        True, description="The same star in every aperture, so its table is read "
                          "once. Off, each aperture names its own (star_name in "
                          "aperture order).")
    beam_switch: bool = Field(
        False, description="Beam-switched data: each aperture comes as an object "
                           "(OFLAG 1) and a sky (OFLAG 0), in either order, and the sky "
                           "is subtracted. Implies samestar.")
    apertures: str = Field("", description="Apertures to measure, as a range list: "
                                           "'1,3-5'. Empty is all.")
    bandwidth: float | None = Field(
        None, gt=0, description="Measure in bands of this width (Angstroms) instead of "
                                "the table's, their magnitudes interpolated in it.")
    bandsep: float | None = Field(
        None, gt=0, description="The distance between those bands' centres. Left out, "
                                "the bandwidth.")
    fnuzero: float = Field(3.68e-20, gt=0,
                           description="The flux per hertz of magnitude zero, which "
                                       "turns the table's magnitudes into fluxes.")
    star_name: str = Field(
        "",
        description="The standard's name, as the calibration directory spells "
                    "it. Read from the OBJECT keyword when left out. With samestar "
                    "off, one per aperture, comma-separated.",
    )
    caldir: str = Field(
        "spec50cal",
        description="Which calibration directory: spec50cal, spec16cal, "
                    "oke1990, iidscal, blackbody... or a directory of your own. "
                    "'harwood standards' lists them.",
    )
    extinction: str = Field(
        "onedstds$kpnoextinct.dat",
        description="Extinction curve for the site. Each pixel is corrected for "
                    "the difference from the band's centre here; the rest is "
                    "sensfunc's.",
    )
    observatory: str = Field(
        "", description="Where the spectra were taken, for an airmass worked out "
                        "from RA, DEC and HA (or ST) when the header has none. Left "
                        "out, the header's OBSERVAT, then the project's default.")
    airmass: float | None = Field(
        None, ge=1,
        description="Airmass for a spectrum whose header has none and gives no way "
                    "to work it out. As IRAF does, it is then written into that "
                    "spectrum's header.",
    )
    exptime: float | None = Field(
        None, gt=0, description="Exposure time for a spectrum whose header has none; "
                                "written into the header, as IRAF does.")
    mag: float | None = Field(None, description="A blackbody calibration (caldir "
                                                "blackbody): the star's magnitude.")
    magband: str = Field("", description="The band of that magnitude: U B V R I J H "
                                         "K L Lprime M. Converted to the table's band "
                                         "through the star's colours.")
    teff: str = Field("", description="The blackbody's temperature (9500) or the "
                                      "star's spectral type (A0V).")
    interact: bool = Field(
        False, description="Show each spectrum with its bands to add some (a at both "
                           "ends) or delete some (d), q to go on. Bands changed stay "
                           "changed for that aperture's next spectrum, as in IRAF. "
                           "Off by default here, on in IRAF.")
    cursor: Path | None = Field(
        None, description="With interact, the keys from a file instead of the window: "
                          "'wavelength y 1 key' per line, IRAF's cursor file.")
    answer: Literal["no", "yes", "N", "Y", "NO", "YES", "NO!", "YES!"] = Field(
        "yes", description="IRAF's answer to 'Edit bandpasses?', given once for every "
                           "spectrum.")
    edge: float = Field(
        0.0, ge=0,
        description="Ignore this many Angstroms at each end of the spectrum, "
                    "where the response falls off. harwood's; IRAF has none.",
    )
    format: Literal["harwood", "iraf"] = Field(
        "harwood", description="The file written: harwood's annotated CSV, or IRAF's "
                               "own text, which IRAF's sensfunc reads. An existing file "
                               "keeps its format.")


@register
class StandardTask(Task[StandardParams]):
    """IRAF's standard, noao/onedspec/t_standard.x: core/stdstar.py."""

    name = "standard"
    package = "onedspec"
    description = "Measure a flux standard in bands, against its known flux."
    Params = StandardParams

    def plan(self, params: StandardParams) -> list[Step]:
        return [
            Step("star", params.star_name or "from the OBJECT keyword",
                 why=STDWHY["star"]),
            Step("bands", "integrate the spectrum over each band" + (
                 f", {params.bandwidth:g} A every {params.bandsep or params.bandwidth:g} A"
                 if params.bandwidth else ""), why=STDWHY["bands"]),
            Step("airmass", "recorded with each band", why=STDWHY["airmass"]),
            Step("write", str(params.output), why=STDWHY["write"]),
        ]

    def execute(self, params: StandardParams) -> TaskResult:
        from ..core import flux, images, ranges, stdfile, stdstar

        names = [str(spec) for spec in images.expand(params.input)]
        if not names:
            advice.fail(f"'{params.input}' matches nothing.",
                        "check the path; 'harwood sections' shows what a template "
                        "expands to.", FileNotFoundError)
        selected = ranges.parse(params.apertures)
        curve = flux.read_extinction(params.extinction) if params.extinction else None
        ext = stdstar.Extinction(curve)
        latitude = _latitude(params)
        notes: list[str] = []
        header_writes: list[str] = []
        spectra = []
        for name in names:
            spectra.extend(_spectra(name, selected, params, latitude, header_writes))
        if not spectra:
            advice.fail(f"no aperture of {', '.join(names)} is in '{params.apertures}'.",
                        "give --apertures as a range of the apertures the files hold.")

        names_given = [n.strip() for n in params.star_name.split(",") if n.strip()]
        asked: list[str] = []

        def calibration(ap: int):
            spec = next(s for s in spectra if s.ap == ap)
            if names_given:
                star = names_given[min(len(asked), len(names_given) - 1)]
            else:
                star = spec.object
            asked.append(star)
            if not star:
                advice.fail(
                    f"{spec.image} has no OBJECT keyword, so the standard cannot be "
                    "named from the file.",
                    "give --star-name. 'harwood standards <caldir>' lists the names "
                    "each calibration directory uses.")
            try:
                cal = stdstar.read_calibration(params.caldir, star, params.mag,
                                               params.magband, params.teff)
            except LookupError:
                advice.fail(*_no_such_star(star, params.caldir))
            cal.star = star
            return cal

        cursor = None
        if params.interact:
            if params.cursor is not None:
                if not Path(params.cursor).exists():
                    advice.fail(f"cursor file '{params.cursor}' does not exist",
                                "write 'wavelength y 1 key' lines, or leave --cursor "
                                "out for the window.", FileNotFoundError)
                cursor = stdstar.CursorFile(params.cursor)
            else:
                from ..viewer.standard_panel import WindowCursor

                cursor = WindowCursor()
        records = stdstar.run(
            spectra, ext, calibration, samestar=params.samestar,
            beam_switch=params.beam_switch, bandwidth=params.bandwidth,
            bandsep=params.bandsep, interact=params.interact,
            answer=lambda spec: params.answer, cursor=cursor, fnuzero=params.fnuzero,
            edge=params.edge, say=notes.append)
        written = [r for r in records if any(f != 0 for f in r.fluxes)]
        if not written:
            first = spectra[0]
            cal = calibration(first.ap)
            advice.fail(*_no_overlap(first, cal, asked[-1]))

        output = Path(params.output)
        kind = stdfile.format_of(output) if output.exists() else params.format
        existing = stdfile.read(output) if output.exists() else None
        if kind == "iraf":
            stdfile.append_iraf(output, records)
        else:
            stdfile.append(output, [m for r in records
                                    for m in stdfile.measurements(r, params.fnuzero)],
                           params.extinction, params.caldir)
        total = stdfile.read(output)

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        stars = sorted({r.star for r in records})
        by_action["star"].measured = {
            "star": ", ".join(stars),
            "table": ", ".join(sorted({f"{params.caldir}/{r.table.name}"
                                       for r in records if r.table})),
        }
        bands_written = sum(1 for r in records for f in r.fluxes if f != 0)
        by_action["bands"].measured = {
            "spectra": len(records),
            "apertures": ", ".join(str(a) for a in sorted({r.obj.ap for r in records})),
            "bands measured": bands_written,
        }
        by_action["airmass"].measured = {
            "airmass": ", ".join(sorted({f"{r.obj.airmass:.3f}" for r in records})),
            "exposure": ", ".join(sorted({f"{r.obj.exptime:.1f} s" for r in records})),
            "extinction curve": params.extinction,
        }
        if header_writes:
            by_action["airmass"].measured["written to headers"] = "; ".join(header_writes)
        by_action["write"].measured = {
            "format": kind,
            "bands in this run": bands_written,
            "bands in the file": len(total),
            "stars in the file": len({item.star for item in total.measurements}),
        }

        warnings = []
        if existing and kind == "harwood" and existing.extinction_name and \
                existing.extinction_name != params.extinction:
            warnings.append(advice.problem(
                f"the file already uses the extinction curve "
                f"'{existing.extinction_name}' and this run gives "
                f"'{params.extinction}'. sensfunc applies one curve to every "
                "band, so mixing sites in one file is a mistake it cannot see.",
                f"start a separate standards file for the other site, or re-run "
                f"with --extinction {existing.extinction_name} if that is the "
                "right curve for all of them."))
        if params.format != kind:
            warnings.append(advice.note(
                f"'{output}' is already in {kind}'s format, so these bands were added "
                f"in it rather than in {params.format}'s.",
                "give another --output to start a file in the other format."))
        negative = sum(1 for r in records for f in r.fluxes if f < 0)
        if negative:
            warnings.append(advice.note(
                f"{negative} bands have no positive counts and will drop "
                "out of the fit. That is usually the ends of the spectrum, "
                "where the response has fallen to nothing.",
                "nothing, if they are at the ends. If they are not, trim the "
                "ends with --edge, in Angstroms, and look at the extraction."))
        for message in notes:
            if "not found" in message and "Effective temperature" not in message:
                warnings.append(advice.warn(
                    message + ": beam-switched data come in object and sky pairs, "
                              "and this one has no partner, so it was not measured.",
                    "check the OFLAG keywords (1 object, 0 sky) and that every "
                    "object has its sky in the list."))

        rows = [{"spectrum": f"{r.obj.image}({r.obj.ap})", "wavelength": round(float(w), 2),
                 "width": round(float(d), 3), "magnitude": round(float(m), 3),
                 "counts": f"{float(f):.5g}"}
                for r in records for w, d, m, f in zip(r.waves, r.dwaves, r.mags,
                                                       r.fluxes, strict=True) if f != 0]
        return TaskResult(
            outputs=[output],
            steps=steps,
            warnings=warnings,
            data={"rows": rows, **({"text": notes} if notes else {})},
            history=[f"standard: {', '.join(stars)} from {params.caldir}, "
                     f"{bands_written} bands from {len(records)} spectra"],
        )


def _latitude(params) -> float | None:
    from ..core import observatory

    site = observatory.find(params.observatory) if params.observatory else None
    if params.observatory and site is None:
        advice.fail(f"unknown observatory '{params.observatory}'.",
                    "'harwood observatory list' shows the sites harwood knows.")
    if site is None:
        site = observatory.default()
    return None if site is None else site.latitude


def _spectra(name: str, selected, params, latitude, writes: list[str]) -> list:
    """shdr_open for each selected aperture of one file: the header's
    exposure time, airmass (or the one worked out, or the one given), OFLAG
    and EX-FLAG."""
    import numpy as np
    from astropy.io import fits

    from ..core import observatory, spectrum, stdstar

    apertures = spectrum.read(name)
    if apertures[0].wavelength is None or not np.any(apertures[0].wavelength):
        advice.fail(
            f"'{name}' is still in pixels. A flux standard has to "
            "be wavelength-calibrated first, or there is no way to know "
            "which band each count belongs to.",
            "run dispcor on it. That needs a dispersion solution, so "
            "identify on an arc comes before it.")
    header = fits.getheader(name)
    exptime = _header_real(header, "EXPOSURE")
    for keyword in ("ITIME", "EXPTIME"):
        exptime = _header_real(header, keyword, exptime)
    if exptime is None:
        if params.exptime is None:
            advice.fail(
                f"{name}: no exposure time. Counts per second is the only form that "
                "can be compared between exposures.",
                "give --exptime (it is then written into the header), or put "
                "EXPTIME there with hedit.")
        exptime = params.exptime
        _write_keyword(name, "EXPTIME", exptime)
        writes.append(f"{name} EXPTIME={exptime:g}")
    airmass = _header_real(header, "AIRMASS")
    if airmass is not None and airmass < 1:
        airmass = None
    if airmass is None:
        lat = latitude
        if lat is None:
            site = observatory.from_header(header)
            lat = site.latitude if site is not None else None
        if lat is not None:
            airmass = stdstar.airmass(*(_header_real(header, k)
                                        for k in ("RA", "DEC", "HA", "ST")), lat)
    if airmass is None:
        if params.airmass is None:
            advice.fail(
                f"{name}: no airmass, and no RA, DEC and HA (or ST) to work one out. "
                "Without it the extinction cannot be removed, and a standard "
                "observed at 1.8 airmasses looks like an instrument 30 per cent "
                "less sensitive than it is.",
                "give --airmass (it is then written into the header), or run "
                "setairmass to put the effective one there.")
        airmass = params.airmass
        _write_keyword(name, "AIRMASS", airmass)
        writes.append(f"{name} AIRMASS={airmass:g}")
    oflag = int(_header_real(header, "OFLAG", stdstar.OBJECT))
    ec = int(_header_real(header, "EX-FLAG", stdstar.ECNO))
    obj = str(header.get("OBJECT", "")).strip()
    out = []
    for aperture in apertures:
        if aperture.number not in selected:
            continue
        spec = stdstar.Spectrum(
            name, aperture.number, aperture.title or obj, aperture.wavelength,
            aperture.flux, float(np.float32(airmass)), float(np.float32(exptime)),
            oflag, ec, f"[SPEC,{aperture.number}]" if len(apertures) > 1 else "", obj)
        out.append(spec)
    return out


def _header_real(header, keyword: str, default=None):
    """shdr_gr: imgetr of the keyword (sexagesimal read as IRAF reads it),
    or the default."""
    from ..core.iraffmt import INDEF, ctod

    value = header.get(keyword)
    if value is None or isinstance(value, bool):
        return default
    number = ctod(str(value))
    if number is None or abs(number) >= INDEF:
        return default
    return float(number)


def _write_keyword(name: str, keyword: str, value: float) -> None:
    from astropy.io import fits

    with fits.open(name, mode="update") as hdul:
        hdul[0].header[keyword] = (float(value), "added by standard")


SENSWHY = {
    "read": "Every band that standard measured, from every star it was run on. "
            "One fit over all of them covers more wavelength than any single "
            "standard does.",
    "extinct": "The extinction is removed band by band, using the airmass of "
               "the exposure that band came from. This is what makes stars "
               "observed at different airmasses comparable: what is left after "
               "the atmosphere is taken out belongs to the instrument, and the "
               "instrument did not change between exposures.",
    "fit": "A smooth curve through the response, one per aperture: the "
           "instrument's sensitivity varies with wavelength because the "
           "grating, the coatings and the detector all do, and all of those are "
           "smooth -- so a low order is right and a high one is fitting the "
           "errors in the standard star's own calibration. It is IRAF's curfit, "
           "in IRAF's single precision, so the curve is IRAF's to the bit.",
    "write": "The fit is sampled from the standards' first to last wavelength at "
             "their finest dispersion, as IRAF's sensfunc writes it: "
             "<name>.0001.fits for aperture 1 and so on (<name>.fits with "
             "ignoreaps), which calibrate finds by itself. The statistics go to "
             "the logfile, as IRAF writes them.",
}


class SensFuncParams(TaskParams):
    standards: Path = Field(
        Path("std"), description="File the standard task wrote (harwood's or IRAF's).",
        json_schema_extra={"positional": True},
    )
    sensitivity: str = Field(
        "sens", description="Root name of the sensitivity spectra: <name>.0001.fits "
                            "for aperture 1, and so on.")
    apertures: str = Field("", description="Apertures to fit, as a range list: "
                                           "'1,3-5'. Empty is all.")
    ignoreaps: bool = Field(
        False, description="One sensitivity function for every aperture, <name>.fits.")
    logfile: str = Field(
        "logfile", description="Where the statistics of each fit are appended, as "
                               "IRAF writes them. Empty for none.")
    extinction: str = Field(
        "", description="Extinction curve. Taken from the standards file when "
                        "left out."
    )
    newextinction: str = Field(
        "extinct.dat", description="Where the extinction curve revised by a residual "
                                   "extinction (key e) is written.")
    function: Literal["chebyshev", "legendre", "spline3", "spline1"] = Field(
        "spline3", description="The curve fitted."
    )
    order: int = Field(
        6, ge=1,
        description="Number of terms, or pieces for a spline. Low: the "
                    "response is smooth.",
    )
    interactive: bool = Field(
        False, description="Work on the stars in IRAF's graphs before each fit is "
                           "written: delete, move, reweight, shift, composite, "
                           "residual extinction. Off by default here, on in IRAF.")
    graphs: str = Field("sr", description="The graphs shown: s sensitivity, r "
                                          "residuals, a residuals against airmass.")
    cursor: Path | None = Field(
        None, description="With interactive, the keys from a file instead of the "
                          "window: 'x y graph key' per line, IRAF's cursor file. The "
                          "weight w asks for and the answer e asks for are the next "
                          "line's ':' text (:0.25, :yes).")
    answer: Literal["no", "yes", "NO", "YES"] = Field(
        "yes", description="IRAF's answer to 'Fit aperture N interactively?', given "
                           "once: YES or yes every aperture, NO or no none.")
    database: Path = Field(Path("."), description="Reduction directory: the "
                                                  "sensitivity spectra and database/ "
                                                  "go there.")


@register
class SensFunc(Task[SensFuncParams]):
    """IRAF's sensfunc, noao/onedspec/sensfunc: core/sensfunc.py."""

    name = "sensfunc"
    package = "onedspec"
    description = "Fit the instrument's response from the measured standards."
    Params = SensFuncParams

    def plan(self, params: SensFuncParams) -> list[Step]:
        return [
            Step("read", str(params.standards), why=SENSWHY["read"]),
            Step("extinct", "per band, at that band's airmass",
                 why=SENSWHY["extinct"]),
            Step("fit", f"{params.function} order {params.order}",
                 why=SENSWHY["fit"]),
            Step("write", f"{params.sensitivity}" +
                 ("" if params.ignoreaps else ".<aperture>"), why=SENSWHY["write"]),
        ]

    def execute(self, params: SensFuncParams) -> TaskResult:
        import getpass
        import platform
        import time

        import numpy as np

        from ..core import database, flux, ranges, stdfile
        from ..core import sensfunc as sf

        if not Path(params.standards).exists():
            advice.fail(
                f"'{params.standards}' does not exist.",
                "it is written by the standard task; run that on a flux "
                "standard first.")
        said: list[str] = []
        aps = ranges.parse(params.apertures)
        if stdfile.format_of(params.standards) == "iraf":
            text = Path(params.standards).read_text(encoding="utf-8", errors="replace")
            stars = sf.read_iraf(text, aps, params.ignoreaps, said.append)
            standard = stdfile.read(params.standards)
        else:
            standard = stdfile.read(params.standards)
            stars = sf.from_measurements(standard, aps, params.ignoreaps)
        if not stars or sum(s.nwaves for s in stars[:-2]) < 2:
            advice.fail(
                f"'{params.standards}' holds no bands to fit"
                + (f" in apertures {params.apertures}" if params.apertures else "") + ".",
                "run standard again -- on another star, or with a smaller "
                "--edge if the spectrum was being trimmed too hard.")

        name = params.extinction or standard.extinction_name
        if not name:
            advice.fail(
                "no extinction curve: the standards file does not name one and "
                "--extinction was not given.",
                "give --extinction onedstds$kpnoextinct.dat, or the curve for "
                "your own site.")
        curve = flux.read_extinction(name)

        from ..core.provenance import _version
        system = (f"harwood {_version()} {getpass.getuser()}@{platform.node()} "
                  f"{time.strftime('%a %H:%M:%S %d-%b-%Y')}")
        mode = "ask" if params.interactive else "no"
        cursor = None
        if params.interactive:
            if params.cursor is not None:
                if not Path(params.cursor).exists():
                    advice.fail(f"cursor file '{params.cursor}' does not exist",
                                "write 'x y graph key' lines, or leave --cursor out "
                                "for the window.", FileNotFoundError)
                cursor = sf.CursorFile(params.cursor)
            else:
                from ..viewer.sensfunc_panel import WindowCursor

                cursor = WindowCursor()

        outputs: list[Path] = []
        ecvs = []
        fits_made = []
        logs: list[str] = []
        for aperture in sf.apertures(stars[:-2]):
            for star in stars[:-2]:
                star.flag = sf.INCLUDE if star.beam == aperture else sf.EXCLUDE
            session = sf.Session(stars, curve, params.function, params.order,
                                 params.graphs, system)
            if mode == "ask":
                said.append(f"Fit aperture {aperture} interactively? {params.answer}")
                if params.answer == "NO":
                    mode = "no"
                elif params.answer == "YES":
                    mode = "all"
            if mode == "all" or (mode == "ask" and params.answer == "yes"):
                cursor.edit(session)
                for message in session.messages:
                    if message.strip("\a"):
                        said.append(message.strip("\a"))
                for target, lines in session.files.items():
                    with open(Path(params.database) / target, "a",
                              encoding="utf-8") as handle:
                        handle.write("\n".join(lines) + "\n")
            w1, dw, nw, beam = sf.sampling(stars)
            values = sf.sample(session.cv, w1, dw, nw)
            root = params.sensitivity if params.ignoreaps else \
                f"{params.sensitivity}.{beam:04d}"
            title = ("Sensitivity function for all apertures" if params.ignoreaps else
                     f"Sensitivity function for aperture {beam}")
            image = flux.SensSpectrum(float(w1), float(dw), values).write(
                Path(params.database) / f"{root}.fits", title=title)
            said.append(f"{title} --> {root}")
            outputs.append(image)
            lines = (sf.stats(stars, session.function, session.order, session.npts,
                              session.rms, system)
                     + sf.vstats(stars, session.cv, curve, session.ecv))
            logs.extend(lines)
            ecvs.append(session.ecv)
            used = [s for s in stars[:-2] if s.flag != sf.EXCLUDE] + \
                [s for s in stars[-2:] if s.flag == sf.INCLUDE]
            waves = np.concatenate([s.waves for s in used])
            sens = np.concatenate([s.sens for s in used])
            kept = np.concatenate([s.wts != 0 for s in used]) & \
                np.concatenate([np.full(s.nwaves, s.flag == sf.INCLUDE) for s in used])
            solution = database.Solution(
                image=root, pixel=waves.astype(float), wavelength=sens.astype(float),
                function=session.function, order=session.order,
                rms=float(session.rms) if session.rms is not None else float("nan"),
                used=kept, units="mag", aperture=beam,
                extra={"sampled": str(image), "w1": float(w1), "dw": float(dw),
                       "nw": int(nw)})
            outputs.append(database.write(solution, params.database, prefix="sens"))
            fits_made.append((beam, session, int(kept.sum()), int(kept.size)))
        # Relative names are the reduction directory's, which is the
        # current one unless --database says otherwise -- as in IRAF.
        if params.logfile.strip():
            log = Path(params.database) / params.logfile.strip()
            with open(log, "a", encoding="utf-8") as handle:
                handle.write("\n".join(logs) + "\n")
            outputs.append(log)
        revised = sf.revised_extinction(curve, ecvs)
        if revised is not None and params.newextinction.strip():
            table = Path(params.database) / params.newextinction.strip()
            table.write_text("\n".join(revised) + "\n")
            outputs.append(table)

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        all_waves = np.concatenate([s.waves for s in stars[:-2]])
        by_action["read"].measured = {
            "observations": len(stars) - 2,
            "bands": int(sum(s.nwaves for s in stars[:-2])),
            "stars": ", ".join(sorted({s.image for s in stars[:-2]})),
            "covering": f"{all_waves.min():.0f} to {all_waves.max():.0f} A",
        }
        airmasses = [float(s.airmass) for s in stars[:-2]]
        by_action["extinct"].measured = {
            "curve": name,
            "airmass": (f"{min(airmasses):.3f}" if min(airmasses) == max(airmasses)
                        else f"{min(airmasses):.3f} to {max(airmasses):.3f}"),
        }
        by_action["fit"].measured = {
            f"aperture {beam}": f"{session.function} {session.order}, {kept} of {total} "
                                f"bands, rms {_rms(session.rms)}"
            for beam, session, kept, total in fits_made}
        by_action["write"].measured = {"spectra": ", ".join(
            str(p.name) for p in outputs if p.suffix == ".fits")}

        warnings = []
        worst = max((s.rms or 0.0) for _, s, _, _ in fits_made)
        percent = 100.0 * (10.0 ** (0.4 * worst) - 1.0)
        warnings.append(advice.note(
            f"the response scatters by {worst:.3f} mag about the fit, which "
            f"is {percent:.1f} per cent in flux. That is the systematic floor "
            "on every spectrum calibrated with this solution -- it is not "
            "reduced by observing the object for longer.",
            f"quote {percent:.1f} per cent as the systematic uncertainty on "
            "every flux from this solution. To make it smaller, observe more "
            "standards, not longer ones."))
        for line in said:
            if "Warning - exposure time" in line:
                warnings.append(advice.warn(line, "give standard the exposure time "
                                                  "(--exptime) and run it again."))
        first = fits_made[0][1]
        return TaskResult(
            outputs=outputs,
            steps=steps,
            warnings=warnings,
            data={"bands": int(sum(t for _, _, _, t in fits_made)),
                  "used": int(sum(k for _, _, k, _ in fits_made)),
                  "rms": round(float(first.rms), 4) if first.rms is not None else None,
                  "text": said},
            history=[f"sensfunc: {params.function} order {params.order}, "
                     f"{len(fits_made)} apertures, rms {_rms(first.rms)}"],
        )


def _rms(value) -> str:
    return "INDEF" if value is None else f"{value:.4f} mag"


CALWHY = {
    "read": "The extracted spectrum, in counts, and the sensitivity solution "
            "that says what this instrument does to flux at each wavelength.",
    "extinct": "The counts are corrected to what they would have been above "
               "the atmosphere, using this exposure's own airmass. This is "
               "where an airmass from the middle of the exposure matters "
               "rather than one from the shutter opening: at 1.8 airmasses in "
               "the blue the difference between them is a few per cent.",
    "apply": "Sensitivity is in magnitudes, so applying it is a division: "
             "F = counts / (exptime * 10^(0.4 S)). What comes out is a flux "
             "density in erg/cm2/s/Angstrom, which is a property of the star "
             "and not of the night it was observed on.",
    "write": "The same file shape as before, with the flux now physical. BUNIT "
             "says so, and the counts are not recoverable from it -- which is "
             "why the calibrated spectrum is a new file rather than an "
             "overwrite.",
}


class CalibrateParams(TaskParams):
    input: str = Field(..., description="Extracted, wavelength-calibrated spectrum.")
    output: Path | None = Field(
        None,
        description="Where to write. Left out, it goes to this run's own "
                    "directory.",
        json_schema_extra={"positional": True},
    )
    sensitivity: str = Field(
        "sens", description="Sensitivity function: the name sensfunc wrote "
                            "(<name>.fits in the reduction directory), or the path of "
                            "a sensitivity spectrum, IRAF's included."
    )
    extinction: str = Field(
        "onedstds$kpnoextinct.dat", description="Extinction curve for the site."
    )
    interp: Literal["nearest", "linear", "poly3", "poly5", "spline3"] = Field(
        "poly5", description="Interpolant in the sensitivity spectrum (onedspec.interp).")
    extinct: bool = Field(
        True, description="Correct for atmospheric extinction."
    )
    flux: bool = Field(
        True, description="Apply the sensitivity function. Off, only the "
                          "extinction is done."
    )
    ignoreaps: bool = Field(
        False, description="One sensitivity function for every aperture. Off, as in "
                           "IRAF, each aperture takes <sensitivity>.0001, .0002, ... "
                           "when they exist, and the single one when they do not.")
    fnu: bool = Field(False, description="Flux per hertz (erg/cm2/s/Hz) instead of per "
                                         "Angstrom.")
    airmass: float | None = Field(
        None, gt=0, description="Airmass. Read from the header when left out."
    )
    exptime: float | None = Field(
        None, gt=0, description="Exposure time. Read from the header when left out."
    )
    database: Path = Field(Path("."), description="Reduction directory holding database/.")


def _sensitivity_file(params, aperture: int | None) -> Path | None:
    name = (params.sensitivity if aperture is None
            else f"{params.sensitivity}.{aperture:04d}")
    candidates = [Path(name), Path(f"{name}.fits"), Path(params.database) / f"{name}.fits",
                  Path(params.database) / name]
    return next((c for c in candidates if c.is_file()), None)


@register
class Calibrate(Task[CalibrateParams]):
    name = "calibrate"
    package = "onedspec"
    description = "Apply the sensitivity function and extinction to a spectrum."
    Params = CalibrateParams

    def plan(self, params: CalibrateParams) -> list[Step]:
        steps = [Step("read", params.input, why=CALWHY["read"])]
        steps.append(Step("extinct", params.extinction, why=CALWHY["extinct"],
                          skipped=not params.extinct,
                          reason="" if params.extinct else "--no-extinct"))
        steps.append(Step("apply", f"database/sens{params.sensitivity}",
                          why=CALWHY["apply"], skipped=not params.flux,
                          reason="" if params.flux else "--no-flux"))
        steps.append(Step("write", str(params.output or "this run's directory"),
                          why=CALWHY["write"]))
        return steps

    def execute(self, params: CalibrateParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import flux, output, spectrum

        apertures = spectrum.read(params.input)
        if not apertures:
            advice.fail(f"'{params.input}' holds no extracted spectra.",
                        "run apall on the frame first; calibrate works on an "
                        "extracted spectrum, not on a 2-D image.")
        if apertures[0].wavelength is None or not any(apertures[0].wavelength):
            advice.fail(
                f"'{params.input}' is still in pixels. Flux calibration needs "
                "wavelengths -- both the sensitivity function and the "
                "extinction are functions of wavelength.",
                "run dispcor on it first.")

        header = fits.getheader(params.input)
        # t_calibrate: what the header says is done is not done again.
        doextinct = params.extinct and int(header.get("EX-FLAG", -1)) != 0
        doflux = params.flux and int(header.get("CA-FLAG", -1)) != 0
        if not (doextinct or doflux):
            advice.fail(f"[{params.input}] is already calibrated: its header has "
                        "EX-FLAG and CA-FLAG 0.",
                        "calibrate the spectrum before calibration, or set the flag "
                        "back with hedit if it is wrong.")
        already = [what for what, asked, done in (("extinction", params.extinct, doextinct),
                                                  ("flux", params.flux, doflux))
                   if asked and not done]
        params = params.model_copy(update={"extinct": doextinct, "flux": doflux})
        airmass = params.airmass or _header_number(header, "AIRMASS")
        exptime = params.exptime or _header_number(header, "EXPTIME", "ITIME")
        if exptime is None:
            advice.fail(
                "no exposure time, and a flux per second cannot be computed "
                "without one.",
                "give --exptime, or put EXPTIME in the header with hedit.")
        if airmass is None and params.extinct:
            advice.fail(
                "no airmass, and the extinction correction needs one.",
                "give --airmass, or run setairmass. --no-extinct turns the "
                "correction off, but then the flux is only right for a star at "
                "the zenith.")

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        curve = flux.read_extinction(params.extinction) if params.extinct else None

        sens = None
        covered = None
        per_aperture: dict[int, object] = {}
        if params.flux:
            if not params.ignoreaps:
                for aperture in apertures:
                    found = _sensitivity_file(params, aperture.number)
                    if found is not None:
                        per_aperture[aperture.number] = flux.SensSpectrum.read(found)
            found = _sensitivity_file(params, None)
            if found is None and len(per_aperture) < len(apertures):
                advice.fail(
                    f"no sensitivity function '{params.sensitivity}' (nor "
                    f"{params.sensitivity}.0001 and on, one per aperture).",
                    "run standard on a flux standard and then sensfunc; sensfunc "
                    "writes <name>.fits.",
                    FileNotFoundError)
            sens = flux.SensSpectrum.read(found) if found is not None else \
                next(iter(per_aperture.values()))
            covered = (sens.w1, sens.w1 + sens.dw * (sens.values.size - 1))

        outside = 0
        for aperture in apertures:
            calibrated, uncertainty, nout = flux.calibrate(
                aperture, per_aperture.get(aperture.number, sens), curve, airmass or 1.0,
                exptime, extinct=params.extinct, interp=params.interp, fnu=params.fnu)
            outside = max(outside, nout)
            aperture.flux = calibrated.astype(float)
            if uncertainty is not None:
                aperture.error = uncertainty.astype(float)

        first = apertures[0]
        wave = np.asarray(first.wavelength, dtype=float)
        by_action["read"].measured = {
            "apertures": len(apertures),
            "range": f"{wave.min():.1f} to {wave.max():.1f} A",
            "exposure": f"{exptime:.1f} s",
        }
        if params.extinct:
            coefficient = flux.extinction_at(wave, curve)
            gain = 10.0 ** (0.4 * coefficient * airmass)
            by_action["extinct"].measured = {
                "airmass": f"{airmass:.4f}",
                "correction at the blue end": f"x{gain[0]:.3f}",
                "correction at the red end": f"x{gain[-1]:.3f}",
            }
        if params.flux and sens is not None and covered is not None:
            by_action["apply"].measured = {
                "sensitivity": f"{params.sensitivity}, {sens.values.size} points",
                "covering": f"{covered[0]:.0f} to {covered[1]:.0f} A",
                "interpolant": params.interp,
            }

        warnings = []
        good = np.isfinite(first.flux) & (first.flux > 0)
        if good.any():
            median = float(np.median(first.flux[good]))
            middle = float(np.median(wave))
            by_action["write"].measured = {
                "median flux": f"{median:.4g} {flux.FLUX_UNIT}",
                "as a magnitude":
                    f"{flux.flux_to_magnitude(median, middle):.2f} at "
                    f"{middle:.0f} A",
            }
        if params.extinct and curve is not None:
            beyond = int(((wave < curve[0][0]) | (wave > curve[0][-1])).sum())
            if beyond:
                warnings.append(advice.warn(
                    f"{beyond} pixels lie outside the extinction table "
                    f"({curve[0][0]:.0f} to {curve[0][-1]:.0f} A). As in IRAF, the "
                    "curve's end parabola is continued there, which can run "
                    "away quickly.",
                    "use an extinction table covering the whole spectrum, or "
                    "trust only the part inside it."))
        if params.flux and covered is not None and outside:
            warnings.append(advice.warn(
                f"{outside} of {wave.size} pixels fall outside the "
                f"{covered[0]:.0f} to {covered[1]:.0f} A the sensitivity function "
                "covers. As in IRAF, they get the sensitivity at its nearer end, "
                "which is a guess.",
                f"trust the spectrum only between {covered[0]:.0f} and "
                f"{covered[1]:.0f} A, or run standard on a calibration star "
                "whose spectrum reaches further."))
        if already:
            warnings.append(advice.note(
                f"the {' and '.join(already)} correction was already applied "
                "(the header's flags say so), so it was not applied again.",
                "nothing, unless the flag is wrong; hedit can set it."))
        if not params.flux and "flux" not in already:
            warnings.append(advice.note(
                "--no-flux was given, so this spectrum is extinction-corrected "
                "counts, not flux. It is comparable between exposures of the "
                "same night and with nothing else.",
                "run standard and sensfunc on a flux standard, then drop "
                "--no-flux, to get a physical flux."))

        destination = params.output or output.named_for(params.input, ".f.fits")
        extra = header.copy()          # IRAF: NEW_COPY of the input
        unit = (flux.FNU_UNIT if params.fnu else flux.FLUX_UNIT) if params.flux else "counts"
        extra["BUNIT"] = (unit, "what the flux column is in")
        for keyword in ("EX-FLAG", "CA-FLAG"):
            if keyword in header:
                extra[keyword] = header[keyword]
        if params.extinct:
            extra["EX-FLAG"] = (0, "extinction corrected (IRAF's flag)")
        if params.flux:
            extra["CA-FLAG"] = (0, "flux calibrated (IRAF's flag)")
        spectrum.write(destination, apertures, header=extra,
                       source=str(params.input))

        return TaskResult(
            outputs=[Path(destination)],
            steps=steps,
            warnings=warnings,
            data={"apertures": len(apertures), "unit": unit},
            history=[f"calibrate: sens{params.sensitivity}"
                     + (f", extinction {params.extinction} at airmass "
                        f"{airmass:.3f}" if params.extinct else
                        ", no extinction correction")],
        )


def _header_number(header, *keywords) -> float | None:
    for keyword in keywords:
        value = header.get(keyword)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _no_such_star(name: str, caldir: str) -> tuple[str, str]:
    """A missing standard is usually a spelling or the wrong directory."""
    from ..core import flux

    elsewhere = flux.where_is(name)
    if elsewhere:
        return (f"'{name}' is not in {caldir}, though it is in "
                f"{', '.join(folder for folder, _ in elsewhere)}.",
                f"give --caldir {elsewhere[0][0]}. 'harwood standards --star "
                f"{name}' shows the band widths each one uses, which is the "
                "thing to choose between.")
    known = flux.stars(caldir)
    close = [item for item in known if name.lower()[:3] in item][:6]
    return (f"no standard called '{name}' in {caldir}.",
            (f"did you mean {', '.join(close)}?" if close else
             f"{caldir} holds {len(known)} stars; 'harwood standards {caldir}' "
             "lists them."))


def _no_overlap(spec, cal, name: str) -> tuple[str, str]:
    """The spectra and the standard's table do not meet."""
    return (
        f"none of {name}'s calibration bands falls inside the spectrum. The "
        f"table covers {min(cal.waves):.0f} to {max(cal.waves):.0f} "
        f"A and the spectrum covers {min(spec.wave):.0f} to "
        f"{max(spec.wave):.0f} A. A band has to be wholly inside the "
        "spectrum to be measured, since a partly covered band looks like a "
        "fainter star.",
        "check the dispersion solution put the spectrum where you expect, then "
        f"pick a calibration directory whose bands reach it -- 'harwood "
        f"standards --star {name}' shows the coverage of each.")
