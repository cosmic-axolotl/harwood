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
    input: str = Field(..., description="Calibrated spectrum of a standard star.")
    output: Path = Field(
        Path("std"),
        description="Where the measurements go. Appended to if it exists.",
        json_schema_extra={"positional": True},
    )
    star_name: str = Field(
        "",
        description="The standard's name, as the calibration directory spells "
                    "it. Read from the OBJECT keyword when left out.",
    )
    caldir: str = Field(
        "spec50cal",
        description="Which calibration directory: spec50cal, spec16cal, "
                    "oke1990, iidscal. 'harwood standards' lists them.",
    )
    extinction: str = Field(
        "onedstds$kpnoextinct.dat",
        description="Extinction curve for the site. Recorded now, applied by "
                    "sensfunc.",
    )
    airmass: float | None = Field(
        None, gt=0,
        description="Airmass of the exposure. Read from the header when left "
                    "out -- setairmass puts the effective one there.",
    )
    exptime: float | None = Field(
        None, gt=0, description="Exposure time. Read from the header when left out."
    )
    aperture: int = Field(1, ge=1, description="Which aperture to measure.")
    edge: float = Field(
        0.0, ge=0,
        description="Ignore this many Angstroms at each end of the spectrum, "
                    "where the response falls off.",
    )


@register
class StandardTask(Task[StandardParams]):
    name = "standard"
    package = "onedspec"
    description = "Measure a flux standard in bands, against its known flux."
    Params = StandardParams

    def plan(self, params: StandardParams) -> list[Step]:
        return [
            Step("star", params.star_name or "from the OBJECT keyword",
                 why=STDWHY["star"]),
            Step("bands", "integrate the spectrum over each band",
                 why=STDWHY["bands"]),
            Step("airmass", "recorded with each band", why=STDWHY["airmass"]),
            Step("write", str(params.output), why=STDWHY["write"]),
        ]

    def execute(self, params: StandardParams) -> TaskResult:
        from astropy.io import fits

        from ..core import flux, spectrum, stdfile

        found = spectrum.read(params.input, number=params.aperture)
        if not found:
            advice.fail(
                f"'{params.input}' has no aperture {params.aperture}; it holds "
                f"{spectrum.count(params.input)}.",
                "use --aperture with a number in that range.")
        aperture = found[0]
        if aperture.wavelength is None or not any(aperture.wavelength):
            advice.fail(
                f"'{params.input}' is still in pixels. A flux standard has to "
                "be wavelength-calibrated first, or there is no way to know "
                "which band each count belongs to.",
                "run dispcor on it. That needs a dispersion solution, so "
                "identify on an arc comes before it.")

        header = fits.getheader(params.input)
        name = params.star_name or str(header.get("OBJECT", "")).strip()
        if not name:
            advice.fail(
                "the header has no OBJECT keyword, so the standard cannot be "
                "named from the file.",
                "give --star-name. 'harwood standards <caldir>' lists the "
                "names each calibration directory uses.")

        path = flux.find_star(name, params.caldir)
        if path is None:
            advice.fail(*_no_such_star(name, params.caldir))

        airmass = params.airmass or _header_number(header, "AIRMASS")
        exptime = params.exptime or _header_number(header, "EXPTIME", "ITIME")
        if airmass is None:
            advice.fail(
                "no airmass. Without it the extinction cannot be removed, and "
                "a standard observed at 1.8 airmasses looks like an instrument "
                "30 per cent less sensitive than it is.",
                "give --airmass, or run setairmass to put the effective one in "
                "the header.")
        if exptime is None:
            advice.fail(
                "no exposure time. Counts per second is the only form that can "
                "be compared between exposures.",
                "give --exptime, or put EXPTIME in the header with hedit.")

        bands = flux.read_bands(path)
        # IRAF's standard corrects each pixel for the extinction difference
        # from the band's centre, so the curve is needed here already.
        curve = flux.read_extinction(params.extinction) if params.extinction else None
        measured = flux.measure(aperture, bands, exptime, airmass,
                                star=name, edge=params.edge, curve=curve)
        if not measured:
            advice.fail(*_no_overlap(aperture, bands, name, path))

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        by_action["star"].measured = {
            "star": name,
            "table": f"{params.caldir}/{path.name}",
            "bands in the table": len(bands),
            "covering": f"{bands[0].wavelength:.0f} to "
                        f"{bands[-1].wavelength:.0f} A",
        }
        by_action["bands"].measured = {
            "bands measured": f"{len(measured)} of {len(bands)}",
            "spectrum covers": f"{min(aperture.wavelength):.0f} to "
                               f"{max(aperture.wavelength):.0f} A",
            "band width": f"{measured[0].width:.0f} A",
        }
        by_action["airmass"].measured = {
            "airmass": f"{airmass:.4f}",
            "exposure": f"{exptime:.1f} s",
            "extinction curve": params.extinction,
        }

        existing = stdfile.read(params.output) if Path(params.output).exists() else None
        total = stdfile.append(params.output, measured, params.extinction,
                               params.caldir)
        by_action["write"].measured = {
            "bands in this run": len(measured),
            "bands in the file": total,
            "stars in the file": len({item.star for item
                                      in stdfile.read(params.output).measurements}),
        }

        warnings = []
        if existing and existing.extinction_name != params.extinction:
            warnings.append(advice.problem(
                f"the file already uses the extinction curve "
                f"'{existing.extinction_name}' and this run gives "
                f"'{params.extinction}'. sensfunc applies one curve to every "
                "band, so mixing sites in one file is a mistake it cannot see.",
                f"start a separate standards file for the other site, or re-run "
                f"with --extinction {existing.extinction_name} if that is the "
                "right curve for all of them."))
        negative = [item for item in measured if item.counts <= 0]
        if negative:
            warnings.append(advice.note(
                f"{len(negative)} bands have no positive counts and will drop "
                "out of the fit. That is usually the ends of the spectrum, "
                "where the response has fallen to nothing.",
                "nothing, if they are at the ends. If they are not, trim the "
                "ends with --edge, in Angstroms, and look at the extraction."))

        return TaskResult(
            outputs=[Path(params.output)],
            steps=steps,
            warnings=warnings,
            data={"rows": [{"wavelength": round(item.wavelength, 1),
                            "width": round(item.width, 1),
                            "magnitude": round(item.magnitude, 3),
                            "flux": f"{item.flux:.4g}",
                            "counts/s/A": f"{item.counts:.5g}"}
                           for item in measured]},
            history=[f"standard: {name} from {params.caldir}, {len(measured)} "
                     f"bands at airmass {airmass:.3f}"],
        )


SENSWHY = {
    "read": "Every band that standard measured, from every star it was run on. "
            "One fit over all of them covers more wavelength than any single "
            "standard does.",
    "extinct": "The extinction is removed band by band, using the airmass of "
               "the exposure that band came from. This is what makes stars "
               "observed at different airmasses comparable: what is left after "
               "the atmosphere is taken out belongs to the instrument, and the "
               "instrument did not change between exposures.",
    "fit": "A smooth curve through the response. The instrument's sensitivity "
           "varies with wavelength because the grating, the coatings and the "
           "detector all do, and all of those are smooth -- so a low order is "
           "right and a high one is fitting the errors in the standard star's "
           "own calibration.",
    "write": "The fit is sampled from the standards' first to last wavelength at "
             "their finest dispersion, as IRAF's sensfunc writes it, into "
             "<name>.fits beside database/; the bands and the fit go to "
             "database/sens<name>, the rejected ones kept.",
}


class SensFuncParams(TaskParams):
    standards: Path = Field(
        Path("std"), description="File the standard task wrote.",
        json_schema_extra={"positional": True},
    )
    sensitivity: str = Field(
        "sens", description="Name for the solution, written to database/sens<name>."
    )
    extinction: str = Field(
        "", description="Extinction curve. Taken from the standards file when "
                        "left out."
    )
    function: str = Field(
        "spline3", description="chebyshev, legendre, spline3 or spline1."
    )
    order: int = Field(
        6, ge=1,
        description="Number of terms, or pieces for a spline. Low: the "
                    "response is smooth.",
    )
    low_reject: float = Field(3.0, ge=0, description="Reject below this many sigma.")
    high_reject: float = Field(3.0, ge=0, description="Reject above this many sigma.")
    niterate: int = Field(
        0, ge=0, description="Rejection passes. IRAF's sensfunc rejects nothing on its "
                             "own -- bands are deleted by hand -- so 0, as there.")
    database: Path = Field(Path("."), description="Reduction directory holding database/.")
    interactive: bool = Field(
        False, description="Fit in the icfit panel, where the order can be changed."
    )


@register
class SensFunc(Task[SensFuncParams]):
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
            Step("write", f"database/sens{params.sensitivity}",
                 why=SENSWHY["write"]),
        ]

    def execute(self, params: SensFuncParams) -> TaskResult:
        import numpy as np

        from ..core import database, flux, icfit, stdfile

        if not Path(params.standards).exists():
            advice.fail(
                f"'{params.standards}' does not exist.",
                "it is written by the standard task; run that on a flux "
                "standard first.")
        standard = stdfile.read(params.standards)
        if len(standard) < 2:
            advice.fail(
                f"'{params.standards}' holds {len(standard)} bands, which is "
                "not enough to fit anything.",
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

        spec = icfit.FitSpec(
            function=params.function, order=params.order,
            low_reject=params.low_reject, high_reject=params.high_reject,
            niterate=params.niterate)

        fit, wavelength, values = flux.sensitivity(standard, curve, spec)
        if params.interactive:
            from ..viewer.icfit_panel import run as run_icfit

            fit = run_icfit(wavelength, values, spec,
                            title=f"sensfunc · {params.sensitivity}",
                            xlabel="wavelength", ylabel="sensitivity (mag)")
        if fit.model is None:
            advice.fail(f"the response could not be fitted: {fit.message}",
                        "lower --order, or look at the bands with "
                        "--interactive.", RuntimeError)

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        by_action["read"].measured = {
            "bands": len(standard),
            "stars": ", ".join(sorted({item.star for item in standard.measurements})),
            "covering": f"{wavelength.min():.0f} to {wavelength.max():.0f} A",
        }
        airmasses = [item.airmass for item in standard.measurements]
        by_action["extinct"].measured = {
            "curve": name,
            "airmass": (f"{min(airmasses):.3f}" if min(airmasses) == max(airmasses)
                        else f"{min(airmasses):.3f} to {max(airmasses):.3f}"),
            "extinction at the ends":
                f"{flux.extinction_at([wavelength.min()], curve)[0]:.3f} to "
                f"{flux.extinction_at([wavelength.max()], curve)[0]:.3f} mag/airmass",
        }

        kept = int(np.sum(fit.mask))
        by_action["fit"].measured = {
            "bands used": f"{kept} of {len(values)}",
            "rms": f"{fit.rms:.4f} mag",
            "response range": f"{values.min():.2f} to {values.max():.2f} mag",
        }

        if flux.sens_range(standard.measurements) is None:
            advice.fail(
                f"'{params.standards}' does not record the standards' wavelength "
                "ranges, which the sensitivity function is sampled over.",
                "run standard again with this version of harwood to rewrite it.")
        sampled = flux.SensSpectrum.from_fit(fit, standard.measurements)
        image = sampled.write(Path(params.database) / f"{params.sensitivity}.fits",
                              title=f"Sensitivity function {params.sensitivity}")
        solution = database.Solution(
            image=params.sensitivity, pixel=wavelength, wavelength=values,
            function=params.function, order=params.order, rms=float(fit.rms),
            used=fit.mask, units="mag",
            extra={"sampled": str(image), "w1": sampled.w1, "dw": sampled.dw,
                   "nw": int(sampled.values.size)},
        )
        destination = database.write(solution, params.database, prefix="sens")

        warnings = []
        # The rms is the honest size of the flux calibration's systematic
        # error, and it is worth saying in per cent because magnitudes hide
        # how large it is.
        percent = 100.0 * (10.0 ** (0.4 * fit.rms) - 1.0)
        warnings.append(advice.note(
            f"the response scatters by {fit.rms:.3f} mag about the fit, which "
            f"is {percent:.1f} per cent in flux. That is the systematic floor "
            "on every spectrum calibrated with this solution -- it is not "
            "reduced by observing the object for longer.",
            f"quote {percent:.1f} per cent as the systematic uncertainty on "
            "every flux from this solution. To make it smaller, observe more "
            "standards, not longer ones."))
        if kept < 0.7 * len(values):
            warnings.append(advice.warn(
                f"only {kept} of {len(values)} bands survived rejection. Either "
                "the order is too low to follow the real response, or some "
                "bands are contaminated.",
                "look at the fit with --interactive. If the curve misses the "
                "shape, raise --order a little; if a few bands sit far off on "
                "their own, they are the problem and not the order."))

        return TaskResult(
            outputs=[destination, image],
            steps=steps,
            warnings=warnings,
            data={"bands": len(values), "used": kept,
                  "rms": round(float(fit.rms), 4)},
            history=[f"sensfunc: {params.function} order {params.order}, "
                     f"{kept} bands, rms {fit.rms:.4f} mag"],
        )


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
        if not params.flux:
            warnings.append(advice.note(
                "--no-flux was given, so this spectrum is extinction-corrected "
                "counts, not flux. It is comparable between exposures of the "
                "same night and with nothing else.",
                "run standard and sensfunc on a flux standard, then drop "
                "--no-flux, to get a physical flux."))

        destination = params.output or output.named_for(params.input, ".f.fits")
        extra = fits.Header()
        unit = (flux.FNU_UNIT if params.fnu else flux.FLUX_UNIT) if params.flux else "counts"
        extra["BUNIT"] = (unit, "what the flux column is in")
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


def _no_overlap(aperture, bands, name: str, path) -> tuple[str, str]:
    """The spectrum and the standard's table do not meet."""
    return (
        f"none of {name}'s calibration bands falls inside the spectrum. The "
        f"table covers {bands[0].wavelength:.0f} to {bands[-1].wavelength:.0f} "
        f"A and the spectrum covers {min(aperture.wavelength):.0f} to "
        f"{max(aperture.wavelength):.0f} A. A band has to be wholly inside the "
        "spectrum to be measured, since a partly covered band looks like a "
        "fainter star.",
        "check the dispersion solution put the spectrum where you expect, then "
        f"pick a calibration directory whose bands reach it -- 'harwood "
        f"standards --star {name}' shows the coverage of each.")
