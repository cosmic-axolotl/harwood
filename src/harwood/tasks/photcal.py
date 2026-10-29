"""photcal -- from instrumental magnitudes to a standard system.

IRAF's noao.digiphot.photcal, and its files: a configuration that says
which column of the catalog and of the observations is which variable and
writes the transformation equations; `fitparams`, which fits them to the
standard stars and appends the coefficients to a parameters file; and
`invertfit`, which turns them round to give every star its standard
magnitudes; and before them `mkapfile`, the aperture corrections from the
curve of growth. The files are IRAF's own, byte for byte, so a reduction can go
back and forth between the two.

The arithmetic is IRAF's (core.photcal), in single precision, and checked
against IRAF's output: the coefficients, their errors and the log to the
last digit printed, the calibrated magnitudes to the seventh decimal, the
aperture corrections, best magnitudes and log byte for byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "config": "The configuration is the whole model: which column holds which "
              "magnitude, colour and airmass, and the equations that relate the "
              "instrumental magnitudes to the standard ones. Nothing about the "
              "photometric system is built in, so the same task fits UBVRI, a "
              "Stromgren system or a single filter.",
    "match": "Each observation is paired with its star in the catalog by the "
             "id in the first column, upper-cased and stripped of everything "
             "but letters, digits, + - and _, so 'pg 2331+055' finds "
             "'PG2331+055'. An observation with no catalog line takes no part "
             "in the fit -- and is listed, because a misspelt standard is the "
             "most common reason a fit has fewer stars than it should.",
    "fit": "Each equation is fitted on its own by non-linear least squares "
           "(IRAF's nlfit, Levenberg-Marquardt), from the initial values in the "
           "configuration. An observation whose equation cannot be evaluated -- "
           "an INDEF magnitude, the log of a negative flux -- gets no weight "
           "rather than stopping the fit. With photometric weighting each "
           "point counts by the inverse of its measured variance, and with "
           "addscatter a scatter term is fitted too, so that stars whose "
           "errors are underestimated do not dominate.",
    "reject": "Rejection removes points beyond low_reject and high_reject times "
              "the rms and fits again from the initial values, nreject times "
              "at most. It needs five points to estimate the rms from.",
    "write": "Each equation's coefficients, their errors and the fit's "
             "statistics are appended to the parameters file as a record "
             "named after the equation -- IRAF's text database, which "
             "invertfit and evalfit read, the last record of a name winning.",
    "invert": "The equations give the instrumental magnitude from the standard "
              "one; for a star of unknown magnitude they have to be solved "
              "the other way. With as many equations as unknowns this is a "
              "root-finding problem, solved star by star by minimising the "
              "rms of the equations from the catalog's mean values.",
    "errors": "obserrors moves each measured magnitude by its error and adds "
              "the changes in the solution in quadrature; equations does the "
              "same with each equation's error expression. Neither includes "
              "the uncertainty of the coefficients themselves.",
}


class FitparamsParams(TaskParams):
    observations: list[str] = Field(
        ..., description="Observation files, as obsfile writes them: one line per "
                         "star, '*' lines continuing it.",
        json_schema_extra={"positional": True})
    catalogs: list[str] = Field(
        ..., description="Standard star catalogs: the id, then the columns the "
                         "configuration's catalog section names.",
        json_schema_extra={"positional": True})
    config: str = Field(..., description="The configuration file.",
                        json_schema_extra={"positional": True})
    parameters: Path = Field(
        ..., description="The parameters file the fit is appended to.",
        json_schema_extra={"positional": True})
    weighting: Literal["uniform", "photometric", "equations"] = Field(
        "uniform", description="uniform: every star the same; photometric: by the "
                               "errors of the magnitudes in the equation; "
                               "equations: by the configuration's weight equations.")
    addscatter: bool = Field(True, description="With weights, fit a scatter term "
                                               "added to every star's variance.")
    tolerance: float = Field(3e-5, ge=0, description="Convergence: the relative "
                                                     "change in chi-squared.")
    maxiter: int = Field(15, ge=4, description="Most fit iterations.")
    nreject: int = Field(0, ge=0, description="Rejection iterations; 0 rejects nothing.")
    low_reject: float = Field(3.0, ge=0, description="Reject below this many sigma.")
    high_reject: float = Field(3.0, ge=0, description="Reject above this many sigma.")
    grow: float = Field(0.0, ge=0, description="Reject the neighbours within this "
                                               "radius too (IRAF's, measured in the "
                                               "first variable).")
    logfile: str = Field("", description="Append IRAF's log here: the unmatched "
                                         "stars, and with log_fit and log_results "
                                         "the fit and each star's residual.")
    log_unmatched: bool = Field(True, description="Log the observations with no "
                                                  "catalog line.")
    log_fit: bool = Field(False, description="Log the fit's settings and statistics.")
    log_results: bool = Field(False, description="Log each star's residual.")
    catdir: str = Field("", description="Look for each catalog here as NAME.dat "
                                        "first.")


@register
class Fitparams(Task[FitparamsParams]):
    """Fit the transformation equations to the standard stars, as IRAF's
    fitparams does non-interactively."""

    name = "fitparams"
    package = "photcal"
    description = "Fit the transformation equations to the standard stars."
    Params = FitparamsParams

    def plan(self, params: FitparamsParams) -> list[Step]:
        steps = [Step("config", params.config, why=WHY["config"]),
                 Step("match", f"{', '.join(params.observations)} against "
                               f"{', '.join(params.catalogs)}", why=WHY["match"]),
                 Step("fit", f"weighting {params.weighting}"
                             + (", with a scatter term" if params.addscatter
                                and params.weighting != "uniform" else ""),
                      why=WHY["fit"])]
        if params.nreject:
            steps.append(Step("reject", f"{params.nreject} iterations at "
                                        f"-{params.low_reject:g}/+{params.high_reject:g} "
                                        "sigma", why=WHY["reject"]))
        steps.append(Step("write", str(params.parameters), why=WHY["write"]))
        return steps

    def execute(self, params: FitparamsParams) -> TaskResult:
        from ..core.photcal import fit, io

        cfg = _parse(params.config)
        existed = params.parameters.is_file()
        try:
            results = fit.fitparams(
                cfg, _files(params.observations), _files(params.catalogs),
                str(params.parameters),
                logfile=params.logfile, catdir=params.catdir,
                log_unmatched=params.log_unmatched, log_fit=params.log_fit,
                log_results=params.log_results, weighting=params.weighting,
                addscatter=params.addscatter, tol=params.tolerance,
                maxiter=params.maxiter, nreject=params.nreject,
                low_reject=params.low_reject, high_reject=params.high_reject,
                grow=params.grow)
        except (ValueError, OSError, io.ContinuationError) as error:
            advice.fail(str(error), "check the observation and catalog files "
                                    "against the configuration's columns.")

        catalog = io.read_catalogs(cfg, _files(params.catalogs), params.catdir) \
            if cfg.mincol > 1 and cfg.ncat and params.catalogs else None
        obs = io.read_observations(cfg, _files(params.observations), catalog)
        steps = self.plan(params)
        by = {s.action: s for s in steps}
        by["config"].measured = {
            "equations": ", ".join(eq.name for eq in cfg.equations),
            "variables": f"{sum(not v.spare for v in cfg.obsvars)} observed, "
                         f"{sum(not v.spare for v in cfg.catvars)} catalog"}
        by["match"].measured = {"matched": len(obs.rows),
                                "unmatched": len(obs.unmatched)}
        warnings = [advice.warn(w, "the configuration still parsed; check the "
                                   "line it names.") for w in cfg.warnings]
        if obs.unmatched:
            ids = sorted({cid for _, cid in obs.unmatched})
            warnings.append(advice.warn(
                f"{len(obs.unmatched)} observations have no catalog line and were "
                f"left out of the fit: {', '.join(ids[:12])}"
                + (" ..." if len(ids) > 12 else ""),
                "that is right for the program stars; a standard among them is "
                "misspelt in one of the two files."))
        data = {}
        for r in results:
            eq = r.eq
            values = {p.name: (round(float(r.params[k]), 7), round(float(r.errors[k]), 7))
                      for k, p in enumerate(eq.params)}
            data[eq.name] = {"status": io.STATUS.get(r.stat, "?"), "rms": float(r.rms),
                             "points": int(len(r.y) - r.deleted), "rejected": r.nrejected,
                             "parameters": values}
            by["fit"].measured[eq.name] = (
                ", ".join(f"{name} = {v:.5g} ± {e:.2g}" for name, (v, e) in values.items())
                + f"; rms {float(r.rms):.4g} from {len(r.y) - r.deleted} stars")
            if r.stat != 0:
                warnings.append(advice.warn(
                    f"{eq.name}: {io.STATUS.get(r.stat, 'unknown error')} -- the "
                    "coefficients written are where the fit stopped.",
                    "check the initial values in the configuration and that "
                    "there are more stars than fitted parameters."))
            if "reject" in by:
                by["reject"].measured[eq.name] = f"{r.nrejected} rejected"
        if existed:
            warnings.append(advice.warn(
                f"{params.parameters} already had records; these were appended.",
                "invertfit and evalfit read the last record of each equation, "
                "so the new ones are the ones used."))
        outputs = [params.parameters] + ([Path(params.logfile)] if params.logfile else [])
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings, data=data)


class InvertfitParams(TaskParams):
    observations: list[str] = Field(
        ..., description="Observation files, standards and program stars.",
        json_schema_extra={"positional": True})
    config: str = Field(..., description="The configuration file.",
                        json_schema_extra={"positional": True})
    parameters: str = Field(..., description="The parameters file fitparams wrote.",
                            json_schema_extra={"positional": True})
    calib: Path = Field(..., description="The calibrated output file.",
                        json_schema_extra={"positional": True})
    catalogs: list[str] = Field(
        default_factory=list,
        description="Standard catalogs: with them the standards get a residual "
                    "against the catalog, and the solution starts from the "
                    "catalog's mean.")
    errors: Literal["undefined", "obserrors", "equations"] = Field(
        "obserrors", description="How the errors are computed: from the "
                                 "observations' errors, from the error equations, "
                                 "or not at all.")
    objects: Literal["all", "program", "standards"] = Field(
        "all", description="Which stars to write, when there are catalogs.")
    print: list[str] = Field(
        default_factory=list,
        description="Variables or set equations to write before the results.")
    format: str = Field("", description="IRAF printf format of each line; left out, "
                                        "%-10s then %-7.3f.")
    append: bool = Field(False, description="Append to the output instead of "
                                            "replacing it.")
    catdir: str = Field("", description="Look for each catalog here as NAME.dat first.")


@register
class Invertfit(Task[InvertfitParams]):
    """Standard magnitudes from instrumental ones, by inverting the fitted
    transformation equations star by star, as IRAF's invertfit."""

    name = "invertfit"
    package = "photcal"
    description = "Turn instrumental magnitudes into standard ones."
    Params = InvertfitParams

    def plan(self, params: InvertfitParams) -> list[Step]:
        return [Step("config", f"{params.config}, coefficients from {params.parameters}",
                     why=WHY["config"]),
                Step("invert", ", ".join(params.observations), why=WHY["invert"]),
                Step("errors", params.errors, why=WHY["errors"]),
                Step("write", str(params.calib))]

    def execute(self, params: InvertfitParams) -> TaskResult:
        from ..core.photcal import invert

        cfg = _parse(params.config)
        try:
            notes = invert.invertfit(
                cfg, _files(params.observations), params.parameters, str(params.calib),
                catalogs=_files(params.catalogs), catdir=params.catdir, errors=params.errors,
                objects=params.objects, print_vars=params.print, format=params.format,
                append=params.append)
        except (ValueError, OSError) as error:
            advice.fail(str(error), "check the configuration's equations: each "
                                    "reference side must hold observed variables "
                                    "only, and there must be as many equations "
                                    "as catalog variables to solve for.")
        rows = [line for line in params.calib.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
        undefined = sum("INDEF" in line.split()[1:2] for line in rows)
        steps = self.plan(params)
        steps[1].measured = {"stars": len(rows), "unsolved": undefined}
        warnings = [advice.warn(note, "check that fitparams converged for every "
                                      "equation and wrote to this file.")
                    for note in notes]
        if undefined:
            warnings.append(advice.warn(
                f"{undefined} of {len(rows)} stars could not be solved and are INDEF.",
                "a star needs a defined magnitude in every filter its equations use."))
        return TaskResult(outputs=[params.calib], steps=steps, warnings=warnings,
                          data={"stars": len(rows), "unsolved": undefined})


class MkapfileParams(TaskParams):
    photfiles: list[str] = Field(
        ..., description="apphot or daophot text files with the magnitudes in "
                         "every aperture (phot's .mag files).",
        json_schema_extra={"positional": True})
    naperts: int = Field(..., ge=2, description="How many apertures to read, from "
                                                "the first.",
                         json_schema_extra={"positional": True})
    apercors: Path = Field(..., description="The aperture corrections file: one line "
                                            "per image, the correction and its error.",
                           json_schema_extra={"positional": True})
    smallap: int = Field(1, ge=1, description="The aperture the correction starts "
                                              "from (1 is the first).")
    largeap: int = Field(0, ge=0, description="The aperture it corrects to; 0 is the "
                                              "last one read.")
    magfile: str = Field("", description="Write each star's best magnitude, "
                                         "corrected to largeap, here.")
    logfile: str = Field("", description="Write IRAF's log here: the model, and "
                                         "each image's curve of growth star by star.")
    append: bool = Field(False, description="Append to the log instead of "
                                            "replacing it.")
    maglim: float = Field(0.10, gt=0, description="A star is used out to the last "
                                                  "aperture before one whose error "
                                                  "exceeds this.")
    nparams: int = Field(3, ge=1, le=5, description="How many of the model's five "
                                                    "parameters to fit, in order.")
    swings: float = Field(1.2, ge=1.0, description="Initial power-law slope of the "
                                                   "stellar wings.")
    pwings: float = Field(0.1, ge=0.0, le=1.0, description="Initial fraction of the "
                                                           "light in the wings.")
    pgauss: float = Field(0.5, ge=0.0, le=1.0, description="Initial fraction of the "
                                                           "core that is Gaussian.")
    rgescale: float = Field(0.9, ge=0.0, description="Initial ratio of the exponential "
                                                     "to the Gaussian radial scale.")
    xwings: float = Field(0.0, description="Initial change of the wings with airmass.")
    names: bool = Field(False, description="Add each star's name from the photometry "
                                           "to magfile as an eleventh column, Obj, for "
                                           "obsfile (incolumns ...,11). IRAF's file has "
                                           "ten.")


WHY.update({
    "read": "Each star is used out to the last aperture whose magnitude is "
            "defined and whose error is under maglim -- a faint star's outer "
            "apertures are mostly sky. What is fitted is the difference "
            "between successive apertures, not the magnitudes, so the star's "
            "brightness drops out and every star, bright or faint, has the "
            "same curve.",
    "model": "Stetson's DAOGROW: the profile is a Moffat wing, a Gaussian core "
             "and an exponential, mixed in proportions that are the model's "
             "parameters and the same in every image; only the seeing radius "
             "changes from image to image. Fitting all the images together "
             "is what lets a frame with three stars borrow the shape from one "
             "with thirty.",
    "adopt": "The curve adopted for each image is the weighted mean of its "
             "stars' differences, pulled towards the model where there are "
             "few stars or they are noisy. Points far from the rest lose "
             "weight progressively, so a star with a neighbour in its outer "
             "apertures does not drag the curve.",
    "correct": "The correction is the adopted curve summed from smallap to "
               "largeap: what has to be added to a magnitude measured in the "
               "small aperture to get the light in the large one. Each star's "
               "best magnitude is the one, of its apertures corrected out to "
               "largeap, with the smallest error.",
})


@register
class Mkapfile(Task[MkapfileParams]):
    """Aperture corrections from the curve of growth, as IRAF's mkapfile
    does non-interactively."""

    name = "mkapfile"
    package = "photcal"
    description = "Aperture corrections from the curve of growth."
    Params = MkapfileParams

    def plan(self, params: MkapfileParams) -> list[Step]:
        large = params.largeap or params.naperts
        names = ("swings", "pwings", "pgauss", "rgescale", "xwings")[:params.nparams]
        return [Step("read", f"{params.naperts} apertures from "
                             f"{', '.join(params.photfiles)}, errors under "
                             f"{params.maglim:g} mag", why=WHY["read"]),
                Step("model", f"fit {', '.join(names)} and a seeing radius per image",
                     why=WHY["model"]),
                Step("adopt", "each image's curve of growth", why=WHY["adopt"]),
                Step("correct", f"from aperture {params.smallap} to {large}, "
                                f"into {params.apercors}", why=WHY["correct"])]

    def execute(self, params: MkapfileParams) -> TaskResult:
        from ..core.photcal import apfile

        if params.largeap and params.largeap > params.naperts:
            advice.fail(f"largeap {params.largeap} is beyond the {params.naperts} "
                        "apertures read.", "lower largeap, or raise naperts.")
        notes: list[str] = []
        try:
            images = apfile.read_photfiles(_files(params.photfiles), params.naperts,
                                           params.maglim, notes)
        except KeyError as error:
            advice.fail(f"a photometry file has no column {error}.",
                        f"naperts ({params.naperts}) must not exceed the apertures "
                        "phot measured; the .mag file's #K APERTURES line lists them.")
        except (OSError, ValueError) as error:
            advice.fail(f"cannot read the photometry: {error}",
                        "the inputs must be phot's text output (.mag files).")
        try:
            result = apfile.apfile(
                images, params.naperts, str(params.apercors), params.smallap,
                params.largeap, magfile=params.magfile, logfile=params.logfile,
                params=(params.swings, params.pwings, params.pgauss, params.rgescale,
                        params.xwings),
                nparams=params.nparams, append=params.append, names=params.names)
        except ValueError as error:
            advice.fail(str(error).removeprefix("Error: "),
                        "with no star measured in at least two apertures there is "
                        "no curve; raise maglim, or check the photometry.")
        steps = self.plan(params)
        by = {s.action: s for s in steps}
        nstars = sum(len(image.stars) for image in result.images)
        by["read"].measured = {"images": len(result.images), "stars": nstars}
        names = ("swings", "pwings", "pgauss", "rgescale", "xwings")
        fit = result.fit
        by["model"].measured = {
            name: f"{float(value):.5g}" for name, value
            in zip(names[:fit.nterms], fit.params, strict=False)}
        data = {"converged": fit.ok, "model": dict(zip(names, map(float, fit.params),
                                                       strict=True)),
                "images": {}}
        warnings = [advice.warn(n.strip(), "set the airmass in the headers "
                                           "(setairmass) before phot.")
                    for n in notes]
        if not fit.ok:
            warnings.append(advice.problem(
                "the curve-of-growth model did not converge; every correction is "
                "INDEF.", "fit fewer parameters (nparams), or start them nearer: "
                          "the log shows where they went."))
        for image in result.images:
            cum, err = result.corrections.get(image.name, (float("nan"),) * 2)
            data["images"][image.name] = {
                "stars": len(image.stars), "seeing_radius": float(image.ro),
                "correction": float(cum), "error": float(err)}
            by["correct"].measured[image.name] = f"{float(cum):.4f} ± {float(err):.4f}"
            if fit.ok and len(image.stars) < 3:
                warnings.append(advice.warn(
                    f"{image.name}: {len(image.stars)} star(s) -- its curve is "
                    "mostly the model's.",
                    "fit it together with other images of the night, which share "
                    "the model."))
        outputs = [params.apercors] + [Path(p) for p in (params.magfile, params.logfile)
                                       if p]
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings, data=data)


class MkimsetsParams(TaskParams):
    imlist: list[str] = Field(
        ..., description="The photometry files (phot's .mag), or with --input "
                         "images, the images.",
        json_schema_extra={"positional": True})
    idfilters: str = Field(..., description="The filters of a set, in the order "
                                            "they are written: B,V.",
                           json_schema_extra={"positional": True})
    imsets: Path = Field(..., description="The image sets file to write.",
                         json_schema_extra={"positional": True})
    imobsparams: str = Field("", description="Also write each image's filter, "
                                             "exposure time, airmass and time here, "
                                             "for obsfile's obsparams.")
    input: Literal["photfiles", "images"] = Field(
        "photfiles", description="Where the filters come from: the photometry "
                                 "files, or the image headers.")
    filter: str = Field("FILTER", description="With --input images, the header "
                                              "keyword that holds the filter.")
    fields: str = Field("", description="More fields to list, for sort and "
                                        "imobsparams.")
    sort: str = Field("", description="Sort the images on this field before "
                                      "grouping: image, the filter, or a field.")
    names: list[str] = Field(
        default_factory=list,
        description="Names for the sets, in order (IRAF asks for them one by "
                    "one with rename); the rest are OBS1, OBS2 ...")
    name_keyword: str = Field(
        "", description="Name each set after this header keyword of its first "
                        "image -- OBJECT names it after the field.")


WHY.update({
    "group": "The images are listed with their filters, sorted -- by name, "
             "which is usually the order they were taken, or by the field "
             "asked for -- and walked in order: an image of each filter makes "
             "a set. A filter seen again before the set is complete closes "
             "it, its missing filters written INDEF; an image whose filter is "
             "not in the list is left out.",
})


@register
class Mkimsets(Task[MkimsetsParams]):
    """The image sets -- a field seen once in each filter -- as IRAF's
    mkimsets makes them without the editor."""

    name = "mkimsets"
    package = "photcal"
    description = "Group the images into sets, one image per filter."
    Params = MkimsetsParams

    def plan(self, params: MkimsetsParams) -> list[Step]:
        source = ("photometry files" if params.input == "photfiles"
                  else f"headers' {params.filter}")
        return [Step("group", f"filters {params.idfilters} from the {source}"
                              + (f", sorted on {params.sort}" if params.sort else ""),
                     why=WHY["group"]),
                Step("write", str(params.imsets))]

    def execute(self, params: MkimsetsParams) -> TaskResult:
        from ..core.photcal import imsets

        files = _files(params.imlist)
        try:
            sets, messages = imsets.mkimsets(
                files, params.idfilters, str(params.imsets), params.imobsparams,
                input=params.input, filter=params.filter, fields=params.fields,
                sort=params.sort, names=params.names)
        except (OSError, ValueError, KeyError) as error:
            advice.fail(f"cannot read the images' filters: {error}",
                        "--input photfiles reads phot's IRAF-format .mag files; "
                        "--input images reads the headers.")
        warnings = []
        if params.name_keyword:
            from astropy.io import fits

            # A set holds the images' names without their directories; with
            # --input images the files given say where they are.
            where = {imsets.imroot(f): f for f in files} if params.input == "images" else {}
            for s in sets[len(params.names):]:
                first = next((i for i in s.images if i != "INDEF"), None)
                image = where.get(first, first)
                try:
                    value = fits.getheader(image).get(params.name_keyword)
                except (OSError, TypeError):
                    value = None
                if value is None:
                    warnings.append(advice.warn(
                        f"{s.name}: no {params.name_keyword} in {image}; kept as {s.name}.",
                        "check that the images are where the photometry says, or "
                        "give the names with --names."))
                    continue
                s.name = "".join(str(value).split())
            imsets.write_sets(params.imsets, sets)
        for m in messages:
            text = m.strip().removeprefix("<Warning>:").removeprefix("<Warning>").strip()
            if text:
                warnings.append(advice.warn(text, "check the filters in the photometry "
                                                  "or headers, and idfilters."))
        steps = self.plan(params)
        steps[0].measured = {f"{k}. {s.name}": " ".join(s.images)
                             for k, s in enumerate(sets, start=1)}
        outputs = [params.imsets] + ([Path(params.imobsparams)] if params.imobsparams
                                     else [])
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings,
                          data={"sets": [{"name": s.name, "images": s.images}
                                         for s in sets]})


class ObsfileParams(TaskParams):
    photfiles: list[str] = Field(
        ..., description="Text files of magnitudes, one star per line: mkapfile's "
                         "best magnitudes, or phot's output dumped to text.",
        json_schema_extra={"positional": True})
    incolumns: str = Field(
        ..., description="The column of image, x, y, filter, exposure time, "
                         "airmass, time, magnitude, error and star name, in that "
                         "order; 0 for one the files do not have. mkapfile's "
                         "files: 1,6,7,2,3,4,5,8,9,0.",
        json_schema_extra={"positional": True})
    idfilters: str = Field(..., description="The filters, in the order of the "
                                            "columns to write: B,V.",
                           json_schema_extra={"positional": True})
    imsets: str = Field(..., description="The image sets file: NAME : image image "
                                         "..., one image per filter (mkimsets).",
                        json_schema_extra={"positional": True})
    observations: Path = Field(..., description="The observations file to write; "
                                                "its format goes to f<name>.dat.",
                               json_schema_extra={"positional": True})
    wrap: bool = Field(True, description="A line per filter, the later ones "
                                         "starting with '*'; off, one long line.")
    obsparams: str = Field("", description="A file of image, filter, exposure "
                                           "time, airmass and time that replaces "
                                           "what the photometry says.")
    obscolumns: str = Field("2 3 4 5", description="obsparams' columns of filter, "
                                                   "exposure time, airmass and time.")
    minmagerr: float = Field(0.001, ge=0, description="The smallest error a "
                                                      "magnitude is given.")
    shifts: str = Field("", description="A file of image, x shift and y shift, "
                                        "added before matching.")
    apercors: str = Field("", description="The aperture corrections mkapfile "
                                          "wrote, added to the magnitudes.")
    normtime: bool = Field(False, description="Correct the magnitudes to an "
                                              "exposure time of 1.")
    tolerance: float = Field(5.0, ge=0, description="Match stars within this many "
                                                    "pixels; 0 matches by order.")
    allfilters: bool = Field(True, description="Write only the stars measured in "
                                               "every filter.")


WHY.update({
    "sets": "An image set is one field observed once in each filter. The sets "
            "file says which images go together and which is which filter; "
            "the stars are only matched within a set.",
    "obsmatch": "Each star of the first filter's image is paired with the "
                "nearest star of each other filter's image within the "
                "tolerance, walking the stars in y order, each star taken at "
                "most once. Shift the images first (shifts) if the telescope "
                "moved between filters by more than the tolerance.",
    "obswrite": "Each star is written as a line per filter -- filter, time, "
                "airmass, x, y, magnitude, error -- the second and later "
                "starting with '*', which is how fitparams and invertfit read "
                "them. Its name is the star's own when the files have a name "
                "column; otherwise the set's name and a number.",
})


@register
class Obsfile(Task[ObsfileParams]):
    """The stars of each image set matched across its filters into an
    observations file, as IRAF's obsfile does non-interactively."""

    name = "obsfile"
    package = "photcal"
    description = "Match the stars of each image set across filters."
    Params = ObsfileParams

    def plan(self, params: ObsfileParams) -> list[Step]:
        how = (f"by position, within {params.tolerance:g} px" if params.tolerance > 0
               else "by order")
        corrections = [w for w, on in (("exposure time", params.normtime),
                                       (f"apertures from {params.apercors}",
                                        bool(params.apercors)),
                                       (f"shifts from {params.shifts}",
                                        bool(params.shifts))) if on]
        return [Step("sets", f"{params.imsets}, filters {params.idfilters}",
                     why=WHY["sets"]),
                Step("obsmatch", how + ("" if params.allfilters
                                        else ", keeping stars missing a filter")
                     + (f"; corrected for {', '.join(corrections)}" if corrections
                        else ""), why=WHY["obsmatch"]),
                Step("obswrite", str(params.observations), why=WHY["obswrite"])]

    def execute(self, params: ObsfileParams) -> TaskResult:
        from ..core.photcal import obsfile

        try:
            result = obsfile.obsfile(
                _files(params.photfiles), params.incolumns, params.idfilters, params.imsets,
                str(params.observations), wrap=params.wrap, obsparams=params.obsparams,
                obscolumns=params.obscolumns, minmagerr=params.minmagerr,
                shifts=params.shifts, apercors=params.apercors,
                normtime=params.normtime, tolerance=params.tolerance,
                allfilters=params.allfilters)
        except (OSError, ValueError) as error:
            advice.fail(str(error), "check the file names, and that incolumns "
                                    "has ten columns.")
        steps = self.plan(params)
        steps[-1].measured = {f"set {k}: {label}": f"{n} stars"
                              for k, (label, n) in enumerate(result.written, start=1)}
        # Comment lines read as an image called '#'; IRAF warns about each,
        # and so does run.yaml, but they are not worth a warning here.
        notes = [n for n in result.notes if "Image: # is not" not in n]
        warnings = [advice.warn(n.strip(), "check the image sets file against the "
                                           "images the photometry names.")
                    for n in notes if n.strip().startswith(("Warning", "Image set"))]
        for label, n in result.written:
            if n == 0:
                warnings.append(advice.warn(
                    f"image set {label}: no star matched in every filter.",
                    "raise the tolerance, give shifts, or check that the set's "
                    "images are the same field."))
        return TaskResult(outputs=[params.observations,
                                   obsfile.format_path(params.observations)],
                          steps=steps, warnings=warnings,
                          data={"sets": [{"label": label, "stars": n}
                                         for label, n in result.written],
                                "notes": notes})


def _files(items: list[str]) -> list[str]:
    """Text files named as images are: @lists, commas and patterns."""
    from ..core import images

    try:
        return [str(spec) for spec in images.expand(items)]
    except FileNotFoundError as error:
        advice.fail(str(error), "check the name; an @list names one file per line.",
                    kind=FileNotFoundError)


def _parse(path: str):
    from ..core.photcal import config

    try:
        return config.parse(path)
    except config.ConfigError as error:
        advice.fail("the configuration could not be parsed:\n" + "\n".join(error.messages),
                    "fix the line each message names; chkconfig lists the "
                    "declarations it understood.")
    except OSError as error:
        advice.fail(f"cannot read the configuration: {error}", "check the path.")
