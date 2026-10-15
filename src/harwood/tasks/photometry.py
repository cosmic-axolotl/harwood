"""Photometry -- detection and aperture measurement.

IRAF names throughout, including the ones that are not obvious: the task is
`phot`, not `apphot` (`apphot` is the *package*); the aperture radii are
`apertures`; the sky annulus is `annulus` and `dannulus`; the gain is `epadu`;
the magnitude zero point is `zmag`. Section 3 is explicit that a different name
costs little and the same name meaning something else ruins a result.

Every coordinate that leaves this module is 1-based and (x, y) -- section 2.2.
`photutils` works 0-based, so the conversion happens at the call, once, through
`core.pixels`. The catalogue written here can be read by someone who has spent
twenty years reading IRAF `.mag` files without a systematic pixel offset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "read": "Pixels come in as floating point whatever the detector wrote. A "
            "measurement is never made on the integers a CCD happens to produce: "
            "a sum of sixteen-bit frames overflows silently, and a difference of "
            "unsigned ones wraps.",
    "background": "The detection threshold is set in units of the noise, not in "
                  "counts, so it means the same thing on a bright night and a dark "
                  "one. The sky level and its scatter are measured with clipping, "
                  "so the stars themselves do not inflate the estimate of the "
                  "noise they are to be detected against.",
    "detect": "DAOFIND convolves the image with a Gaussian of the given width and "
              "looks for maxima in the result. The width is the whole parameter: "
              "a filter matched to the stars suppresses everything that is not "
              "star-shaped, so a cosmic ray, which is sharper, and a galaxy, which "
              "is broader, both respond less than a star of the same brightness.",
    "filter": "Sharpness and roundness are shape statistics, and they separate "
              "things a brightness cut cannot. A cosmic ray is too sharp to be a "
              "point source through this atmosphere; a galaxy is too broad; a bad "
              "column is too elongated.",
    "border": "The convolution needs whole pixels around a source to judge it, so "
              "anything closer to the edge than its own width is an artefact of "
              "the truncated filter rather than a star. On real frames this is a "
              "fifth of all detections.",
    "sky": "The sky is measured in an annulus around each star rather than once "
           "for the frame, because it is not flat: there are gradients, scattered "
           "light, and the wings of neighbours. The annulus starts outside the "
           "aperture so the star does not measure its own background.",
    "measure": "The aperture sum minus the sky over the same area. The radius "
               "trades light against noise: too small and a seeing change moves "
               "the answer, too large and the sky noise over the extra area "
               "swamps the star. Four or five times the FWHM catches essentially "
               "all the light.",
    "calibrate": "Instrumental magnitude is -2.5 log10(counts per second) plus a "
                 "zero point, which is arbitrary until standards tie it to a real "
                 "system. The uncertainty has three parts: photon noise from the "
                 "star, sky noise over the aperture, and the uncertainty in the "
                 "mean sky itself -- and in a crowded field the third one, which "
                 "a naive implementation drops, is what dominates.",
    "write": "Coordinates leave here 1-based and (x, y), as IRAF and DS9 write "
             "them, with the convention recorded in the file so nothing "
             "downstream has to assume it.",
}


class DaoFindParams(TaskParams):
    image: str = Field(..., description="Image to search. Accepts 'obj.fits[SCI,1]'.")
    output: Path | None = Field(
        None,
        description="Coordinate catalogue to write. Left out, it goes to this "
                    "run's own directory under output/, named after the frame -- "
                    "obj.fits becomes obj.coo, as in IRAF.",
        json_schema_extra={"positional": True},
    )

    fwhm: float = Field(
        2.5, gt=0,
        description="FWHM of the point spread function, in pixels. The single most "
                    "important parameter: measure it with imexamine before guessing.",
    )
    sigma: float | None = Field(
        None, gt=0,
        description="Standard deviation of the background. Measured from the image "
                    "when not given.",
    )
    threshold: float = Field(
        4.0, gt=0, description="Detection threshold above background, in units of sigma."
    )
    sharplo: float = Field(0.2, description="Lower sharpness bound; rejects cosmic rays.")
    sharphi: float = Field(1.0, description="Upper sharpness bound; rejects galaxies.")
    roundlo: float = Field(-1.0, description="Lower roundness bound.")
    roundhi: float = Field(1.0, description="Upper roundness bound.")
    border: float | None = Field(
        None, ge=0,
        description="Ignore detections within this many pixels of the edge. "
                    "Defaults to twice the FWHM. The detection filter needs whole "
                    "pixels around a source to judge it, so anything closer to the "
                    "edge than its own width is an artefact of the truncated "
                    "convolution, not a star. Set 0 to keep them.",
    )


@register
class DaoFind(Task[DaoFindParams]):
    name = "daofind"
    package = "apphot"
    description = "Detect point sources with the DAOFIND algorithm."
    Params = DaoFindParams

    def plan(self, params: DaoFindParams) -> list[Step]:
        return [
            Step("read", params.image, why=WHY["read"]),
            Step("background", "sigma-clipped statistics"
                 if params.sigma is None else f"sigma={params.sigma} as given",
                 why=WHY["background"]),
            Step("detect", f"DAOFIND, fwhm={params.fwhm}, "
                           f"threshold={params.threshold} sigma",
                 why=WHY["detect"]),
            Step("filter", f"sharpness {params.sharplo}..{params.sharphi}, "
                           f"roundness {params.roundlo}..{params.roundhi}",
                 why=WHY["filter"]),
            Step("border", f"drop anything within "
                           f"{_border(params):g} px of the edge",
                 why=WHY["border"]),
            Step("write", f"{params.output or 'obj.coo in this run''s directory'}, "
                          "coordinates 1-based (x,y)", why=WHY["write"]),
        ]

    def execute(self, params: DaoFindParams) -> TaskResult:
        import numpy as np
        from astropy.stats import sigma_clipped_stats
        from astropy.table import Table
        from photutils.detection import DAOStarFinder

        from ..core import catalog, images, output, pixels, provenance

        data, header = images.read(params.image)
        mean, median, sigma = sigma_clipped_stats(data, sigma=3.0)
        if params.sigma is not None:
            sigma = params.sigma

        finder = DAOStarFinder(fwhm=params.fwhm, threshold=params.threshold * sigma,
                               **_shape_bounds(DAOStarFinder, params))
        found = finder(data - median)
        if found is None or len(found) == 0:
            raise RuntimeError(
                f"no sources found. The background sigma measured {sigma:.2f}; try a "
                f"lower --threshold than {params.threshold}, or check --fwhm."
            )

        # The one conversion. photutils counts from 0; everything harwood
        # writes counts from 1.
        position = pixels.from_numpy(_column(found, "y_centroid", "ycentroid"),
                                     _column(found, "x_centroid", "xcentroid"))
        border = _border(params)
        ny, nx = data.shape
        inside = ((position.x > border) & (position.x <= nx - border)
                  & (position.y > border) & (position.y <= ny - border))
        dropped = int((~inside).sum())
        if not inside.any():
            raise RuntimeError(
                f"every one of the {len(found)} detections was within {border:g} px "
                "of the edge. Lower --border if the field really does run to the edge."
            )
        found, position = found[inside], pixels.Position(position.x[inside],
                                                         position.y[inside])
        # Brightest first. It is the one ordering two frames of the same field
        # agree on, which is what xyxymatch's triangle search needs when it
        # takes only the brightest nmatch stars from each list.
        order = np.argsort(np.asarray(_column(found, "flux")))[::-1]
        found, position = found[order], pixels.Position(position.x[order],
                                                        position.y[order])
        catalogue = Table({
            "id": range(1, len(found) + 1),
            "x": position.x,
            "y": position.y,
            "sharpness": _column(found, "sharpness"),
            "roundness": _column(found, "roundness1", "roundness"),
            "peak": _column(found, "peak", "max_value"),
            "flux": _column(found, "flux"),
        })
        destination = params.output or output.named_for(params.image, ".coo")
        catalog.write(catalogue, destination,
                      harwood_image=str(params.image),
                      harwood_background=float(median),
                      harwood_sigma=float(sigma),
                      harwood_call=provenance.current_stamp() or "")

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        peaks = np.asarray(_column(found, "peak", "max_value"), dtype=float)
        by_action["background"].measured = {
            "sky": round(float(median), 3), "sigma": round(float(sigma), 3),
            "threshold": f"{params.threshold * sigma:.2f} counts above sky",
        }
        by_action["detect"].measured = {
            "found": len(found) + dropped,
            "peaks": f"{peaks.min():.1f} to {peaks.max():.1f} counts",
        }
        by_action["border"].measured = {
            "dropped": f"{dropped} of {len(found) + dropped}", "kept": len(found),
        }

        return TaskResult(
            outputs=[destination],
            steps=steps,
            data={"sources": len(catalogue), "dropped_at_border": dropped,
                  "background": float(median), "sigma": float(sigma)},
            history=[f"daofind: {len(catalogue)} sources, fwhm={params.fwhm}, "
                     f"threshold={params.threshold} sigma (sky={median:.2f}, "
                     f"sigma={sigma:.2f}), {dropped} dropped within "
                     f"{border:g} px of the edge"],
        )


# ---------------------------------------------------------- the psets
#
# IRAF splits apphot's parameters into psets -- datapars, centerpars,
# fitskypars, photpars -- shared between center, fitsky and phot. harwood has
# no psets, so the same fields are declared once here and each task takes the
# groups it uses: the names, defaults and meanings are IRAF 2.18's.


class _DataPars(TaskParams):
    scale: float = Field(1.0, gt=0, description="Pixels per scale unit. cbox, "
                         "annulus, dannulus and apertures are in scale units.")
    fwhmpsf: float = Field(2.5, gt=0, description="FWHM of the stars, in scale units.")
    sigma: float | None = Field(
        None, gt=0, description="Standard deviation of the sky, in counts. Empty is "
                                "IRAF's INDEF: it enters only the centring S/N and "
                                "cthreshold.")
    datamin: float | None = Field(
        None, description="Lowest good pixel value. Empty accepts any.")
    datamax: float | None = Field(
        None, description="Highest good pixel value -- the saturation level. "
                          "Empty accepts any.")
    epadu: float = Field(1.0, gt=0, description="Gain in electrons per ADU.")
    readnoise: float = Field(
        0.0, ge=0, description="Read noise in electrons. As in IRAF it does not "
                               "enter the error of a star: the measured scatter of "
                               "the sky already contains it.")
    exposure: str = Field(
        "", description="Header keyword with the exposure time. Empty uses itime, "
                        "as IRAF does; name EXPTIME to put magnitudes per second.")
    itime: float = Field(1.0, gt=0, description="Exposure time when no keyword is named.")


class _CenterPars(TaskParams):
    calgorithm: Literal["centroid", "none"] = Field(
        "centroid", description="Centring: the marginal centroid, IRAF's default, or "
                                "none to keep the coordinates as given. gauss and "
                                "ofilter are not ported yet.")
    cbox: float = Field(5.0, gt=0, description="Width of the centring box, in scale units.")
    cthreshold: float = Field(
        0.0, ge=0, description="Threshold above the box's lowest row median, in "
                               "units of sigma. 0 uses the box minimum.")
    minsnratio: float = Field(1.0, ge=0, description="Below this S/N the centre is "
                                                     "flagged as unreliable.")
    cmaxiter: int = Field(10, ge=1, description="Most centring iterations.")
    maxshift: float = Field(1.0, gt=0, description="A centre that moves further than "
                            "this, in scale units, is flagged -- and kept.")


class _SkyPars(TaskParams):
    salgorithm: Literal["centroid", "mode", "median", "mean", "constant"] = Field(
        "centroid", description="Sky estimator. centroid, IRAF's default, is the "
                                "centroid of the histogram of the annulus; mode is "
                                "3 median - 2 mean.")
    annulus: float = Field(10.0, gt=0, description="Inner radius of the sky annulus, "
                                                   "in scale units.")
    dannulus: float = Field(10.0, gt=0, description="Width of the sky annulus, in "
                                                    "scale units.")
    skyvalue: float = Field(0.0, description="The sky, when salgorithm is constant.")
    smaxiter: int = Field(10, ge=0, description="Most rejection passes of the "
                                                "centroid algorithm.")
    snreject: int = Field(50, ge=0, description="Most rejection passes of mode, "
                                                "median and mean.")
    sloreject: float = Field(3.0, gt=0, description="Lower rejection limit, in sky sigma.")
    shireject: float = Field(3.0, gt=0, description="Upper rejection limit, in sky sigma.")
    khist: float = Field(3.0, gt=0, description="Half width of the centroid "
                                                "algorithm's histogram, in sky sigma.")
    binsize: float = Field(0.1, gt=0, description="Its bin size, in sky sigma.")


def _datapars(params, header):
    from ..core import advice, apphot

    itime = params.itime
    if params.exposure:
        if params.exposure not in header:
            advice.fail(f"the header has no '{params.exposure}'",
                        "name the keyword that holds the exposure time, or leave "
                        "exposure empty and give itime")
        itime = float(header[params.exposure])
    return apphot.DataPars(scale=params.scale, fwhmpsf=params.fwhmpsf,
                           sigma=params.sigma, datamin=params.datamin,
                           datamax=params.datamax, epadu=params.epadu,
                           readnoise=params.readnoise, itime=itime)


def _centerpars(params):
    from ..core import apphot

    return apphot.CenterPars(params.calgorithm, params.cbox, params.cthreshold,
                             params.minsnratio, params.cmaxiter, params.maxshift)


def _skypars(params):
    from ..core import apphot

    return apphot.SkyPars(params.salgorithm, params.annulus, params.dannulus,
                          params.skyvalue, params.smaxiter, params.snreject,
                          params.sloreject, params.shireject, params.khist,
                          params.binsize)


def _flags(codes, what: str, fix: str):
    """One warning for every star a stage flagged, by reason."""
    from collections import Counter

    from ..core import advice, apphot

    counts = Counter(int(c) for c in codes if int(c) != 0)
    return [advice.warn(f"{n} star{'s' if n > 1 else ''}: "
                        f"{apphot.MESSAGES.get(code, f'error {code}')} ({what} {code})",
                        fix) for code, n in sorted(counts.items())]


APWHY = {
    "center": "Each star is recentred on the marginal distributions of a small "
              "box around it, using only what stands above the box's floor, "
              "and again while the centre keeps moving a pixel or more. A centre "
              "that ends up further than maxshift is flagged and kept: IRAF "
              "leaves the decision to you, and so does harwood.",
    "sky": "The sky comes from the annulus around the *new* centre, after "
           "rejecting pixels beyond sloreject/shireject sigma -- stars, cosmic "
           "rays, a neighbour's wings. centroid takes the centroid of the "
           "histogram of what remains, which a few bright pixels cannot drag "
           "the way they drag a mean.",
    "measure": "Each aperture sums the pixels within it, the last one counting "
               "for the fraction the circle covers (IRAF's linear approximation, "
               "so the sums are IRAF's), minus the sky over the same area.",
    "calibrate": "mag = zmag - 2.5 log10(flux) + 2.5 log10(itime). The error "
                 "has three terms: the star's own photon noise, the sky's scatter "
                 "over the aperture, and the uncertainty of the sky level itself, "
                 "which grows as the annulus shrinks.",
}


class CenterParams(_DataPars, _CenterPars):
    image: str = Field(..., description="Image to measure.")
    coords: Path = Field(..., description="Coordinates to start from, 1-based (x, y).")
    output: Path | None = Field(
        None, description="Table of centres. Left out: obj.ctr in this run's "
                          "directory, as in IRAF.", json_schema_extra={"positional": True})


@register
class CenterTask(Task[CenterParams]):
    """Recentre a list of stars. The table keeps where each one started, where
    it ended, the shift and its error, and IRAF's code for anything flagged --
    and its x and y are the new centres, so it can be handed straight to phot
    with calgorithm=none.
    """

    name = "center"
    package = "apphot"
    description = "Recentre stars on the marginal distributions, as IRAF's center."
    Params = CenterParams

    def plan(self, params: CenterParams) -> list[Step]:
        return [Step("read", f"{params.image} and {params.coords}", why=WHY["read"]),
                Step("center", f"{params.calgorithm} in a {params.cbox:g} px box, "
                               f"maxshift {params.maxshift:g}", why=APWHY["center"]),
                Step("write", str(params.output or "obj.ctr in this run's directory"),
                     why=WHY["write"])]

    def execute(self, params: CenterParams) -> TaskResult:
        import numpy as np
        from astropy.table import Table

        from ..core import apphot, catalog, images, output, provenance

        data, header = images.read(params.image)
        table = catalog.read(params.coords)
        x0, y0 = catalog.coordinates(table, params.coords)
        dpars, cpars = _datapars(params, header), _centerpars(params)
        found = [apphot.center(data, a, b, cpars, dpars) for a, b in zip(x0, y0, strict=True)]
        out = Table({"id": _ids(table, len(x0)), "xinit": x0, "yinit": y0,
                     "x": [c.x for c in found], "y": [c.y for c in found],
                     "xshift": [c.xshift for c in found], "yshift": [c.yshift for c in found],
                     "xerr": [c.xerr for c in found], "yerr": [c.yerr for c in found],
                     "cier": [c.ier for c in found]})
        destination = params.output or output.named_for(params.image, ".ctr")
        catalog.write(out, destination, harwood_image=str(params.image),
                      harwood_call=provenance.current_stamp() or "")
        steps = self.plan(params)
        shifts = np.hypot(out["xshift"], out["yshift"])
        steps[1].measured = {"stars": len(out), "median shift": f"{np.median(shifts):.2f} px",
                             "flagged": int((out["cier"] != 0).sum())}
        return TaskResult(outputs=[destination], steps=steps,
                          data={"stars": len(out), "flagged": int((out["cier"] != 0).sum())},
                          warnings=_flags(out["cier"], "cier", _CENTER_FIX))


class FitskyParams(_DataPars, _SkyPars):
    image: str = Field(..., description="Image to measure.")
    coords: Path = Field(..., description="Positions to measure the sky around, "
                                          "1-based (x, y). Not recentred.")
    output: Path | None = Field(
        None, description="Table of sky values. Left out: obj.sky in this run's "
                          "directory, as in IRAF.", json_schema_extra={"positional": True})


@register
class FitskyTask(Task[FitskyParams]):
    """Measure the sky in an annulus around each position, as phot does
    before it sums an aperture -- level, scatter, skew, how many pixels were
    used and how many rejected. The positions are used as given.
    """

    name = "fitsky"
    package = "apphot"
    description = "Fit the sky in an annulus around each position, as IRAF's fitsky."
    Params = FitskyParams

    def plan(self, params: FitskyParams) -> list[Step]:
        return [Step("read", f"{params.image} and {params.coords}", why=WHY["read"]),
                Step("sky", f"{params.salgorithm} over {params.annulus:g}.."
                            f"{params.annulus + params.dannulus:g} px", why=APWHY["sky"]),
                Step("write", str(params.output or "obj.sky in this run's directory"),
                     why=WHY["write"])]

    def execute(self, params: FitskyParams) -> TaskResult:
        import numpy as np
        from astropy.table import Table

        from ..core import apphot, catalog, images, output, provenance

        data, header = images.read(params.image)
        table = catalog.read(params.coords)
        x, y = catalog.coordinates(table, params.coords)
        dpars, spars = _datapars(params, header), _skypars(params)
        skies = [apphot.fitsky(data, a, b, spars, dpars) for a, b in zip(x, y, strict=True)]
        out = Table({"id": _ids(table, len(x)), "x": x, "y": y,
                     "msky": [s.value for s in skies], "stdev": [s.sigma for s in skies],
                     "sskew": [s.skew for s in skies], "nsky": [s.nsky for s in skies],
                     "nsrej": [s.nreject for s in skies], "sier": [s.ier for s in skies]})
        destination = params.output or output.named_for(params.image, ".sky")
        catalog.write(out, destination, harwood_image=str(params.image),
                      harwood_call=provenance.current_stamp() or "")
        steps = self.plan(params)
        steps[1].measured = {"sky": f"{np.nanmedian(out['msky']):.2f} counts",
                             "scatter": f"{np.nanmedian(out['stdev']):.2f} counts",
                             "flagged": int((out["sier"] != 0).sum())}
        return TaskResult(outputs=[destination], steps=steps,
                          data={"stars": len(out),
                                "sky_median": float(np.nanmedian(out["msky"]))},
                          warnings=_flags(out["sier"], "sier", _SKY_FIX))


class PhotParams(_DataPars, _CenterPars, _SkyPars):
    image: str = Field(..., description="Image to measure.")
    coords: Path = Field(..., description="Coordinate catalogue, 1-based (x,y), from daofind.")
    output: Path | None = Field(
        None,
        description="Photometry table to write. Left out, it goes to this run's "
                    "own directory under output/, named after the frame -- "
                    "obj.fits becomes obj.mag, as in IRAF.",
        json_schema_extra={"positional": True},
    )
    apertures: list[float] = Field(
        [3.0], description="Aperture radii in scale units. Several may be given."
    )
    zmag: float = Field(25.0, description="Zero point of the instrumental magnitude scale.")

    @model_validator(mode="after")
    def _check_radii(self) -> PhotParams:
        if self.salgorithm != "constant" and max(self.apertures) >= self.annulus:
            raise ValueError(
                f"the sky annulus starts at {self.annulus} px but the largest aperture "
                f"is {max(self.apertures)} px: the sky would be measured over the star. "
                "Raise --annulus or lower --apertures."
            )
        return self


@register
class Phot(Task[PhotParams]):
    """IRAF's phot: centre each star, fit the sky around the new centre, sum
    the apertures. The arithmetic is a port of IRAF 2.18's apphot, held by a
    test to what IRAF prints for the same pixels -- so these magnitudes are
    IRAF's magnitudes. The table has IRAF's columns: the starting and final
    positions, the shift and its error, msky, stdev, sskew, nsky, nsrej, and
    per aperture the sum, area, flux, mag and merr, with cier, sier and pier
    saying what, if anything, went wrong at each stage.
    """

    name = "phot"
    package = "apphot"
    description = "Aperture photometry with a local sky annulus."
    Params = PhotParams

    def plan(self, params: PhotParams) -> list[Step]:
        radii = ", ".join(f"{r:g}" for r in params.apertures)
        return [
            Step("read", f"{params.image} and {params.coords}", why=WHY["read"]),
            Step("center", f"{params.calgorithm}"
                           + (f" in a {params.cbox:g} px box" if params.calgorithm != "none"
                              else ": coordinates used as given"),
                 why=APWHY["center"]),
            Step("sky", f"{params.salgorithm} over the annulus {params.annulus:g}.."
                        f"{params.annulus + params.dannulus:g} px", why=APWHY["sky"]),
            Step("measure", f"apertures {radii} px", why=APWHY["measure"]),
            Step("calibrate", f"zmag={params.zmag:g}, epadu={params.epadu:g}, itime "
                              + (f"from {params.exposure}" if params.exposure
                                 else f"{params.itime:g}"),
                 why=APWHY["calibrate"]),
            Step("write", str(params.output or "obj.mag in this run's directory"),
                 why=WHY["write"]),
        ]

    def execute(self, params: PhotParams) -> TaskResult:
        import numpy as np
        from astropy.table import Table

        from ..core import apphot, catalog, images, output, provenance

        data, header = images.read(params.image)
        catalogue = catalog.read(params.coords)
        x0, y0 = catalog.coordinates(catalogue, params.coords)
        dpars = _datapars(params, header)
        cpars, spars = _centerpars(params), _skypars(params)
        radii = sorted(params.apertures)

        centres, skies, measured = [], [], []
        for a, b in zip(x0, y0, strict=True):
            c = apphot.center(data, a, b, cpars, dpars)
            s = apphot.fitsky(data, c.x, c.y, spars, dpars)
            centres.append(c)
            skies.append(s)
            measured.append(apphot.apertures(data, c.x, c.y, radii, s, params.zmag, dpars))

        out = Table({"id": _ids(catalogue, len(x0)), "xinit": x0, "yinit": y0,
                     "x": [c.x for c in centres], "y": [c.y for c in centres],
                     "xshift": [c.xshift for c in centres],
                     "yshift": [c.yshift for c in centres],
                     "xerr": [c.xerr for c in centres], "yerr": [c.yerr for c in centres],
                     "cier": [c.ier for c in centres],
                     "msky": [s.value for s in skies], "stdev": [s.sigma for s in skies],
                     "sskew": [s.skew for s in skies], "nsky": [s.nsky for s in skies],
                     "nsrej": [s.nreject for s in skies], "sier": [s.ier for s in skies],
                     "itime": np.full(len(x0), dpars.itime)})
        for k, radius in enumerate(radii):
            suffix = "" if len(radii) == 1 else f"_{radius:g}"
            for column in ("sums", "areas", "flux", "mag", "merr"):
                name = {"sums": "sum", "areas": "area"}.get(column, column)
                out[f"{name}{suffix}"] = [getattr(m, column)[k] for m in measured]
            out[f"pier{suffix}"] = [m.ier for m in measured]

        destination = params.output or output.named_for(params.image, ".mag")
        catalog.write(out, destination,
                      harwood_image=str(params.image),
                      harwood_itime=dpars.itime,
                      harwood_zmag=params.zmag,
                      harwood_call=provenance.current_stamp() or "")

        first = "" if len(radii) == 1 else f"_{radii[0]:g}"
        magnitudes = np.asarray(out[f"mag{first}"], dtype=float)
        errors = np.asarray(out[f"merr{first}"], dtype=float)
        good = np.isfinite(magnitudes)
        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        shifts = np.hypot(out["xshift"], out["yshift"])
        by_action["center"].measured = {
            "median shift": f"{np.median(shifts):.2f} px",
            "flagged": int((out["cier"] != 0).sum())}
        by_action["sky"].measured = {
            "sky": f"{np.nanmedian(out['msky']):.2f} counts",
            "scatter": f"{np.nanmedian(out['stdev']):.2f} counts",
            "pixels used": f"{np.median(out['nsky']):.0f}",
            "rejected": f"{np.median(out['nsrej']):.0f}"}
        by_action["measure"].measured = {
            "stars": len(out), "measured": int(good.sum()),
            "aperture area": f"{np.nanmedian(out[f'area{first}']):.1f} px"}
        if good.any():
            by_action["calibrate"].measured = {
                "itime": f"{dpars.itime:g}",
                "magnitudes": f"{magnitudes[good].min():.2f} to {magnitudes[good].max():.2f}",
                "best error": f"{np.nanmin(errors[good]) * 1000:.0f} mmag",
                "median error": f"{np.nanmedian(errors[good]) * 1000:.0f} mmag"}

        warnings = (_flags(out["cier"], "cier", _CENTER_FIX)
                    + _flags(out["sier"], "sier", _SKY_FIX)
                    + _flags(out[f"pier{first}"], "pier", _APERT_FIX))
        return TaskResult(
            outputs=[destination], steps=steps, warnings=warnings,
            data={"stars": len(out), "measured": int(good.sum()),
                  "sky_median": float(np.nanmedian(out["msky"]))},
            history=[f"phot: apertures={radii} annulus={params.annulus} "
                     f"dannulus={params.dannulus} salgorithm={params.salgorithm} "
                     f"calgorithm={params.calgorithm} zmag={params.zmag} "
                     f"epadu={params.epadu} itime={dpars.itime:g}"],
        )


_CENTER_FIX = ("look at the flagged rows (cier) in the table; for coordinates "
               "taken from another frame raise --maxshift, and for faint stars "
               "lower --minsnratio")
_SKY_FIX = ("look at the flagged rows (sier); widen --dannulus, or move "
            "--annulus away from a neighbour")
_APERT_FIX = ("look at the flagged rows (pier); a star near the edge loses its "
              "largest apertures, and bad pixels follow --datamin/--datamax")


def _ids(table, n):
    return table["id"] if "id" in table.colnames else list(range(1, n + 1))


def _border(params) -> float:
    """How close to the edge a detection may sit.

    IRAF's daofind skips the border because its convolution kernel does not
    fit there; photutils will happily report the artefact. On the M92 frames
    a fifth of all detections came from the truncated edge, so the default
    matters more than it looks.
    """
    return float(params.border) if params.border is not None else 2.0 * params.fwhm


def _shape_bounds(finder_class, params) -> dict:
    """Pass the sharpness and roundness limits under whichever names apply.

    photutils 3.0 replaced `sharplo`/`sharphi` with `sharpness_range`. harwood
    keeps IRAF's four names on its own surface -- they are what a user types --
    and adapts here, at the boundary, where the ecosystem's churn belongs.
    """
    import inspect

    accepted = inspect.signature(finder_class).parameters
    if "sharpness_range" in accepted:
        return {"sharpness_range": (params.sharplo, params.sharphi),
                "roundness_range": (params.roundlo, params.roundhi)}
    return {"sharplo": params.sharplo, "sharphi": params.sharphi,
            "roundlo": params.roundlo, "roundhi": params.roundhi}


def _column(table, *names):
    """First column that exists, so a photutils rename is not an outage."""
    for name in names:
        if name in table.colnames:
            return table[name]
    raise KeyError(
        f"photutils returned none of {names}; it has {', '.join(table.colnames)}. "
        "This is a photutils API change -- harwood needs updating."
    )
