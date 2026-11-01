"""Photometry -- detection and aperture measurement.

IRAF names throughout, including the ones that are not obvious: the task is
`phot`, not `apphot` (`apphot` is the *package*); the aperture radii are
`apertures`; the sky annulus is `annulus` and `dannulus`; the gain is `epadu`;
the magnitude zero point is `zmag`. Section 3 is explicit that a different name
costs little and the same name meaning something else ruins a result.

Every coordinate that leaves this module is 1-based and (x, y) -- section 2.2.
Both daofind and phot are IRAF's own arithmetic (core.daofind, core.apphot),
which counts from 1 as well. The catalogue written here can be read by someone who has spent
twenty years reading IRAF `.mag` files without a systematic pixel offset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "read": "Pixels come in as floating point whatever the detector wrote. A "
            "measurement is never made on the integers a CCD happens to produce: "
            "a sum of sixteen-bit frames overflows silently, and a difference of "
            "unsigned ones wraps.",
    "background": "The detection threshold is set in units of the noise, not in "
                  "counts, so it means the same thing on a bright night and a dark "
                  "one. The scatter is measured with clipping, so the stars "
                  "themselves do not inflate the estimate of the noise they are to "
                  "be detected against. The sky level itself is not needed: the "
                  "detection kernel sums to zero.",
    "detect": "DAOFIND convolves the image with a Gaussian of the given width and "
              "looks for maxima in the result. The width is the whole parameter: "
              "a filter matched to the stars suppresses everything that is not "
              "star-shaped, so a cosmic ray, which is sharper, and a galaxy, which "
              "is broader, both respond less than a star of the same brightness.",
    "filter": "Sharpness and roundness are shape statistics, and they separate "
              "things a brightness cut cannot. A cosmic ray is too sharp to be a "
              "point source through this atmosphere; a galaxy is too broad; a bad "
              "column is too elongated.",
    "border": "IRAF extends the image by its nearest pixel, so the filter is "
              "defined to the edge and a star half off the frame is still found. "
              "Its centre is less certain there, and phot loses its outer "
              "apertures, which is the reason to drop them.",
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
    "findkeys": "The threshold is a judgement about this frame: too low and the "
                "noise is stars, too high and the faint ones are gone. Trying it on the "
                "image is quicker than reading a list, and f only looks, so that what "
                "is written is what was decided on.",
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

    fwhmpsf: float = Field(
        2.5, gt=0,
        description="FWHM of the point spread function, in pixels. The single most "
                    "important parameter: measure it with imexamine before guessing.",
    )
    sigma: float | None = Field(
        None, gt=0,
        description="Standard deviation of the background, in counts. IRAF takes it "
                    "from datapars and, left INDEF, detects at a threshold of zero; "
                    "harwood measures it from the image instead (sigma-clipped "
                    "standard deviation) and says what it measured.",
    )
    threshold: float = Field(
        4.0, gt=0, description="Detection threshold above background, in units of sigma."
    )
    nsigma: float = Field(
        1.5, gt=0, description="Width of the detection kernel, in sigmas of the gaussian."
    )
    ratio: float = Field(
        1.0, ge=0, le=1,
        description="Ratio of the kernel's minor to major sigma (1 for round stars).",
    )
    theta: float = Field(
        0.0, ge=0, le=180,
        description="Position angle of the kernel's major axis, degrees from +x.",
    )
    sharplo: float = Field(0.2, description="Lower sharpness bound; rejects cosmic rays.")
    sharphi: float = Field(1.0, description="Upper sharpness bound; rejects galaxies.")
    roundlo: float = Field(-1.0, description="Lower bound on both roundnesses.")
    roundhi: float = Field(1.0, description="Upper bound on both roundnesses.")
    epadu: float = Field(
        1.0, gt=0,
        description="Gain, electrons per count. With readnoise, it only enters the "
                    "sky estimate that sharpness and the centres subtract.",
    )
    readnoise: float = Field(0.0, ge=0, description="Readout noise, in electrons.")
    border: float = Field(
        0.0, ge=0,
        description="Drop detections within this many pixels of the edge. IRAF "
                    "keeps them -- it extends the image by its nearest pixel, so "
                    "the filter is defined right up to the edge -- and so does "
                    "harwood by default.",
    )
    interactive: bool = Field(
        False, description="Try parameters on the image in the window, with IRAF's "
                           "keys: f finds, the space bar finds and writes, q ends.")
    icommands: str = Field(
        "", description="The same keys from a file, 'x y wcs key' per line.")


@register
class DaoFind(Task[DaoFindParams]):
    name = "daofind"
    package = "apphot"
    description = "Detect point sources with the DAOFIND algorithm."
    Params = DaoFindParams

    def _by_the_keys(self, params, data, header, sigma, measured) -> TaskResult:
        """Daofind's keys, from --icommands or the window: a list for each time
        the space bar found something new."""
        import numpy as np

        from ..core import catalog, daofindsession, output, photsession, provenance

        if params.icommands and not Path(params.icommands).exists():
            advice.fail(f"cursor file '{params.icommands}' does not exist",
                        "write 'x y 1 key' lines, or use --interactive for the window",
                        FileNotFoundError)
        destination = params.output or output.named_for(params.image, ".coo")
        session = daofindsession.Session(
            data, str(params.image), str(destination),
            dict(fwhmpsf=params.fwhmpsf, sigma=sigma, threshold=params.threshold,
                 nsigma=params.nsigma, ratio=params.ratio, theta=params.theta,
                 sharplo=params.sharplo, sharphi=params.sharphi, roundlo=params.roundlo,
                 roundhi=params.roundhi, epadu=params.epadu, readnoise=params.readnoise,
                 scale=1.0))
        if params.icommands:
            for x, y, key in photsession.read_commands(params.icommands):
                session.colon(key) if key.startswith(":") else session.key(key, x, y)
                if session.done:
                    break
        else:
            _find_in_window(session, data, header, params)
        outputs, sources = [], 0
        for name, found, _ in session.written:
            keep = np.flatnonzero(_inside(found, data.shape, params.border))
            sources = keep.size
            outputs.append(catalog.write(
                _catalogue(found, keep), name, harwood_image=str(params.image),
                harwood_sigma=float(session.params["sigma"]),
                harwood_threshold=found["threshold"],
                harwood_call=provenance.current_stamp() or ""))
        warnings = []
        if measured:
            warnings.append(advice.note(
                f"the background sigma was measured from the image: {sigma:.3f}.",
                "set another with :sigma, or give it with --sigma."))
        if not session.written:
            warnings.append(advice.warn(
                "nothing was written: f only finds, and the space bar writes.",
                "press the space bar before q when the list looks right."))
        refits = [name for name, _, refit in session.written if refit]
        if refits:
            warnings.append(advice.warn(
                f"the sharpness in {', '.join(refits)} is what IRAF gives when it finds "
                "again on the same kernel, which counts one pixel too few, so some stars "
                "may have passed or failed sharplo by it.",
                "to have the sharpness a plain run gives, change a kernel parameter or "
                "run it without the keys."))
        steps = [Step("read", params.image, why=WHY["read"]),
                 Step("keys", f"the keys, from {params.icommands or 'the window'}",
                      why=WHY["findkeys"]),
                 Step("write", ", ".join(str(path) for path in outputs) or "nothing",
                      why=WHY["write"])]
        data_out = {"sources": sources, "sigma": float(session.params["sigma"])}
        if session.said:
            data_out["text"] = session.said
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings, data=data_out)

    def plan(self, params: DaoFindParams) -> list[Step]:
        steps = [
            Step("read", params.image, why=WHY["read"]),
            Step("background", "sigma-clipped standard deviation"
                 if params.sigma is None else f"sigma={params.sigma} as given",
                 why=WHY["background"]),
            Step("detect", f"DAOFIND, fwhmpsf={params.fwhmpsf}, "
                           f"threshold={params.threshold} sigma, "
                           f"nsigma={params.nsigma}",
                 why=WHY["detect"]),
            Step("filter", f"sharpness {params.sharplo}..{params.sharphi}, "
                           f"roundness {params.roundlo}..{params.roundhi}",
                 why=WHY["filter"]),
        ]
        if params.border > 0:
            steps.append(Step("border", f"drop anything within {params.border:g} px "
                                        "of the edge", why=WHY["border"]))
        steps.append(Step("write", f"{params.output or 'obj.coo in this run''s directory'}, "
                                   "coordinates 1-based (x,y), brightest first",
                          why=WHY["write"]))
        return steps

    def execute(self, params: DaoFindParams) -> TaskResult:
        import numpy as np
        from astropy.stats import sigma_clipped_stats

        from ..core import catalog, daofind, images, output, provenance

        data, header = images.read(params.image)
        if params.sigma is not None:
            sigma, measured = float(params.sigma), False
        else:
            sigma, measured = float(sigma_clipped_stats(data, sigma=3.0)[2]), True
        if not np.isfinite(sigma) or sigma <= 0:
            raise RuntimeError(
                f"the background sigma measured {sigma}; the frame may be blank or "
                "constant. Give it with --sigma."
            )

        if params.interactive or params.icommands:
            return self._by_the_keys(params, data, header, sigma, measured)

        found = daofind.find(
            data, params.fwhmpsf, sigma, threshold=params.threshold,
            nsigma=params.nsigma, ratio=params.ratio, theta=params.theta,
            sharplo=params.sharplo, sharphi=params.sharphi,
            roundlo=params.roundlo, roundhi=params.roundhi,
            epadu=params.epadu, readnoise=params.readnoise)
        x = found["x"]
        if x.size == 0:
            raise RuntimeError(
                f"no sources found. The background sigma "
                f"{'measured' if measured else 'given'} is {sigma:.2f}; try a lower "
                f"--threshold than {params.threshold}, or check --fwhmpsf."
            )

        border = params.border
        inside = _inside(found, data.shape, border)
        dropped = int((~inside).sum())
        if not inside.any():
            raise RuntimeError(
                f"every one of the {x.size} detections was within {border:g} px "
                "of the edge. Lower --border if the field really does run to the edge."
            )
        keep = np.flatnonzero(inside)
        catalogue = _catalogue(found, keep)
        destination = params.output or output.named_for(params.image, ".coo")
        catalog.write(catalogue, destination,
                      harwood_image=str(params.image),
                      harwood_sigma=sigma,
                      harwood_threshold=found["threshold"],
                      harwood_call=provenance.current_stamp() or "")

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        by_action["background"].measured = {
            "sigma": round(sigma, 3),
            "threshold": f"{params.threshold * sigma:.2f} counts of fitted peak",
        }
        mags = np.asarray(catalogue["mag"])
        by_action["detect"].measured = {
            "found": int(x.size),
            "mag": f"{mags.min():.2f} to {mags.max():.2f} (0 at the threshold)",
        }
        if "border" in by_action:
            by_action["border"].measured = {
                "dropped": f"{dropped} of {x.size}", "kept": int(keep.size),
            }

        return TaskResult(
            outputs=[destination],
            steps=steps,
            data={"sources": len(catalogue), "dropped_at_border": dropped,
                  "sigma": sigma},
            history=[f"daofind: {len(catalogue)} sources, fwhmpsf={params.fwhmpsf}, "
                     f"threshold={params.threshold} sigma (sigma={sigma:.2f}"
                     f"{', measured' if measured else ''})"
                     + (f", {dropped} dropped within {border:g} px of the edge"
                        if border > 0 else "")],
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
    gain: str = Field("", description="Header keyword with the gain, in electrons "
                                      "per ADU; empty uses epadu.")
    ccdread: str = Field("", description="Header keyword with the read noise; empty "
                                         "uses readnoise.")
    airmass: str = Field("", description="Header keyword with the airmass, recorded "
                                         "with each star for photcal; empty uses "
                                         "xairmass.")
    xairmass: float | None = Field(None, gt=0, description="Airmass when no keyword is "
                                                           "named. Empty is INDEF.")
    filter: str = Field("", description="Header keyword with the filter, recorded "
                                        "with each star; empty uses ifilter.")
    ifilter: str = Field("INDEF", description="Filter when no keyword is named.")
    obstime: str = Field("", description="Header keyword with the time of the "
                                         "observation, recorded with each star; empty "
                                         "uses otime.")
    otime: str = Field("INDEF", description="Time of the observation when no keyword "
                                            "is named.")
    emission: bool = Field(True, description="The features are above the sky. Off "
                                             "(IRAF's emission=no), center finds dips.")
    noise: Literal["poisson", "constant"] = Field(
        "poisson", description="Noise model for the centring S/N and the clean: the "
                               "star's photons plus the sky's sigma (poisson), or the "
                               "sky's sigma alone (constant).")


class _CenterPars(TaskParams):
    calgorithm: Literal["centroid", "gauss", "ofilter", "none"] = Field(
        "centroid", description="Centring: the marginal centroid (IRAF's default); "
                                "gauss, a gaussian of width fwhmpsf fitted to each "
                                "marginal; ofilter, optimal filtering of the marginals "
                                "with a triangle of that width; or none to keep the "
                                "coordinates as given.")
    cbox: float = Field(5.0, gt=0, description="Width of the centring box, in scale units.")
    cthreshold: float = Field(
        0.0, ge=0, description="Threshold above the box's lowest row median, in "
                               "units of sigma. 0 uses the box minimum.")
    minsnratio: float = Field(1.0, ge=0, description="Below this S/N the centre is "
                                                     "flagged as unreliable.")
    cmaxiter: int = Field(10, ge=1, description="Most centring iterations.")
    maxshift: float = Field(1.0, gt=0, description="A centre that moves further than "
                            "this, in scale units, is flagged -- and kept.")
    clean: bool = Field(False, description="Symmetry-clean the centring box first: a "
                        "pixel brighter than its mirror image through the peak by more "
                        "than the noise allows takes the mirror's value. Needs sigma.")
    rclean: float = Field(1.0, ge=0, description="Inside this radius, in scale units, "
                                                  "nothing is cleaned.")
    rclip: float = Field(2.0, ge=0, description="Beyond rclean and inside this radius the "
                         "noise allowed includes the star's photons (noise=poisson); "
                         "beyond it, only kclean sky sigmas.")
    kclean: float = Field(3.0, gt=0, description="Cleaning limit, in sky sigmas.")


class _SkyPars(TaskParams):
    salgorithm: Literal["centroid", "mode", "median", "mean", "gauss", "ofilter",
                        "crosscor", "histplot", "radplot", "constant", "file"] = Field(
        "centroid", description="Sky estimator. centroid, IRAF's default, is the "
                                "centroid of the histogram of the annulus; mode is "
                                "3 median - 2 mean; gauss fits a skewed gaussian to the "
                                "histogram; ofilter filters it with a triangle; "
                                "crosscor takes the peak of its correlation with a "
                                "gaussian; histplot and radplot are marked by hand on "
                                "the histogram or the radial profile; file reads phot's "
                                "--skyfile.")
    gcommands: str = Field(
        "", description="Graphics cursor commands for histplot and radplot, 'x y wcs "
                        "key' per line, each star reading until q. Empty: the window.")
    annulus: float = Field(10.0, gt=0, description="Inner radius of the sky annulus, "
                                                   "in scale units.")
    dannulus: float = Field(10.0, gt=0, description="Width of the sky annulus, in "
                                                    "scale units.")
    skyvalue: float = Field(0.0, description="The sky, when salgorithm is constant.")
    smaxiter: int = Field(10, ge=0, description="Most rejection passes of centroid, "
                                                "ofilter and crosscor, and most "
                                                "iterations of each gauss fit.")
    snreject: int = Field(50, ge=0, description="Most rejection passes of mode, "
                                                "median, mean and gauss.")
    sloclip: float = Field(0.0, ge=0, lt=100, description="Percent of the annulus' "
                           "lowest pixels dropped before fitting.")
    shiclip: float = Field(0.0, ge=0, lt=100, description="Percent of its highest "
                           "pixels dropped before fitting.")
    sloreject: float | None = Field(3.0, gt=0, description="Lower rejection limit, in "
                                    "sky sigma. Empty is IRAF's INDEF: no lower cut.")
    shireject: float | None = Field(3.0, gt=0, description="Upper rejection limit, in "
                                    "sky sigma. Empty: no upper cut.")
    khist: float = Field(3.0, gt=0, description="Half width of the histogram of the "
                                                "histogram algorithms, in sky sigma.")
    binsize: float = Field(0.1, gt=0, description="Its bin size, in sky sigma.")
    smooth: bool = Field(False, description="Boxcar-smooth the histogram before "
                                            "centroid, gauss or ofilter read it.")
    rgrow: float = Field(0.0, ge=0, description="Reject, with each rejected pixel, "
                         "every pixel within this radius (scale units): a cosmic ray "
                         "or a faint star goes whole, wings included. 0 rejects pixels "
                         "one by one.")


def _datapars(params, header):
    from ..core import advice, apphot

    itime = params.itime
    if params.exposure:
        if params.exposure not in header:
            advice.fail(f"the header has no '{params.exposure}'",
                        "name the keyword that holds the exposure time, or leave "
                        "exposure empty and give itime")
        itime = float(header[params.exposure])
    epadu, readnoise = params.epadu, params.readnoise
    for keyword, what in ((params.gain, "gain"), (params.ccdread, "read noise")):
        if keyword and keyword not in header:
            advice.fail(f"the header has no '{keyword}'",
                        f"name the keyword that holds the {what}, or leave it empty "
                        "and give the value")
    if params.gain:
        epadu = float(header[params.gain])
    if params.ccdread:
        readnoise = float(header[params.ccdread])
    return apphot.DataPars(scale=params.scale, fwhmpsf=params.fwhmpsf,
                           sigma=params.sigma, datamin=params.datamin,
                           datamax=params.datamax, epadu=epadu,
                           readnoise=readnoise, itime=itime, noise=params.noise,
                           emission=params.emission)


def _observed(params, header) -> tuple[dict, list[str]]:
    """The filter, airmass and time of the exposure, from the keywords
    datapars names or else its values -- what IRAF writes with every star
    (IFILTER, XAIRMASS, OTIME), and photcal reads to group the images and
    correct for the atmosphere. A keyword named but missing falls back to
    the value, with a note, as in IRAF."""
    notes = []

    def get(keyword, default):
        if not keyword:
            return default
        if keyword not in header:
            notes.append(f"the header has no '{keyword}'; {default} used")
            return default
        return header[keyword]

    airmass = get(params.airmass, params.xairmass)
    try:
        airmass = float(airmass) if airmass is not None else float("nan")
    except (TypeError, ValueError):
        notes.append(f"'{params.airmass}' is not a number; the airmass is INDEF")
        airmass = float("nan")
    if not airmass > 0:
        airmass = float("nan")
    ifilter = "".join(str(get(params.filter, params.ifilter)).split()) or "INDEF"
    otime = "".join(str(get(params.obstime, params.otime)).split()) or "INDEF"
    return {"ifilter": ifilter, "xairmass": airmass, "otime": otime}, notes


def _centerpars(params):
    from ..core import apphot

    return apphot.CenterPars(params.calgorithm, params.cbox, params.cthreshold,
                             params.minsnratio, params.cmaxiter, params.maxshift,
                             params.clean, params.rclean, params.rclip, params.kclean)


def _need_coords(params):
    if params.coords is None:
        advice.fail("no coordinates to measure",
                    "give the list daofind wrote, or measure with the cursor: "
                    "--interactive, or --icommands with a cursor file")
    return params.coords


def _by_cursor(task, params, kind: str) -> TaskResult:
    """phot or center driven by the cursor -- the window's, or IRAF's
    icommands file -- through core.photsession; the rows written are the
    ones IRAF's keys write."""
    import numpy as np
    from astropy.table import Table

    from ..core import apphot, catalog, images, output, photsession, provenance

    data, header = images.read(params.image)
    stars = []
    if params.coords is not None:
        table = catalog.read(params.coords)
        x, y = catalog.coordinates(table, params.coords)
        stars = [(float(a), float(b)) for a, b in zip(x, y, strict=True)]
    dpars, cpars = _datapars(params, header), _centerpars(params)
    if kind == "phot":
        spars = _skypars(params)
        cursor, skyfile = _sky_sources(params)
        radii, zmag = sorted(params.apertures), params.zmag
    else:
        spars, cursor, skyfile, radii, zmag = apphot.SkyPars(), None, None, [], 25.0
    session = photsession.Session(data, dpars, cpars, spars, radii, zmag, stars, kind,
                                  cursor, skyfile)
    if params.icommands:
        if not Path(params.icommands).exists():
            advice.fail(f"cursor file '{params.icommands}' does not exist",
                        "write 'x y 1 key' lines, or use --interactive for the window",
                        FileNotFoundError)
        for x, y, key in photsession.read_commands(params.icommands):
            session.colon(key) if key.startswith(":") else session.key(key, x, y)
            if session.done:
                break
    else:
        _in_window(session, data, header, params)

    rows = session.rows
    out = Table({"id": [r.id for r in rows], "lid": [r.lid for r in rows],
                 "xinit": [r.xinit for r in rows], "yinit": [r.yinit for r in rows],
                 "x": [r.center.x for r in rows], "y": [r.center.y for r in rows],
                 "xshift": [r.center.xshift for r in rows],
                 "yshift": [r.center.yshift for r in rows],
                 "xerr": [r.center.xerr for r in rows], "yerr": [r.center.yerr for r in rows],
                 "cier": [r.center.ier for r in rows]})
    if kind == "phot":
        for name, attr in (("msky", "value"), ("stdev", "sigma"), ("sskew", "skew"),
                           ("nsky", "nsky"), ("nsrej", "nreject"), ("sier", "ier")):
            out[name] = [getattr(r.sky, attr) for r in rows]
        out["itime"] = np.full(len(rows), dpars.itime)
        for name, value in _observed(params, header)[0].items():
            out[name] = [value] * len(rows)
        every = sorted({radius for r in rows for radius in r.radii})
        for radius in every:
            suffix = "" if len(every) == 1 else f"_{radius:g}"
            for column in ("sums", "areas", "flux", "mag", "merr"):
                name = {"sums": "sum", "areas": "area"}.get(column, column)
                out[f"{name}{suffix}"] = [
                    getattr(r.mags, column)[r.radii.index(radius)] if radius in r.radii
                    else np.nan for r in rows]
            out[f"pier{suffix}"] = [r.mags.ier if radius in r.radii else -1 for r in rows]
    destination = params.output or output.named_for(params.image,
                                                    ".mag" if kind == "phot" else ".ctr")
    catalog.write(out, destination, harwood_image=str(params.image),
                  harwood_call=provenance.current_stamp() or "")
    steps = task.plan(params)
    data_out: dict = {"stars": len(out)}
    if session.messages:
        data_out["text"] = session.messages
    return TaskResult(outputs=[destination], steps=steps, data=data_out,
                      warnings=_flags(out["cier"], "cier", _CENTER_FIX) if len(out) else [])


def _in_window(session, data, header, params) -> None:
    """The image in the window and its keys handed to the session until q;
    each measurement circled where it was centred."""
    from ..core import display

    screen = display.get(None)
    window = getattr(screen, "window", None)
    if window is None or not hasattr(window, "wait_for_examine"):
        advice.fail("--interactive needs the window",
                    "run it where there is a screen, or give the keys in a file with "
                    "--icommands")
    screen.show(data, header, name=str(params.image))
    shown = [0]

    def redraw() -> None:
        frame = window.frames.get(window.current)
        if frame is None:
            return
        rows = session.rows[shown[0]:]
        radius = max(session.radii) if session.radii else 3.0
        frame.markers += [display.Marker(r.center.x, r.center.y, "circle", radius,
                                         label=str(r.id), colour="accent", tag="phot")
                          for r in rows]
        shown[0] = len(session.rows)
        for message in session.messages[getattr(redraw, "said", 0):]:
            window.log("  " + message)
        redraw.said = len(session.messages)
        if rows:
            last = rows[-1]
            what = (f"  {last.id}: x={last.center.x:.3f} y={last.center.y:.3f}"
                    + (f" sky={last.sky.value:.2f} mag="
                       + " ".join(f"{m:.3f}" for m in last.mags.mag)
                       if last.mags is not None else ""))
            window.log(what)
            if last.profile is not None and last.profile.intensity.size:
                _plot_profile(window, last)
        window._redraw()

    def key(name: str, x: float, y: float) -> bool:
        session.key(name, x, y)
        redraw()
        if session.done:
            window.examine_done.emit()
        return True

    def colon(line: str) -> str:
        session.colon(line)
        redraw()
        return ""

    window.key_hook, window.colon_hook = key, colon
    from ..core import photsession

    window.set_task_keys(session.task, photsession.HELP)
    window.log(f"{session.task}: space measures and writes, n the next star, l the "
               "rest, :annulus 12 sets a parameter, q ends -- the Keys panel has "
               "them all.")
    try:
        window.wait_for_examine()
    finally:
        window.key_hook = window.colon_hook = None
        window.set_task_keys(None)


def _plot_profile(window, row) -> None:
    """ap_rpplot: the profile and the light inside each radius, against the radius."""
    import pyqtgraph as pg

    from ..viewer.theme import palette

    profile = row.profile
    window.plot.clear()
    window.plot.setTitle(f"radprof · star {row.id}: FWHM {profile.fwhm:.3f}", size="10pt")
    window.plot.setLabel("bottom", "radius")
    window.plot.setLabel("left", "intensity")
    window.plot.plot(profile.radius, profile.intensity, pen=pg.mkPen(palette().sky, width=1))
    dashed = pg.mkPen(palette().accent, width=1, style=pg.QtCore.Qt.PenStyle.DashDotLine)
    window.plot.plot(profile.radius, profile.tintensity, pen=dashed)


def _sky_sources(params):
    """The graphics cursor histplot and radplot read, and phot's sky file:
    --gcommands, or the window when there is a screen; None otherwise, and
    the stars are flagged as IRAF flags them."""
    from ..core import advice, display, fitsky

    cursor = skyfile = None
    if params.salgorithm in ("histplot", "radplot"):
        if params.gcommands:
            if not Path(params.gcommands).exists():
                advice.fail(f"graphics cursor file '{params.gcommands}' does not exist",
                            "write 'x y 1 key' lines, q ending each star")
            cursor = fitsky.Cursor(params.gcommands)
        elif display._autodetect() == "native":
            from ..viewer.graph_panel import mark

            cursor = fitsky.Cursor(show=mark)
    if cursor is not None:
        cursor.image = str(params.image)
    if params.salgorithm == "file" and getattr(params, "skyfile", ""):
        skyfile = fitsky.SkyFile(params.skyfile)
    return cursor, skyfile


def _skypars(params):
    from ..core import apphot

    return apphot.SkyPars(params.salgorithm, params.annulus, params.dannulus,
                          params.skyvalue, params.smaxiter, params.snreject,
                          params.sloreject, params.shireject, params.khist,
                          params.binsize, params.smooth, params.sloclip,
                          params.shiclip, params.rgrow)


def _sky_extras(params) -> str:
    """What the sky step does beyond the default, for the plan."""
    extras = []
    if params.sloclip or params.shiclip:
        extras.append(f"clipping {params.sloclip:g}%/{params.shiclip:g}%")
    if params.rgrow:
        extras.append(f"growing rejections by {params.rgrow:g}")
    if params.smooth and params.salgorithm in ("centroid", "gauss", "ofilter"):
        extras.append("smoothed histogram")
    return "".join(", " + e for e in extras)


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


class _Interactive(TaskParams):
    interactive: bool = Field(
        False, description="Measure with the cursor in the window, IRAF's keys: space "
                           "measures and writes, n the next star of the list, l the "
                           "rest, :annulus 12 sets a parameter, q ends.")
    icommands: str = Field(
        "", description="Image cursor commands instead of the window: 'x y 1 key' or "
                        "'x y 1 :command' per line, IRAF's icommands. \\040 is the "
                        "space bar.")


def _inside(found: dict, shape: tuple[int, int], border: float):
    """The detections further than `border` from the edge."""
    ny, nx = shape
    x, y = found["x"], found["y"]
    return ((x > border + 0.5) & (x < nx + 0.5 - border)
            & (y > border + 0.5) & (y < ny + 0.5 - border))


def _catalogue(found: dict, keep):
    """A detection list as a table. Brightest first, where IRAF writes them line
    by line: it is the one ordering two frames of the same field agree on, which
    is what xyxymatch's triangle search needs when it takes only the brightest
    nmatch stars from each list."""
    import numpy as np
    from astropy.table import Table

    keep = keep[np.argsort(found["mag"][keep], kind="stable")]
    return Table({
        "id": range(1, keep.size + 1),
        **{name: found[name][keep]
           for name in ("x", "y", "mag", "sharpness", "sround", "ground")},
    })


def _find_in_window(session, data, header, params) -> None:
    """The image in the window and daofind's keys handed to the session until q;
    the stars of the last find circled, and d the radial profile."""
    from ..core import daofindsession, display

    screen = display.get(None)
    window = getattr(screen, "window", None)
    if window is None or not hasattr(window, "wait_for_examine"):
        advice.fail("--interactive needs the window",
                    "run it where there is a screen, or give the keys in a file with "
                    "--icommands")
    screen.show(data, header, name=str(params.image))
    said = [0]

    def redraw() -> None:
        for line in session.said[said[0]:]:
            window.log("  " + line)
        said[0] = len(session.said)
        frame = window.frames.get(window.current)
        if frame is not None and session.found is not None:
            found = session.found
            radius = float(session.params["fwhmpsf"])
            frame.markers = [m for m in frame.markers if m.tag != "daofind"] + [
                display.Marker(float(x), float(y), "circle", radius, colour="accent",
                               tag="daofind")
                for x, y in zip(found["x"], found["y"], strict=True)]
        window._redraw()

    def key(name: str, x: float, y: float) -> bool:
        session.key(name, x, y)
        if name == "d":
            window.examine_at("r", x, y)
        redraw()
        if session.done:
            window.examine_done.emit()
        return True

    def colon(line: str) -> str:
        session.colon(line)
        redraw()
        return ""

    window.key_hook, window.colon_hook = key, colon
    window.set_task_keys("daofind", daofindsession.HELP)
    window.log("daofind: f finds, the space bar finds and writes, :threshold 6 sets a "
               "parameter, q ends -- the Keys panel has them all.")
    try:
        window.wait_for_examine()
    finally:
        window.key_hook = window.colon_hook = None
        window.set_task_keys(None)


class CenterParams(_DataPars, _CenterPars, _Interactive):
    image: str = Field(..., description="Image to measure.")
    coords: Path | None = Field(None, description="Coordinates to start from, 1-based "
                                                  "(x, y). Optional with the cursor.",
                                json_schema_extra={"positional": True})
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
                               f"maxshift {params.maxshift:g}"
                               + (", symmetry-cleaned" if params.clean else ""),
                     why=APWHY["center"]),
                Step("write", str(params.output or "obj.ctr in this run's directory"),
                     why=WHY["write"])]

    def execute(self, params: CenterParams) -> TaskResult:
        import numpy as np
        from astropy.table import Table

        from ..core import apphot, catalog, images, output, provenance

        if params.interactive or params.icommands:
            return _by_cursor(self, params, "center")
        data, header = images.read(params.image)
        table = catalog.read(_need_coords(params))
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
                            f"{params.annulus + params.dannulus:g} px" + _sky_extras(params),
                     why=APWHY["sky"]),
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
        cursor, _ = _sky_sources(params)
        skies = [apphot.fitsky(data, a, b, spars, dpars, cursor)
                 for a, b in zip(x, y, strict=True)]
        out = Table({"id": _ids(table, len(x)), "x": x, "y": y,
                     "msky": [s.value for s in skies], "stdev": [s.sigma for s in skies],
                     "sskew": [s.skew for s in skies], "nsky": [s.nsky for s in skies],
                     "nsrej": [s.nreject for s in skies], "sier": [s.ier for s in skies]})
        destination = params.output or output.named_for(params.image, ".sky")
        catalog.write(out, destination, harwood_image=str(params.image),
                      harwood_call=provenance.current_stamp() or "")
        steps = self.plan(params)
        good = np.isfinite(np.asarray(out["msky"], dtype=float))
        median = float(np.median(out["msky"][good])) if good.any() else float("nan")
        scatter = float(np.median(out["stdev"][good])) if good.any() else float("nan")
        steps[1].measured = {"sky": f"{median:.2f} counts",
                             "scatter": f"{scatter:.2f} counts",
                             "flagged": int((out["sier"] != 0).sum())}
        return TaskResult(outputs=[destination], steps=steps,
                          data={"stars": len(out), "sky_median": median},
                          warnings=_flags(out["sier"], "sier", _SKY_FIX))


class PhotParams(_DataPars, _CenterPars, _SkyPars, _Interactive):
    image: str = Field(..., description="Image to measure.")
    coords: Path | None = Field(None, description="Coordinate catalogue, 1-based (x,y), "
                                                  "from daofind. Optional with the cursor.",
                                json_schema_extra={"positional": True})
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
    skyfile: str = Field(
        "", description="Sky values for salgorithm file: one 'x y msky stdev sskew nsky "
                        "nsrej' line per star, in the order of the coordinates -- a "
                        "fitsky .sky table's columns.")
    format: Literal["harwood", "iraf"] = Field(
        "harwood", description="harwood writes an ECSV table at full precision; iraf "
                               "writes IRAF's own .mag text file, which IRAF's tasks "
                               "read -- rounded as IRAF rounds it.")

    @model_validator(mode="after")
    def _check_radii(self) -> PhotParams:
        if self.salgorithm not in ("constant", "file") and \
                max(self.apertures) > self.annulus:
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
                              else ": coordinates used as given")
                           + (", symmetry-cleaned" if params.clean
                              and params.calgorithm != "none" else ""),
                 why=APWHY["center"]),
            Step("sky", f"{params.salgorithm} over the annulus {params.annulus:g}.."
                        f"{params.annulus + params.dannulus:g} px" + _sky_extras(params),
                 why=APWHY["sky"]),
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

        from ..core import apphot, catalog, images, mef, output, provenance

        if params.interactive or params.icommands:
            return _by_cursor(self, params, "phot")
        data, header = images.read(params.image)
        _, _, variance, quality = mef.load(params.image)
        catalogue = catalog.read(_need_coords(params))
        x0, y0 = catalog.coordinates(catalogue, params.coords)
        dpars = _datapars(params, header)
        cpars, spars = _centerpars(params), _skypars(params)
        radii = sorted(params.apertures)

        centres, skies, measured = [], [], []
        cursor, skyfile = _sky_sources(params)
        for a, b in zip(x0, y0, strict=True):
            c = apphot.center(data, a, b, cpars, dpars)
            s = apphot.fitsky(data, c.x, c.y, spars, dpars, cursor, skyfile)
            centres.append(c)
            skies.append(s)
            measured.append(apphot.apertures(data, c.x, c.y, radii, s, params.zmag, dpars,
                                             variance, quality))

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
        observed, observed_notes = _observed(params, header)
        for name, value in observed.items():
            out[name] = [value] * len(x0)
        for k, radius in enumerate(radii):
            suffix = "" if len(radii) == 1 else f"_{radius:g}"
            for column in ("sums", "areas", "flux", "mag", "merr"):
                name = {"sums": "sum", "areas": "area"}.get(column, column)
                out[f"{name}{suffix}"] = [getattr(m, column)[k] for m in measured]
            out[f"pier{suffix}"] = [m.ier for m in measured]
            if variance is not None:
                # Section 12: the error the image's own variance gives, flat
                # field and all; IRAF's, from epadu and the sky alone, is kept
                # beside it for anyone comparing with an IRAF reduction.
                ferr = [m.flux_error(sky, k) for m, sky in zip(measured, skies, strict=True)]
                out[f"merr_model{suffix}"] = out[f"merr{suffix}"]
                out[f"ferr{suffix}"] = ferr
                out[f"merr{suffix}"] = [
                    1.0857 * e / m.flux[k] if np.isfinite(e) and m.flux[k] > 0 else np.nan
                    for m, e in zip(measured, ferr, strict=True)]
            if quality is not None and np.any(quality):
                out[f"pdq{suffix}"] = [int(m.dq[k]) for m in measured]

        destination = params.output or output.named_for(params.image, ".mag")
        if params.format == "iraf":
            _write_iraf(destination, params, dpars, observed, centres, skies, measured,
                        x0, y0)
        else:
            catalog.write(out, destination,
                          harwood_image=str(params.image),
                          harwood_itime=dpars.itime,
                          harwood_zmag=params.zmag,
                          harwood_apertures=[float(r) for r in radii],
                          harwood_merr="the image's VAR" if variance is not None
                          else "IRAF's model: epadu and the sky",
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
                "filter, airmass, time": f"{observed['ifilter']}, "
                                         f"{observed['xairmass']:.3f}, {observed['otime']}",
                "magnitudes": f"{magnitudes[good].min():.2f} to {magnitudes[good].max():.2f}",
                "best error": f"{np.nanmin(errors[good]) * 1000:.0f} mmag",
                "median error": f"{np.nanmedian(errors[good]) * 1000:.0f} mmag"}

        warnings = (_flags(out["cier"], "cier", _CENTER_FIX)
                    + _flags(out["sier"], "sier", _SKY_FIX)
                    + _flags(out[f"pier{first}"], "pier", _APERT_FIX)
                    + [advice.warn(n, "name the keyword the headers use, or give the "
                                      "value (xairmass, ifilter, otime).")
                       for n in observed_notes])
        if f"pdq{first}" in out.colnames:
            touched = np.asarray(out[f"pdq{first}"]) != 0
            if touched.any():
                from ..core import dq

                example = int(np.asarray(out[f"pdq{first}"])[touched][0])
                warnings.append(advice.warn(
                    f"{int(touched.sum())} of {len(out)} stars have pixels the DQ "
                    f"flags inside the aperture -- for one, {dq.describe(example)}. "
                    "Their magnitudes are measured on values that are not all "
                    "measurements.",
                    "look at the pdq column; a saturated star needs a shorter "
                    "exposure, and a cosmic ray can be cleaned (cosmicrays) or "
                    "left out by combining frames (imcombine)."))
        if variance is None:
            by_action["measure"].measured["errors"] = (
                "IRAF's model (epadu and the sky): the image has no VAR")
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


def _write_iraf(destination, params, dpars, observed, centres, skies, measured, x0, y0):
    """IRAF's .mag text file: what IRAF's phot writes for the same stars. Its
    errors are IRAF's model (epadu and the sky), whatever the image carries,
    and everything is rounded where IRAF's formats round it."""
    from ..core import apphot_text

    p = params.model_dump()
    p.update(epadu=dpars.epadu, readnoise=dpars.readnoise,
             apertures=",".join(f"{r:g}" for r in params.apertures))
    stars = []
    for k, (c, sky, m) in enumerate(zip(centres, skies, measured, strict=True), start=1):
        stars.append(apphot_text.Star(
            xinit=float(x0[k - 1]), yinit=float(y0[k - 1]), id=k, lid=k,
            x=c.x, y=c.y, xshift=c.xshift, yshift=c.yshift, xerr=c.xerr, yerr=c.yerr,
            cier=c.ier, msky=sky.value, stdev=sky.sigma, sskew=sky.skew, nsky=sky.nsky,
            nsrej=sky.nreject, sier=sky.ier, itime=dpars.itime,
            xairmass=observed["xairmass"], ifilter=observed["ifilter"],
            otime=observed["otime"], radii=list(m.radii), sums=list(m.sums),
            areas=list(m.areas), flux=list(m.flux), mag=list(m.mag), merr=list(m.merr),
            pier=m.ier, nmin=m.nmin, nmax=m.nmax))
    apphot_text.write(destination, "phot", p, Path(params.image).name,
                      Path(params.coords).name if params.coords else "", stars)


class QphotParams(TaskParams):
    image: str = Field(..., description="Image to measure.",
                       json_schema_extra={"positional": True})
    cbox: float = Field(..., gt=0, description="Width of the centring box, in pixels.",
                        json_schema_extra={"positional": True})
    annulus: float = Field(..., gt=0, description="Inner radius of the sky annulus, in "
                                                  "pixels.",
                           json_schema_extra={"positional": True})
    dannulus: float = Field(..., gt=0, description="Width of the sky annulus, in pixels.",
                            json_schema_extra={"positional": True})
    apertures: list[float] = Field(..., description="Aperture radii, in pixels.",
                                   json_schema_extra={"positional": True})
    coords: Path | None = Field(None, description="Coordinate catalogue, 1-based (x,y), "
                                                  "from daofind. Optional with the cursor.")
    output: Path | None = Field(
        None, description="Photometry table to write. Left out, it goes to this run's "
                          "own directory under output/, as obj.mag.")
    zmag: float = Field(25.0, description="Zero point of the instrumental magnitude scale.")
    exposure: str = Field("", description="Header keyword with the exposure time.")
    airmass: str = Field("", description="Header keyword with the airmass.")
    filter: str = Field("", description="Header keyword with the filter.")
    obstime: str = Field("", description="Header keyword with the time of observation.")
    epadu: float = Field(1.0, gt=0, description="Gain in electrons per count.")
    interactive: bool = Field(False, description="Measure with the cursor in the window, "
                                                 "with phot's keys.")
    icommands: str = Field("", description="Those keys from a file, 'x y 1 key' per line.")
    format: Literal["harwood", "iraf"] = Field(
        "harwood", description="harwood writes an ECSV table at full precision; iraf "
                               "writes IRAF's own .mag text file, rounded as IRAF rounds it.")


@register
class Qphot(Phot):
    """IRAF's qphot: phot with what is to be decided already decided. Centroid
    centring in the box given, the sky from the centroid of a smoothed histogram
    of the annulus, a constant weight and poisson noise; everything else is at
    phot's defaults, the same arithmetic and the same table."""

    name = "qphot"
    description = "Quick aperture photometry: phot with its choices made."
    Params = QphotParams

    @staticmethod
    def _phot(params) -> PhotParams:
        if isinstance(params, PhotParams):      # phot's own steps, on their way through
            return params
        return PhotParams(calgorithm="centroid", salgorithm="centroid", smooth=True,
                          **params.model_dump())

    def plan(self, params) -> list[Step]:
        return super().plan(self._phot(params))

    def execute(self, params: QphotParams) -> TaskResult:
        return super().execute(self._phot(params))


class RadprofParams(_DataPars, _CenterPars, _SkyPars):
    image: str = Field(..., description="Image to measure.",
                       json_schema_extra={"positional": True})
    radius: float = Field(..., gt=0, description="How far out to fit the profile, in "
                                                 "scale units.",
                          json_schema_extra={"positional": True})
    step: float = Field(..., gt=0, description="The step of the profile written, in scale "
                                               "units.",
                        json_schema_extra={"positional": True})
    coords: Path | None = Field(None, description="Coordinate catalogue, 1-based (x,y), "
                                                  "from daofind.")
    output: Path | None = Field(
        None, description="The table to write. Left out, it goes to this run's own "
                          "directory under output/, as obj.prf.")
    apertures: list[float] = Field([3.0], description="Aperture radii for the magnitudes, "
                                                      "in scale units.")
    zmag: float = Field(25.0, description="Zero point of the instrumental magnitude scale.")
    order: int = Field(5, ge=1, description="Pieces of the spline fitted to the profile.")
    kreject: float = Field(3.0, ge=0, description="Reject pixels beyond this many times "
                                                  "the fit's rms; 0 rejects nothing.")
    nreject: int = Field(1, ge=0, description="Most rejection passes.")
    format: Literal["harwood", "iraf"] = Field(
        "harwood", description="harwood writes an ECSV table at full precision; iraf writes "
                               "IRAF's own .prf text file, rounded as IRAF rounds it.")
    interactive: bool = Field(False, description="Measure with the cursor in the window, with "
                                                 "phot's keys and each profile in the plot.")
    icommands: str = Field("", description="Those keys from a file, 'x y 1 key' per line.")


RPWHY = {
    "profile": "Every pixel within the radius, less the sky, is a point of intensity "
               "against distance from the centre, so that a star is measured by all its "
               "pixels and not along a cut. A spline through them is the profile; pixels "
               "far from it, a neighbour or a cosmic ray, are taken out and it is fitted "
               "again. Too many pieces follow the noise, too few miss the core.",
    "fwhm": "The FWHM is twice the radius at which the fitted profile falls to half of "
            "its value at the centre, in a straight line between the two steps on either "
            "side: the step limits it, so keep it well under the FWHM.",
}


@register
class Radprof(Task[RadprofParams]):
    """IRAF's radprof: each star's centre, sky and magnitudes as in phot, and its
    radial profile, the integral of it, and the FWHM."""

    name = "radprof"
    package = "apphot"
    description = "The radial profile and FWHM of each star."
    Params = RadprofParams

    def plan(self, params: RadprofParams) -> list[Step]:
        return [Step("read", f"{params.image} and {params.coords}", why=WHY["read"]),
                Step("center", params.calgorithm, why=APWHY["center"]),
                Step("sky", f"{params.salgorithm} over the annulus {params.annulus:g}.."
                            f"{params.annulus + params.dannulus:g}" + _sky_extras(params),
                     why=APWHY["sky"]),
                Step("profile", f"a spline of {params.order} pieces out to "
                                f"{params.radius:g}, rejecting at {params.kreject:g} sigma",
                     why=RPWHY["profile"]),
                Step("fwhm", f"from the profile, in steps of {params.step:g}",
                     why=RPWHY["fwhm"]),
                Step("write", str(params.output or "obj.prf in this run's directory"),
                     why=WHY["write"])]

    def execute(self, params: RadprofParams) -> TaskResult:
        import numpy as np

        from ..core import apphot, catalog, images, photsession, radprof

        data, header = images.read(params.image)
        if params.coords is None and not (params.interactive or params.icommands):
            advice.fail("no coordinates to measure", "give the list daofind wrote, with "
                                                     "--coords.")
        stars = []
        catalogue = None
        if params.coords is not None:
            catalogue = catalog.read(params.coords)
            x0, y0 = catalog.coordinates(catalogue, params.coords)
            stars = [(float(a), float(b)) for a, b in zip(x0, y0, strict=True)]
        dpars = _datapars(params, header)
        cpars, spars = _centerpars(params), _skypars(params)
        radii = sorted(params.apertures)
        cursor, skyfile = _sky_sources(params)
        fitter = radprof.Fitter(params.radius, params.step, params.order, params.kreject,
                                params.nreject)
        messages = []
        if params.interactive or params.icommands:
            session = photsession.Session(data, dpars, cpars, spars, radii, params.zmag, stars,
                                          "radprof", cursor, skyfile, fitter=fitter)
            if params.icommands:
                if not Path(params.icommands).exists():
                    advice.fail(f"cursor file '{params.icommands}' does not exist",
                                "write 'x y 1 key' lines, or use --interactive for the window",
                                FileNotFoundError)
                for x, y, key in photsession.read_commands(params.icommands):
                    session.colon(key) if key.startswith(":") else session.key(key, x, y)
                    if session.done:
                        break
            else:
                _in_window(session, data, header, params)
            messages = session.messages
            rows = [(r.center, r.sky, r.profile) for r in session.rows]
            settings = [(r.settings, r.cards) for r in session.rows]
            x0 = np.array([r.xinit for r in session.rows])
            y0 = np.array([r.yinit for r in session.rows])
            ids = [r.id for r in session.rows]
            lids = [r.lid for r in session.rows]
        else:
            rows = []
            for a, b in stars:
                c = apphot.center(data, a, b, cpars, dpars)
                s = apphot.fitsky(data, c.x, c.y, spars, dpars, cursor, skyfile)
                rows.append((c, s, fitter.fit(data, c.x, c.y, s, radii, params.zmag, dpars)))
            x0, y0 = np.asarray(x0), np.asarray(y0)
            ids = list(_ids(catalogue, len(stars)))
            lids = list(range(1, len(stars) + 1))
            same = {"radius": params.radius, "step": params.step, "order": params.order,
                    "kreject": params.kreject, "nreject": params.nreject}
            settings = [(same, []) for _ in rows]
        return self._written(params, dpars, header, fitter, rows, x0, y0, ids, lids, radii,
                             messages, settings)

    def _written(self, params, dpars, header, fitter, rows, x0, y0, ids, lids, radii,
                 messages, settings) -> TaskResult:
        import numpy as np
        from astropy.table import Table

        from ..core import apphot, catalog, output, provenance

        npts = fitter.npts
        blank = np.full(npts, np.nan)
        out = Table({"id": ids, "lid": lids, "xinit": x0, "yinit": y0,
                     "x": [c.x for c, _, _ in rows], "y": [c.y for c, _, _ in rows],
                     "cier": [c.ier for c, _, _ in rows],
                     "msky": [s.value for _, s, _ in rows],
                     "stdev": [s.sigma for _, s, _ in rows],
                     "sier": [s.ier for _, s, _ in rows]})
        none = np.full(len(radii), np.nan)
        for k, radius in enumerate(radii):
            suffix = "" if len(radii) == 1 else f"_{radius:g}"
            found = [p.measured for _, _, p in rows]
            out[f"mag{suffix}"] = [m.mag[k] if m else none[k] for m in found]
            out[f"merr{suffix}"] = [m.merr[k] if m else none[k] for m in found]
            out[f"pier{suffix}"] = [m.ier if m else apphot.APERT_NOAPERT for m in found]
        out["pfwhm"] = [p.fwhm for _, _, p in rows]
        out["inorm"] = [p.inorm for _, _, p in rows]
        out["tinorm"] = [p.tinorm for _, _, p in rows]
        out["rier"] = [p.ier for _, _, p in rows]
        out["intensity"] = [p.intensity if p.intensity.size else blank for _, _, p in rows]
        out["tintensity"] = [p.tintensity if p.tintensity.size else blank for _, _, p in rows]
        destination = params.output or output.named_for(params.image, ".prf")
        if params.format == "iraf":
            _write_rprof(destination, params, dpars, rows, x0, y0, lids, settings)
        else:
            catalog.write(out, destination, harwood_image=str(params.image),
                          harwood_radius=(np.arange(npts) * params.step).tolist(),
                          harwood_apertures=[float(r) for r in radii],
                          harwood_call=provenance.current_stamp() or "")
        outputs = [destination]

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        fwhm = np.asarray(out["pfwhm"], dtype=float)
        good = np.isfinite(fwhm)
        by_action["profile"].measured = {
            "stars": len(out),
            "pixels used": f"{np.median([p.ndata for _, _, p in rows]):.0f}",
            "rejected": f"{np.median([p.nreject for _, _, p in rows]):.0f}"}
        if good.any():
            by_action["fwhm"].measured = {
                "median": f"{np.median(fwhm[good]):.3f}",
                "range": f"{fwhm[good].min():.3f} to {fwhm[good].max():.3f}"}
        first = "" if len(radii) == 1 else f"_{radii[0]:g}"
        warnings = (_flags(out["cier"], "cier", _CENTER_FIX)
                    + _flags(out["sier"], "sier", _SKY_FIX)
                    + _flags(out[f"pier{first}"], "pier", _APERT_FIX)
                    + _flags(out["rier"], "rier", "a star whose profile failed needs a "
                                                  "smaller --order, or is off the image; "
                                                  "IRAF's code also stays set for the stars "
                                                  "after it."))
        if max(radii) > params.radius:
            warnings.append(advice.warn(
                f"an aperture ({max(radii):g}) is larger than the radius ({params.radius:g}): "
                "its sum stops at the profile's box, as in IRAF, and is short of light.",
                "raise radius, or keep the apertures within it."))
        data_out = {"stars": len(out),
                    "fwhm": float(np.median(fwhm[good])) if good.any() else None}
        if messages:
            data_out["text"] = messages
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings, data=data_out)


def _write_rprof(destination, params, dpars, rows, x0, y0, lids, settings):
    """IRAF's .prf."""
    import numpy as np

    from ..core import apphot, apphot_text

    p = params.model_dump()
    p.update(epadu=dpars.epadu, readnoise=dpars.readnoise,
             apertures=",".join(f"{r:g}" for r in params.apertures))
    for key in ("exposure", "airmass", "filter", "obstime", "gain", "ccdread"):
        p.setdefault(key, "")
    radii = sorted(params.apertures)
    stars = []
    for k, (c, sky, profile) in enumerate(rows, start=1):
        m = profile.measured
        if m is None:
            pier = (apphot.APERT_NOSKYMODE if not np.isfinite(sky.value)
                    else apphot.APERT_NOAPERT)
        else:
            pier = m.ier
        nothing = [0.0] * len(radii)
        star = apphot_text.Star(
            xinit=float(x0[k - 1]), yinit=float(y0[k - 1]), id=k, lid=lids[k - 1],
            x=c.x, y=c.y, xshift=c.xshift, yshift=c.yshift, xerr=c.xerr, yerr=c.yerr,
            cier=c.ier,
            msky=sky.value, stdev=sky.sigma, sskew=sky.skew, nsky=sky.nsky,
            nsrej=sky.nreject, sier=sky.ier, itime=dpars.itime, xairmass=float("nan"),
            ifilter="INDEF", otime="INDEF",
            radii=list(m.radii) if m else [float(r) for r in radii],
            sums=list(m.sums) if m else nothing, areas=list(m.areas) if m else nothing,
            flux=list(m.flux) if m else nothing,
            mag=list(m.mag) if m else [float("nan")] * len(radii),
            merr=list(m.merr) if m else [float("nan")] * len(radii),
            pier=pier, nmin=m.nmin if m else 0, nmax=m.nmax if m else 0)
        done = profile.intensity.size > 0
        now, cards = settings[k - 1]
        npts = int(np.float32(np.float32(now["radius"]) / np.float32(now["step"]))) + 1
        grid = (np.arange(npts, dtype=np.float32) * np.float32(now["step"])).astype(np.float32)
        indef = np.full(npts, np.nan)
        stars.append((star, {"fwhm": profile.fwhm, "inorm": profile.inorm,
                             "tinorm": profile.tinorm, "ier": profile.ier,
                             "radius": profile.radius if done else grid,
                             "intensity": profile.intensity if done else indef,
                             "tintensity": profile.tintensity if done else indef},
                      now, cards))
    first = settings[0][0] if settings else {"radius": params.radius, "step": params.step,
                                             "order": params.order, "kreject": params.kreject,
                                             "nreject": params.nreject}
    apphot_text.write_rprof(destination, p, Path(params.image).name,
                            Path(params.coords).name if params.coords else "", stars, first)


def _ids(table, n):
    """The stars' ids from the coordinates -- IRAF's .coo calls the column ID."""
    name = next((c for c in table.colnames if c.lower() == "id"), None)
    return table[name] if name else list(range(1, n + 1))
