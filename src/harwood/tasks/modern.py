"""What IRAF has no task for, and is done today: the sky as a surface (Bertin & Arnouts 1996),
photometry with an effective PSF (Anderson & King 2000), image subtraction by ZOGY (Zackay,
Ofek & Gal-Yam 2016), proper coaddition (Zackay & Ofek 2017), drizzle (Fruchter & Hook 2002),
a plate from a catalogue with the pairs found. With no IRAF to compare with, each is held to a
known answer and, where there is one, to the implementation others use. See core/skymesh.py,
epsf.py, zogy.py, coadd.py, drizzle.py and platesolve.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "mesh": "Each box of the image gives a level from its clipped pixels, so that the stars "
            "leave it alone; a surface through the boxes gives every pixel a sky, where an "
            "annulus around a star only says what the sky is there.",
    "psf": "An effective PSF is the image of a star as the pixels saw it, built from the "
           "stars of the frame at their different phases, with no function assumed.",
    "fit": "Each star's flux and position against that PSF, its neighbours taken out by the "
           "fits so far, and all of them gone round again: stars an aperture would add "
           "together are told apart.",
    "subtract": "Neither image is blurred to match the other. The difference is worked out "
                "from both PSFs and both noises, so that it keeps what the sharper image had, "
                "and its noise is white and flat, so that one threshold means the same "
                "everywhere.",
    "search": "Nothing says which way up a frame is. Every pair of stars votes for an offset, "
              "at each rotation, mirror and scale: the right one makes a peak nothing else "
              "makes.",
    "plate": "The pairs give a plane through the sky, and they are found again until they "
             "stop changing, what stands off being left out. A distortion is what is left.",
    "judge": "A plate fits anything a little, and a list at random has pairs too. It has to "
             "beat what chance gives, or no WCS is written.",
    "tilt": "A sky line leans across the slit, by as much as this: it is what sets where a "
            "pixel is along the sky's spectrum, and what the spline has to work with.",
    "spline": "The sky is one spectrum, and each pixel of the slit samples it at its own "
              "place. A spline in that coordinate, evaluated back at each pixel, leaves the "
              "noise of the pixel, where a fit by lines leaves what it cannot see of a tilted "
              "line.",
    "flexure": "A spectrograph bends as the telescope moves, and the arc from the start of "
               "the night is not where the sky is now. The sky's own lines are in the frame, "
               "and say by how much, against a frame whose wavelengths are known.",
    "drizzle": "Each pixel, shrunk to a drop, lands on a finer grid where the dither put it, "
               "and adds to what is under it by the area they share. Nothing is interpolated, "
               "the flux is kept, and what the camera's pixels lost comes back.",
    "coadd": "A mean of the pixels counts a frame as a whole. Here each counts at each "
             "spatial frequency for what it knows there, so that the sharp frames give the "
             "detail and the deep ones the depth, and nothing they say of a source is lost.",
}


def _write(path, data, header=None) -> Path:
    import numpy as np
    from astropy.io import fits

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.asarray(data, np.float32), header=header).writeto(path, overwrite=True)
    return path


def _stamped(header):
    from astropy.io import fits

    out = fits.Header()
    for card in header.cards:
        if card.keyword not in ("SIMPLE", "BITPIX", "EXTEND", "COMMENT", "HISTORY", "BZERO",
                                "BSCALE", "") and not card.keyword.startswith("NAXIS"):
            out.append(card)
    return out


class SkymodelParams(TaskParams):
    image: str = Field(..., description="The image.", json_schema_extra={"positional": True})
    output: Path | None = Field(None, description="The sky, an image. Left out, this run's "
                                                  "directory under output/, as obj.sky.fits.")
    subtracted: Path | None = Field(None, description="The image less its sky.")
    rms: Path | None = Field(None, description="The noise of the sky at each pixel, an image.")
    box: int = Field(64, ge=3, description="The side of a box of the mesh, in pixels: larger "
                                           "than a star and smaller than what the sky does.")
    filter_size: int = Field(3, ge=1, description="Boxes in the median filter of the mesh; 1 "
                                                  "is none.")
    exclude: float = Field(0.5, gt=0, le=1, description="A box with less than this of its "
                                                        "pixels left goes by its neighbours.")


@register
class Skymodel(Task[SkymodelParams]):
    """The sky of an image as a surface, from a mesh of clipped boxes, SExtractor's method."""

    name = "skymodel"
    package = "harwood"
    description = "The sky of an image as a surface, from a mesh of boxes."
    Params = SkymodelParams

    def plan(self, params: SkymodelParams) -> list[Step]:
        return [Step("mesh", f"boxes of {params.box} px, filter {params.filter_size}",
                     why=WHY["mesh"]),
                Step("write", str(params.output or "obj.sky.fits in this run's directory"))]

    def execute(self, params: SkymodelParams) -> TaskResult:
        from ..core import mef, output, skymesh

        data, header, _, quality = mef.load(params.image)
        model = skymesh.model(data, params.box, params.filter_size, mask=quality != 0,
                              exclude=params.exclude)
        keep = _stamped(header)
        destination = params.output or output.named_for(params.image, ".sky.fits")
        outputs = [_write(destination, model.sky, keep)]
        if params.subtracted:
            outputs.append(_write(params.subtracted, data - model.sky, keep))
        if params.rms:
            outputs.append(_write(params.rms, model.rms, keep))
        steps = self.plan(params)
        import numpy as np

        steps[0].measured = {"boxes": f"{model.grid.shape[1]} x {model.grid.shape[0]}",
                             "from their neighbours": model.replaced,
                             "sky": f"{float(model.sky.min()):.2f} to "
                                    f"{float(model.sky.max()):.2f}",
                             "noise": f"{float(np.median(model.rms)):.3f}"}
        return TaskResult(outputs=outputs, steps=steps,
                          data={"sky": float(np.median(model.sky)),
                                "rms": float(np.median(model.rms)),
                                "replaced": model.replaced})


class PsfphotParams(TaskParams):
    image: str = Field(..., description="The image.", json_schema_extra={"positional": True})
    coords: Path = Field(..., description="The stars, from daofind: 1-based (x, y).",
                         json_schema_extra={"positional": True})
    output: Path | None = Field(None, description="The table. Left out, this run's "
                                                  "directory under output/, as obj.psf.ecsv.")
    psf: Path | None = Field(None, description="An effective PSF to use, from an earlier run. "
                                               "Left out one is built from this image.")
    psfout: Path | None = Field(None, description="Where to keep the one that is built.")
    psfstars: Path | None = Field(None, description="The stars to build it from. Left out, "
                                                    "the brightest with room around them.")
    nstars: int = Field(50, ge=3, description="How many, when they are chosen here.")
    datamax: float | None = Field(None, description="Above this a pixel is saturated and its "
                                                    "star is not used for the PSF.")
    size: int = Field(21, ge=7, description="The side of the PSF, in pixels: its wings beyond "
                                            "go into the zero point.")
    oversampling: int = Field(2, ge=1, le=8, description="Grid points to a pixel. 2 for a few "
                                                         "dozen stars; 4 wants some hundreds.")
    iterations: int = Field(6, ge=1, description="Passes to build it. It stops by itself when "
                                                 "it has settled.")
    radius: float = Field(4.0, gt=0, description="The pixels fitted, around each star.")
    passes: int = Field(3, ge=1, description="Times round the list, with the neighbours out.")
    box: int = Field(64, ge=3, description="The mesh of the sky, as skymodel's.")
    sky: float | None = Field(None, description="The sky, one number. Left out, the mesh.")
    zmag: float = Field(25.0, description="The zero point of the magnitudes.")
    residual: Path | None = Field(None, description="The image with every star taken out.")


def _bright(data, sky, stars, params):
    """The stars for the PSF, when they are not named: those with the most light in the fit's
    radius, none of their pixels saturated, and nothing brighter than a tenth of them near."""
    import numpy as np

    half = params.size // 2
    ny, nx = data.shape
    measured = []
    for x, y in stars:
        ix, iy = int(round(x)) - 1, int(round(y)) - 1
        if ix - half < 0 or iy - half < 0 or ix + half >= nx or iy + half >= ny:
            continue
        box = data[iy - half:iy + half + 1, ix - half:ix + half + 1]
        if not np.isfinite(box).all():
            continue
        if params.datamax is not None and box.max() >= params.datamax:
            continue
        jj, ii = np.mgrid[-half:half + 1, -half:half + 1]
        level = sky if np.ndim(sky) == 0 else sky[iy, ix]
        flux = float((box - level)[ii * ii + jj * jj <= params.radius ** 2].sum())
        measured.append((flux, x, y))
    measured.sort(reverse=True)
    chosen = []
    for flux, x, y in measured:
        crowded = any(0 < (x - a) ** 2 + (y - b) ** 2 < (half * 0.75) ** 2 and f > 0.1 * flux
                      for f, a, b in measured)
        if flux > 0 and not crowded:
            chosen.append((x, y))
        if len(chosen) == params.nstars:
            break
    return chosen


def write_psf(path, psf) -> Path:
    from astropy.io import fits

    header = fits.Header()
    header["OVERSAMP"] = (psf.oversampling, "grid points to a pixel")
    header["PSFSIZE"] = (psf.size, "its side, in pixels")
    header["NSTARS"] = (psf.nstars, "stars it was built from")
    return _write(path, psf.grid, header)


def read_psf(path):
    import numpy as np
    from astropy.io import fits

    from ..core import epsf

    if not Path(path).exists():
        advice.fail(f"no PSF '{path}'", "psfphot --psfout writes one", FileNotFoundError)
    data, header = fits.getdata(path, header=True, memmap=False)
    if "OVERSAMP" not in header:
        advice.fail(f"{path} is not an effective PSF", "psfphot --psfout writes one")
    return epsf.EPSF(np.asarray(data, np.float64), int(header["OVERSAMP"]),
                     int(header["PSFSIZE"]), int(header.get("NSTARS", 0)))


@register
class Psfphot(Task[PsfphotParams]):
    """Photometry with an effective PSF, Anderson & King's method, built from the frame's own
    stars and fitted with the neighbours out."""

    name = "psfphot"
    package = "harwood"
    description = "Photometry with an effective PSF built from the frame's stars."
    Params = PsfphotParams

    def plan(self, params: PsfphotParams) -> list[Step]:
        return [Step("psf", str(params.psf) if params.psf else
                     f"from {params.psfstars or f'the {params.nstars} best stars'}, "
                     f"{params.size} px, oversampled {params.oversampling}", why=WHY["psf"]),
                Step("fit", f"within {params.radius:g} px, {params.passes} passes",
                     why=WHY["fit"]),
                Step("write", str(params.output or "obj.psf.ecsv in this run's directory"))]

    def execute(self, params: PsfphotParams) -> TaskResult:
        import math

        import numpy as np
        from astropy.table import Table

        from ..core import catalog, epsf, mef, output, provenance, skymesh

        data, header, variance, quality = mef.load(params.image)
        table = catalog.read(params.coords)
        xs, ys = catalog.coordinates(table, params.coords)
        stars = [(float(a), float(b)) for a, b in zip(xs, ys, strict=True)]
        if not stars:
            advice.fail(f"{params.coords} has no stars", "daofind makes the list")
        work = np.where(quality != 0, np.nan, data)
        sky = (float(params.sky) if params.sky is not None
               else skymesh.model(work, params.box, mask=quality != 0).sky)
        outputs = []
        if params.psf:
            psf = read_psf(params.psf)
        else:
            if params.psfstars:
                named = catalog.read(params.psfstars)
                a, b = catalog.coordinates(named, params.psfstars)
                chosen = [(float(u), float(v)) for u, v in zip(a, b, strict=True)]
            else:
                chosen = _bright(data, sky, stars, params)
            psf = epsf.build(work, chosen, sky=sky, size=params.size,
                             oversampling=params.oversampling, iterations=params.iterations)
            if params.psfout:
                outputs.append(write_psf(params.psfout, psf))
        fits_done, residual = epsf.photometry(work, psf, [(x, y, None) for x, y in stars], sky,
                                              variance, params.radius, params.passes)
        flux = np.array([f.flux for f in fits_done])
        ferr = np.array([f.ferr for f in fits_done])
        with np.errstate(divide="ignore", invalid="ignore"):
            positive = np.where(flux > 0, flux, 1.0)
            mag = np.where(flux > 0, params.zmag - 2.5 * np.log10(positive), np.nan)
            merr = np.where(flux > 0, 2.5 / math.log(10.0) * ferr / positive, np.nan)
        at = [(int(round(f.y)) - 1, int(round(f.x)) - 1) for f in fits_done]
        ny, nx = data.shape
        msky = [float(sky) if np.ndim(sky) == 0 else
                (float(sky[j, i]) if 0 <= j < ny and 0 <= i < nx else float("nan"))
                for j, i in at]
        out = Table({"id": list(range(1, len(stars) + 1)),
                     "xinit": [s[0] for s in stars], "yinit": [s[1] for s in stars],
                     "x": [f.x for f in fits_done], "y": [f.y for f in fits_done],
                     "xerr": [f.xerr for f in fits_done], "yerr": [f.yerr for f in fits_done],
                     "flux": flux, "ferr": ferr, "mag": mag, "merr": merr, "msky": msky,
                     "chi": [f.chi for f in fits_done], "ier": [f.ier for f in fits_done]})
        destination = params.output or output.named_for(params.image, ".psf.ecsv")
        catalog.write(out, destination, harwood_image=str(params.image),
                      harwood_zmag=params.zmag, harwood_psf_size=psf.size,
                      harwood_merr="the image's VAR" if variance is not None
                      else "the scatter about the fit",
                      harwood_call=provenance.current_stamp() or "")
        outputs.insert(0, destination)
        if params.residual:
            outputs.append(_write(params.residual, residual, _stamped(header)))
        steps = self.plan(params)
        steps[0].measured = {"stars": psf.nstars}
        good = np.array([f.ier == 0 for f in fits_done]) & np.isfinite(mag)
        steps[1].measured = {"fitted": f"{int(good.sum())} of {len(stars)}",
                             "median chi": f"{float(np.median(out['chi'][good])):.3g}"
                             if good.any() else "none"}
        warnings = []
        if not good.all():
            warnings.append(advice.warn(
                f"{int((~good).sum())} of {len(stars)} stars have no fit",
                "ier 1 is off the image, 2 too few pixels; a negative flux has no magnitude."))
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings,
                          data={"stars": len(stars), "fitted": int(good.sum()),
                                "psf_stars": psf.nstars})


class ZogyParams(TaskParams):
    new: str = Field(..., description="The new image.", json_schema_extra={"positional": True})
    reference: str = Field(..., description="The reference, on the same pixels: imalign "
                                            "first.",
                           json_schema_extra={"positional": True})
    output: Path = Field(..., description="The difference, whose noise is white and of unit "
                                          "variance.", json_schema_extra={"positional": True})
    scorr: Path | None = Field(None, description="The score in sigma at each pixel, with "
                                                 "the sources' noise in it: to threshold.")
    flux: Path | None = Field(None, description="The flux of a point source at each pixel, in "
                                                "the units where the zero points are 1.")
    psf_new: Path | None = Field(None, description="The new image's effective PSF, from "
                                                   "psfphot --psfout.")
    psf_reference: Path | None = Field(None, description="The reference's, the same way.")
    fwhm_new: float | None = Field(None, gt=0, description="Or the new image's FWHM, a "
                                                           "Gaussian.")
    fwhm_reference: float | None = Field(None, gt=0, description="Or the reference's.")
    zero_new: float = Field(1.0, gt=0, description="The new image's flux for a source of "
                                                   "unit flux: its zero point, as a factor.")
    zero_reference: float = Field(1.0, gt=0, description="The reference's. Only their ratio "
                                                         "matters to what is subtracted.")
    coords: Path | None = Field(None, description="Stars seen in both, to find that ratio "
                                                  "from their light. Overrides --zero-new.")
    gain: float | None = Field(None, gt=0, description="Electrons per ADU, to add the "
                                                       "sources' noise to the score. An "
                                                       "image's VAR says it already.")
    box: int = Field(64, ge=3, description="The mesh of the sky of each, as skymodel's.")
    astrometric_x: float = Field(0.0, ge=0, description="The rms of the alignment in x, in "
                                                        "pixels: it keeps bright stars out "
                                                        "of the score.")
    astrometric_y: float = Field(0.0, ge=0, description="The same in y.")
    threshold: float = Field(5.0, gt=0, description="The score of a candidate, in sigma.")
    margin: int | None = Field(None, ge=0, description="No candidates this close to the edge, "
                                                       "in pixels. Left out, a PSF's width.")
    align: bool = Field(True, description="With --coords, measure what is left of the "
                                          "alignment on those stars and take it out, by a "
                                          "fraction of a pixel in Fourier space.")
    psf_model: Literal["moffat", "epsf"] = Field(
        "moffat", description="What to use of an effective PSF: moffat fits one to it, "
                              "tapered; epsf uses it as it is, with its noise and its sharp "
                              "edge.")
    datamax: float | None = Field(None, description="Above this a pixel is saturated, and no "
                                                    "candidate is taken near it.")
    candidates: Path | None = Field(None, description="The candidates, a table: position, "
                                                      "score and flux.")


def _psf_for(params, which: str, fwhm, path):
    from ..core import zogy

    if path:
        from ..core import epsf

        sampled = read_psf(path).sampled()
        return epsf.smooth(sampled) if params.psf_model == "moffat" else sampled
    if fwhm:
        return zogy.gaussian(fwhm)
    advice.fail(f"the PSF of the {which} image",
                f"give --psf-{which} from psfphot --psfout, or --fwhm-{which}")


def _ratio(new, reference, path) -> tuple[float, int]:
    """The median of the ratio of what stars have within a few pixels, the sky already off."""
    import numpy as np

    from ..core import catalog

    table = catalog.read(path)
    xs, ys = catalog.coordinates(table, path)
    ny, nx = new.shape
    half = 8
    jj, ii = np.mgrid[-half:half + 1, -half:half + 1]
    round_ = ii * ii + jj * jj <= half * half
    ratios = []
    for x, y in zip(xs, ys, strict=True):
        ix, iy = int(round(float(x))) - 1, int(round(float(y))) - 1
        if ix - half < 0 or iy - half < 0 or ix + half >= nx or iy + half >= ny:
            continue
        a = new[iy - half:iy + half + 1, ix - half:ix + half + 1][round_].sum()
        b = reference[iy - half:iy + half + 1, ix - half:ix + half + 1][round_].sum()
        if b > 0 and a > 0:
            ratios.append(a / b)
    if len(ratios) < 3:
        advice.fail(f"only {len(ratios)} stars of {path} are measured in both",
                    "give more, or the ratio with --zero-new")
    return float(np.median(ratios)), len(ratios)


def _variance(data, sky, gain):
    """The noise of the image: what the sky has, measured, and the sources' above it."""
    import numpy as np

    return sky.rms ** 2 + np.maximum(data - sky.sky, 0.0) / gain


def _peaks(scorr, threshold: float, margin: int, bad):
    """Pixels above the threshold that are the highest of their eight neighbours. Not within
    a PSF of the edge, where the transforms wrap round, and not where `bad` says, nor next to
    it: saturated stars do not subtract."""
    import numpy as np
    from scipy.ndimage import binary_dilation, maximum_filter

    top = maximum_filter(scorr, size=3)
    allowed = np.zeros(scorr.shape, bool)
    allowed[margin:-margin or None, margin:-margin or None] = True
    if bad is not None and bad.any():
        allowed &= ~binary_dilation(bad, iterations=max(margin // 2, 2))
    found = np.argwhere((scorr == top) & (scorr >= threshold) & allowed)
    return found[np.argsort(-scorr[tuple(found.T)])]


@register
class Zogy(Task[ZogyParams]):
    """Image subtraction by ZOGY (Zackay, Ofek & Gal-Yam 2016): neither image blurred, a
    difference with white noise, and a score that says in sigma what has changed."""

    name = "zogy"
    package = "harwood"
    description = "Subtract a reference from a new image and score what has changed."
    Params = ZogyParams

    def plan(self, params: ZogyParams) -> list[Step]:
        return [Step("subtract", f"{params.new} - {params.reference}", why=WHY["subtract"]),
                Step("write", str(params.output))]

    def execute(self, params: ZogyParams) -> TaskResult:
        import numpy as np
        from astropy.table import Table

        from ..core import catalog, mef, skymesh, zogy

        new, header, var_new, dq_new = mef.load(params.new)
        reference, _, var_ref, dq_ref = mef.load(params.reference)
        if new.shape != reference.shape:
            advice.fail(f"{params.new} is {new.shape} and {params.reference} "
                        f"{reference.shape}",
                        "put them on the same pixels first, with imalign")
        sky_new = skymesh.model(new, params.box, mask=dq_new != 0)
        sky_ref = skymesh.model(reference, params.box, mask=dq_ref != 0)
        flat_new, flat_ref = new - sky_new.sky, reference - sky_ref.sky
        zero_new, used = params.zero_new, 0
        if params.coords:
            zero_new, used = _ratio(flat_new, flat_ref, params.coords)
            zero_new *= params.zero_reference
        limit = np.inf if params.datamax is None else params.datamax
        bad_new = (dq_new != 0) | (new >= limit)
        bad_ref = (dq_ref != 0) | (reference >= limit)
        offset, astrometric, measured_on = (0.0, 0.0), (params.astrometric_x,
                                                        params.astrometric_y), 0
        if params.coords and params.align:
            table = catalog.read(params.coords)
            xs, ys = catalog.coordinates(table, params.coords)
            found = zogy.centre_offsets(flat_new, flat_ref, zip(xs, ys, strict=True),
                                        saturated=bad_new | bad_ref)
            if found.shape[0] >= 3:
                # The brightest half says it best, as the others' centroids are noise.
                best = found[found[:, 2] >= np.median(found[:, 2])]
                offset = (float(np.median(best[:, 0])), float(np.median(best[:, 1])))
                measured_on = best.shape[0]
                if not any(astrometric):
                    spread = np.abs(best[:, :2] - np.array(offset))
                    astrometric = tuple(1.4826 * np.median(spread, axis=0).astype(float))
                flat_new = zogy.shifted(flat_new, -offset[0], -offset[1])
                if var_new is not None:
                    var_new = np.maximum(zogy.shifted(var_new, -offset[0], -offset[1]), 0.0)
        sigma_new = float(np.median(sky_new.rms))
        sigma_ref = float(np.median(sky_ref.rms))
        if params.gain:
            if var_new is None:
                var_new = _variance(new, sky_new, params.gain)
            if var_ref is None:
                var_ref = _variance(reference, sky_ref, params.gain)
        psf_new = _psf_for(params, "new", params.fwhm_new, params.psf_new)
        psf_ref = _psf_for(params, "reference", params.fwhm_reference, params.psf_reference)
        done = zogy.subtract(
            flat_new, flat_ref, psf_new, psf_ref,
            sigma_new, sigma_ref, zero_new, params.zero_reference, var_new, var_ref,
            astrometric)
        keep = _stamped(header)
        keep["ZOGYFD"] = (done.F_D, "the difference's zero point")
        outputs = [_write(params.output, done.D, keep)]
        for path, plane in ((params.scorr, done.Scorr), (params.flux, done.alpha)):
            if path:
                outputs.append(_write(path, plane, keep))
        margin = (max(psf_new.shape[0], psf_ref.shape[0]) if params.margin is None
                  else params.margin)
        peaks = _peaks(done.Scorr, params.threshold, margin, bad_new | bad_ref)
        rows = [{"x": float(i + 1), "y": float(j + 1), "score": float(done.Scorr[j, i]),
                 "flux": float(done.alpha[j, i]), "ferr": float(done.alpha_err[j, i])}
                for j, i in peaks]
        if params.candidates:
            table = Table(rows=rows, names=("x", "y", "score", "flux", "ferr")) if rows else (
                Table(names=("x", "y", "score", "flux", "ferr")))
            catalog.write(table, params.candidates, harwood_threshold=params.threshold)
            outputs.append(params.candidates)
        steps = self.plan(params)
        d = done.D
        scatter = float(1.4826 * np.median(np.abs(d - np.median(d))))
        steps[0].measured = {"noise of the difference": f"{scatter:.3f}, and it should be 1",
                             "noise of the images": f"{sigma_new:.3g} and {sigma_ref:.3g}",
                             "zero point ratio": f"{zero_new / params.zero_reference:.4g}"
                             + (f" from {used} stars" if used else ""),
                             "candidates": len(rows)}
        if measured_on:
            steps[0].measured["alignment left"] = (
                f"{offset[0]:+.3f}, {offset[1]:+.3f} px on {measured_on} stars, taken out")
        if any(astrometric):
            steps[0].measured["astrometric rms"] = (f"{astrometric[0]:.3f}, "
                                                    f"{astrometric[1]:.3f} px")
        warnings = []
        if var_new is None and var_ref is None:
            warnings.append(advice.warn(
                "the score has no noise of the sources in it, so what is not perfect in the "
                "PSFs shows on bright stars as candidates.",
                "give --gain, or images with a VAR, as ccdproc writes them."))
        if scatter > 1.5 or scatter < 0.7:
            warnings.append(advice.warn(
                f"the noise of the difference is {scatter:.2f}, and it should be 1",
                "a wrong PSF, a wrong zero point, or images not on the same pixels would do "
                "this; look at the difference around the bright stars."))
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings,
                          data={"rows": rows, "candidates": len(rows), "noise": scatter,
                                "zero_ratio": zero_new / params.zero_reference,
                                "offset": list(offset)})


class CoaddParams(TaskParams):
    input: list[str] = Field(..., description="The frames, on the same pixels: imalign first. "
                                              "Globs and @lists expand.",
                             json_schema_extra={"positional": True})
    output: Path = Field(..., description="The coadd, with white noise, as one exposure with "
                                          "the first frame's zero point.")
    coords: Path | None = Field(None, description="Stars seen in all the frames: what is left "
                                                  "of the alignment and the zero points come "
                                                  "from them, and the PSFs, if not given.")
    psf: str = Field("", description="The frames' effective PSFs from psfphot --psfout, with "
                                     "commas, one for each.")
    fwhm: str = Field("", description="Or their FWHM, Gaussians, with commas.")
    zero: str = Field("", description="Or their zero points as factors, with commas, the "
                                      "first being 1. Left out, from --coords, or all 1.")
    box: int = Field(64, ge=3, description="The mesh of the sky of each, as skymodel's.")
    align: bool = Field(True, description="With --coords, measure what is left of the "
                                          "alignment on those stars and take it out, by a "
                                          "fraction of a pixel in Fourier space.")
    psf_model: Literal["moffat", "epsf"] = Field(
        "moffat", description="What to use of an effective PSF: moffat fits one to it, "
                              "tapered; epsf uses it as it is, with its noise and its sharp "
                              "edge.")
    datamax: float | None = Field(None, description="Above this a pixel is saturated, and "
                                                    "takes its value from the others.")
    score: Path | None = Field(None, description="The match to a point source, in sigma of "
                                                 "the background at each pixel, an image: "
                                                 "to threshold.")
    flux: Path | None = Field(None, description="The flux of a point source at each pixel, "
                                                "an image, in the first frame's units.")
    psfout: Path | None = Field(None, description="The coadd's own PSF, for psfphot --psf.")


def _each(text: str, count: int, name: str, convert=str) -> list:
    words = [word.strip() for word in text.split(",") if word.strip()]
    if len(words) != count:
        advice.fail(f"{count} frames and {len(words)} of {name}",
                    f"give one of {name} for each frame, with commas")
    return [convert(word) for word in words]


@register
class Coaddition(Task[CoaddParams]):
    """Proper coaddition (Zackay & Ofek 2017): each frame counts at each spatial frequency for
    what it knows there, its sharpness and its depth, and the noise comes out white."""

    name = "coadd"
    package = "harwood"
    description = "Add frames of different seeing and depth, and lose nothing they say."
    Params = CoaddParams

    def plan(self, params: CoaddParams) -> list[Step]:
        return [Step("coadd", ", ".join(params.input), why=WHY["coadd"]),
                Step("write", str(params.output))]

    def execute(self, params: CoaddParams) -> TaskResult:
        import numpy as np

        from ..core import catalog, coadd, epsf, images, mef, skymesh, zogy

        names = images.expand(params.input)
        loaded = [mef.load(name) for name in names]
        shapes = {item[0].shape for item in loaded}
        if len(shapes) > 1:
            advice.fail(f"the frames are not all the same shape: {sorted(shapes)}",
                        "put them on the same pixels first, with imalign")
        limit = np.inf if params.datamax is None else params.datamax
        bad = [(item[3] != 0) | (item[0] >= limit) for item in loaded]
        skies = [skymesh.model(item[0], params.box, mask=flagged)
                 for item, flagged in zip(loaded, bad, strict=True)]
        flats = [item[0] - sky.sky for item, sky in zip(loaded, skies, strict=True)]
        stars = None
        if params.coords:
            table = catalog.read(params.coords)
            stars = list(zip(*catalog.coordinates(table, params.coords), strict=True))
        zeros, used = [1.0] * len(flats), 0
        if params.zero:
            zeros = _each(params.zero, len(flats), "--zero", float)
        elif params.coords:
            for at in range(1, len(flats)):
                zeros[at], used = _ratio(flats[at], flats[0], params.coords)
        shifts = [(0.0, 0.0)] * len(flats)
        if stars and params.align:
            for at in range(1, len(flats)):
                found = zogy.centre_offsets(flats[at], flats[0], stars,
                                            saturated=bad[at] | bad[0])
                if found.shape[0] >= 3:
                    best = found[found[:, 2] >= np.median(found[:, 2])]
                    shifts[at] = (float(np.median(best[:, 0])), float(np.median(best[:, 1])))
                    flats[at] = zogy.shifted(flats[at], -shifts[at][0], -shifts[at][1])
        if params.psf:
            paths = _each(params.psf, len(flats), "--psf")
            psfs = [read_psf(path).sampled() for path in paths]
            if params.psf_model == "moffat":
                psfs = [epsf.smooth(psf) for psf in psfs]
        elif params.fwhm:
            psfs = [zogy.gaussian(fwhm) for fwhm in _each(params.fwhm, len(flats), "--fwhm",
                                                          float)]
        elif stars:
            psfs = []
            for flat in flats:
                sampled = epsf.build(flat, stars, sky=0.0).sampled()
                psfs.append(epsf.smooth(sampled) if params.psf_model == "moffat" else sampled)
        else:
            advice.fail("the frames' PSFs", "give --coords, to build them from its stars, "
                                            "or --psf from psfphot --psfout, or --fwhm")
        sigmas = [float(np.median(sky.rms)) for sky in skies]
        done = coadd.coadd(flats, psfs, sigmas, zeros, bad)
        keep = _stamped(loaded[0][1])
        keep["COADDN"] = (len(flats), "frames in the proper coaddition")
        keep["COADDSIG"] = (done.sigma, "the noise it should have")
        outputs = [_write(params.output, done.image, keep)]
        for path, plane in ((params.score, done.score), (params.flux, done.flux)):
            if path:
                outputs.append(_write(path, plane, keep))
        if params.psfout:
            cy, cx = done.psf.shape[0] // 2, done.psf.shape[1] // 2
            half = min(max(psf.shape[0] for psf in psfs) // 2 + 2, cy - 1, cx - 1)
            cut = done.psf[cy - half:cy + half + 1, cx - half:cx + half + 1]
            outputs.append(write_psf(params.psfout, epsf.EPSF(cut / cut.sum(), 1, 2 * half + 1,
                                                              len(flats))))
        scatter = float(1.4826 * np.median(np.abs(done.image - np.median(done.image))))
        rows = [{"image": Path(name.path).name, "noise": round(sigma, 3),
                 "zero point": round(zero, 4), "dx": round(shift[0], 3),
                 "dy": round(shift[1], 3)}
                for name, sigma, zero, shift in zip(names, sigmas, zeros, shifts, strict=True)]
        steps = self.plan(params)
        steps[0].measured = {"noise of the coadd": f"{scatter:.3g}, and it should be "
                                                   f"{done.sigma:.3g}",
                             "frames": len(flats)}
        if used:
            steps[0].measured["zero points"] = f"from {used} stars"
        if done.filled or done.unfilled:
            steps[0].measured["bad pixels filled"] = f"{done.filled} from the other frames"
        warnings = []
        if done.unfilled:
            warnings.append(advice.warn(
                f"{done.unfilled} pixels are bad in every frame and are the sky in the coadd.",
                "look at where they are before measuring anything near them."))
        if scatter > 1.5 * done.sigma:
            warnings.append(advice.warn(
                f"the noise of the coadd is {scatter:.3g}, and it should be {done.sigma:.3g}",
                "a crowded field keeps its faintest stars in the noise; otherwise a wrong PSF "
                "or zero point, or frames not on the same pixels, would do this."))
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings,
                          data={"rows": rows, "noise": scatter, "sigma": done.sigma,
                                "zeros": zeros, "shifts": [list(s) for s in shifts],
                                "filled": done.filled, "unfilled": done.unfilled})


class DrizzleParams(TaskParams):
    input: list[str] = Field(..., description="The frames, dithered. Globs and @lists expand.",
                             json_schema_extra={"positional": True})
    output: Path = Field(..., description="The drizzled image, in counts per output pixel: a "
                                          "star has the flux it has in the first frame.")
    scale: float = Field(0.5, gt=0, description="An output pixel, in input pixels.")
    pixfrac: float = Field(0.8, gt=0, le=1, description="The side of the drop, in input "
                                                        "pixels: smaller is sharper and "
                                                        "leaves holes.")
    shifts: Path | None = Field(None, description="Where each frame is against the first, a "
                                                  "table of dx and dy, a row a frame.")
    coords: Path | None = Field(None, description="Or stars seen in all the frames, to "
                                                  "measure that from, within 3 pixels.")
    use_wcs: bool = Field(False, description="Or what the frames' WCS says, which takes a "
                                             "rotation and a distortion too.")
    zero: str = Field("", description="The frames' zero points as factors, with commas, the "
                                      "first being 1. Left out, from --coords, or all 1.")
    sky: Literal["match", "surface", "none"] = Field(
        "match", description="match brings the level of each frame to the first's; surface "
                             "takes each one's sky off by skymodel's mesh; none leaves them.")
    box: int = Field(64, ge=3, description="The mesh of the sky of each, as skymodel's.")
    datamax: float | None = Field(None, description="Above this a pixel is saturated, and "
                                                    "lands with no weight.")
    weight: Path | None = Field(None, description="The weight at each output pixel, an image: "
                                                  "zero where nothing landed.")


def _dither(params, flats, stars, bad) -> list[tuple[float, float]]:
    """Where each frame is against the first, from --shifts or the stars."""
    import numpy as np

    from ..core import catalog, zogy

    if params.shifts:
        table = catalog.read(params.shifts)
        names = table.colnames
        if len(names) < 2:
            advice.fail(f"{params.shifts} has {len(names)} columns", "give dx and dy")
        first, second = ("dx", "dy") if {"dx", "dy"} <= set(names) else (names[0], names[1])
        if len(table) != len(flats):
            advice.fail(f"{len(table)} rows in {params.shifts} and {len(flats)} frames",
                        "give a row for each frame, the first's being 0 0")
        return [(float(a), float(b)) for a, b in zip(table[first], table[second], strict=True)]
    if not stars:
        advice.fail("where the frames are against each other",
                    "give --shifts, or --coords with stars seen in all, or --use-wcs")
    shifts = [(0.0, 0.0)]
    for at in range(1, len(flats)):
        found = zogy.centre_offsets(flats[at], flats[0], stars, saturated=bad[at] | bad[0])
        if found.shape[0] < 3:
            advice.fail(f"only {found.shape[0]} stars of {params.coords} are measured in "
                        f"frame {at + 1}", "they have to be within 3 pixels; give --shifts "
                                           "otherwise")
        best = found[found[:, 2] >= np.median(found[:, 2])]
        shifts.append((float(np.median(best[:, 0])), float(np.median(best[:, 1]))))
    return shifts


@register
class Drizzling(Task[DrizzleParams]):
    """Drizzle (Fruchter & Hook 2002): dithered frames onto a finer grid, with no
    interpolation, and the flux kept."""

    name = "drizzle"
    package = "harwood"
    description = "Put dithered frames on a finer grid, getting back what the pixels lost."
    Params = DrizzleParams

    def plan(self, params: DrizzleParams) -> list[Step]:
        return [Step("drizzle", f"scale {params.scale}, pixfrac {params.pixfrac}",
                     why=WHY["drizzle"]),
                Step("write", str(params.output))]

    def execute(self, params: DrizzleParams) -> TaskResult:
        import numpy as np

        from ..core import catalog, drizzle, images, mef, skymesh

        names = images.expand(params.input)
        loaded = [mef.load(name) for name in names]
        shapes = {item[0].shape for item in loaded}
        if len(shapes) > 1:
            advice.fail(f"the frames are not all the same shape: {sorted(shapes)}",
                        "trim them to a common section first")
        limit = np.inf if params.datamax is None else params.datamax
        bad = [(item[3] != 0) | (item[0] >= limit) | ~np.isfinite(item[0]) for item in loaded]
        skies = [skymesh.model(item[0], params.box, mask=flagged)
                 for item, flagged in zip(loaded, bad, strict=True)]
        flats = [item[0] - sky.sky for item, sky in zip(loaded, skies, strict=True)]
        stars = None
        if params.coords:
            table = catalog.read(params.coords)
            stars = list(zip(*catalog.coordinates(table, params.coords), strict=True))
        zeros, used = [1.0] * len(flats), 0
        if params.zero:
            zeros = _each(params.zero, len(flats), "--zero", float)
        elif params.coords:
            for at in range(1, len(flats)):
                zeros[at], used = _ratio(flats[at], flats[0], params.coords)
        shape = loaded[0][0].shape
        size = (int(np.ceil(shape[0] / params.scale)), int(np.ceil(shape[1] / params.scale)))
        keep = drizzle.scaled_header(_stamped(loaded[0][1]), params.scale)
        shifts = None
        if not params.use_wcs:
            shifts = _dither(params, flats, stars, bad)
        grid = drizzle.Grid(size, params.pixfrac, params.scale)
        level = float(np.median(skies[0].sky))
        noises = []
        for at, (item, sky) in enumerate(zip(loaded, skies, strict=True)):
            data = {"none": item[0], "surface": flats[at]}.get(
                params.sky, item[0] - float(np.median(sky.sky)) + level * zeros[at])
            noise = float(np.median(sky.rms))
            noises.append(noise)
            # A frame of one value has no noise to weigh it by
            variance = (item[2] if item[2] is not None
                        else np.full(shape, noise ** 2 if noise > 0 else 1.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = np.where(bad[at] | (variance <= 0), 0.0, zeros[at] ** 2 / variance)
            transform = (drizzle.wcs_transform(_stamped(item[1]), keep) if params.use_wcs else
                         drizzle.shifted(*shifts[at], params.scale))
            grid.add(data / zeros[at], transform, weights)
        image = grid.image
        keep["NDRIZZLE"] = (len(flats), "frames drizzled")
        keep["DRIZSCAL"] = (params.scale, "an output pixel, in input pixels")
        keep["DRIZFRAC"] = (params.pixfrac, "the side of the drop, in input pixels")
        factor = drizzle.noise_correlation(params.pixfrac, params.scale)
        keep["DRIZNCOR"] = (factor, "true noise over what the pixels show")
        outputs = [_write(params.output, image, keep)]
        if params.weight:
            outputs.append(_write(params.weight, grid.weight, keep))
        inner = grid.weight[size[0] // 10:-size[0] // 10 or None,
                            size[1] // 10:-size[1] // 10 or None]
        holes = float((inner <= 0).mean())
        steps = self.plan(params)
        steps[0].measured = {"frames": len(flats), "output": f"{size[1]} by {size[0]}",
                             "holes": f"{100 * holes:.2f}%",
                             "noise factor": f"{factor:.2f}"}
        if used:
            steps[0].measured["zero points"] = f"from {used} stars"
        warnings = []
        if holes > 0.001:
            warnings.append(advice.warn(
                f"{100 * holes:.1f}% of the inner output pixels had nothing land on them, and "
                "are zero.",
                "too few frames, or dithers too much alike, for this grid: raise --pixfrac or "
                "--scale, or add frames."))
        if len(flats) > 1 and shifts is not None:
            phases = {(round((a % 1) * 4), round((b % 1) * 4)) for a, b in shifts}
            if len(phases) == 1:
                warnings.append(advice.warn(
                    "the frames are at the same fraction of a pixel, so that they sample the "
                    "same phases and nothing of what the pixels lost comes back.",
                    "use imcombine for these; drizzling wants dithers by fractions of a "
                    "pixel."))
        warnings.append(advice.note(
            f"the noise between neighbouring pixels is correlated: the true noise is about "
            f"{factor:.2f} times what their scatter shows.",
            "use the factor, in the header's DRIZNCOR, in any error taken from the scatter."))
        rows = [{"image": Path(name.path).name, "dx": None if shifts is None else
                 round(shifts[at][0], 3), "dy": None if shifts is None else
                 round(shifts[at][1], 3), "noise": round(noises[at], 3),
                 "zero point": round(zeros[at], 4)} for at, name in enumerate(names)]
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings,
                          data={"rows": rows, "holes": holes, "noise_factor": factor,
                                "shifts": shifts, "zeros": zeros})


class PlatesolveParams(TaskParams):
    image: str = Field(..., description="The image.", json_schema_extra={"positional": True})
    coords: Path = Field(..., description="Stars of the frame, from daofind: brightest first "
                                          "by its mag column, or as they come with none.")
    catalog: Path | None = Field(None, description="The catalogue, a table of ra and dec in "
                                                   "degrees, with mag, pmra, pmdec if it has "
                                                   "them, brightest first.")
    gaia: bool = Field(False, description="Or Gaia DR3, from ESA or Heidelberg: only the "
                                          "place and the radius leave the machine.")
    ra: str = Field("", description="About where it is: degrees, or hours as hh:mm:ss. Left "
                                    "out, the header's RA or CRVAL1.")
    dec: str = Field("", description="The same: degrees, or dd:mm:ss.")
    scale: float | None = Field(None, gt=0, description="About its scale, arcsec a pixel. "
                                                        "Left out, what its CD says.")
    scale_error: float = Field(0.05, ge=0, lt=1, description="How far --scale can be off, as "
                                                             "a fraction: 0.05 is 5%.")
    rotation: float | None = Field(None, description="Its rotation in degrees, if it is "
                                                     "known, to try only that and near it.")
    flip: Literal["both", "yes", "no"] = Field("both", description="Whether to try the sky's "
                                                                   "mirror image.")
    radius: float | None = Field(None, gt=0, description="The catalogue's radius, degrees. "
                                                         "Left out, enough for the frame and "
                                                         "as much again.")
    nstars: int = Field(100, ge=4, description="The frame's brightest stars to search with.")
    epoch: float | None = Field(None, description="The year of the frame, for proper motions. "
                                                  "Left out, from DATE-OBS.")
    catalog_epoch: float = Field(2016.0, description="The year of the catalogue's positions.")
    sip: int = Field(0, ge=0, le=5, description="The order of a SIP distortion, 2 or more. "
                                                "None by default.")
    tolerance: float = Field(3.0, gt=0, description="Pairs within this at the last, pixels.")
    significance: float = Field(20.0, gt=0, description="How many sigma of chance the pairs "
                                                        "have to reach to be a solution: the "
                                                        "search takes the best of many, and "
                                                        "beats 5 with nothing there.")
    output: Path | None = Field(None, description="A copy of the image with the WCS in it.")
    matched: Path | None = Field(None, description="The pairs, a table, with what is left of "
                                                   "each.")


def _sky_guess(params, header):
    """The place to start from, in ICRS. What a header says is at its own equinox."""
    from astropy import units as u
    from astropy.coordinates import FK5, SkyCoord
    from astropy.time import Time

    def number(text, unit):
        return float(text) * u.deg if ":" not in str(text) else _sexagesimal(text, unit)

    def _sexagesimal(text, unit):
        return u.Quantity(SkyCoord(f"{text} 0" if unit == "ra" else f"0 {text}",
                                   unit=(u.hourangle, u.deg)).ra.deg if unit == "ra" else
                          SkyCoord(f"0 {text}", unit=(u.hourangle, u.deg)).dec.deg, u.deg)

    if params.ra and params.dec:
        return number(params.ra, "ra").value, number(params.dec, "dec").value
    if "CRVAL1" in header and "CTYPE1" in header and "RA" in str(header["CTYPE1"]):
        return float(header["CRVAL1"]), float(header["CRVAL2"])
    if "RA" in header and "DEC" in header:
        ra, dec = number(header["RA"], "ra"), number(header["DEC"], "dec")
        equinox = header.get("EQUINOX", header.get("EPOCH"))
        if equinox:
            place = SkyCoord(ra, dec, frame=FK5(equinox=Time(float(equinox), format="jyear")))
            return float(place.icrs.ra.deg), float(place.icrs.dec.deg)
        return ra.value, dec.value
    advice.fail("where the image is", "give --ra and --dec, as its header has neither RA and "
                                      "DEC nor a WCS")


def _frame_year(params, header) -> float | None:
    from astropy.time import Time

    if params.epoch is not None:
        return params.epoch
    value = str(header.get("DATE-OBS", "")).strip()
    if "/" in value:
        day, month, year = value.split("/")[:3]
        year = int(year)
        year += 0 if year >= 100 else 1900 if year >= 50 else 2000
        value = f"{year}-{month}-{day}"
    try:
        return float(Time(value).jyear) if value else None
    except ValueError:
        return None


def _frame_scale(params, header) -> float:
    import math

    if params.scale:
        return params.scale
    if "CD1_1" in header:
        det = header["CD1_1"] * header.get("CD2_2", 0.0) - (header.get("CD1_2", 0.0)
                                                            * header.get("CD2_1", 0.0))
        if det:
            return math.sqrt(abs(det)) * 3600.0
    advice.fail("about what scale the image has", "give --scale, in arcsec a pixel: a wide "
                                                  "--scale-error will do, and takes longer")


@register
class Platesolve(Task[PlatesolveParams]):
    """Where an image is on the sky: stars against a catalogue, with the pairs found, TAN and
    SIP. IRAF's ccmap fits a plate from stars already paired by hand."""

    name = "platesolve"
    package = "harwood"
    description = "Find where an image is on the sky, against a catalogue, and write its WCS."
    Params = PlatesolveParams

    def plan(self, params: PlatesolveParams) -> list[Step]:
        return [Step("search", "every rotation, mirror and scale in the range",
                     why=WHY["search"]),
                Step("fit", f"TAN{f', SIP order {params.sip}' if params.sip else ''}",
                     why=WHY["plate"]),
                Step("judge", f"{params.significance} sigma of chance", why=WHY["judge"]),
                Step("write", str(params.output) if params.output else "nothing")]

    def execute(self, params: PlatesolveParams) -> TaskResult:
        import math

        import numpy as np
        from astropy.io import fits
        from astropy.table import Table

        from ..core import catalog, images, platesolve

        if bool(params.catalog) == bool(params.gaia):
            advice.fail("which catalogue", "give --catalog with a table, or --gaia")
        if params.sip == 1:
            advice.fail("a SIP of order 1", "a plate is order 1 already: give 2 or more")
        data, header = images.read(params.image)
        ny, nx = data.shape
        centre = _sky_guess(params, header)
        pixel = _frame_scale(params, header)
        table = catalog.read(params.coords)
        xs, ys = (np.asarray(column, float) for column in catalog.coordinates(table,
                                                                              params.coords))
        if "mag" in table.colnames:
            order = np.argsort(np.asarray(table["mag"], float), kind="stable")
            xs, ys = xs[order], ys[order]
        if xs.size < 4:
            advice.fail(f"{xs.size} stars in {params.coords}",
                        "it takes more; lower daofind's threshold")
        half_diagonal = 0.5 * math.hypot(nx, ny) * pixel * (1 + params.scale_error) / 3600.0
        radius = params.radius or 2.0 * half_diagonal
        service = ""
        if params.gaia:
            stars, service = platesolve.gaia(*centre, radius)
        else:
            stars = catalog.read(params.catalog)
            if not {"ra", "dec"} <= set(stars.colnames):
                advice.fail(f"{params.catalog} has {', '.join(stars.colnames)}",
                            "it needs ra and dec, in degrees")
        ra, dec = np.asarray(stars["ra"], float), np.asarray(stars["dec"], float)
        year = _frame_year(params, header)
        moved = "pmra" in stars.colnames and year is not None
        if moved:
            motion = [np.ma.filled(np.ma.asarray(stars[name], float), np.nan)
                      for name in ("pmra", "pmdec")]
            ra, dec = platesolve.proper_motion(ra, dec, *motion, params.catalog_epoch, year)
        for name in ("mag", "phot_g_mean_mag"):
            if name in stars.colnames:
                light = np.ma.filled(np.ma.asarray(stars[name], float), np.inf)
                by_light = np.argsort(light, kind="stable")
                ra, dec = ra[by_light], dec[by_light]
                break
        xi, eta = platesolve.project(ra, dec, *centre)
        near = np.hypot(xi, eta) <= radius
        if near.sum() < 4:
            advice.fail(f"{int(near.sum())} catalogue stars within {radius:.3g} degrees of "
                        f"{centre[0]:.4f} {centre[1]:.4f}",
                        "check --ra and --dec, or give a larger --radius")
        ra, dec, xi, eta = ra[near], dec[near], xi[near], eta[near]
        # About as dense on the sky as the search's stars are on the frame: more only adds to
        # what agrees by chance
        count = min(params.nstars, xs.size)
        area = math.pi * radius ** 2
        frame_area = nx * ny * (pixel / 3600.0) ** 2
        wanted = max(int(2.0 * count * area / frame_area), 20)
        rotations = None
        if params.rotation is not None:
            rotations = -params.rotation + np.arange(-3.0, 3.01, 0.25)
        mirrors = {"both": (False, True), "yes": (True,), "no": (False,)}[params.flip]
        crpix = (0.5 * (nx + 1), 0.5 * (ny + 1))
        guess = platesolve.search(xs[:count] - crpix[0], ys[:count] - crpix[1], xi[:wanted],
                                  eta[:wanted], pixel, params.scale_error, rotations=rotations,
                                  mirrors=mirrors)
        deep = 3 * wanted
        solved = platesolve.fit(xs, ys, ra[:deep], dec[:deep], guess, crpix, centre,
                                params.sip, params.tolerance)
        found, chance, worth = platesolve.judged(solved, xs.size, xi[:deep], eta[:deep],
                                                 data.shape, params.tolerance)
        steps = self.plan(params)
        steps[0].measured = {"frame stars": count, "catalogue stars": min(wanted, xi.size),
                             "votes": f"{guess.votes}, and {guess.runner_up} elsewhere"}
        steps[1].measured = {"pairs": found, "rms": f"{solved.rms:.3f} arcsec",
                             "rejected": solved.rejected,
                             "scale": f"{solved.pixel_scale:.4f} arcsec a pixel",
                             "rotation": f"{solved.rotation:.2f} degrees",
                             "mirror": "yes" if solved.flipped else "no",
                             "middle": f"{solved.crval[0]:.6f} {solved.crval[1]:.6f}"}
        steps[2].measured = {"by chance": f"{chance:.1f} pairs", "sigma": f"{worth:.1f}"}
        # Pairs at random fill the radius, at 0.7 of it, where true ones sit in its middle
        tight = solved.rms < 0.55 * params.tolerance * solved.pixel_scale
        if found < 5 or worth < params.significance or not tight:
            advice.fail(
                f"no solution: {found} pairs, where chance gives {chance:.1f} ({worth:.1f} "
                f"sigma), with an rms of {solved.rms:.2f} arcsec",
                "this is what a wrong place or scale looks like, and no WCS was written: "
                "widen --radius and --scale-error, check how far the pointing can be off, "
                "give a deeper list of stars, or a shallower catalogue where the field is "
                "crowded. A strong distortion wants --sip.")
        outputs = []
        if params.output:
            outputs.append(_written_with_wcs(params, solved, fits))
        if params.matched:
            frame, other = solved.pairs.T
            back_ra, back_dec = solved.sky(xs[frame], ys[frame])
            cos_dec = np.cos(np.radians(back_dec))
            pairs = Table({"x": xs[frame], "y": ys[frame], "ra": ra[other], "dec": dec[other],
                           "dra": (back_ra - ra[other]) * cos_dec * 3600.0,
                           "ddec": (back_dec - dec[other]) * 3600.0})
            outputs.append(catalog.write(pairs, params.matched))
        warnings = []
        if params.gaia and not moved:
            warnings.append(advice.note(
                "no year for the frame, so no proper motions were applied.",
                "give --epoch: at 2016 and older, stars have moved."))
        if service:
            steps[0].measured["catalogue"] = service
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings,
                          data={"pairs": found, "rms": solved.rms, "chance": chance,
                                "significance": worth, "scale": solved.pixel_scale,
                                "rotation": solved.rotation, "flipped": solved.flipped,
                                "crval": list(solved.crval), "rejected": solved.rejected,
                                "votes": guess.votes, "runner_up": guess.runner_up})


def _written_with_wcs(params, solved, fits) -> Path:
    """The file as it was, with any old WCS out of the image's header and the new one in."""
    old = ("WCSDIM", "LTV1", "LTV2", "LTM1_1", "LTM1_2", "LTM2_1", "LTM2_2", "CDELT1",
           "CDELT2", "CROTA1", "CROTA2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "WCSAXES",
           "RADESYS", "LONPOLE", "LATPOLE", "EQUINOX", "A_ORDER", "B_ORDER", "AP_ORDER",
           "BP_ORDER", "WCSSOL")
    path = Path(params.image.split("[")[0])
    Path(params.output).parent.mkdir(parents=True, exist_ok=True)
    with fits.open(path, memmap=False) as hdus:
        at = next(number for number, hdu in enumerate(hdus)
                  if hdu.data is not None and hdu.data.ndim == 2)
        header = hdus[at].header
        for key in list(header):
            if key in old or key.startswith(("WAT", "A_", "B_", "AP_", "BP_")):
                del header[key]
        header["RADESYS"] = ("ICRS", "from the catalogue")
        header["EQUINOX"] = 2000.0
        for key, value, comment in solved.cards():
            header[key] = (value, comment)
        header["PLATERMS"] = (solved.rms, "rms of the pairs, arcsec")
        header["PLATEN"] = (solved.pairs.shape[0], "pairs in the plate")
        hdus.writeto(params.output, overwrite=True)
    return Path(params.output)


class SkysplineParams(TaskParams):
    image: str = Field(..., description="The long-slit frame, reduced.",
                       json_schema_extra={"positional": True})
    output: Path = Field(..., description="The frame less its sky.")
    sky: Path | None = Field(None, description="The sky itself, an image.")
    tilt_frame: str = Field("", description="A frame with sharp features to measure the tilt "
                                            "from, an arc. Left out, the image itself.")
    dispaxis: int | None = Field(None, ge=1, le=2, description="1 if the dispersion is along "
                                                               "the lines, 2 along the "
                                                               "columns. Left out, the "
                                                               "header's.")
    slit: str = Field("", description="The columns across the dispersion the slit lights, as "
                                      "c1:c2 from 1. Left out, found from the profile.")
    object: str = Field("", description="Those with an object on, as c1:c2,c3:c4. Left out, "
                                        "found from the profile.")
    gain: float | str | None = Field(None, description="Electrons per count, or its keyword. "
                                                       "Left out, the header's GAIN, else 1.")
    rdnoise: float | str | None = Field(None, description="Read noise in electrons, or its "
                                                          "keyword. Left out, the header's "
                                                          "RDNOISE, else 0.")
    spacing: float | None = Field(None, gt=0, description="The spline's knots, pixels. Left "
                                                          "out, 0.5 where the tilt samples "
                                                          "the sky that finely, else 1.")
    order: int = Field(3, ge=0, description="Of the slit's illumination, a polynomial.")
    clip: float = Field(4.0, gt=0, description="Pixels this many sigma off stay out of the "
                                               "fit: cosmic rays, an object's wings.")
    xorder: int = Field(2, ge=1, description="Of the tilt across the slit.")
    yorder: int = Field(2, ge=0, description="Of the tilt along the dispersion.")


def _columns(text: str, slit: tuple[int, int]):
    """c1:c2,c3:c4, from 1, as a mask of the slit's columns."""
    import numpy as np

    mask = np.zeros(slit[1] - slit[0] + 1, bool)
    for part in text.split(","):
        if ":" not in part:
            advice.fail(f"--object {text}", "give columns as c1:c2,c3:c4, from 1")
        first, last = (int(word) - 1 for word in part.split(":"))
        mask[max(first - slit[0], 0):max(last - slit[0] + 1, 0)] = True
    return mask


def _gain_noise(params, header) -> tuple[float, float, list]:
    said = []
    values = []
    for given, keyword, fallback in ((params.gain, "GAIN", 1.0),
                                     (params.rdnoise, "RDNOISE", 0.0)):
        if given is not None:
            try:
                values.append(float(given))
            except ValueError:
                if given not in header:
                    advice.fail(f"'{given}' is neither a number nor a keyword in the header.",
                                f"give the {keyword.lower()} as a number")
                values.append(float(header[given]))
        elif keyword in header:
            values.append(float(header[keyword]))
        else:
            values.append(fallback)
            said.append(advice.note(f"no {keyword} in the header and none given, so "
                                    f"{fallback:g} was used.",
                                    f"give --{keyword.lower()}, from the detector's sheet."))
    return values[0], values[1], said


@register
class Skyspline(Task[SkysplineParams]):
    """The sky of a long-slit frame as one spectrum (Kelson 2003): a spline in where each pixel
    is along it, with nothing resampled. IRAF fits each line across the slit."""

    name = "skyspline"
    package = "harwood"
    description = "Take the sky off a long-slit frame, as one spectrum sampled by every pixel."
    Params = SkysplineParams

    def plan(self, params: SkysplineParams) -> list[Step]:
        return [Step("tilt", params.tilt_frame or params.image, why=WHY["tilt"]),
                Step("spline", "the sky's spectrum and the slit's illumination",
                     why=WHY["spline"]),
                Step("write", str(params.output))]

    def execute(self, params: SkysplineParams) -> TaskResult:
        import numpy as np

        from ..core import apextract, images, mef, skyspline

        data, header, _, quality = mef.load(params.image)
        axis = params.dispaxis
        if axis is None:
            try:
                axis = int(header.get("DISPAXIS", 0))
            except (TypeError, ValueError):
                axis = 0
        if axis not in (1, 2):
            advice.fail("which way the dispersion runs",
                        "give --dispaxis: 1 along the lines, 2 along the columns")
        frame = apextract.frame(data, axis)
        bad = apextract.frame(quality != 0, axis)
        gain, rdnoise, said = _gain_noise(params, header)
        if params.slit:
            first, last = (int(word) - 1 for word in params.slit.split(":"))
            slit = (first, last)
        else:
            slit = skyspline.slit_columns(frame)
        if slit[1] - slit[0] < 4:
            advice.fail(f"a slit of {slit[1] - slit[0] + 1} columns",
                        "there is not enough of it: check --dispaxis, or give --slit")
        objects = (_columns(params.object, slit) if params.object
                   else skyspline.object_columns(frame, slit))
        tilted = frame
        if params.tilt_frame:
            other, _ = images.read(params.tilt_frame)
            if other.shape != data.shape:
                advice.fail(f"{params.tilt_frame} is {other.shape} and the image {data.shape}",
                            "give an arc taken with the same setup")
            tilted = apextract.frame(other, axis)
        tilt = skyspline.measure_tilt(tilted, slit, params.xorder, params.yorder,
                                      objects=objects)
        reach = 2.0 * tilt.largest
        spacing = params.spacing or (0.5 if reach >= 2.0 else 1.0)
        done = skyspline.fit(frame, tilt, slit, objects, gain, rdnoise, spacing, params.order,
                             params.clip, mask=bad)
        keep = _stamped(header)
        keep["SKYSPLIN"] = (spacing, "the knots of the sky's spline, pixels")
        keep["SKYTILT"] = (tilt.largest, "the tilt at the slit's edge, pixels")
        restored = done.subtracted if axis == 2 else done.subtracted.T
        outputs = [_write(params.output, restored, keep)]
        if params.sky:
            outputs.append(_write(params.sky, done.sky if axis == 2 else done.sky.T, keep))
        steps = self.plan(params)
        steps[0].measured = {"at the slit's edge": f"{tilt.largest:.3f} pixels",
                             "measures": tilt.measured, "their scatter": f"{tilt.rms:.3f}"}
        steps[1].measured = {"sky pixels": done.used, "left out": done.rejected,
                             "knots": f"every {spacing:g} pixels",
                             "what remains": f"{done.rms:.2f} of their noise",
                             "slit": f"{slit[0] + 1}:{slit[1] + 1}",
                             "columns with an object": int(np.count_nonzero(objects))}
        warnings = list(said)
        if reach < 1.0:
            warnings.append(advice.note(
                f"the slit's lines lean by only {reach:.2f} pixels across it, so that its "
                "pixels sample the sky at about the same places, and this is no better than a "
                "fit by lines, which is what it comes to.",
                "this is what gets better with tilt; on an instrument with none, either "
                "does."))
        level = float(np.median(done.sky[:, slit[0]:slit[1] + 1]))
        if level < 3.0 * rdnoise / gain:
            warnings.append(advice.note(
                f"there is hardly any sky in this frame: {level:.1f} counts a pixel, against "
                f"a read noise of {rdnoise / gain:.1f}.",
                "taking it off changes little; the object's own noise is what counts here."))
        if done.rms > 1.5:
            warnings.append(advice.warn(
                f"what remains of the sky is {done.rms:.2f} times its noise.",
                "a flat that does not take the pixels out would do this, or a wrong --gain, "
                "or an object not left out: look at the output, around the sky's lines."))
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings,
                          data={"tilt": tilt.largest, "remains": done.rms, "spacing": spacing,
                                "slit": [slit[0] + 1, slit[1] + 1], "rejected": done.rejected,
                                "objects": int(np.count_nonzero(objects))})


class FlexureParams(TaskParams):
    image: str = Field(..., description="The frame, or a spectrum as an image, with sky in "
                                        "it.", json_schema_extra={"positional": True})
    reference: str = Field(..., description="The frame whose wavelengths are known: the arc's "
                                            "neighbour in time, or a sky spectrum.",
                           json_schema_extra={"positional": True})
    dispaxis: int | None = Field(None, ge=1, le=2, description="As in skyspline. A spectrum "
                                                               "of one line needs none.")
    columns: str = Field("", description="The columns of the sky across the dispersion, as "
                                         "c1:c2 from 1, the same in both. Left out, the slit "
                                         "but for an object.")
    reach: int = Field(10, ge=1, description="How far to look, pixels.")
    width: int = Field(61, ge=3, description="Of a running median taken off each, pixels, to "
                                             "leave the lines.")
    output: Path | None = Field(None, description="A copy of the frame with its wavelengths "
                                                  "moved by what was found: CRVAL or CRPIX.")


@register
class Flexure(Task[FlexureParams]):
    """The shift of a frame's wavelengths, from its own sky lines against a reference's."""

    name = "flexure"
    package = "harwood"
    description = "Measure how far the spectrograph has bent, from the sky's own lines."
    Params = FlexureParams

    def plan(self, params: FlexureParams) -> list[Step]:
        return [Step("correlate", f"{params.image} against {params.reference}",
                     why=WHY["flexure"]),
                Step("write", str(params.output) if params.output else "nothing")]

    def execute(self, params: FlexureParams) -> TaskResult:
        from astropy.io import fits

        from ..core import images, skyspline

        data, header = images.read(params.image)
        other, _ = images.read(params.reference)
        if data.shape != other.shape:
            advice.fail(f"{params.image} is {data.shape} and {params.reference} {other.shape}",
                        "both need the same setup, or the same spectrum's length")
        axis = params.dispaxis or int(header.get("DISPAXIS", 0) or 0)
        if data.ndim == 2 and data.shape[0] > 1 and data.shape[1] > 1 and axis not in (1, 2):
            advice.fail("which way the dispersion runs",
                        "give --dispaxis: 1 along the lines, 2 along the columns")
        one, slit_used = skyspline.sky_spectrum(data, axis, params.columns)
        two, _ = skyspline.sky_spectrum(other, axis, params.columns)
        found = skyspline.flexure(one, two, params.reach, params.width)
        cdelt = _dispersion(header, axis)
        steps = self.plan(params)
        steps[0].measured = {"shift": f"{found.shift:+.3f} +- {found.error:.3f} pixels",
                             "peak": f"{found.peak:.2f} of the correlation",
                             "columns": slit_used}
        if cdelt:
            steps[0].measured["in wavelength"] = f"{found.shift * cdelt:+.4g}"
        warnings = []
        if found.peak < 0.5:
            warnings.append(advice.warn(
                f"the frames agree only {found.peak:.2f} at the best shift.",
                "they may have no lines in common, or the sky may be too faint in one: this "
                "shift is not to be trusted."))
        outputs = []
        if params.output:
            axis_number = axis if axis in (1, 2) else 1
            with fits.open(params.image.split("[")[0], memmap=False) as hdus:
                at = next(number for number, hdu in enumerate(hdus) if hdu.data is not None)
                card = f"CRPIX{axis_number}"
                hdus[at].header[card] = hdus[at].header.get(card, 1.0) + found.shift
                hdus[at].header["FLEXURE"] = (found.shift, "pixels, against the reference")
                hdus.writeto(params.output, overwrite=True)
            outputs.append(params.output)
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings,
                          data={"shift": found.shift, "error": found.error,
                                "peak": found.peak})


def _dispersion(header, axis: int) -> float:
    axis = axis if axis in (1, 2) else 1
    for key in (f"CD{axis}_{axis}", f"CDELT{axis}"):
        if key in header:
            return float(header[key])
    return 0.0
