"""ccdtest: IRAF's test package of ccdred -- artificial CCD observations to try a reduction
on, as ccdred's demonstration makes them (setinstrument demo sets these tasks'
parameters).

    mkimage      make an image of a sloped plane and noise, or edit one (replace, add,
                 multiply), whole or a section
    artobs       a CCD observation: noise, sky, flat field, zero level and dark, bias,
                 bad pixels, and the header ccdproc reads
    subsection   a frame read out from a section of the chip, from a full one

From noao/imred/ccdred/ccdtest (t_mkimage.x, artobs.cl, subsection.cl). See core/ccdtest.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

#: Where harwood keeps the files ccdtest$ names (badpix.dat).
CCDTEST = Path(__file__).resolve().parent.parent / "data" / "iraf" / "ccdtest"


def _iraf_name(name: str) -> str:
    """ccdtest$badpix.dat as a path harwood can open."""
    prefix, sep, rest = name.partition("$")
    return str(CCDTEST / rest) if sep and prefix == "ccdtest" else name


def _split(name: str) -> tuple[Path, str]:
    """An image name, as a file (.fits added when it has no extension) and a section."""
    text = name.strip()
    section = ""
    if text.endswith("]") and "[" in text:
        text, _, rest = text.partition("[")
        section = "[" + rest
    path = Path(text)
    if not path.suffix:
        path = path.with_suffix(".fits")
    return path, section


def _exists(name: str) -> bool:
    return _split(name)[0].exists()


def _access(name: str) -> bool:
    """The CL's access(), as artobs asks it of imdata: the name as given, or with IRAF's
    old extensions .imh and .hhh -- not .fits: a FITS image named without its extension
    is not found, and IRAF's artobs then copies no data in."""
    return any(Path(name + ext).is_file() for ext in ("", ".imh", ".hhh"))


def _region(shape, section: str):
    from ..core import pixels

    if not section:
        return tuple(slice(None) for _ in shape)
    return pixels.parse_section(section, shape)


def edit(name: str, option: str, value: float, slope: float = 0.0, sigma: float = 0.0,
         seed: int = 0) -> Path:
    """mkimage replace, add or multiply, on an image or a section of it, in place: the
    pixels keep their type."""
    import numpy as np
    from astropy.io import fits

    from ..core import ccdtest

    path, section = _split(name)
    if not path.exists():
        advice.fail(f"Cannot open image ({name})", "make it first: mkimage NAME make ...",
                    FileNotFoundError)
    ccdtest.seed(seed)
    with fits.open(path, mode="update", do_not_scale_image_data=False) as hdul:
        hdu = hdul[0]
        data = np.asarray(hdu.data)
        cut = _region(data.shape, section)
        current = np.asarray(data[cut], np.float32)
        made = ccdtest.plane(option, current, value, slope, sigma)
        data[cut] = ccdtest.stored(made, data.dtype)
        hdu.data = data
    return path


def make(name: str, value: float, ndim: int, dims: str, pixtype: str = "real",
         slope: float = 0.0, sigma: float = 0.0, seed: int = 0) -> Path:
    """mkimage make: a new image of ndim dimensions, dims long, of pixtype."""
    import numpy as np
    from astropy.io import fits

    from ..core import ccdtest

    try:
        sizes = [int(word) for word in dims.split()]
    except ValueError:
        sizes = []
    if len(sizes) < ndim:
        advice.fail("Bad dimension string", f"give {ndim} lengths, as '132 100'")
    dtype = ccdtest.PIXTYPES.get(pixtype[:1])
    if dtype is None:
        advice.fail("Bad pixel type", "give short, int, long, real or double")
    ccdtest.seed(seed)
    shape = tuple(reversed(sizes[:ndim]))
    made = ccdtest.plane("make", np.zeros(shape, np.float32), value, slope, sigma,
                         new=True)
    path, _ = _split(name)
    fits.PrimaryHDU(ccdtest.stored(made, dtype)).writeto(path, overwrite=True)
    return path


def copy_into(source: str, target: str) -> None:
    """imcopy of an image or section into a new image, or into a section of one that is
    there: the target's type kept."""
    import numpy as np
    from astropy.io import fits

    from ..core import ccdtest, images

    path, section = _split(target)
    if not section:
        # imcopy's own: the type kept, and the section's world coordinates written
        from ..core.runner import run as run_task

        run_task("imcopy", input=[_iraf_name(source)], output=str(path))
        return
    pixels, header = images.read(_iraf_name(source))
    with fits.open(path, mode="update") as hdul:
        data = np.asarray(hdul[0].data)
        cut = _region(data.shape, section)
        data[cut] = ccdtest.stored(np.asarray(pixels, np.float32), data.dtype)
        hdul[0].data = data


# ------------------------------------------------------------------ mkimage

class MkImageParams(TaskParams):
    image: str = Field(..., description="The image to make or edit; a section edits only "
                                        "that part.", json_schema_extra={"positional": True})
    option: Literal["make", "replace", "add", "multiply"] = Field(
        ..., description="make a new image, or replace, add to or multiply its pixels.",
        json_schema_extra={"positional": True})
    value: float = Field(..., description="The mean value of the plane.",
                         json_schema_extra={"positional": True})
    ndim: int = Field(2, ge=1, le=7, description="make: the number of dimensions.")
    dims: str = Field("", description="make: the lengths, '132 100'.")
    pixtype: Literal["short", "int", "long", "real", "double"] = Field(
        "real", description="make: the pixel type.")
    slope: float = Field(0.0, description="The plane's slope: per pixel, along lines "
                                          "and columns.")
    sigma: float = Field(0.0, ge=0, description="Gaussian noise's sigma.")
    seed: int = Field(0, ge=0, description="The noise's seed; 0 goes on from the last "
                                           "call's.")


@register
class MkImage(Task[MkImageParams]):
    name = "mkimage"
    package = "ccdtest"
    description = "Make an image of a sloped plane and noise, or edit one with it."
    Params = MkImageParams

    def plan(self, params: MkImageParams) -> list[Step]:
        return [Step(params.option, f"{params.image}: {params.value:g}, slope "
                                    f"{params.slope:g}, sigma {params.sigma:g}",
                     why="A test frame whose every value is known: the plane gives "
                         "levels and gradients, the noise gives something to reject.")]

    def execute(self, params: MkImageParams) -> TaskResult:
        if params.option == "make":
            path = make(params.image, params.value, params.ndim, params.dims,
                        params.pixtype, params.slope, params.sigma, params.seed)
        else:
            path = edit(params.image, params.option, params.value, params.slope,
                        params.sigma, params.seed)
        return TaskResult(outputs=[path], steps=self.plan(params))


# ------------------------------------------------------------------- artobs

class ArtObsParams(TaskParams):
    image: str = Field(..., description="The observation to make.",
                       json_schema_extra={"positional": True})
    exptime: float = Field(..., description="Its exposure time.",
                           json_schema_extra={"positional": True})
    ccdtype: str = Field(..., description="Its type: object, zero, dark, flat... (through "
                                          "the instrument's translation).",
                         json_schema_extra={"positional": True})
    ncols: int = Field(132, ge=1, description="Columns.")
    nlines: int = Field(100, ge=1, description="Lines.")
    filter: str = Field("", description="The filter, the subset.")
    datasec: str = Field("[1:100,1:100]", description="The data section.")
    trimsec: str = Field("[3:98,3:98]", description="The trim section.")
    biassec: str = Field("[103:130,*]", description="The bias (overscan) section.")
    imdata: str = Field("", description="An image whose data section is the scene.")
    skyrate: float = Field(0.0, description="Sky counts a second.")
    badpix: str = Field("", description="Bad pixel regions, c1 c2 l1 l2 a line.")
    biasval: float = Field(500.0, description="The bias level.")
    badval: float = Field(500.0, description="The bad pixels' value.")
    zeroval: float = Field(100.0, description="The zero level.")
    darkrate: float = Field(1.0, description="Dark counts a second.")
    zeroslope: float = Field(0.01, description="The zero level's slope.")
    darkslope: float = Field(0.002, description="The dark rate's slope.")
    flatslope: float = Field(3e-4, description="The flat field's slope.")
    sigma: float = Field(5.0, ge=0, description="Gaussian noise.")
    seed: int = Field(0, ge=0, description="The noise's seed; 0 goes on from the last.")
    overwrite: bool = Field(False, description="Make it again if it is there.")
    instrument: str | None = Field(None, description="Translation for the header; the "
                                                     "project's when not given.")


@register
class ArtObs(Task[ArtObsParams]):
    name = "artobs"
    package = "ccdtest"
    description = "Make an artificial CCD observation, as ccdred's demonstration does."
    Params = ArtObsParams

    def plan(self, params: ArtObsParams) -> list[Step]:
        return [Step("make", f"{params.image}: {params.ccdtype}, {params.exptime:g}s",
                     why="Noise, sky, the flat field's slope, zero level and dark, bias "
                         "over the whole frame, bad pixels: what ccdproc takes away.")]

    def execute(self, params: ArtObsParams) -> TaskResult:
        im = params.image
        if _exists(im):
            if not params.overwrite:
                return TaskResult(outputs=[], steps=self.plan(params))
            _split(im)[0].unlink()
        exp = params.exptime
        make(im, 0.0, 2, f"{params.ncols} {params.nlines}", "short", 0.0, params.sigma,
             params.seed)
        if params.imdata.strip() and _access(params.imdata.strip()):
            copy_into(params.imdata + params.datasec, im + params.datasec)
        value = exp * params.skyrate
        if value != 0:
            edit(im + params.datasec, "add", value)
        if params.flatslope != 0:
            edit(im + params.datasec, "multiply", 1.0, params.flatslope)
        value = params.zeroval + exp * params.darkrate
        slope = params.zeroslope + exp * params.darkslope
        if value != 0 and slope != 0:
            edit(im + params.datasec, "add", value, slope)
        if params.biasval != 0:
            edit(im, "add", params.biasval, 0.0, params.sigma, 0)
        badpix = _iraf_name(params.badpix.strip())
        if badpix and Path(badpix).is_file():
            from ..core import ccdtest

            for c1, c2, l1, l2 in ccdtest.read_regions(badpix):
                c1, c2 = max(1, c1), min(params.ncols, c2)
                l1, l2 = max(1, l1), min(params.nlines, l2)
                edit(f"{im}[{c1}:{c2},{l1}:{l2}]", "replace", params.badval)
        hedit = _hedit(params.instrument)
        hedit(im, "exptime", exp, "real")
        if params.ccdtype:
            hedit(im, "imagetyp", params.ccdtype, "string")
        for key in ("datasec", "trimsec", "biassec"):
            if getattr(params, key):
                hedit(im, key, getattr(params, key), "string")
        if params.filter:
            hedit(im, "subset", params.filter, "string")
        return TaskResult(outputs=[_split(im)[0]], steps=self.plan(params))


def _hedit(instrument):
    """ccdhedit with the translation given, or the project's."""
    from .ccdred import ccdhedit

    def edit_one(image, parameter, value, kind):
        ccdhedit([image], parameter, str(value) if not isinstance(value, float)
                 else repr(value), kind, instrument)
    return edit_one


# --------------------------------------------------------------- subsection

class SubsectionParams(TaskParams):
    subimage: str = Field(..., description="The frame to make.",
                          json_schema_extra={"positional": True})
    image: str = Field(..., description="The full frame it is read from.",
                       json_schema_extra={"positional": True})
    ncols: int = Field(82, ge=1, description="Its columns.")
    nlines: int = Field(50, ge=1, description="Its lines.")
    ccdsec: str = Field("[26:75,26:75]", description="The part of the chip read out.")
    datasec: str = Field("[1:50,1:50]", description="Where it goes in the new frame.")
    trimsec: str = Field("", description="Its trim section.")
    biassec: str = Field("[51:82,1:50]", description="Its bias section.")
    overwrite: bool = Field(False, description="Make it again if it is there.")
    biasval: float | None = Field(None, description="The bias level; left out, "
                                                    "artobs's (as IRAF's reads it).")
    sigma: float | None = Field(None, ge=0, description="The bias's noise; left out, "
                                                        "artobs's.")
    instrument: str | None = Field(None, description="Translation for the header; the "
                                                     "project's when not given.")


@register
class Subsection(Task[SubsectionParams]):
    name = "subsection"
    package = "ccdtest"
    description = "Make a frame read out from a section of the chip, from a full one."
    Params = SubsectionParams

    def plan(self, params: SubsectionParams) -> list[Step]:
        return [Step("make", f"{params.subimage}: {params.image}{params.ccdsec}",
                     why="A detector read out in part: the header says where the frame "
                         "sits on the chip (CCDSEC), and the calibrations of the whole "
                         "chip still apply.")]

    def execute(self, params: SubsectionParams) -> TaskResult:
        im = params.subimage
        if _exists(im):
            if not params.overwrite:
                return TaskResult(outputs=[], steps=self.plan(params))
            _split(im)[0].unlink()
        copy_into(f"{params.image}[1:{params.ncols},1:{params.nlines}]", im)
        copy_into(params.image + params.ccdsec, im + params.datasec)
        biasval, sigma = _artobs(params)
        if biasval != 0:
            edit(im + params.biassec, "replace", biasval, 0.0, sigma, 0)
        hedit = _hedit(params.instrument)
        for key in ("ccdsec", "datasec", "trimsec", "biassec"):
            hedit(im, key, getattr(params, key), "string")
        return TaskResult(outputs=[_split(im)[0]], steps=self.plan(params))


def _artobs(params) -> tuple[float, float]:
    """subsection reads artobs.biasval and artobs.sigma: the project's artobs, else its
    defaults -- unless given here."""
    from ..core import params as store

    found, _ = store.project_params("artobs")
    defaults = ArtObsParams.model_fields
    biasval = params.biasval if params.biasval is not None else float(
        found.get("biasval", defaults["biasval"].default))
    sigma = params.sigma if params.sigma is not None else float(
        found.get("sigma", defaults["sigma"].default))
    return biasval, sigma
