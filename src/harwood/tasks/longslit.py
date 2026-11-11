"""longslit -- IRAF's twodspec.longslit: the calibrations of a long-slit frame.

`response` and `illumination` are what a long-slit flat needs: the dome flat
divided by a smooth fit to its own spectrum, which leaves the pixel-to-pixel
response, and the variation along the slit from a twilight sky, which the dome
does not light as the sky does. The product of the two is the flat ccdproc
divides by. The arithmetic is core/longslit.py, a port of IRAF's.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

FUNCTIONS = Literal["spline3", "legendre", "chebyshev", "spline1"]

WHY = {
    "normalization": "The normalization image averaged across the slit: its spectrum -- "
                     "the lamp's colour times the spectrograph's response along the "
                     "dispersion.",
    "fit": "A smooth curve through that spectrum, ICFIT's: what the flat varies by from "
           "wavelength to wavelength, without the pixel-to-pixel part.",
    "divide": "The calibration over the fitted spectrum is the response: near 1, with the "
              "pixel-to-pixel variation and the slit's profile left in it.",
    "bins": "The image averaged along the dispersion in bins: the profile along the slit "
            "at a few wavelengths.",
    "fits": "Each bin's profile fitted, normalized near the middle of the slit, and "
            "interpolated between the bins over the whole image.",
    "write": "In real, a new image the size of the whole input, ones outside a section; an "
             "image that already exists is written only where the section is, as IRAF "
             "writes it.",
}


def _fit_fields():
    return dict(
        sample=Field("*", description="Points to fit, as ranges: '10:50 60:110'."),
        naverage=Field(1, description="Points averaged (positive) or medianed "
                                      "(negative) into each one fitted."),
        function=Field("spline3", description="The fitting function."),
        order=Field(1, ge=1, description="Order: polynomial terms, or spline pieces."),
        low_reject=Field(0.0, ge=0.0, description="Low rejection, in sigmas of the fit. "
                                                  "0: none."),
        high_reject=Field(0.0, ge=0.0, description="High rejection, in sigmas of the fit. "
                                                   "0: none."),
        niterate=Field(1, ge=0, description="Rejection iterations."),
        grow=Field(0.0, ge=0.0, description="Rejection growing radius, in pixels."),
    )


class _Common(TaskParams):
    interactive: bool = Field(False, description="Fit by hand, in the icfit window (IRAF's "
                                                 "interactive=yes).")
    cursor: str = Field("", description="The fits' keys from a file, 'x y 1 key' or "
                                        "':command' a line, each fit up to its q -- IRAF's "
                                        "graphics cursor file.")
    sample: str = _fit_fields()["sample"]
    naverage: int = _fit_fields()["naverage"]
    function: FUNCTIONS = _fit_fields()["function"]
    order: int = _fit_fields()["order"]
    low_reject: float = _fit_fields()["low_reject"]
    high_reject: float = _fit_fields()["high_reject"]
    niterate: int = _fit_fields()["niterate"]
    grow: float = _fit_fields()["grow"]
    dispaxis: Literal[1, 2] | None = Field(
        None, description="1 if the dispersion runs along the lines, 2 along the columns. "
                          "Left out, the DISPAXIS keyword.")


class ResponseParams(_Common):
    calibration: list[str] = Field(
        ..., description="Calibration images (the flats). A section is fitted and "
                         "written as a section.",
        json_schema_extra={"positional": True})
    normalization: list[str] = Field(
        ..., description="Normalization images, one each: usually the flats themselves.",
        json_schema_extra={"positional": True})
    response: list[str] = Field(
        ..., description="Response images, one each, the extension written.",
        json_schema_extra={"positional": True})
    threshold: float | None = Field(
        None, description="Below this -- in the fit or in the pixel -- the response is "
                          "1. Empty is IRAF's INDEF: none.")


class IlluminationParams(_Common):
    images: list[str] = Field(
        ..., description="Calibration images (the sky flats). A section is fitted and "
                         "written as a section.",
        json_schema_extra={"positional": True})
    illuminations: list[str] = Field(
        ..., description="Illumination images, one each, the extension written.",
        json_schema_extra={"positional": True})
    bins: str = Field("", description="Bins along the dispersion, as ranges: '1:200 "
                                      "400:600'. Empty: nbins equal ones.")
    nbins: int = Field(5, ge=1, description="Equal bins when --bins is empty.")
    interpolator: Literal["nearest", "linear", "poly3", "poly5", "spline3"] = Field(
        "poly3", description="Between the bins. poly3 and spline3 need four bins and "
                             "poly5 six: with fewer, IRAF uses a lower one.")


# ------------------------------------------------------------------ reading

class _Image:
    def __init__(self, spec):
        import numpy as np

        from ..core import images, imsum, pixels

        hdul, hdu = images.open_hdu(images.ImageSpec(spec.path, spec.extension))
        try:
            stored = np.array(hdu.data)
            self.header = images.inherited(hdul, hdu)
        finally:
            hdul.close()
        if stored.ndim != 2:
            advice.fail(f"{spec} is not a 2-D image",
                        "response and illumination calibrate long-slit frames")
        self.full = imsum.convert(imsum.ieee_in(stored), np.float32)
        self.cut = (pixels.parse_section(spec.section.strip("[]"), stored.shape)
                    if spec.section else (slice(None), slice(None)))
        self.data = self.full[self.cut]
        self.spec = spec


def _lists(*groups):
    from ..core import images

    expanded = [images.expand(g) for g in groups]
    if any(not e for e in expanded):
        advice.fail("an image list matches nothing", "check the names",
                    FileNotFoundError)
    return expanded


def _laxis(image: _Image, given) -> int:
    from ..core import longslit

    try:
        return longslit.dispersion_axis(image.header, given)
    except longslit.LongslitError:
        advice.fail(f"{image.spec} has no DISPAXIS keyword",
                    "give --dispaxis 1 (dispersion along the lines) or 2 (along the "
                    "columns), or add DISPAXIS with hedit")


def _write(target: Path, image: _Image, values, cards: dict, history: str,
           warnings: list):
    """ls_immap: a new image the size of the input's root, ones outside the
    section; an existing one written only in the section."""
    import numpy as np

    from ..core import images, mef

    if target.exists():
        current, header = images.read(str(target))
        out = np.array(current, dtype=np.float32)
        warnings.append(advice.note(
            f"{target} existed: written where {image.spec} is, the rest left as it was",
            "remove it first for a new image, ones outside the section"))
    else:
        out = np.ones(image.full.shape, dtype=np.float32)
        header = image.header.copy()
    out[image.cut] = values
    for key, value in cards.items():
        header[key] = value
    # IRAF's FITS kernel stamps every write: ccdproc reads it as the image's time,
    # and a CCDMEANT copied from the input is older -- the mean is computed again.
    header["IRAF-TLM"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    header.add_history(history)
    files, notes = mef.save(target, out, header, None, None, keep_type=True)
    warnings.extend(notes)
    return files


_CURSORS: dict[str, list[list[str]]] = {}


def _keys(cursor: str) -> list[str]:
    """The next fit's keys from the cursor file, up to and with its q."""
    from .onedtools import _SFT_CURSORS, _sft_keys

    _SFT_CURSORS.update(_CURSORS)
    keys = _sft_keys(cursor)
    _CURSORS.update(_SFT_CURSORS)
    return keys


def _spec(params) -> dict:
    return {k: getattr(params, k) for k in ("function", "order", "sample", "naverage",
                                            "low_reject", "high_reject", "niterate",
                                            "grow")}


def _by_hand(params, x, y, spec: dict, title: str, xlabel: str):
    """icg_fit: the window or the cursor file; the parameters it leaves (the next
    fit starts from them, as IRAF's ic structure does) and the weights."""
    import numpy as np

    from ..core import icfit, icfitsession

    fitspec = icfit.FitSpec(**spec)
    if params.cursor:
        session = icfitsession.IcfitSession(x.astype(float), y.astype(float), fitspec,
                                            np.ones(x.size), title)
        session.run_commands("\n".join(_keys(params.cursor)))
    else:
        from ..viewer import icfit_panel

        session = icfit_panel.make(x.astype(float), y.astype(float), fitspec,
                                   title=title, weights=np.ones(x.size))
    done = session.spec
    return ({key: getattr(done, key) for key in spec},
            np.asarray(session.fitted_wts, np.float32))


def _fit(params, x, y, spec: dict, title: str, xlabel: str, text: list):
    from ..core import longslit

    weights = None
    if params.interactive or params.cursor:
        spec, weights = _by_hand(params, x, y, spec, title, xlabel)
    values, ic = longslit.fitted(x, y, spec, weights)
    text.extend(ic.messages)
    return values, spec


# ----------------------------------------------------------------- response

@register
class Response(Task[ResponseParams]):
    """IRAF's response, from noao/twodspec/longslit/response.x: the calibration
    image over an ICFIT fit to the normalization image's spectrum. Checked pixel
    for pixel against IRAF 2.18."""

    name = "response"
    package = "longslit"
    description = "The response of a long-slit flat: the flat over a fit to its spectrum."
    Params = ResponseParams

    def plan(self, params: ResponseParams) -> list[Step]:
        return [Step("normalization", ", ".join(params.normalization),
                     why=WHY["normalization"]),
                Step("fit", f"{params.function} order {params.order}"
                     + (" by hand" if params.interactive or params.cursor else ""),
                     why=WHY["fit"]),
                Step("divide", ", ".join(params.calibration)
                     + (f", 1 below {params.threshold:g}" if params.threshold is not None
                        else ""), why=WHY["divide"]),
                Step("write", ", ".join(params.response), why=WHY["write"])]

    def execute(self, params: ResponseParams) -> TaskResult:
        from ..core import longslit

        cals, norms = _lists(params.calibration, params.normalization)
        outs = [Path(name) for group in params.response for name in group.split(",")
                if name.strip()]
        if not (len(cals) == len(norms) == len(outs)):
            advice.fail("Image lists do not match",
                        f"{len(cals)} calibration, {len(norms)} normalization and "
                        f"{len(outs)} response images: give one of each for each flat")
        spec, text, written, warnings, rows = _spec(params), [], [], [], []
        for cal_spec, norm_spec, target in zip(cals, norms, outs, strict=True):
            cal, norm = _Image(cal_spec), _Image(norm_spec)
            laxis = _laxis(cal, params.dispaxis)
            x, spectrum = longslit.normalization(norm.data, laxis)
            title = (f"Fit the normalization spectrum for {cal_spec}\n"
                     f"{str(cal.header.get('OBJECT', '')).strip()}")
            fit, spec = _fit(params, x, spectrum, spec, title,
                             "Column" if laxis == 1 else "Line", text)
            values = longslit.normalize(cal.data, fit, laxis, params.threshold)
            written += _write(target, cal, values, {"CCDMEAN": 1.0},
                              f"Response determined from {norm_spec}.", warnings)
            rows.append({"calibration": str(cal_spec), "normalization": str(norm_spec),
                         "response": str(target), "function": spec["function"],
                         "order": spec["order"]})
        data: dict = {"rows": rows}
        if text:
            data["text"] = text
        return TaskResult(outputs=written, steps=self.plan(params), data=data,
                          warnings=warnings, history=[f"response: {len(rows)} image(s)"])


# ------------------------------------------------------------- illumination

@register
class Illumination(Task[IlluminationParams]):
    """IRAF's illumination, from noao/twodspec/longslit/illumination.x: the slit
    profile in bins along the dispersion, each fitted with ICFIT, normalized and
    interpolated over the image. Checked pixel for pixel against IRAF 2.18."""

    name = "illumination"
    package = "longslit"
    description = "The illumination along the slit of a long-slit sky flat."
    Params = IlluminationParams

    def plan(self, params: IlluminationParams) -> list[Step]:
        return [Step("bins", params.bins or f"{params.nbins} equal", why=WHY["bins"]),
                Step("fits", f"{params.function} order {params.order}, "
                             f"{params.interpolator} between the bins"
                     + (" by hand" if params.interactive or params.cursor else ""),
                     why=WHY["fits"]),
                Step("write", ", ".join(params.illuminations), why=WHY["write"])]

    def execute(self, params: IlluminationParams) -> TaskResult:
        import numpy as np

        from ..core import longslit, msi

        (ims,) = _lists(params.images)
        outs = [Path(name) for group in params.illuminations for name in group.split(",")
                if name.strip()]
        if len(ims) != len(outs):
            advice.fail("The number of input and output images are not the same.",
                        f"{len(ims)} images and {len(outs)} illuminations: give one each")
        spec, text, written, warnings, rows = _spec(params), [], [], [], []
        for im_spec, target in zip(ims, outs, strict=True):
            image = _Image(im_spec)
            laxis = _laxis(image, params.dispaxis)
            slit = 2 if laxis == 1 else 1
            npts = image.data.shape[0] if slit == 1 else image.data.shape[1]
            given = params.bins.strip()
            if params.cursor:
                given = _cursor_bins(params.cursor, given, npts, params.nbins, text)
            try:
                bins = longslit.ranges(given or longslit.default_bins(npts, params.nbins),
                                       1, npts)
            except longslit.LongslitError:
                advice.fail("Bad range string for parameter bins",
                            "give ranges along the dispersion, '1:200 400:600'")
            if not bins:
                advice.fail("Bad range string for parameter bins",
                            f"the bins are outside 1..{npts}, the dispersion axis")
            x, centres, profiles = longslit.binned(image.data, slit, bins)
            fits_ = []
            for number, profile in enumerate(profiles, start=1):
                title = (f"Determine illumination interactively for {im_spec} at bin "
                         f"{number}\n{str(image.header.get('OBJECT', '')).strip()}")
                values, spec = _fit(params, x, profile, spec, title,
                                    "Column" if slit == 1 else "Line", text)
                fits_.append(values)
            try:
                values = longslit.expand(image.data.shape, slit, x, centres,
                                         np.vstack(fits_), params.interpolator)
            except msi.MsiError as error:
                advice.fail(str(error), "use more bins, or a lower --interpolator")
            written += _write(target, image, values,
                              {"CCDMEAN": 1.0, "MKILLUM": "Illumination correction "
                               f"determined from {im_spec}."},
                              f"illumination: {len(bins)} bins of {im_spec}", warnings)
            rows.append({"image": str(im_spec), "illumination": str(target),
                         "bins": " ".join(f"{a}:{b}" for a, b in bins),
                         "interpolator": longslit.interpolator(params.interpolator,
                                                               len(bins))})
        data: dict = {"rows": rows}
        if text:
            data["text"] = text
        return TaskResult(outputs=written, steps=self.plan(params), data=data,
                          warnings=warnings, history=[f"illumination: {len(rows)} image(s)"])


def _cursor_bins(cursor: str, given: str, npts: int, nbins: int, text: list) -> str:
    """il_gsetbins from the cursor file: the keys before the bins graph's q --
    ':bins ranges' sets them, ':bins' and ':show' print them, i sets '*'."""
    from ..core import longslit

    current = given or longslit.default_bins(npts, nbins)
    for line in _keys(cursor):
        key = line.split(None, 3)[3].strip()
        if key == "q":
            break
        if key == "i":
            current = "*"
        elif key.startswith(":"):
            words = key[1:].split(None, 1)
            name = words[0] if words else ""
            rest = words[1].strip() if len(words) > 1 else ""
            if name in ("show",) or (name == "bins" and not rest):
                text.append(f"bins = {current}")
            elif name == "bins":
                current = rest
        elif key == "s":
            advice.fail("the s key needs the bins graph's window",
                        "set the bins with ':bins 1:200 400:600' in the cursor file, or "
                        "--bins")
    return current


# ---------------------------------------------------------------- fitcoords

FCWHY = {
    "read": "The features identify and reidentify measured on the arc's lines or columns: "
            "the pixel each lamp line fell on and its wavelength, from database/id<image>.",
    "fit": "One surface through all of them, the wavelength as a function of the column "
           "and the line: a low-order fit first, then the full orders with cross terms on "
           "what it leaves, the two added (gsurfit, in double precision).",
    "write": "Appended to database/fc<fitname>, the surface as gsurfit saves it: what "
             "transform reads.",
}


class FitcoordsParams(TaskParams):
    images: list[str] = Field(
        ..., description="Arcs whose identify records are fitted, by name: arc or "
                         "arc.fits.", json_schema_extra={"positional": True})
    fitname: str = Field("", description="Name of the fit. Each image's fit is "
                                         "<fitname><image>; with --combine, the one fit's.")
    interactive: bool = Field(False, description="Fit by hand (IRAF's interactive=yes): "
                                                 "not ported yet.")
    combine: bool = Field(False, description="One fit of every image's features.")
    database: Path = Field(Path("."), description="Reduction directory holding database/.")
    deletions: str = Field("deletions.db", description="Points fitted with weight zero, "
                                                       "'user position' a line. Empty: none.")
    function: Literal["chebyshev", "legendre"] = Field("chebyshev",
                                                       description="The surface's functions.")
    xorder: int = Field(6, ge=2, description="Order of the surface in x (the columns).")
    yorder: int = Field(6, ge=2, description="Order of the surface in y (the lines).")
    logfiles: str = Field("STDOUT,logfile", description="Where the log goes: STDOUT is "
                                                        "shown, a file is appended to.")


def _sysid() -> str:
    import getpass
    import socket

    from ..core import iraffmt
    from ..core.provenance import _version

    return (f"harwood {_version()} {getpass.getuser()}@{socket.gethostname()} "
            f"{iraffmt.cnvtime()}")


def _log(logfiles: str, lines: list[str], shown: list[str], workdir: Path = Path(".")):
    """IRAF's logfiles: STDOUT shown, any other name appended to."""
    for name in (n.strip() for n in logfiles.replace(" ", ",").split(",")):
        if not name:
            continue
        if name in ("STDOUT", "STDERR"):
            shown.extend(lines)
            continue
        with open(workdir / name, "a", encoding="utf-8") as out:
            out.writelines(line + "\n" for line in lines)


def _header_of(root: str):
    """The frame a record names, as fc_getcoords maps it: header and shape."""
    from astropy.io import fits

    from ..core import fitcoords as fc

    for candidate in (root, f"{root}.fits", f"{root}.fit", f"{root}.fz"):
        if Path(candidate).is_file():
            with fits.open(candidate, memmap=False) as hdul:
                hdu = next((h for h in hdul if h.data is not None and h.data.ndim == 2),
                           None)
                if hdu is None:
                    raise fc.FitcoordsError(f"{candidate}: not a 2-D image")
                return hdu.header.copy(), hdu.data.shape
    raise fc.FitcoordsError(f"Cannot open image ({root})")


@register
class Fitcoords(Task[FitcoordsParams]):
    """IRAF's fitcoords, from noao/twodspec/longslit/lstrans: a surface fitted to the
    wavelengths of the features identify and reidentify measured along a long slit,
    written for transform. Checked against IRAF 2.18: the surface's coefficients to
    the last digit, and the log."""

    name = "fitcoords"
    package = "longslit"
    description = "Fit a surface to the wavelengths measured along a long-slit arc."
    Params = FitcoordsParams

    def plan(self, params: FitcoordsParams) -> list[Step]:
        name = params.fitname if params.combine else f"{params.fitname}<image>"
        return [Step("read", ", ".join(params.images), why=FCWHY["read"]),
                Step("fit", f"{params.function}, orders {params.xorder} and "
                            f"{params.yorder}", why=FCWHY["fit"]),
                Step("write", f"database/fc{name}", why=FCWHY["write"])]

    def execute(self, params: FitcoordsParams) -> TaskResult:
        from ..core import fitcoords as fc

        if params.interactive:
            advice.fail("fitcoords by hand (IRAF's igsfit graph) is not ported yet",
                        "run it with --no-interactive, and look at the fit's residuals "
                        "in the log; delete points with --deletions")
        fitname = params.fitname.strip()
        if params.combine and not fitname:
            advice.fail("Fit name not specified", "give --fitname for the combined fit")
        names = [fc.strip_extension(str(spec).split("[")[0])
                 for group in params.images for spec in group.split(",") if spec.strip()]
        groups = [(fitname, names)] if params.combine else \
            [(f"{fitname}{n}", [n]) for n in names]
        dbdir = Path(params.database) / "database"
        shown: list[str] = []
        warnings: list[str] = []
        written: list[Path] = []
        rows = []
        deletions = Path(params.deletions) if params.deletions.strip() else None
        for name, images in groups:
            _log(params.logfiles, fc.header_lines(_sysid(), name, "database", images),
                 shown)
            try:
                coords = fc.get_coords(images, dbdir, _header_of)
                fc.read_deletions(deletions, coords)
                sf, said = fc.fit(coords, params.function, params.xorder, params.yorder)
            except (fc.FitcoordsError, ValueError) as error:
                if params.combine:
                    advice.fail(str(error), "identify and reidentify the arc first: "
                                "the features are read from database/id<image>")
                warnings.append(advice.warn(f"{name}: {error}",
                                            "identify and reidentify the arc first: the "
                                            "features are read from database/id<image>"))
                continue
            for text in coords.warnings + said:
                warnings.append(advice.warn(text, "check the database's records"))
            _log(params.logfiles, fc.result_lines(coords, sf, params.function,
                                                   params.xorder, params.yorder), shown)
            path = fc.write_fc(dbdir, name, coords.axis, coords.units, sf)
            if path not in written:
                written.append(path)
            pts = coords.points
            used = pts[fc.W] > 0
            rms = float(((pts[fc.R][used].astype(float) ** 2).mean()) ** 0.5) \
                if used.any() else float("nan")
            rows.append({"fit": name, "axis": coords.axis, "points": coords.n,
                         "rms": round(rms, 6), "units": coords.units or ""})
        data: dict = {"rows": rows}
        if shown:
            data["text"] = shown
        return TaskResult(outputs=written, steps=self.plan(params), data=data,
                          warnings=warnings, history=[f"fitcoords: {len(rows)} fit(s)"])


# ---------------------------------------------------------------- transform

TRWHY = {
    "fits": "The surfaces fitcoords wrote, database/fc<fitname>: the wavelength (and, "
            "with a fit of the other axis, the position along the slit) of every input "
            "pixel.",
    "grid": "The surfaces inverted on every tenth output pixel by Newton's method, and "
            "interpolated bilinearly in between: where each output pixel falls on the "
            "input.",
    "resample": "The input interpolated there; with flux on, multiplied by the input area "
                "the output pixel covers, so that counts are kept.",
    "write": "A new image on straight lines of constant wavelength, with a linear world "
             "system for onedspec.",
}


class TransformParams(TaskParams):
    input: list[str] = Field(..., description="Images to transform.",
                             json_schema_extra={"positional": True})
    output: list[str] = Field(..., description="Output images, one each.",
                              json_schema_extra={"positional": True})
    fitnames: str = Field(..., description="Fits in database/fc<name>, comma separated: "
                                           "one for the wavelength, and one for the slit "
                                           "if there is one. Empty (''): each image's "
                                           "own linear world system.",
                          json_schema_extra={"positional": True})
    database: Path = Field(Path("."), description="Reduction directory holding database/.")
    minput: str = Field("", description="Masks of the inputs (above zero bad), one or one "
                                        "each.")
    moutput: str = Field("", description="Output masks, one each: 1 where the input was "
                                         "bad or off the image.")
    interptype: Literal["nearest", "linear", "poly3", "poly5", "spline3"] = Field(
        "spline3", description="How the input is interpolated.")
    x1: float | None = Field(None, description="First output x coordinate. Empty: from "
                                               "the fits' range.")
    x2: float | None = Field(None, description="Last output x coordinate.")
    dx: float | None = Field(None, description="Output x interval; given, it sets nx.")
    nx: int | None = Field(None, description="Output columns. Empty: the input's.")
    xlog: bool = Field(False, description="Logarithmic x intervals.")
    y1: float | None = Field(None, description="First output y coordinate.")
    y2: float | None = Field(None, description="Last output y coordinate.")
    dy: float | None = Field(None, description="Output y interval; given, it sets ny.")
    ny: int | None = Field(None, description="Output lines. Empty: the input's.")
    ylog: bool = Field(False, description="Logarithmic y intervals.")
    flux: bool = Field(True, description="Conserve flux per pixel.")
    blank: float | None = Field(None, description="Value of output pixels off the input. "
                                                  "Empty: the input's edge, extended.")
    logfiles: str = Field("STDOUT,logfile", description="Where the log goes: STDOUT is "
                                                        "shown, a file is appended to.")
    dispaxis: Literal[1, 2] | None = Field(
        None, description="1 if the dispersion runs along the lines, 2 along the columns "
                          "(IRAF's longslit.dispaxis). Left out, the DISPAXIS keyword.")


def _mask_name(name: str) -> tuple[Path, str]:
    """xt_maskname: a .pl file, or a FITS file holding the mask in a pl extension
    -- what the BPM card then names."""
    if name.endswith(".pl"):
        return Path(name), name
    path = Path(name if name.endswith((".fits", ".fit")) else f"{name}.fits")
    return path, f"{path}[pl,type=mask]"


def _write_mask(path: Path, mask, header) -> None:
    """A mask as IRAF writes one into a FITS file: an empty primary and the pixels,
    PLIO-compressed, in an extension named pl."""
    import numpy as np
    from astropy.io import fits

    ext = fits.CompImageHDU(np.asarray(mask, np.int32), header=header,
                            compression_type="PLIO_1", name="pl")
    fits.HDUList([fits.PrimaryHDU(), ext]).writeto(path, overwrite=True)


@register
class Transform(Task[TransformParams]):
    """IRAF's transform, from noao/twodspec/longslit/lstrans: a long-slit frame
    resampled onto the coordinates of fitcoords' surfaces -- the lines of the lamp
    straight, the wavelength linear. Checked pixel for pixel against IRAF 2.18."""

    name = "transform"
    package = "longslit"
    description = "Resample a long-slit frame onto straight lines of constant wavelength."
    Params = TransformParams

    def plan(self, params: TransformParams) -> list[Step]:
        return [Step("fits", params.fitnames, why=TRWHY["fits"]),
                Step("grid", "every 10th output pixel", why=TRWHY["grid"]),
                Step("resample", f"{params.interptype}"
                     + (", flux conserved" if params.flux else ""), why=TRWHY["resample"]),
                Step("write", ", ".join(params.output), why=TRWHY["write"])]

    def execute(self, params: TransformParams) -> TaskResult:
        import numpy as np

        from ..core import fitcoords as fc
        from ..core import images, imsum, iraffmt, longslit, mef, pixels, pixmask
        from ..core import transform as tr

        ins, given = [], []
        for group in params.input:
            for piece in images.split_list(group):
                found = images.expand(piece)
                ins += found
                given += [piece] if len(found) == 1 else [str(s) for s in found]
        outs = [p for g in params.output for p in images.split_list(g) if p.strip()]
        if len(ins) != len(outs):
            advice.fail("Number of input and output images differ",
                        f"{len(ins)} inputs and {len(outs)} outputs: give one each")
        minputs = images.split_list(params.minput) if params.minput.strip() else []
        moutputs = images.split_list(params.moutput) if params.moutput.strip() else []
        if len(minputs) > 1 and len(minputs) != len(ins):
            advice.fail("Can't associate input masks with input images",
                        "give one mask, or one for each input")
        if moutputs and len(moutputs) != len(ins):
            advice.fail("Number output masks differ from input",
                        "give one output mask for each input")
        names = [n.strip() for n in params.fitnames.split(",") if n.strip()]
        usewcs = not names              # no fits: each image's own world system
        dbdir = Path(params.database) / "database"
        wanted = tr.Output(params.x1, params.x2, params.dx, params.nx, params.xlog,
                           params.y1, params.y2, params.dy, params.ny, params.ylog)
        setup = None
        if not usewcs:
            try:
                fits_ = [fc.read_fc(dbdir, n) for n in names]
                usf, vsf, un = tr.surfaces(fits_)
                setup = tr.setup(usf, vsf, un, wanted, params.flux)
            except (fc.FitcoordsError, tr.TransformError, ValueError) as error:
                advice.fail(str(error), "run fitcoords first: transform reads "
                                        "database/fc<fitname>")
        shown, written, warnings, rows = [], [], [], []
        listed = False                  # clgfil's list is read once, never rewound
        minname = mname = moutname = ""
        for k, (spec, target) in enumerate(zip(ins, outs, strict=True)):
            if k < len(minputs):
                minname = minputs[k]
            if k < len(moutputs):
                moutname = moutputs[k]
            hdul, hdu = images.open_hdu(images.ImageSpec(spec.path, spec.extension))
            try:
                stored = np.array(hdu.data)
                header = images.inherited(hdul, hdu)
            finally:
                hdul.close()
            if spec.section:
                stored = stored[pixels.parse_section(spec.section.strip("[]"),
                                                     stored.shape)]
            if stored.ndim != 2:
                advice.fail(f"{spec} is not a 2-D image", "transform resamples long-slit "
                                                          "frames")
            data = imsum.convert(imsum.ieee_in(stored), np.float32)
            if usewcs:
                try:
                    setup = tr.setup_wcs(tr.ImageWcs(header, data.shape[1],
                                                     data.shape[0]), wanted, params.flux)
                except (ValueError, np.linalg.LinAlgError) as error:
                    advice.fail(f"{spec}: {error}", "the image's world system has to be "
                                "linear and invertible to transform by it")
            o = setup.output
            mask = None
            if minname:
                mask = pixmask.read(minname, header, data.shape)
                mname = minname
            mpath = mcard = None
            if moutname:
                mpath, mcard = _mask_name(moutname)
            lines = ["", _sysid(), f"  Transform {given[k]} to {target}."]
            if mpath is not None:
                lines.append(f"  Transform mask {mname} to {mcard}." if mask is not None
                             else f"  Output mask is {mcard}.")
            if params.flux:
                lines.append("  Conserve flux per pixel.")
            lines.append("  Transforming using image WCS." if usewcs else
                         "  User coordinate transformations:")
            # the names go to the first log of the first image only: IRAF reads the
            # list to its end there and never goes back
            for name in params.logfiles.replace(" ", ",").split(","):
                if not name.strip():
                    continue
                these = list(lines)
                if not listed and not usewcs:
                    these += [f"    {n}" for n in names]
                    listed = True
                these.append(f"  Interpolation is {params.interptype}.")
                these.append(f"  Out of bounds pixel value is {iraffmt.g(params.blank)}."
                             if params.blank is not None else
                             "  Using edge extension for out of bounds pixel values.")
                these.append("  Output coordinate parameters are:")
                these.append(_coords_line("x", o.x1, o.x2, o.dx, o.nx, o.xlog))
                these.append(_coords_line("y", o.y1, o.y2, o.dy, o.ny, o.ylog))
                _log(name, these, shown)
            out, mout = tr.transform(data, setup, params.interptype, params.flux,
                                     params.blank, mask, want_mask=mpath is not None)
            laxis = None
            if not usewcs:
                try:
                    laxis = longslit.dispersion_axis(header, params.dispaxis)
                except longslit.LongslitError:
                    advice.fail(f"{spec} has no DISPAXIS keyword", "give --dispaxis 2 "
                                "(the dispersion along the columns) or 1 (along the "
                                "lines), as IRAF's longslit.dispaxis, or add DISPAXIS "
                                "with hedit")
            for key in tr.OLD_WCS + (() if usewcs else tr.OLD_DISPERSION):
                header.remove(key, ignore_missing=True, remove_all=True)
            header.update(tr.wcs_cards(setup, laxis))
            if mcard is not None:
                header["BPM"] = mcard
            header["IRAF-TLM"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            path = Path(target if target.endswith((".fits", ".fit")) else f"{target}.fits")
            stored_type = stored.dtype if stored.dtype.kind in "iuf" else np.float32
            values = imsum.convert(out, stored_type)
            files, notes = mef.save(path, values, header, None, None, keep_type=True)
            warnings.extend(notes)
            written += files
            if mpath is not None:
                mheader = header.copy()
                for key in ("BPM", "DCLOG1", "DC-FLAG"):
                    mheader.remove(key, ignore_missing=True)
                _write_mask(mpath, imsum.convert(mout, np.int32).clip(0), mheader)
                written.append(mpath)
            rows.append({"input": str(spec), "output": str(path), "nx": o.nx, "ny": o.ny,
                         "y1": round(o.y1, 6), "dy": round(o.dy, 8)})
        data_out: dict = {"rows": rows}
        if shown:
            data_out["text"] = shown
        return TaskResult(outputs=written, steps=self.plan(params), data=data_out,
                          warnings=warnings, history=[f"transform: {len(rows)} image(s)"])


def _coords_line(axis: str, a1, a2, d, n, log) -> str:
    """'    x1 = %10.4g, x2 = %10.4g, dx = %10.4g, nx = %4d, xlog = %b'."""
    from ..core import iraffmt

    def g(v):
        return f"{iraffmt.printf(float(v), 'g', 4, 10):>10s}"

    return (f"    {axis}1 = {g(a1)}, {axis}2 = {g(a2)}, d{axis} = {g(d)}, "
            f"n{axis} = {n:4d}, {axis}log = {'yes' if log else 'no'}")
