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
