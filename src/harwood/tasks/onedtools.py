"""onedspec -- the tools that work on a spectrum once it has wavelengths.

dopcor takes it to the rest frame, specshift moves its zero point, sflip turns it
round, deredden takes the dust out, slist says what each aperture is, rspectext
reads one from text, names spells out a record list, sbands measures bands, lcalib
lists the calibration tables, mkspec makes an artificial one, sinterp
interpolates a table into one, scoords gives one its wavelengths from a list. Each is
IRAF's
task of the same name, ported from its .x and held to what IRAF 2.18 writes, on
harwood's spectra: a table per aperture, the wavelengths in a column
(core/spectrum.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

DOPWHY = {
    "read": "A spectrum with wavelengths: dopcor moves its scale, so it has to have one.",
    "redshift": "A velocity is turned into a redshift relativistically, "
                "sqrt((1+v/c)/(1-v/c)) - 1, so that a fast galaxy and a slow star are "
                "corrected the same way. A redshift read from the header can be undone "
                "with a minus sign: -Z applies 1/(1+z) - 1.",
    "correct": "The rest wavelength is the observed one over 1+z. The correction "
               "replaces the one the spectrum already has, unless --add puts the two "
               "together, (1+z)(1+zold) - 1, as for a heliocentric correction and then "
               "a galaxy's own velocity.",
    "flux": "A flux per unit wavelength at redshift z is (1+z)^3 fainter than at rest "
            "(the factor's default); per unit frequency, (1+z)^1. Off by default: most "
            "of the time the wavelengths are what is wanted.",
    "write": "The header gets a DOPCORnn card saying what was done, as IRAF's does, and "
             "each aperture keeps its z (APZ), so that a second dopcor knows it.",
}


class DopcorParams(TaskParams):
    input: str = Field(..., description="Spectra with wavelengths. Commas, globs and "
                                        "@lists expand.")
    output: str | None = Field(
        None, description="One name per input; the input's own name corrects it in "
                          "place, as IRAF's default does. Left out, each goes to this "
                          "run's own directory.",
        json_schema_extra={"positional": True})
    redshift: str = Field(..., description="A redshift, a velocity in km/s with "
                                           "--isvelocity, or a header keyword holding "
                                           "one; a minus before a keyword undoes it.")
    isvelocity: bool = Field(False, description="The redshift is a velocity, km/s.")
    add: bool = Field(False, description="Add to the correction the spectrum already "
                                         "has, instead of replacing it.")
    dispersion: bool = Field(True, description="Correct the wavelengths.")
    flux: bool = Field(False, description="Correct the fluxes by (1+z)^factor.")
    factor: float = Field(3.0, description="The power of 1+z the fluxes are corrected "
                                           "by.")
    apertures: str = Field("", description="Apertures to correct (1,3-5); all when "
                                           "empty. The others are copied as they are.")
    verbose: bool = Field(False, description="Say what was done to each spectrum.")


@register
class Dopcor(Task[DopcorParams]):
    name = "dopcor"
    package = "onedspec"
    description = "Correct spectra for a redshift or a velocity."
    Params = DopcorParams

    def plan(self, params: DopcorParams) -> list[Step]:
        return [Step("read", params.input, why=DOPWHY["read"]),
                Step("redshift", f"{params.redshift}"
                     + (" km/s" if params.isvelocity else ""), why=DOPWHY["redshift"]),
                Step("correct", "the wavelengths", why=DOPWHY["correct"],
                     skipped=not params.dispersion, reason="--no-dispersion"),
                Step("flux", f"by (1+z)^{params.factor:g}", why=DOPWHY["flux"],
                     skipped=not params.flux, reason="--flux not given"),
                Step("write", str(params.output or "this run's directory"),
                     why=DOPWHY["write"])]

    def execute(self, params: DopcorParams) -> TaskResult:
        from astropy.io import fits

        from ..core import dopcor, images, output, ranges, spectrum

        if not params.dispersion and not params.flux:
            advice.fail("No correction specified",
                        "leave --dispersion on, or give --flux, or both.")
        aps = ranges.parse(params.apertures, -ranges.RNG_MAXINT, ranges.RNG_MAXINT)
        apstr = params.apertures.strip() or "all"
        inputs = [name if Path(name).exists() or not Path(name + ".fits").exists()
                  else name + ".fits"
                  for name in (str(spec) for spec in images.expand(params.input))]
        # IRAF's image names leave the extension out, and are said as they were typed.
        given = [name.removesuffix(".fits") if ".fits" not in params.input else name
                 for name in inputs]
        if not inputs:
            advice.fail(f"'{params.input}' matches nothing.",
                        "check the path; 'harwood sections' shows what a template "
                        "expands to.", FileNotFoundError)
        outs = [part if Path(part).suffix else part + ".fits"
                for part in (p.strip() for p in images.split_list(str(params.output)))] \
            if params.output else []

        steps = self.plan(params)
        said: list[str] = []
        warnings: list = []
        written: list[Path] = []
        for index, name in enumerate(inputs):
            target = Path(outs[index]) if index < len(outs) else \
                output.named_for(name, ".dop.fits")
            try:
                header = fits.getheader(name)
                apertures = spectrum.read(name)
                if any(a.wavelength is None or not len(a.wavelength) or
                       not any(a.wavelength) for a in apertures):
                    advice.fail(f"[{given[index]}] has no dispersion function",
                                "run dispcor on it first: dopcor moves a wavelength "
                                "scale, and this spectrum is still in pixels.")
                z, given = dopcor.redshift(params.redshift, header, params.isvelocity)
            except (KeyError, ValueError) as error:
                # erract (EA_WARN): the image is skipped, the others go on.
                text = str(error.args[0] if error.args else error).splitlines()[0]
                warnings.append(advice.warn(text, "the spectrum was left as it was."))
                said.append(f"Warning: {text}")
                continue
            fcval = dopcor.flux_factor(z, params.factor) if params.flux else None
            zold = znew = 0.0
            for aperture in apertures:
                if aperture.number in aps:
                    zold, znew = dopcor.correct(aperture, z, params.add, params.dispersion,
                                                fcval)
            out_header = header.copy()
            card = dopcor.card_name(out_header)
            out_header[card] = dopcor.log_card(z, given, params.isvelocity,
                                               params.factor if params.flux else None,
                                               apstr)
            spectrum.write(target, apertures, header=out_header, source=name)
            written.append(target)
            if params.verbose:
                said.extend(dopcor.verbose_lines(
                    target.name.removesuffix(".fits") if index < len(outs) else str(target),
                    apstr, z, given, params.isvelocity,
                    params.factor if params.flux else 0.0, params.add, zold, znew))

        steps[-1].measured = {"written": ", ".join(str(p) for p in written) or "nothing"}
        return TaskResult(outputs=written, steps=steps, warnings=warnings,
                          data={"text": said},
                          history=[f"dopcor: z={params.redshift}"
                                   + (" (km/s)" if params.isvelocity else "")])


class SpecshiftParams(TaskParams):
    spectra: str = Field(..., description="Spectra to shift, in place, as IRAF's does. "
                                          "Commas, globs and @lists expand.")
    shift: float = Field(..., description="What to add to the dispersion coordinates: "
                                          "Angstroms, before any dopcor.")
    apertures: str = Field("", description="Apertures to shift (1,3-5); all when empty.")
    verbose: bool = Field(False, description="Say each aperture's start before and after.")


@register
class Specshift(Task[SpecshiftParams]):
    name = "specshift"
    package = "onedspec"
    description = "Shift the wavelength zero point of spectra."
    Params = SpecshiftParams

    def plan(self, params: SpecshiftParams) -> list[Step]:
        return [Step("shift", f"{params.shift:g} added to the dispersion of "
                     f"{params.spectra}",
                     why="The zero point of a dispersion solution is its weakest part: a "
                         "night's flexure or a sky line seen in the wrong place shifts "
                         "every wavelength by the same amount, and this puts it back. "
                         "The shift is in the dispersion's coordinates, so a spectrum "
                         "dopcor has corrected moves by shift/(1+z)."),
                Step("write", "in place", why="As IRAF's specshift, which has no output.")]

    def execute(self, params: SpecshiftParams) -> TaskResult:
        from astropy.io import fits

        from ..core import dopcor, images, ranges, spectrum

        aps = ranges.parse(params.apertures, -ranges.RNG_MAXINT, ranges.RNG_MAXINT)
        names = [name if Path(name).exists() or not Path(name + ".fits").exists()
                 else name + ".fits"
                 for name in (str(spec) for spec in images.expand(params.spectra))]
        if not names:
            advice.fail(f"'{params.spectra}' matches nothing.",
                        "check the path; 'harwood sections' shows what a template "
                        "expands to.", FileNotFoundError)
        steps = self.plan(params)
        said: list[str] = []
        warnings: list = []
        written: list[Path] = []
        g = dopcor.iraffmt.g
        for name in names:
            try:
                apertures = spectrum.read(name)
                header = fits.getheader(name)
            except (OSError, ValueError) as error:
                text = str(error.args[0] if error.args else error).splitlines()[0]
                warnings.append(advice.warn(text, "the spectrum was left as it was."))
                said.append(f"Warning: {text}")
                continue
            for aperture in apertures:
                if aperture.number not in aps or aperture.wavelength is None:
                    continue
                before, after = dopcor.shift_zero(aperture, params.shift)
                # sshift's "%s: shift = %g" waits for a count it never makes, so only
                # the apertures' lines are said.
                if params.verbose:
                    said.append(f"  Aperture {aperture.number}:  {g(before, double=True)}"
                                f" --> {g(after, double=True)}")
            spectrum.write(name, apertures, header=header)
            written.append(Path(name))
        return TaskResult(outputs=written, steps=steps, warnings=warnings,
                          data={"text": said},
                          history=[f"specshift: {params.shift:g}"])


#: The bands of a spectrum that are data, and the columns that are its coordinates.
_DATA = ("flux", "error", "sky", "raw", "dq")
_COORDINATES = ("pixel", "wavelength")


class SflipParams(TaskParams):
    input: str = Field(..., description="Spectra to flip. Commas, globs and @lists expand.")
    output: str | None = Field(
        None, description="One name per input; the input's own name flips it in place. "
                          "Left out, each goes to this run's own directory.",
        json_schema_extra={"positional": True})
    coord_flip: bool = Field(False, description="Flip the coordinates: the first pixel "
                                                "takes the last one's wavelength and "
                                                "position.")
    data_flip: bool = Field(True, description="Flip the data: the first pixel takes the "
                                              "last one's flux, error, sky and raw.")


@register
class Sflip(Task[SflipParams]):
    name = "sflip"
    package = "onedspec"
    description = "Turn spectra round: their data, their coordinates, or both."
    Params = SflipParams

    def plan(self, params: SflipParams) -> list[Step]:
        what = " and ".join(part for part, on in (("the coordinates", params.coord_flip),
                                                  ("the data", params.data_flip)) if on)
        return [Step("flip", what or "nothing: a copy",
                     why="Both, and every value keeps its wavelength while the pixels run "
                         "the other way -- red to the right, say, for a spectrograph that "
                         "puts it on the left. The data alone, and they are put against "
                         "the coordinates the other way round; the coordinates alone, the "
                         "same from the other side. Those two are for a spectrum whose "
                         "data or whose solution was read backwards."),
                Step("write", str(params.output or "this run's directory"),
                     why="A new spectrum, unless the output is the input's own name.")]

    def execute(self, params: SflipParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import images, output, spectrum

        names = [name if Path(name).exists() or not Path(name + ".fits").exists()
                 else name + ".fits"
                 for name in (str(spec) for spec in images.expand(params.input))]
        if not names:
            advice.fail(f"'{params.input}' matches nothing.",
                        "check the path; 'harwood sections' shows what a template "
                        "expands to.", FileNotFoundError)
        outs = [part if Path(part).suffix else part + ".fits"
                for part in (p.strip() for p in images.split_list(str(params.output)))] \
            if params.output else []
        written: list[Path] = []
        for index, name in enumerate(names):
            if outs and index >= len(outs):
                break               # t_sflip stops when the output list runs out
            target = Path(outs[index]) if outs else output.named_for(name, ".flip.fits")
            header = fits.getheader(name)
            apertures = spectrum.read(name)
            columns = (_COORDINATES if params.coord_flip else ()) + \
                (_DATA if params.data_flip else ())
            for aperture in apertures:
                for column in columns:
                    values = getattr(aperture, column, None)
                    if values is not None:
                        setattr(aperture, column, np.asarray(values)[::-1].copy())
            spectrum.write(target, apertures, header=header, source=name)
            written.append(target)
        return TaskResult(outputs=written, steps=self.plan(params),
                          history=["sflip: " + ", ".join(
                              part for part, on in (("coordinates", params.coord_flip),
                                                    ("data", params.data_flip)) if on)])


class DereddenParams(TaskParams):
    input: str = Field(..., description="Spectra with wavelengths. Commas, globs and "
                                        "@lists expand.")
    output: str | None = Field(
        None, description="One name per input; the input's own name corrects it in "
                          "place. Left out, each goes to this run's own directory.",
        json_schema_extra={"positional": True})
    value: float = Field(..., description="The extinction: A(V), E(B-V) or c, as --type "
                                          "says.")
    R: float = Field(3.1, description="A(V)/E(B-V): 3.1 in the diffuse interstellar "
                                      "medium of the Galaxy.")
    type: Literal["A(V)", "E(B-V)", "c"] = Field(
        "E(B-V)", description="What --value is: A(V), E(B-V), or c, the logarithmic "
                              "extinction at H-beta.")
    apertures: str = Field("", description="Apertures to correct (1,3-5); all when "
                                           "empty. The others are copied as they are.")
    override: bool = Field(False, description="Correct a spectrum whose header says it "
                                              "has been corrected already.")
    uncorrect: bool = Field(True, description="With --override, take the old correction "
                                              "out first, as its DEREDDEN card says it.")


@register
class Deredden(Task[DereddenParams]):
    name = "deredden"
    package = "onedspec"
    description = "Correct spectra for interstellar extinction."
    Params = DereddenParams

    def plan(self, params: DereddenParams) -> list[Step]:
        return [Step("law", f"Cardelli, Clayton & Mathis, R = {params.R:g}",
                     why="Dust dims the blue more than the red, by a law that depends on "
                         "the grains along the line of sight; R = A(V)/E(B-V) says which, "
                         "3.1 for the Galaxy's diffuse medium. CCM (1989) is IRAF's "
                         "law, from 1000 to 33333 A."),
                Step("correct", f"{params.type} = {params.value:g}",
                     why="Each pixel times 10^(0.4 A(V) A(lambda)/A(V)). c, the "
                         "logarithmic extinction at H-beta, is A(V) = R c (0.61 + 0.024 c)."),
                Step("write", str(params.output or "this run's directory"),
                     why="With a DEREDDEN card that says what was done and when, so that "
                         "the spectrum is not corrected twice.")]

    def execute(self, params: DereddenParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import deredden, images, output, ranges, spectrum

        av, words = deredden.to_av(params.value, params.R, params.type)
        log = f"{deredden.cnvdate()} {words}"
        aps = ranges.parse(params.apertures, -ranges.RNG_MAXINT, ranges.RNG_MAXINT)
        if params.apertures.strip():
            log += f" ap={params.apertures.strip()}"
        names = [name if Path(name).exists() or not Path(name + ".fits").exists()
                 else name + ".fits"
                 for name in (str(spec) for spec in images.expand(params.input))]
        if not names:
            advice.fail(f"'{params.input}' matches nothing.",
                        "check the path; 'harwood sections' shows what a template "
                        "expands to.", FileNotFoundError)
        shown = [name.removesuffix(".fits") if ".fits" not in params.input else name
                 for name in names]
        outs = [part if Path(part).suffix else part + ".fits"
                for part in (p.strip() for p in images.split_list(str(params.output)))] \
            if params.output else []
        said: list[str] = []
        warnings: list = []
        written: list[Path] = []
        for index, name in enumerate(names):
            if outs and index >= len(outs):
                break
            target = Path(outs[index]) if outs else output.named_for(name, ".dered.fits")
            try:
                header = fits.getheader(name)
                apertures = spectrum.read(name)
                if any(a.wavelength is None or not any(a.wavelength) for a in apertures):
                    advice.fail(f"[{shown[index]}] has no dispersion function",
                                "run dispcor on it first: the extinction is a function "
                                "of wavelength.")
                avold, rvold = 0.0, params.R
                if "DEREDDEN" in header:
                    if not params.override:
                        advice.fail(f"[{shown[index]}] has already been corrected",
                                    "give --override to correct it again; the old "
                                    "correction is taken out first.")
                    if params.uncorrect:
                        avold, rvold = deredden.previous(header["DEREDDEN"], params.R)
                # Said as it goes: an error in an aperture comes after its line.
                said.extend([f"[{target.name.removesuffix('.fits') if outs else target}]: "
                             f"{header.get('OBJECT', '')}", f"  {log}"])
                for aperture in apertures:
                    if aperture.number not in aps:
                        continue
                    said.append(f"  Ap {aperture.number}: "
                                f"{aperture.title or header.get('OBJECT', '')}")
                    wave = np.asarray(aperture.wavelength, dtype=np.float32)
                    for band in ("flux", "error", "sky", "raw"):
                        values = getattr(aperture, band, None)
                        if values is not None:
                            setattr(aperture, band, deredden.correct(
                                wave, values, av, params.R, avold, rvold).astype(float))
            except (KeyError, ValueError) as error:
                text = str(error.args[0] if error.args else error).splitlines()[0]
                warnings.append(advice.warn(text, "the spectrum was left as it was."))
                said.append(f"Warning: {text}")
                continue
            header["DEREDDEN"] = log
            spectrum.write(target, apertures, header=header, source=name)
            written.append(target)
        return TaskResult(outputs=written, steps=self.plan(params), warnings=warnings,
                          data={"text": said}, history=[f"deredden: {log}"])


class SlistParams(TaskParams):
    images: str = Field(..., description="Spectra to list. Commas, globs and @lists "
                                         "expand.")
    apertures: str = Field("", description="Apertures to list (1,3-5); all when empty.")
    long_header: bool = Field(False, description="Each spectrum's exposure, times, place "
                                                 "and airmass first.")


def _tab(text: str, column: int) -> str:
    """fmtio's %Nt: on to column N (1-based)."""
    return text.ljust(column - 1)


def _header_real(header, key: str):
    """What shdr reads of a keyword, in real: sexagesimal too; None for INDEF."""
    import numpy as np

    from ..core import iraffmt

    if key not in header:
        return None
    value = iraffmt.ctod(str(header[key]))
    if value is None or value > 1e300:
        return None
    return float(np.float32(value))


@register
class Slist(Task[SlistParams]):
    name = "slist"
    package = "onedspec"
    description = "List the apertures of spectra: number, beam, dispersion, title."
    Params = SlistParams

    def plan(self, params: SlistParams) -> list[Step]:
        return [Step("list", params.images,
                     why="One line an aperture: its line in the file, its aperture and "
                         "beam numbers, its dispersion (-1 pixels, 0 linear, 1 log, 2 "
                         "other), the first wavelength and the mean step to the last, "
                         "the number of pixels, and the title.")]

    def execute(self, params: SlistParams) -> TaskResult:
        from astropy.io import fits

        from ..core import images, iraffmt, ranges, spectrum

        aps = ranges.parse(params.apertures, -ranges.RNG_MAXINT, ranges.RNG_MAXINT)
        g = lambda v: iraffmt.g(float(v))  # noqa: E731

        def h(value, places):
            return "INDEF" if value is None else iraffmt.dtoc(value, places, "h")

        said: list[str] = []
        for token in images.split_list(params.images):
            for spec in images.expand(token) or [token]:
                name = str(spec)
                path = name if Path(name).exists() or not Path(name + ".fits").exists() \
                    else name + ".fits"
                shown = name.removesuffix(".fits")
                try:
                    header = fits.getheader(path)
                    apertures = spectrum.read(path)
                except (OSError, ValueError):
                    said.append(f"Warning: Cannot open image ({shown})")
                    continue
                title = str(header.get("OBJECT", ""))
                if params.long_header:
                    exptime = _header_real(header, "EXPTIME")
                    said.append(f"{shown}: {title}")
                    said.append(_tab(_tab(
                        "    EXPTIME = " + ("INDEF" if exptime is None else f"{exptime:.2f}"),
                        24) + f"UT = {h(_header_real(header, 'UT'), 1)}", 44)
                        + f"ST = {h(_header_real(header, 'ST'), 1)}")
                    airmass = _header_real(header, "AIRMASS")
                    said.append(_tab(_tab(_tab(
                        f"    RA = {h(_header_real(header, 'RA'), 2)}", 24)
                        + f"DEC = {h(_header_real(header, 'DEC'), 1)}", 44)
                        + f"HA = {h(_header_real(header, 'HA'), 2)}", 64)
                        + "AIRMASS = " + ("INDEF" if airmass is None else f"{airmass:5.3f}"))
                for line, aperture in enumerate(apertures, 1):
                    if aperture.number not in aps:
                        continue
                    npts = len(aperture.flux)
                    w0, wp = spectrum.start_and_step(aperture.wavelength, npts)
                    beam = aperture.number if aperture.beam is None else aperture.beam
                    said.append(("   " if params.long_header else shown) +
                                f" {line} {aperture.number} {beam} "
                                f"{spectrum.dispersion_type(aperture.wavelength)} {g(w0)} "
                                f"{g(wp)} {npts} {aperture.title or title}")
        return TaskResult(outputs=[], steps=self.plan(params), data={"text": said})


class RspectextParams(TaskParams):
    input: str = Field(..., description="Text spectra: lines of wavelength and flux, "
                                        "after a FITS header if there is one.")
    output: str = Field(..., description="One spectrum per input; one that exists is "
                                         "not written over.")
    title: str = Field("", description="The title, when the text has no header.")
    flux: bool = Field(False, description="The fluxes are calibrated (CA-FLAG and EX-FLAG "
                                          "0), when the text has no header.")
    dtype: Literal["none", "linear", "log", "nonlinear", "interp"] = Field(
        "linear", description="The dispersion, when the text has no header: none, linear "
                              "or log from --crval1 and --cdelt1 (the first column is not "
                              "read), nonlinear (the first column is each pixel's "
                              "wavelength) or interp (the same, resampled onto an even "
                              "grid).")
    crval1: float = Field(1.0, description="Wavelength of the first pixel (its log10 "
                                           "for log).")
    cdelt1: float = Field(1.0, description="Wavelength per pixel (in log10 for log).")


@register
class Rspectext(Task[RspectextParams]):
    name = "rspectext"
    package = "onedspec"
    description = "Read a text spectrum into a spectrum."
    Params = RspectextParams

    def plan(self, params: RspectextParams) -> list[Step]:
        return [Step("read", params.input,
                     why="Two columns, wavelength and flux; the lines that are not two "
                         "numbers are skipped. Both go through seven figures (%g), as "
                         "IRAF's script passes them on as text."),
                Step("dispersion", params.dtype,
                     why="With a FITS header in front (wspectext's), the header says it. "
                         "Without, linear or log take --crval1 and --cdelt1 and not the "
                         "first column; nonlinear keeps the first column as each pixel's "
                         "wavelength, and interp resamples onto an even grid between its "
                         "ends, as dispcor does, the fluxes not conserved."),
                Step("write", params.output, why="A spectrum of one aperture.")]

    def execute(self, params: RspectextParams) -> TaskResult:
        import re

        import numpy as np
        from astropy.io import fits

        from ..core import images, rspectext, spectrum

        inputs = [str(spec) for spec in images.expand(params.input)]
        outs = [part if Path(part).suffix else part + ".fits"
                for part in (p.strip() for p in images.split_list(params.output))]
        said: list[str] = []
        written: list[Path] = []
        for name, target in zip(inputs, outs, strict=False):    # join's shortest
            target = Path(target)
            if target.exists():
                # The script looks for the name, .imh and .hhh, and so not for a FITS
                # file, which rtextimage then refuses: the task stops there.
                advice.fail(f"Operation would overwrite existing image ({target.name})",
                            "name another output, or delete this one first.",
                            FileExistsError)
            cards, x, y = rspectext.read(Path(name).read_text(encoding="utf-8"))
            header = fits.Header()
            if cards is not None:
                title = ""
                for line in cards:
                    keyword = line[:8].rstrip()
                    if not re.fullmatch(r"[A-Z0-9_-]{1,8}", keyword):
                        continue                    # IRAF-B/P and the like: not FITS
                    card = fits.Card.fromstring(line.ljust(80)[:80])
                    if keyword == "OBJECT":         # rt_rheader: the last one is it
                        title = str(card.value).rstrip()
                        continue
                    if keyword in ("", "END", "BITPIX", "NDIM", "FORMAT", "IRAFTYPE",
                                   "SIMPLE", "EXTEND") or keyword.startswith(("NAXIS", "LEN")):
                        continue
                    try:
                        header.append(card)
                    except ValueError:
                        continue
                header["OBJECT"] = title
                wave = rspectext.header_dispersion(header.cards, y.size)
            else:
                header["OBJECT"] = params.title
                title = params.title
                if params.dtype == "interp":
                    wave, y = rspectext.interpolated(x, y)
                else:
                    wave = rspectext.wavelengths(params.dtype, x, params.crval1,
                                                 params.cdelt1)
                if params.dtype in ("linear", "log"):
                    header["DC-FLAG"] = 0 if params.dtype == "linear" else 1
                if params.flux:
                    header["CA-FLAG"] = 0
                    header["EX-FLAG"] = 0
            pixel = np.arange(1.0, y.size + 1.0)
            spectrum.write(target, [spectrum.Aperture(
                number=1, centre=0.0, low=0.0, high=0.0, pixel=pixel, flux=y,
                wavelength=wave, title=title)], header=header, source=name)
            written.append(target)
        return TaskResult(outputs=written, steps=self.plan(params), data={"text": said})


class NamesParams(TaskParams):
    input: str = Field(..., description="Root names, as onedspec's record format has "
                                        "them: night1/sp, which records make sp.0001...")
    records: str = Field(..., description="Record numbers: 1-5,7 or 10-20x2; empty, the "
                                          "roots themselves.")
    append: str = Field("", description="Text put after each name, a section such as "
                                        "[1:500].")
    check: bool = Field(False, description="Only the names of images that exist.")


@register
class Names(Task[NamesParams]):
    name = "names"
    package = "onedspec"
    description = "Spell out the image names of a root and record numbers."
    Params = NamesParams

    def plan(self, params: NamesParams) -> list[Step]:
        return [Step("expand", f"{params.input} with records {params.records or '(none)'}",
                     why="IRAF's old spectra were one to a file, named by a root and a "
                         "number of four figures. The numbers come out in increasing "
                         "order, each once, as xtools' decode_ranges has them.")]

    def execute(self, params: NamesParams) -> TaskResult:
        from ..core import images, odrecords

        roots: list[str] = []
        for token in images.split_list(params.input):
            found = [str(spec) for spec in images.expand(token)] \
                if any(c in token for c in "*?[@") else []
            roots.extend(found or [token])
        said = []
        for name in odrecords.names(roots, params.records):
            if params.check and not (Path(name).exists() or Path(name + ".fits").exists()):
                continue
            said.append(f"{name}{params.append}")
        return TaskResult(outputs=[], steps=self.plan(params), data={"text": said})


class SbandsParams(TaskParams):
    input: str = Field(..., description="Spectra with wavelengths. Commas, globs and "
                                        "@lists expand.")
    output: str = Field(..., description="The file the measurements are added to; STDOUT "
                                         "to say them.")
    bands: str = Field(..., description="The band file: lines of id, centre, width and "
                                        "filter (a file of wavelength and response, or "
                                        "none), one band or a band and one or two "
                                        "continua.")
    apertures: str = Field("", description="Apertures to measure (1,3-5); all when empty.")
    normalize: bool = Field(True, description="Divide each band's flux by its filter's "
                                              "summed response.")
    mag: bool = Field(False, description="Magnitudes, magzero - 2.5 log10 flux, instead of "
                                         "fluxes.")
    magzero: float = Field(0.0, description="The magnitudes' zero point.")
    verbose: bool = Field(True, description="A header first: the parameters, the bands "
                                            "and the columns.")


@register
class Sbands(Task[SbandsParams]):
    name = "sbands"
    package = "onedspec"
    description = "Measure fluxes, indices and equivalent widths in bandpasses."
    Params = SbandsParams

    def plan(self, params: SbandsParams) -> list[Step]:
        return [Step("measure", f"the bands of {params.bands} on {params.input}",
                     why="A band's flux is the sum of the pixels it covers, the end "
                         "pixels by the fraction inside it, each weighed by the filter's "
                         "response; with continua beside it, the index is the flux over "
                         "the continuum at the band's centre and the equivalent width "
                         "(1 - index) times its width -- a Lick index, say."),
                Step("write", params.output, why="Added to the file, as IRAF appends.")]

    def execute(self, params: SbandsParams) -> TaskResult:
        import getpass
        import socket

        import numpy as np

        from ..core import images, ranges, sbands, spectrum
        from ..core.deredden import cnvdate

        bandfile = Path(params.bands)
        if not bandfile.exists():
            advice.fail(f"Cannot open file ({params.bands})",
                        "give the band file: lines of id, centre, width and filter.",
                        FileNotFoundError)
        table = sbands.read_bands(bandfile.read_text(encoding="utf-8"), bandfile.parent)
        aps = ranges.parse(params.apertures, -ranges.RNG_MAXINT, ranges.RNG_MAXINT)
        lines: list[str] = []
        warnings: list = []
        opened = 0
        for token in images.split_list(params.input):
            for spec in images.expand(token) or [token]:
                name = str(spec)
                path = name if Path(name).exists() or not Path(name + ".fits").exists() \
                    else name + ".fits"
                try:
                    apertures = spectrum.read(path)
                except (OSError, ValueError):
                    shown = name.removesuffix(".fits")
                    lines.append(f"Warning: Cannot open image ({shown})")
                    warnings.append(advice.warn(f"Cannot open image ({shown})",
                                                "check the name."))
                    continue
                opened += 1
                if opened == 1 and params.verbose:
                    from ..core.provenance import _version

                    sysid = (f"harwood {_version()} {getpass.getuser()}@"
                             f"{socket.gethostname()} {cnvdate()}")
                    lines.extend(sbands.header(sysid, params.bands, table, params.normalize,
                                               params.mag, params.magzero))
                for line, aperture in enumerate(apertures, 1):
                    if aperture.number not in aps or aperture.wavelength is None:
                        continue
                    wave = np.asarray(aperture.wavelength, dtype=float)
                    to_pixel = sbands.inverse(wave)
                    # IRAF names it by its line of the multispec image: in[*,2](2).
                    lines.extend(sbands.measure(
                        f"{name.removesuffix('.fits')}[*,{line}]({aperture.number})", wave,
                        aperture.flux, table, to_pixel, params.normalize, params.mag,
                        params.magzero))
        said = [line for line in lines if line.startswith("Warning:")]
        written = [line for line in lines if not line.startswith("Warning:")]
        outputs = []
        if params.output == "STDOUT":
            said = lines
        elif written:
            with open(params.output, "a", encoding="utf-8") as out:
                out.write("\n".join(written) + "\n")
            outputs.append(Path(params.output))
        return TaskResult(outputs=outputs, steps=self.plan(params), warnings=warnings,
                          data={"text": said, "lines": written})


class LcalibParams(TaskParams):
    option: Literal["bands", "ext", "mags", "fnu", "flam", "stars"] = Field(
        ..., description="What to list: a star's bands, magnitudes, F-nu or F-lambda; the "
                         "extinction table; or the stars of a calibration directory.")
    star_name: str = Field("", description="The star, as its calibration file is named.")
    caldir: str = Field("onedstds$redcal/", description="The calibration directory, IRAF's "
                                                        "onedstds$ ones with harwood.")
    extinction: str = Field("onedstds$kpnoextinct.dat",
                            description="The extinction table, for ext.")
    fnuzero: float = Field(3.68e-20, description="The flux of magnitude zero, erg/cm2/s/Hz.")
    mag: float | None = Field(None, description="A blackbody's magnitude.")
    magband: str = Field("", description="Its band: U B V R I J H K L Lprime M.")
    teff: str = Field("", description="Its temperature, or spectral type.")


@register
class Lcalib(Task[LcalibParams]):
    name = "lcalib"
    package = "onedspec"
    description = "List the calibration tables standard, sensfunc and calibrate read."
    Params = LcalibParams

    def plan(self, params: LcalibParams) -> list[Step]:
        return [Step("list", params.option,
                     why="What standard reads for a star -- its bands' centres and widths "
                         "and its magnitudes, and so its flux, F-nu = fnuzero 10^(-0.4 m) "
                         "and F-lambda = F-nu c / lambda^2 -- and what calibrate reads for "
                         "the site's extinction, in real, as IRAF prints them.")]

    def execute(self, params: LcalibParams) -> TaskResult:
        import numpy as np

        from ..core import flux, iraffmt, stdstar

        f32 = np.float32
        said: list[str] = []
        errors: list[str] = []

        def row(wave, value):
            return (f"{iraffmt.printf(float(f32(wave)), 'f', width=6).rjust(6)} "
                    f"{iraffmt.printf(float(f32(value)), 'g', 5, width=12).rjust(12)}")

        if params.option == "ext":
            waves, extns = flux.read_extinction(params.extinction)
            said = [row(w, e) for w, e in zip(waves, extns, strict=True)]
        elif params.option == "stars":
            path = flux.resolve(params.caldir.rstrip("/") + "/standards.men")
            errors = path.read_text(encoding="utf-8").splitlines()
        else:
            cal = stdstar.read_calibration(params.caldir, params.star_name, params.mag,
                                           params.magband, params.teff)
            errors = [note for note in cal.notes]
            for wave, band, mag in zip(cal.waves, cal.dwaves, cal.mags, strict=True):
                if params.option == "bands":
                    value = band
                elif params.option == "mags":
                    value = mag
                else:
                    fnu = f32(f32(params.fnuzero) * f32(10.0) ** (f32(-0.4) * f32(mag)))
                    value = fnu if params.option == "fnu" else \
                        f32(fnu * f32(2.997925e18) / f32(wave) ** 2)
                said.append(row(wave, value))
        # What IRAF prints on the error stream (the stars, a blackbody's line) first.
        return TaskResult(outputs=[], steps=self.plan(params),
                          data={"text": errors + said, "stdout": said, "errors": errors})


class MkspecParams(TaskParams):
    image_name: str = Field(..., description="The spectrum to make.")
    image_title: str = Field("", description="Its title.")
    ncols: int = Field(..., ge=2, description="Pixels along the dispersion.")
    nlines: int = Field(1, ge=1, description="Lines: more than one makes a frame, the "
                                             "same spectrum on every line.")
    function: int = Field(1, ge=1, le=3, description="1 flat, 2 a ramp, 3 a blackbody.")
    constant: float = Field(0.0, description="The flat spectrum's level.")
    start_level: float = Field(0.0, description="The ramp's first value.")
    end_level: float = Field(0.0, description="The ramp's last value.")
    start_wave: float = Field(4000.0, gt=0, description="The blackbody's first wavelength, "
                                                        "A.")
    end_wave: float = Field(7000.0, gt=0, description="Its last.")
    temperature: float = Field(5800.0, gt=0, description="Its temperature, K.")


@register
class Mkspec(Task[MkspecParams]):
    name = "mkspec"
    package = "onedspec"
    description = "Make an artificial spectrum: flat, a ramp, or a blackbody."
    Params = MkspecParams

    def plan(self, params: MkspecParams) -> list[Step]:
        kind = {1: f"flat at {params.constant:g}",
                2: f"a ramp from {params.start_level:g} to {params.end_level:g}",
                3: f"a {params.temperature:g} K blackbody from {params.start_wave:g} to "
                   f"{params.end_wave:g} A"}[params.function]
        return [Step("make", kind,
                     why="A blackbody is divided by its peak, 1.2865e-4 T^5 (Wien's), so "
                         "that it reaches 1 where the peak is in the range. In real, as "
                         "IRAF's, its pixels to the bit. No wavelengths: dispcor or "
                         "rspectext give it some."),
                Step("write", params.image_name,
                     why="A spectrum of one aperture; with more lines, a frame of them.")]

    def execute(self, params: MkspecParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import mkspec, spectrum

        line = mkspec.row(params.ncols, params.function, params.constant,
                          params.start_level, params.end_level, params.start_wave,
                          params.end_wave, params.temperature)
        target = Path(params.image_name if Path(params.image_name).suffix
                      else params.image_name + ".fits")
        header = fits.Header()
        header["OBJECT"] = params.image_title
        if params.nlines == 1:
            spectrum.write(target, [spectrum.Aperture(
                number=1, centre=0.0, low=0.0, high=0.0,
                pixel=np.arange(1.0, params.ncols + 1.0), flux=line.astype(float),
                title=params.image_title)], header=header)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            fits.writeto(target, np.tile(line, (params.nlines, 1)), header, overwrite=True)
        return TaskResult(outputs=[target], steps=self.plan(params))


class SinterpParams(TaskParams):
    tbl_file: str = Field(..., description="The table: lines of x and y.")
    input: str = Field("STDIN", description="A file of x's to interpolate at, one a line, "
                                            "when neither --curve-gen nor --make-image.")
    image: str = Field("", description="The spectrum --make-image makes.")
    order: int = Field(5, ge=1, description="The fit's order, for legendre, chebyshev, "
                                            "spline3 and spline1.")
    x1: float = Field(0.0, description="The range's first x.")
    x2: float = Field(0.0, description="Its last; 0 to work it out from dx and npts.")
    dx: float = Field(0.0, description="Its step; 0 to work it out from x2 and npts.")
    npts: int = Field(0, ge=0, description="Its number of points.")
    curve_gen: bool = Field(False, description="Interpolate along the range, and say it.")
    make_image: bool = Field(False, description="Interpolate along the range into a "
                                                "spectrum, its wavelengths the x's.")
    tbl_size: int = Field(1024, ge=200, description="The most points the table may hold.")
    interp_mode: str = Field("chebyshev", description="linear, curves, legendre, "
                                                      "chebyshev, spline3 or spline1.")


@register
class Sinterp(Task[SinterpParams]):
    name = "sinterp"
    package = "onedspec"
    description = "Interpolate a table of x and y: at x's, along a range, into a spectrum."
    Params = SinterpParams

    def plan(self, params: SinterpParams) -> list[Step]:
        return [Step("interpolate", f"{params.tbl_file} by {params.interp_mode}",
                     why="linear is a straight line between the two points around x; "
                         "curves blends the two parabolas through the points around it "
                         "by how far apart they are; the others fit a curve of the "
                         "order through every point and evaluate it. In real, as "
                         "IRAF's.")]

    def execute(self, params: SinterpParams) -> TaskResult:
        import sys

        import numpy as np
        from astropy.io import fits

        from ..core import iraffmt, sinterp, spectrum

        f32 = np.float32
        xtab, ytab = sinterp.read_table(Path(params.tbl_file).read_text(encoding="utf-8"),
                                        params.tbl_size)
        mode = sinterp.mode_of(params.interp_mode)
        at = sinterp.Interpolant(xtab, ytab, mode, params.order)
        said: list[str] = []
        outputs: list[Path] = []
        if params.curve_gen or params.make_image:
            x2, dx, n = sinterp.grid(params.x1, params.x2, params.dx, params.npts)
            xs = [f32(f32(params.x1) + f32(i) * f32(dx)) for i in range(n)]
            ys = [f32(at(x)) for x in xs]
            if params.make_image:
                target = Path(params.image if Path(params.image).suffix
                              else params.image + ".fits")
                # imaddr writes a real into the header with its seven figures, and the
                # wavelengths are the WCS's from those, in double; the pixels are the
                # interpolant at x's in real.
                start = float(iraffmt.g(float(f32(params.x1))))
                step = float(iraffmt.g(float(f32(dx))))
                header = fits.Header()
                header["CRVAL1"], header["CDELT1"] = start, step
                header["CD1_1"], header["CRPIX1"], header["DC-FLAG"] = step, 1.0, 0
                spectrum.write(target, [spectrum.Aperture(
                    number=1, centre=0.0, low=0.0, high=0.0,
                    pixel=np.arange(1.0, n + 1.0), flux=np.array(ys, float),
                    wavelength=start + step * np.arange(n))], header=header)
                outputs.append(target)
            else:
                said = [sinterp.line(x, y) for x, y in zip(xs, ys, strict=True)]
        else:
            text = sys.stdin.read() if params.input == "STDIN" else \
                Path(params.input).read_text(encoding="utf-8")
            x = f32(0)
            for row in text.splitlines():
                words = row.split()
                value = iraffmt.ctod(words[0]) if words else None
                if value is not None and value < 1e300:
                    x = f32(value)          # gargr leaves x as it was on a word it can't read
                said.append(sinterp.line(x, at(x)))
        return TaskResult(outputs=outputs, steps=self.plan(params), data={"text": said})


class ScoordsParams(TaskParams):
    images: str = Field(..., description="Spectra of one aperture, given wavelengths in "
                                         "place.")
    coords: str = Field(..., description="Files of wavelengths, one a line and one a pixel: "
                                         "one for every spectrum, or one for each.")
    label: str = Field("", description="The axis's label, as IRAF's takes it; harwood's "
                                       "wavelength column has none to set.")
    units: str = Field("", description="Its units: the same. IRAF's writes the label "
                                       "there instead.")
    verbose: bool = Field(True, description="Say each spectrum and its file.")


@register
class Scoords(Task[ScoordsParams]):
    name = "scoords"
    package = "onedspec"
    description = "Give spectra their wavelengths from a list, one a pixel."
    Params = ScoordsParams

    def plan(self, params: ScoordsParams) -> list[Step]:
        return [Step("coordinates", f"{params.coords} onto {params.images}",
                     why="Each pixel's wavelength, from a list that had them: a lookup "
                         "table, as IRAF's multispec pixel array function, the wavelength "
                         "of each whole pixel the list's own.")]

    def execute(self, params: ScoordsParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import images, iraffmt, spectrum

        specs = [str(s) for s in images.expand(params.images)]
        coords = [c.strip() for c in images.split_list(params.coords)]
        if len(coords) > 1 and len(coords) != len(specs):
            advice.fail("Image and coordinate lists do not match",
                        "give one file of coordinates, or one for each spectrum.")
        said: list[str] = []
        warnings: list = []
        written: list[Path] = []
        for index, name in enumerate(specs):
            coord = coords[min(index, len(coords) - 1)]
            path = name if Path(name).exists() or not Path(name + ".fits").exists() \
                else name + ".fits"
            try:
                apertures = spectrum.read(path)
                header = fits.getheader(path)
                n = len(apertures[0].flux)
                values = []
                for line in Path(coord).read_text(encoding="utf-8").splitlines():
                    words = line.split()
                    value = iraffmt.ctod(words[0]) if words else None
                    if value is not None and value < 1e300:
                        values.append(value)
                    if len(values) > n:
                        break
                if len(values) != n:
                    advice.fail("Wrong number of coordinate values in file",
                                f"the file needs {n} wavelengths, one a line.")
                if params.verbose:
                    shown = name.removesuffix(".fits") if ".fits" not in params.images \
                        else name
                    said.append(f"SCOORDS: Setting coordinates for {shown} from coordinate "
                                f"file {coord}.")
                if len(apertures) != 1:
                    advice.fail("scoords: image must be one dimensional",
                                "give a spectrum of one aperture.")
                dw = (values[-1] - values[0]) / (n - 1)
                # scoords checks from the third pixel on: the first step goes unchecked.
                for i in range(2, n):
                    if (values[i] - values[i - 1]) * dw <= 0:
                        advice.fail("Coordinates are not monotonic",
                                    "the wavelengths must rise, or fall, all the way.")
                # Written into the WCS with %g, and read back from it.
                wave = np.array([float(iraffmt.g(v, double=True)) for v in values])
            except (OSError, ValueError) as error:
                text = str(error.args[0] if error.args else error).splitlines()[0]
                said.append(f"Warning: {text}")
                warnings.append(advice.warn(text, "the spectrum was left as it was."))
                continue
            aperture = apertures[0]
            aperture.wavelength, aperture.number, aperture.beam = wave, 1, None
            spectrum.write(path, [aperture], header=header)
            written.append(Path(path))
        return TaskResult(outputs=written, steps=self.plan(params), warnings=warnings,
                          data={"text": said})
