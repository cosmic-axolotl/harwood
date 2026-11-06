"""onedspec -- the tools that work on a spectrum once it has wavelengths.

dopcor takes it to the rest frame, specshift moves its zero point, sflip turns it
round, deredden takes the dust out, slist says what each aperture is, rspectext
reads one from text, names spells out a record list, sbands measures bands, lcalib
lists the calibration tables, mkspec makes an artificial one, sinterp
interpolates a table into one, scoords gives one its wavelengths from a list,
sapertures changes what its apertures are, ndprep makes a neutral density filter's
correction, disptrans puts its dispersion in other units, sarith and scopy copy,
cut and operate on spectra, scombine and odcombine combine them. Each is IRAF's
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
                                                "check the name, and that it is a spectrum."))
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


class SaperturesParams(TaskParams):
    input: str = Field(..., description="Spectra whose apertures to change, in place.")
    apertures: str = Field("", description="Apertures to change (1,3-5); all when empty.")
    apidtable: str = Field("", description="A table of aperture, beam, dtype, w1, dw, z, "
                                           "aplow, aphigh and title, a line each, or of "
                                           "aperture, beam and title.")
    wcsreset: bool = Field(False, description="Put every aperture back in pixels, the "
                                              "table and the defaults ignored.")
    verbose: bool = Field(False, description="Say each change.")
    beam: int | None = Field(None, description="Default beam number.")
    dtype: int | None = Field(None, ge=-1, le=1, description="Default dispersion type: -1 "
                                                             "pixels, 0 linear, 1 log.")
    w1: float | None = Field(None, description="Default first wavelength (a wavelength, "
                                               "for a log dispersion too).")
    dw: float | None = Field(None, description="Default step, in wavelength.")
    z: float | None = Field(None, description="Default redshift.")
    aplow: float | None = Field(None, description="Default lower limit across the slit.")
    aphigh: float | None = Field(None, description="Default upper limit across the slit.")
    title: str = Field("INDEF", description="Default title; INDEF keeps the aperture's.")


@register
class Sapertures(Task[SaperturesParams]):
    name = "sapertures"
    package = "onedspec"
    description = "Change the apertures' beams, dispersions, limits and titles."
    Params = SaperturesParams

    def plan(self, params: SaperturesParams) -> list[Step]:
        return [Step("change", params.apidtable or "the defaults given",
                     why="What an aperture is -- its beam, its dispersion, its place on "
                         "the slit, its title -- set by hand: a fibre's object from the "
                         "observing table, a dispersion known from elsewhere, a redshift. "
                         "w1 and dw are wavelengths, a log dispersion's too, as IRAF's.")]

    def execute(self, params: SaperturesParams) -> TaskResult:
        from astropy.io import fits

        from ..core import images, iraffmt, ranges, sapertures, spectrum

        aps = ranges.parse(params.apertures, -ranges.RNG_MAXINT, ranges.RNG_MAXINT)
        errors: list[str] = []
        if params.wcsreset:
            rows = []
            default = sapertures.Values(dtype=-1, w1=1.0, dw=1.0, z=0.0, title="")
        else:
            rows = []
            if params.apidtable.strip():
                rows, errors = sapertures.read_table(
                    Path(params.apidtable.strip()).read_text(encoding="utf-8"),
                    params.verbose)
            beam = params.beam
            if beam is not None and beam < 0:
                if params.verbose:
                    errors.append("Negative default beam number ignored.")
                beam = None
            # clgetd reads the parameter's text with IRAF's ctod, 1e-4 a hair above.
            read = lambda v: None if v is None else iraffmt.ctod(repr(float(v)))  # noqa: E731
            default = sapertures.Values(None, beam, params.dtype, read(params.w1),
                                        read(params.dw), read(params.z), params.aplow,
                                        params.aphigh, params.title)
        said: list[str] = []
        written: list[Path] = []
        for spec in images.expand(params.input):
            name = str(spec)
            path = name if Path(name).exists() or not Path(name + ".fits").exists() \
                else name + ".fits"
            header = fits.getheader(path)
            apertures = spectrum.read(path)
            lines: list[str] = []
            for aperture in apertures:
                if aperture.number not in aps:
                    continue
                row = next((r for r in rows if r.ap == aperture.number), default)
                change = sapertures.change(aperture, row,
                                           aperture.title or str(header.get("OBJECT", "")))
                if change is not None:
                    lines.extend(change)
            if lines and params.verbose:
                shown = name.removesuffix(".fits") if ".fits" not in params.input else name
                said.extend([f"{shown}:"] + lines)
            spectrum.write(path, apertures, header=header)
            written.append(Path(path))
        return TaskResult(outputs=written, steps=self.plan(params),
                          data={"text": errors + said, "stdout": said, "errors": errors})


class NdprepParams(TaskParams):
    filter_curve: str = Field(..., description="The filter curve, in --directory: "
                                               "nd1m.250mag.dat and the like.")
    output: str = Field(..., description="The correction to make.")
    w0: float = Field(..., description="Its first wavelength, A.")
    dw: float = Field(..., description="Its step, A.")
    nw: int = Field(..., ge=2, description="Its number of wavelengths.")
    nspace: int = Field(0, ge=0, description="Points across the slit: 0 for a spectrum, "
                                             "more for a frame.")
    logarithm: bool = Field(False, description="Resample onto wavelengths even in log10.")
    flux: bool = Field(True, description="Conserve flux in that resampling.")
    dispaxis: int = Field(2, ge=1, le=2, description="The frame's dispersion axis: 1 along "
                                                     "lines, 2 along columns.")
    directory: str = Field("onedstds$ctio/", description="Where the filter curves are.")


@register
class Ndprep(Task[NdprepParams]):
    name = "ndprep"
    package = "onedspec"
    description = "Make a neutral density filter's correction from its curve."
    Params = NdprepParams

    def plan(self, params: NdprepParams) -> list[Step]:
        return [Step("interpolate", f"{params.filter_curve} onto {params.nw} wavelengths "
                     f"from {params.w0:g} by {params.dw:g}",
                     why="sinterp's curves, as IRAF's ndprep: the filter's transmission "
                         "at each wavelength, two parabolas through the curve's points "
                         "blended. Divide a spectrum taken through the filter by it."),
                Step("resample", "onto log10 wavelengths", why="dispcor, as IRAF's does.",
                     skipped=not params.logarithm, reason="--logarithm not given"),
                Step("write", params.output, why="A spectrum, or with --nspace a frame.")]

    def execute(self, params: NdprepParams) -> TaskResult:
        import math

        import numpy as np
        from astropy.io import fits

        from ..core import dispcor, flux, sinterp, spectrum

        directory = params.directory
        if params.filter_curve == "?":
            menu = flux.resolve(directory.rstrip("/") + "/ndfilters.men")
            return TaskResult(outputs=[], steps=self.plan(params),
                              data={"text": menu.read_text(encoding="utf-8").splitlines()})
        try:
            curve = flux.resolve(directory.rstrip("/") + "/" + params.filter_curve)
        except (OSError, ValueError):
            curve = Path(directory) / params.filter_curve
        if not Path(curve).is_file():
            advice.fail(f"Filter curve {directory}{params.filter_curve} not found",
                        "give ? as the curve to list the filters of the directory.",
                        FileNotFoundError)
        xtab, ytab = sinterp.read_table(Path(curve).read_text(encoding="utf-8"))
        values, start, step = sinterp.make_image(xtab, ytab, "curves", 5, params.w0, 0.0,
                                                 params.dw, params.nw)
        n = values.size
        wave = start + step * np.arange(n)
        if params.logarithm:
            w1, w2 = wave[0], wave[-1]
            l1 = math.log10(w1)
            ldw = (math.log10(w2) - l1) / (n - 1)
            values = dispcor.resample(values, sinterp.Linear(start, step), l1, ldw, n,
                                      conserve=params.flux, blank=0.0, log=True)
            wave = 10 ** (l1 + ldw * np.arange(n))
        target = Path(params.output if Path(params.output).suffix
                      else params.output + ".fits")
        header = fits.Header()
        header["OBJECT"] = ""
        header["DC-FLAG"] = 1 if params.logarithm else 0
        if params.nspace == 0:
            spectrum.write(target, [spectrum.Aperture(
                number=1, centre=0.0, low=0.0, high=0.0, pixel=np.arange(1.0, n + 1.0),
                flux=values.astype(float), wavelength=wave)], header=header)
        else:
            axis = params.dispaxis
            header["DISPAXIS"] = axis
            header[f"CRVAL{axis}"] = math.log10(wave[0]) if params.logarithm else wave[0]
            header[f"CDELT{axis}"] = (math.log10(wave[-1]) - math.log10(wave[0])) / (n - 1) \
                if params.logarithm else step
            header[f"CRPIX{axis}"] = 1.0
            frame = np.tile(values.astype(np.float32), (params.nspace, 1))
            target.parent.mkdir(parents=True, exist_ok=True)
            fits.writeto(target, frame if axis == 1 else frame.T.copy(), header,
                         overwrite=True)
        return TaskResult(outputs=[target], steps=self.plan(params))


class DisptransParams(TaskParams):
    input: str = Field(..., description="Spectra with a dispersion.")
    output: str | None = Field(
        None, description="One name per input; the input's own name changes it in place. "
                          "Left out, each goes to this run's own directory.",
        json_schema_extra={"positional": True})
    units: str = Field(..., description="The new units: nm, microns, hertz, ev, inverse "
                                        "centimeters, km/s 6563 angstroms...")
    error: float = Field(0.01, ge=1e-6, description="How far, in pixels, the dispersion "
                                                    "function may be from a pixel's new "
                                                    "coordinate.")
    linearize: bool = Field(False, description="Resample onto an even grid in the new "
                                               "units, the fluxes not conserved.")
    verbose: bool = Field(True, description="Say each spectrum's new units.")
    air: Literal["none", "air2vac", "vac2air"] = Field(
        "none", description="From air to vacuum, or back.")
    t: float = Field(15.0, description="The air's temperature, C.")
    p: float = Field(760.0, description="Its pressure, mmHg.")
    f: float = Field(4.0, description="Its water vapour pressure, mmHg.")


@register
class Disptrans(Task[DisptransParams]):
    name = "disptrans"
    package = "onedspec"
    description = "Change the units of spectra's dispersion, and air to vacuum."
    Params = DisptransParams

    def plan(self, params: DisptransParams) -> list[Step]:
        return [Step("transform", f"to {params.units}"
                     + (f", {params.air}" if params.air != "none" else ""),
                     why="Each pixel's coordinate in the new units -- a frequency or an "
                         "energy goes as one over the wavelength, a velocity "
                         "relativistically from its line -- and, for air and vacuum, "
                         "Edlen's index of refraction. A straight line is fitted to them, "
                         "or a cubic spline of as many pieces as it takes to keep within "
                         "--error pixels, and that is the dispersion."),
                Step("write", str(params.output or "this run's directory"),
                     why="The wavelength column in the new units, which the spectrum's "
                         "header says (APUNITS). The other tasks read it in Angstroms.")]

    def execute(self, params: DisptransParams) -> TaskResult:
        from astropy.io import fits

        from ..core import disptrans, images, output, spectrum, units

        if units.decode(params.units).kind == 0:
            # shdr_units' error, which stops the task.
            advice.fail("Cannot convert to specified units",
                        "give units IRAF knows: angstroms, nm, microns, hertz, ev, "
                        "inverse centimeters, km/s 6563 angstroms...")
        names = [name if Path(name).exists() or not Path(name + ".fits").exists()
                 else name + ".fits"
                 for name in (str(spec) for spec in images.expand(params.input))]
        outs = [part if Path(part).suffix else part + ".fits"
                for part in (p.strip() for p in images.split_list(str(params.output)))] \
            if params.output else []
        said: list[str] = []
        warnings: list = []
        written: list[Path] = []
        for index, name in enumerate(names):
            target = Path(outs[index]) if index < len(outs) else \
                output.named_for(name, ".dt.fits")
            try:
                header = fits.getheader(name)
                apertures = spectrum.read(name)
                shown_units = ""
                for aperture in apertures:
                    if aperture.wavelength is None:
                        advice.fail(f"[{name}] has no dispersion function",
                                    "run dispcor on it first.")
                    wave, values, _, un = disptrans.transform(
                        aperture.wavelength, aperture.flux, aperture.units, params.units,
                        params.air, params.t, params.p, params.f, params.error,
                        params.linearize)
                    aperture.wavelength, aperture.flux = wave, values
                    aperture.units, aperture.redshift = params.units, 0.0
                    shown_units = un.units
            except (OSError, ValueError) as error:
                text = str(error.args[0] if error.args else error).splitlines()[0]
                said.append(f"Warning: {text}")
                warnings.append(advice.warn(text, "the spectrum was left as it was."))
                continue
            spectrum.write(target, apertures, header=header, source=name)
            written.append(target)
            if params.verbose:
                label = target.name.removesuffix(".fits") if index < len(outs) else \
                    str(target)
                line = f"{label}: Dispersion transformed to {shown_units}"
                if params.air == "none":
                    said.append(line + ".")
                else:
                    four = lambda v: iraffmt_g(v, 4)  # noqa: E731
                    said.append(line + (" in vacuum with" if params.air == "air2vac"
                                        else " in air with"))
                    said.append(f"  t = {four(params.t)} C, p = {iraffmt_g(params.p, 6)} "
                                f"mmHg, f = {four(params.f)} mmHg.")
        return TaskResult(outputs=written, steps=self.plan(params), warnings=warnings,
                          data={"text": said})


def iraffmt_g(value: float, digits: int) -> str:
    """%.Ng of a double."""
    from ..core import iraffmt

    return iraffmt.printf(float(value), "g", digits, double=True)


class SarithParams(TaskParams):
    input1: str = Field(..., description="Spectra to operate on. Commas, globs and @lists "
                                         "expand.")
    op: Literal["abs", "copy", "dex", "exp", "flam", "fnu", "inv", "ln", "log", "lum",
                "mag", "sqrt", "replace", "+", "-", "*", "/", "^"] = Field(
        ..., description="The operation: one of the spectrum alone, or with a second "
                         "operand (replace + - * / ^).")
    input2: str = Field("", description="The second operand: spectra, resampled onto the "
                                        "first's pixels, or a number.")
    output: str = Field(..., description="The result: one spectrum, or one per input.")
    w1: float | None = Field(None, description="The region's first wavelength.")
    w2: float | None = Field(None, description="Its last.")
    apertures: str = Field("", description="Apertures to take (1,3-5); ! before takes the "
                                           "others.")
    bands: str = Field("", description="Bands to take: 1 the spectrum, 2 raw, 3 the sky, "
                                       "4 the error, of those there are.")
    beams: str = Field("", description="Beams to take.")
    apmodulus: int = Field(0, ge=0, description="Aperture numbers taken modulo this for "
                                                "--apertures.")
    reverse: bool = Field(False, description="The second operand first.")
    ignoreaps: bool = Field(False, description="Pair the second operand's spectra by line, "
                                               "not by aperture number.")
    format: Literal["multispec", "onedspec"] = Field(
        "multispec", description="One spectrum of all the apertures, or one each "
                                 "(output.0001...).")
    renumber: bool = Field(False, description="Number the output's apertures from 1.")
    offset: int = Field(0, description="Add this to the output's aperture numbers.")
    clobber: bool = Field(False, description="Write over an output that exists.")
    merge: bool = Field(False, description="With --clobber, add the apertures to an output "
                                           "that exists.")
    rebin: bool = Field(True, description="Cut the region at its wavelengths exactly, "
                                          "resampled; off, at the nearest whole pixels.")
    errval: float = Field(0.0, description="What cannot be computed is this.")
    verbose: bool = Field(False, description="Say each spectrum done.")
    interp: Literal["nearest", "linear", "poly3", "poly5", "spline3", "sinc"] = Field(
        "poly5", description="Interpolant of the resampling (onedspec.interp).")


@register
class Sarith(Task[SarithParams]):
    name = "sarith"
    package = "onedspec"
    description = "Operate on spectra: copy, cut, resample, and arithmetic."
    Params = SarithParams

    def plan(self, params: SarithParams) -> list[Step]:
        return [Step("select", params.input1,
                     why="The apertures, beams and bands asked for; a region cut at its "
                         "wavelengths, resampled onto as many pixels as it covered, or at "
                         "the nearest whole pixels with --no-rebin."),
                Step("operate", params.op + (f" {params.input2}" if params.input2 else ""),
                     why="A second spectrum is resampled onto the first's pixels -- the "
                         "mean of the interpolant over each -- and paired by aperture "
                         "number; one that has none is copied. The error is only ever "
                         "copied. In real, as IRAF's; what cannot be computed is errval."),
                Step("write", params.output, why="One spectrum, or one per aperture.")]

    def execute(self, params: SarithParams) -> TaskResult:
        from ..core import images

        if params.op in ("replace", "+", "-", "*", "/", "^") and not params.input2.strip():
            advice.fail("Error in second operand", "give --input2, a spectrum or a number.")
        ins = [str(s) for s in images.expand(params.input1)]
        seconds = [str(s) for s in images.expand(params.input2)] if params.input2 else []
        if params.input2 and not seconds:
            seconds = [params.input2]
        outs = [o.strip() for o in images.split_list(params.output)]
        if len(outs) > 1 and len(outs) != len(ins):
            advice.fail("Input and output image lists don't make sense",
                        "give one output, or one for each input.")
        said: list[str] = []
        written: list[Path] = []
        groups = [([i], [seconds[min(k, len(seconds) - 1)]] if seconds else [], outs[k])
                  for k, i in enumerate(ins)] if len(outs) > 1 else \
            [(ins, seconds, outs[0])]
        for group, operands, out in groups:
            try:
                if params.format == "multispec":
                    written += _sa_ms(group, operands, out, params, said)
                else:
                    written += _sa_1d(group, operands, out, params, said)
            except (OSError, ValueError) as error:
                text = str(error.args[0] if error.args else error)
                said.extend(f"Warning: {text}".split("\n"))
        return TaskResult(outputs=written, steps=self.plan(params), data={"text": said})


class ScopyParams(TaskParams):
    input: str = Field(..., description="Spectra to copy.")
    output: str = Field(..., description="The copy: one spectrum, or one per input.")
    w1: float | None = Field(None, description="The region's first wavelength.")
    w2: float | None = Field(None, description="Its last.")
    apertures: str = Field("", description="Apertures to take; ! before takes the others.")
    bands: str = Field("", description="Bands to take: 1 the spectrum, 2 raw, 3 the sky, "
                                       "4 the error.")
    beams: str = Field("", description="Beams to take.")
    apmodulus: int = Field(0, ge=0, description="Aperture numbers taken modulo this.")
    format: Literal["multispec", "onedspec"] = Field("multispec",
                                                     description="One spectrum, or one "
                                                                 "per aperture.")
    renumber: bool = Field(False, description="Number the output's apertures from 1.")
    offset: int = Field(0, description="Add this to the output's aperture numbers.")
    clobber: bool = Field(False, description="Write over an output that exists.")
    merge: bool = Field(False, description="With --clobber, add to an output that exists.")
    rebin: bool = Field(True, description="Cut the region at its wavelengths exactly.")
    verbose: bool = Field(False, description="Say each spectrum copied.")


@register
class Scopy(Task[ScopyParams]):
    name = "scopy"
    package = "onedspec"
    description = "Copy spectra: apertures, bands, a wavelength region."
    Params = ScopyParams

    def plan(self, params: ScopyParams) -> list[Step]:
        return [Step("copy", f"{params.input} to {params.output}",
                     why="sarith with copy, as IRAF's scopy script is.")]

    def execute(self, params: ScopyParams) -> TaskResult:
        values = params.model_dump()
        values["input1"] = values.pop("input")
        result = Sarith().execute(SarithParams(op="copy", input2="", reverse=False,
                                               ignoreaps=False, errval=0.0, **values))
        result.steps = self.plan(params)
        return result


def _selected(ap: int, beam: int, aps, beams, apmod: int, complement: bool) -> bool:
    j = ap % apmod if apmod > 1 else ap
    chosen = j in aps and beam in beams
    return chosen != complement


def _ranges(params):
    from ..core import ranges

    complement = False
    out = []
    for text, what in ((params.apertures, "aperture/column/line"), (params.bands, "band"),
                       (params.beams, "beam")):
        text = text.strip()
        if text.startswith("!"):
            complement, text = True, text[1:]
        try:
            out.append(ranges.parse(text, -ranges.RNG_MAXINT, ranges.RNG_MAXINT))
        except ValueError:
            advice.fail(f"Bad {what} list", "write it as 1,3-5.")
    return (*out, complement)


def _read(name: str):
    from astropy.io import fits

    from ..core import spectrum

    path = name if Path(name).exists() or not Path(name + ".fits").exists() \
        else name + ".fits"
    return path, fits.getheader(path), spectrum.read(path)


def _band_names(aperture) -> list[str]:
    from ..core.sarith import BANDS

    return [band for band in BANDS if getattr(aperture, band, None) is not None]


def _operand(text: str):
    """A second operand: a spectrum, or a number."""
    from ..core import iraffmt

    name = text if Path(text).exists() or not Path(text + ".fits").exists() else text + ".fits"
    if Path(name).exists():
        path, header, apertures = _read(text)
        return apertures, None
    value = iraffmt.ctod(text)
    if value is None or value > 1e300:
        advice.fail("Error in second operand", "a spectrum, or a number.")
    return None, value


def _sections(nlines: int, nbands: int, line: int, band: int) -> str:
    if nbands > 1:
        return f"[*,{line},{band}]"
    if nlines > 1:
        return f"[*,{line}]"
    return ""


def _process(sh, params, operand, aperture, band_name, line, band):
    """The band of one aperture extracted and operated on: data, wavelengths, the
    operation done, and the second operand's view."""
    from ..core import sarith

    sarith.extract(sh, params.w1, params.w2, params.rebin, params.interp)
    op, other, const = params.op, None, None
    if op in sarith.BINARY:
        apertures, const, opname = operand
        if apertures is not None:
            if params.ignoreaps:
                match = apertures[line - 1] if line - 1 < len(apertures) else None
            else:
                match = next((a for a in apertures if a.number == sh.ap), None)
            if match is None or (not params.ignoreaps and match.number != sh.ap):
                op = "copy"
            else:
                names = _band_names(match)
                other = sarith.open_aperture(match, opname, _sections(
                    len(apertures), len(names), apertures.index(match) + 1, band),
                    names[min(band, len(names)) - 1])
                sarith.rebin(other, sh, params.interp)
    if band_name == "error":
        op = "copy"                     # "For now just copy noise band."
    data = sarith.arith(op, sh, other, const, params.reverse, params.errval)
    return data, op, other, const


def _wavelengths(sh) -> object:
    """The output's world coordinates for the band: sa_ms's w and dw from shdr_lw."""
    import math

    import numpy as np

    n = sh.n
    if sh.dc == 2:
        # A nonlinear dispersion keeps its function, cut or not: its own wavelengths.
        return sh.wave.copy()
    w, wb = sh.lw(1.0) * (1 + sh.z), sh.lw(float(n)) * (1 + sh.z)
    if n == 1:
        return np.array([w / (1 + sh.z)])
    if sh.dc == 1:
        lw, lwb = math.log10(w), math.log10(wb)
        dw = (lwb - lw) / (n - 1)
        world = 10 ** (lw + dw * np.arange(n))
    else:
        dw = (wb - w) / (n - 1)
        world = w + dw * np.arange(n)
    return world / (1 + sh.z)


def _sa_ms(group, operands, output, params, said) -> list[Path]:
    """sa_ms: the spectra of the group into one output."""
    import numpy as np

    from ..core import sarith, spectrum

    target = Path(output if Path(output).suffix else output + ".fits")
    aps, bands, beams, complement = _ranges(params)
    existing = []
    if target.exists():
        if not params.clobber:
            advice.fail(f"Output spectrum {output} already exists",
                        "give --clobber to write over it, with --merge to add to it.")
        if params.merge:
            _, _, existing = _read(str(target))
            for i, aperture in enumerate(existing, 1):
                if params.renumber:
                    aperture.number = i + params.offset
    outaps = [a.number for a in existing]
    naps = len(outaps)
    selected = []                   # (input, its apertures, header, aperture, new number)
    for name in group:
        _, header, apertures = _read(name)
        for aperture in apertures:
            if not _selected(aperture.number, aperture.number if aperture.beam is None
                             else aperture.beam, aps, beams, params.apmodulus, complement):
                continue
            ap = (naps + 1 if params.renumber else aperture.number) + params.offset
            if ap not in outaps:
                naps += 1
            selected.append((name, apertures, header, aperture, ap))
    if not selected:
        advice.fail("No spectra selected", "check --apertures, --beams and --bands.")
    numbers = [ap for *_, ap in selected]
    if len(set(numbers)) != len(numbers):
        advice.fail("Output spectra cannot have the same aperture number.\n\tUse renumber "
                    "parameter.", "give --renumber, or --offset to one of them.")
    nbands = max(len([b for i, b in enumerate(_band_names(a), 1) if i in bands])
                 for _, _, _, a, _ in selected)
    nbands = max(nbands, *[len(_band_names(a)) for a in existing]) if existing else nbands
    results = list(existing)
    last = len(outaps)
    operand_at = 0
    for name in group:
        names_here = [s for s in selected if s[0] == name]
        if not names_here:
            continue
        operand = None
        if params.op in sarith.BINARY:
            text = operands[min(operand_at, len(operands) - 1)]
            apertures2, const = _operand(text)
            operand = (apertures2, const, text.removesuffix(".fits"))
        operand_at += 1
        for _, apertures, _header, aperture, _ in names_here:
            ap = (last + 1 if params.renumber else aperture.number) + params.offset
            if ap in outaps:
                position = outaps.index(ap)
            else:
                position, last = last, last + 1
            line = apertures.index(aperture) + 1
            columns = _band_names(aperture)
            beam = aperture.number if aperture.beam is None else aperture.beam
            made = spectrum.Aperture(number=ap, centre=aperture.centre, low=aperture.low,
                                     high=aperture.high, title=aperture.title,
                                     beam=beam if beam != ap else None)
            k = 0
            for band, column in enumerate(columns, 1):
                if band not in bands:
                    continue
                k += 1
                sh = sarith.open_aperture(aperture, name.removesuffix(".fits"),
                                          _sections(len(apertures), len(columns), line,
                                                    band), column)
                data, op, other, const = _process(sh, params, operand, aperture, column,
                                                  line, band)
                setattr(made, column, data.astype(float))
                if made.wavelength is None:
                    made.wavelength = _wavelengths(sh)
                    made.pixel = np.arange(1.0, sh.n + 1.0)
                if params.verbose:
                    said.append(sarith.said(sh, other, output.removesuffix(".fits"),
                                            _sections(naps, nbands, position + 1, k), ap, op,
                                            const, params.reverse))
            if made.flux is None:
                for column in ("raw", "sky", "error"):
                    if getattr(made, column) is not None:
                        made.flux = getattr(made, column)
                        setattr(made, column, None)
                        break
            if position < len(results):
                results[position] = made
            else:
                results.append(made)
    header = selected[0][2].copy()
    spectrum.write(target, results, header=header, source=group[0])
    return [target]


def _sa_1d(group, operands, output, params, said) -> list[Path]:
    """sa_1d: each aperture and band of the group to a spectrum of its own."""
    import re

    import numpy as np

    from ..core import sarith, spectrum

    aps, bands, beams, complement = _ranges(params)
    root = re.sub(r"\.\d{4}$", "", output.removesuffix(".fits"))
    written: list[Path] = []
    naps = 0
    for at, name in enumerate(group):
        _, header, apertures = _read(name)
        operand = None
        if params.op in sarith.BINARY:
            text = operands[min(at, len(operands) - 1)]
            apertures2, const = _operand(text)
            operand = (apertures2, const, text.removesuffix(".fits"))
        for line, aperture in enumerate(apertures, 1):
            if not _selected(aperture.number, aperture.number if aperture.beam is None
                             else aperture.beam, aps, beams, params.apmodulus, complement):
                continue
            if params.renumber:
                naps += 1
            ap = (naps if params.renumber else aperture.number) + params.offset
            columns = _band_names(aperture)
            for band, column in enumerate(columns, 1):
                if band not in bands:
                    continue
                stem = f"{root}.{band - 1}{ap:03d}"
                target = Path(stem + ".fits")
                if target.exists() and not params.clobber:
                    said.append(f"Warning: Output spectrum {output} already exists")
                    continue
                sh = sarith.open_aperture(aperture, name.removesuffix(".fits"),
                                          _sections(len(apertures), len(columns), line,
                                                    band), column)
                data, op, other, const = _process(sh, params, operand, aperture, column,
                                                  line, band)
                beam = aperture.number if aperture.beam is None else aperture.beam
                made = spectrum.Aperture(number=ap, centre=aperture.centre,
                                         low=aperture.low, high=aperture.high,
                                         title=aperture.title, flux=data.astype(float),
                                         wavelength=_wavelengths(sh),
                                         pixel=np.arange(1.0, sh.n + 1.0),
                                         beam=beam if beam != ap else None)
                spectrum.write(target, [made], header=header.copy(), source=name)
                written.append(target)
                if params.verbose:
                    said.append(sarith.said(sh, other, stem, "", ap, op, const,
                                            params.reverse))
    return written


SCWHY = {
    "group": "With group apertures, each aperture number is combined across the "
             "inputs into an aperture of one output; with all, every spectrum into one; "
             "with images, the apertures of each input into an output of its own.",
    "grid": "Unless --first, the output runs over all the inputs' wavelengths in the "
            "smallest of their steps, or as --w1/--w2/--dw/--nw say; each spectrum is "
            "resampled onto it, the mean of its interpolant over each pixel, and where "
            "it does not reach is left out.",
    "scale": "Scales, zero offsets and weights -- from the spectra's mode, median or "
             "mean over --sample, their exposure, a file or a header keyword -- are "
             "made relative to their mean before combining.",
    "reject": "The rejection is IRAF's onedspec one, older than imcombine's: values put "
              "back after a sigma clip are looked for among all those rejected, and "
              "grow is a number of pixels along the spectrum.",
}


class ScombineParams(TaskParams):
    input: str = Field(..., description="Spectra to combine. Commas, globs and @lists "
                                        "expand.")
    output: str = Field(..., description="The combined spectrum; with --group images, "
                                         "one per input.")
    noutput: str = Field("", description="Spectra of the number combined at each pixel.")
    logfile: str = Field("STDOUT", description="The log: STDOUT, a file it is added to, "
                                               "or empty for none.")
    apertures: str = Field("", description="Apertures to combine (1,3-5).")
    group: Literal["all", "images", "apertures"] = Field(
        "apertures", description="What is combined with what.")
    combine: Literal["average", "median", "sum"] = Field("average",
                                                         description="How.")
    reject: Literal["none", "minmax", "ccdclip", "crreject", "sigclip", "avsigclip",
                    "pclip"] = Field("none", description="Rejection.")
    first: bool = Field(False, description="The output on the first spectrum's "
                                           "dispersion.")
    w1: float | None = Field(None, description="The output's first wavelength.")
    w2: float | None = Field(None, description="Its last.")
    dw: float | None = Field(None, description="Its step.")
    nw: int | None = Field(None, description="Its length.")
    log: bool = Field(False, description="Steps even in log wavelength.")
    scale: str = Field("none", description="none, mode, median, mean, exposure, @file or "
                                           "!keyword.")
    zero: str = Field("none", description="none, mode, median, mean, @file or !keyword.")
    weight: str = Field("none", description="none, mode, median, mean, exposure, @file "
                                            "or !keyword.")
    sample: str = Field("", description="Wavelength regions for the statistics, "
                                        "w1:w2,...")
    lthreshold: float | None = Field(None, description="Values below are left out.")
    hthreshold: float | None = Field(None, description="Values above are left out.")
    nlow: int = Field(1, ge=0, description="minmax: the lowest to reject.")
    nhigh: int = Field(1, ge=0, description="minmax: the highest to reject.")
    nkeep: int = Field(1, description="The fewest to keep, or negative, the most to "
                                      "reject.")
    mclip: bool = Field(True, description="Clip about the median, not the average.")
    lsigma: float = Field(3.0, ge=0, description="Low clipping factor.")
    hsigma: float = Field(3.0, ge=0, description="High clipping factor.")
    rdnoise: str = Field("0.", description="ccdclip: read noise (electrons), or a "
                                           "keyword.")
    gain: str = Field("1.", description="ccdclip: gain (electrons/DN), or a keyword.")
    snoise: str = Field("0.", description="ccdclip: sensitivity noise (fraction), or a "
                                          "keyword.")
    sigscale: float = Field(0.1, ge=0, description="Scales or zeros differing by more "
                                                   "than this correct the sigmas.")
    pclip: float = Field(-0.5, description="pclip: the percentile, or a number of "
                                           "places from the median.")
    grow: int = Field(0, ge=0, description="Pixels either side rejected with one.")
    blank: float = Field(0.0, description="The value where nothing is left.")
    interp: Literal["nearest", "linear", "poly3", "poly5", "spline3", "sinc"] = Field(
        "poly5", description="Interpolant of the resampling (onedspec.interp).")


@register
class Scombine(Task[ScombineParams]):
    name = "scombine"
    package = "onedspec"
    description = "Combine spectra: resampled, scaled, rejected, averaged."
    Params = ScombineParams

    def plan(self, params: ScombineParams) -> list[Step]:
        return [Step("group", params.group, why=SCWHY["group"]),
                Step("resample", "first" if params.first else "common grid",
                     why=SCWHY["grid"]),
                Step("scale", f"scale {params.scale}, zero {params.zero}, weight "
                              f"{params.weight}", why=SCWHY["scale"]),
                Step("combine", f"{params.combine}, reject {params.reject}",
                     why=SCWHY["reject"]),
                Step("write", params.output, why="NCOMBINE in the header; the log as "
                                                 "IRAF's.")]

    def execute(self, params: ScombineParams) -> TaskResult:
        from ..core import images, ranges, scombine

        ins = [str(s) for s in images.expand(params.input)]
        outs = [o.strip() for o in images.split_list(params.output)]
        nouts = [o.strip() for o in images.split_list(params.noutput)] \
            if params.noutput.strip() else []
        reject1 = "none" if params.combine == "sum" else params.reject
        if params.pclip == 0 and reject1 == "pclip":
            advice.fail("Pclip parameter may not be zero", "give --pclip, e.g. -0.5.")
        big = float(scombine.MAX_REAL)
        st = scombine.State(
            combine=params.combine, blank=params.blank, lsigma=params.lsigma,
            hsigma=params.hsigma, grow=params.grow, mclip=params.mclip,
            sigscale=params.sigscale,
            dothresh=params.lthreshold is not None or params.hthreshold is not None,
            lthresh=-big if params.lthreshold is None else float(params.lthreshold),
            hthresh=big if params.hthreshold is None else float(params.hthreshold))
        texts = {"scale": params.scale, "zero": params.zero, "weight": params.weight,
                 "rdnoise": params.rdnoise, "gain": params.gain, "snoise": params.snoise,
                 "sample": params.sample, "interp": params.interp}
        aps = ranges.parse(params.apertures, -ranges.RNG_MAXINT, ranges.RNG_MAXINT)
        said: list[str] = []
        log: list[str] = []
        written: list[Path] = []
        k = o = 0
        while k < len(ins):
            if o >= len(outs):
                said.append("No output image")
                break
            out_name, nout_name = outs[o], nouts[o] if o < len(nouts) else ""
            o += 1
            groups: list[list] = []
            numbers: list[int] = []
            while k < len(ins):
                name = ins[k]
                k += 1
                try:
                    specs = _sc_read(name, params, reject1)
                except (OSError, ValueError) as error:
                    if params.group != "images":
                        raise
                    said.append(f"Warning: {error}")
                    continue
                header = specs[0][2] if specs else None
                for spec, aperture_number, _ in specs:
                    if aperture_number not in aps:
                        continue
                    if params.group == "apertures":
                        if aperture_number not in numbers:
                            numbers.append(aperture_number)
                            groups.append([])
                        groups[numbers.index(aperture_number)].append((spec, header))
                    else:
                        if not groups:
                            groups.append([])
                        groups[0].append((spec, header))
                if params.group == "images":
                    break
            if not groups:
                said.append("No input spectra to combine")
                continue
            written += _sc_output(st, groups, out_name, nout_name, params, reject1, texts,
                                  said, log)
        text = said + log if params.logfile.strip() == "STDOUT" else said
        if params.logfile.strip() and params.logfile.strip() != "STDOUT":
            with open(params.logfile.strip(), "a", encoding="utf-8") as handle:
                handle.write("".join(line + "\n" for line in log))
        return TaskResult(outputs=written, steps=self.plan(params), data={"text": text})


def _sc_header_real(header, key: str):
    from ..core import iraffmt

    if key not in header:
        return None
    value = header[key]
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return iraffmt.ctod(str(value).strip())


def _sc_read(name: str, params, reject1: str):
    """Each aperture of an input as t_scombine reads it: band 1, and the header values
    it may want."""
    from ..core import sarith, scombine, spectrum

    path, header, apertures = _read(name)
    shown = name.removesuffix(".fits")
    exptime = None
    for key in ("EXPOSURE", "ITIME", "EXPTIME"):
        value = _sc_header_real(header, key)
        if value is not None:
            exptime = value
    keys = {}
    for param, text in (("scale", params.scale), ("zero", params.zero),
                        ("weight", params.weight)):
        text = "".join(text.split())
        if text.startswith("!"):
            keys[param] = _sc_header_real(header, text[1:])
            if keys[param] is None:
                advice.fail(f"Image header parameter not found ({text[1:]})",
                            f"the header needs {text[1:]}, or give another {param}.")
    for param in ("rdnoise", "gain", "snoise"):
        text = getattr(params, param)
        keys[param] = None
        if reject1 in ("ccdclip", "crreject") and scombine._leading_number(text) is None:
            keys[param] = _sc_header_real(header, text.strip())
            if keys[param] is None:
                advice.fail(f"Image header parameter not found ({text.strip()})",
                            f"the header needs {text.strip()}, or give --{param} a "
                            "number.")
    out = []
    nlines = len(apertures)
    for line, aperture in enumerate(apertures, 1):
        names = _band_names(aperture)
        sh = sarith.open_aperture(aperture, shown,
                                  _sections(nlines, len(names), line, 1), names[0])
        spec = scombine.Spectrum(sh, shown, spectrum.dispersion_type(aperture.wavelength),
                                 aperture, exptime, dict(keys))
        out.append((spec, aperture.number, header))
    return out


def _sc_output(st, groups, out_name, nout_name, params, reject1, texts, said, log):
    """scb_output and the combining: an output aperture for each group."""
    import numpy as np

    from ..core import sarith, scombine, spectrum

    target = Path(out_name if Path(out_name).suffix else out_name + ".fits")
    if target.exists():
        advice.fail(f"Output spectrum {out_name} already exists",
                    "give another name, or remove it.")
    header = groups[0][0][1].copy()
    if len(groups) == 1:
        # One aperture is a one-dimensional spectrum, whose title is the aperture's.
        header["OBJECT"] = groups[0][0][0].aperture.title
    nheader = header.copy()             # made before NCOMBINE is added, as IRAF's
    nheader["OBJECT"] = f"Number combined for {out_name}"
    made, counts = [], []
    for members in groups:
        specs = [spec for spec, _ in members]
        first = specs[0]
        a1 = first.aperture
        beam = a1.number if a1.beam is None else a1.beam
        if params.first:
            wave = None if a1.wavelength is None else np.asarray(a1.wavelength, float)
            z = float(a1.redshift or 0.0)
            dtype, nw = first.dc, first.sh.n
        else:
            dtype, w1, dw, nw = scombine.default_grid(specs, params.w1, params.w2,
                                                      params.dw, params.nw, params.log)
            wave, z = scombine.grid_wavelengths(dtype, w1, dw, nw), 0.0
        aperture = spectrum.Aperture(number=a1.number, centre=a1.centre, low=a1.low,
                                     high=a1.high, title=a1.title,
                                     pixel=np.arange(1.0, nw + 1.0),
                                     flux=np.zeros(nw), wavelength=wave, redshift=z,
                                     beam=beam if beam != a1.number else None)
        out = sarith.open_aperture(aperture, out_name.removesuffix(".fits"), "", "flux")
        n = len(specs)
        st.reject, st.nkeep = reject1, params.nkeep
        if st.nkeep < 0:
            st.nkeep = n + st.nkeep
        if st.reject == "pclip":
            half = int((n - 1) / 2.0)
            p = scombine.f32(params.pclip)
            if abs(p) < 1:
                p = scombine.f32(p * scombine.f32(half))
            p = min(-1, max(-half, int(p))) if p < 0 else max(1, min(half, int(p)))
            st.pclip = float(p)
        if st.reject == "minmax":
            f32 = scombine.f32
            flow, fhigh = f32(params.nlow), f32(params.nhigh)
            if flow >= 1:
                flow = f32(flow / f32(n))
            if fhigh >= 1:
                fhigh = f32(fhigh / f32(n))
            st.flow, st.fhigh = float(flow), float(fhigh)
            total = int(f32(f32(flow * f32(n)) + f32(fhigh * f32(n))))
            if total == 0:
                st.reject = "none"
            elif total >= n:
                said += ["Bad minmax rejection parameters", "Using no rejection"]
                st.reject = "none"
        logged = {"rdnoise": params.rdnoise, "gain": params.gain, "snoise": params.snoise,
                  "sample": params.sample, "output": out_name.removesuffix(".fits"),
                  "out": out, "dtype": dtype}
        result, number, scaling = scombine.combine_group(st, specs, out, texts, logged)
        said += scaling.said
        log += scaling.log
        aperture.flux = result.astype(float)
        made.append(aperture)
        counts.append(spectrum.Aperture(number=aperture.number, centre=aperture.centre,
                                        low=aperture.low, high=aperture.high,
                                        title=aperture.title, pixel=aperture.pixel,
                                        flux=number.astype(float),
                                        wavelength=aperture.wavelength, redshift=z,
                                        beam=aperture.beam))
        header["NCOMBINE"] = scaling.ncombine
        if scaling.expname:
            header["EXPTIME"] = float(f"{scaling.exposure:.7g}")
    written = [spectrum.write(target, made, header=header, source=groups[0][0][0].name)]
    if nout_name:
        ntarget = Path(nout_name if Path(nout_name).suffix else nout_name + ".fits")
        written.append(spectrum.write(ntarget, counts, header=nheader, source=out_name))
    return written


ODWHY = {
    "group": "With group apertures, each aperture number is combined across the "
             "inputs into an output of its own, root.NNNN; with all, every spectrum into "
             "one; with images, the apertures of each input into an output of its own.",
    "grid": "Unless --first, the output runs over all the inputs' wavelengths in the "
            "smallest of their steps, or as --w1/--w2/--dw/--nw say. Each spectrum is "
            "resampled onto it, the mean of its interpolant over each pixel, and kept "
            "only where it reaches; the output then starts where the first one starts "
            "and ends where the last one ends.",
    "mask": "A bad pixel mask named by BPM in an input's header is resampled with its "
            "spectrum, as 0 and 1000, and is bad where 10 or more of that is left. As "
            "IRAF's odcombine: only the last spectrum taken of each output when the "
            "image is read gets one.",
    "combine": "imcombine's code then does the rest -- scaling, rejection, weights -- "
               "with every spectrum's header values read from the first's image, as "
               "IRAF's: a scale by exposure, or by a keyword, is the same for all.",
}


class OdcombineParams(TaskParams):
    input: str = Field(..., description="Spectra to combine. Commas, globs and @lists "
                                        "expand.")
    output: str = Field(..., description="The combined spectrum: a root for "
                                         "root.NNNN by aperture, one per input with "
                                         "--group images.")
    bpmasks: str = Field("", description="Masks of the output's pixels with no data "
                                         "(root.NNNN.pl by aperture).")
    rejmasks: str = Field("", description="Masks of what each spectrum lost, a line "
                                          "per spectrum.")
    nrejmasks: str = Field("", description="Masks of the number rejected or left out at "
                                           "each pixel.")
    expmasks: str = Field("", description="Masks of the exposure (--expname) at each "
                                          "pixel.")
    sigmas: str = Field("", description="Spectra of the sigma about the combination.")
    logfile: str = Field("STDOUT", description="The log: STDOUT, a file it is added to, "
                                               "or empty for none.")
    apertures: str = Field("", description="Apertures to combine (1,3-5).")
    group: Literal["all", "images", "apertures"] = Field(
        "apertures", description="What is combined with what.")
    first: bool = Field(False, description="The output on the first spectrum's "
                                           "dispersion.")
    w1: float | None = Field(None, description="The output's first wavelength.")
    w2: float | None = Field(None, description="Its last.")
    dw: float | None = Field(None, description="Its step.")
    nw: int | None = Field(None, description="Its length.")
    log: bool = Field(False, description="Steps even in log wavelength.")
    combine: Literal["average", "median", "sum"] = Field("average", description="How.")
    reject: Literal["none", "minmax", "ccdclip", "crreject", "sigclip", "avsigclip",
                    "pclip"] = Field("none", description="Rejection.")
    outtype: Literal["short", "ushort", "integer", "long", "real", "double"] = Field(
        "real", description="The output's pixel type.")
    smaskformat: Literal["bpmpixel", "bpmspectrum"] = Field(
        "bpmspectrum", description="An input's mask: a line for each of its lines, or "
                                   "a spectrum for each aperture on its own dispersion.")
    smasktype: Literal["none", "goodvalue", "badvalue", "goodbits", "badbits"] = Field(
        "none", description="What in the mask says a pixel is good or bad.")
    smaskvalue: int = Field(0, description="The mask value, or bits.")
    blank: float = Field(0.0, description="The value where nothing is left.")
    scale: str = Field("none", description="none, mode, median, mean, exposure, @file or "
                                           "!keyword.")
    zero: str = Field("none", description="none, mode, median, mean, @file or !keyword.")
    weight: str = Field("none", description="none, mode, median, mean, exposure, @file "
                                            "or !keyword.")
    statsec: str = Field("", description="The section for the statistics: [x1:x2], "
                                         "input [..], output [..] or overlap.")
    expname: str = Field("", description="The header keyword of the exposure time.")
    lthreshold: float | None = Field(None, description="Values below are left out.")
    hthreshold: float | None = Field(None, description="Values above are left out.")
    nlow: int = Field(1, ge=0, description="minmax: the lowest to reject.")
    nhigh: int = Field(1, ge=0, description="minmax: the highest to reject.")
    nkeep: int = Field(1, description="The fewest to keep, or negative, the most to "
                                      "reject.")
    mclip: bool = Field(True, description="Clip about the median, not the average.")
    lsigma: float = Field(3.0, ge=0, description="Low clipping factor.")
    hsigma: float = Field(3.0, ge=0, description="High clipping factor.")
    rdnoise: str = Field("0.", description="ccdclip: read noise (electrons), or a "
                                           "keyword.")
    gain: str = Field("1.", description="ccdclip: gain (electrons/DN), or a keyword.")
    snoise: str = Field("0.", description="ccdclip: sensitivity noise (fraction), or a "
                                          "keyword.")
    sigscale: float = Field(0.1, ge=0, description="Scales differing from one by more "
                                                   "than this correct the sigmas.")
    pclip: float = Field(-0.5, description="pclip: the percentile, or a number of "
                                           "places from the median.")
    grow: float = Field(0.0, ge=0, description="Radius in pixels rejected around a "
                                               "rejected one.")
    interp: Literal["nearest", "linear", "poly3", "poly5", "spline3", "sinc"] = Field(
        "poly5", description="Interpolant of the resampling (onedspec.interp).")


@register
class Odcombine(Task[OdcombineParams]):
    name = "odcombine"
    package = "onedspec"
    description = "Combine spectra: resampled onto one grid, then imcombine's code."
    Params = OdcombineParams

    def plan(self, params: OdcombineParams) -> list[Step]:
        steps = [Step("group", params.group, why=ODWHY["group"]),
                 Step("resample", "first" if params.first else "common grid",
                      why=ODWHY["grid"])]
        if params.smasktype != "none":
            steps.append(Step("mask", f"{params.smaskformat}, {params.smasktype} "
                                      f"{params.smaskvalue}", why=ODWHY["mask"]))
        steps += [Step("combine", f"{params.combine}, reject {params.reject}, scale "
                                  f"{params.scale}, zero {params.zero}, weight "
                                  f"{params.weight}", why=ODWHY["combine"]),
                  Step("write", params.output, why="IMCMBnnn and NCOMBINE in the "
                                                   "header; the log as IRAF's.")]
        return steps

    def execute(self, params: OdcombineParams) -> TaskResult:
        from ..core import images, ranges

        ins = [str(s) for s in images.expand(params.input)]
        lists = {key: [o.strip() for o in images.split_list(getattr(params, key))]
                 if getattr(params, key).strip() else []
                 for key in ("output", "bpmasks", "rejmasks", "nrejmasks", "expmasks",
                             "sigmas")}
        if not ins:
            advice.fail("No input images to combine", "give --input spectra.")
        what = {"output": "output images", "bpmasks": "bad pixel masks",
                "rejmasks": "rejection masks", "nrejmasks": "number rejected masks",
                "expmasks": "exposure masks", "sigmas": "sigma images"}
        for key, names in lists.items():
            if params.group == "images":
                wrong = len(names) != len(ins) if key == "output" else \
                    len(names) > (1 if key == "sigmas" else 0) and len(names) != len(ins)
            else:
                wrong = len(names) != 1 if key == "output" else len(names) > 1
            if wrong:
                advice.fail(f"Wrong number of {what[key]}",
                            "one for all with --group all or apertures, one per input "
                            "with --group images.")
        if params.reject == "pclip" and params.pclip == 0:
            advice.fail("Pclip parameter may not be zero", "give --pclip, e.g. -0.5.")
        if params.outtype == "double":
            # IRAF's icombine then combines in double; harwood's engine is its real one.
            advice.fail("outtype double is not harwood's: IRAF combines in double "
                        "precision then, and harwood only as IRAF does in real",
                        "use --outtype real; the output table holds the values in double "
                        "anyway.")
        mtype = params.smasktype
        if mtype == "badbits" and params.smaskvalue == 0:
            mtype = "none"
        aps = ranges.parse(params.apertures, -ranges.RNG_MAXINT, ranges.RNG_MAXINT)
        text: list[str] = []
        written: list[Path] = []
        index = k = 0
        while k < len(ins):
            groups: list[list] = []
            while k < len(ins):
                name = ins[k]
                k += 1
                try:
                    _odc_read(groups, name, aps, params, mtype)
                except (OSError, ValueError) as error:
                    if params.group != "images":
                        raise
                    text.append(f"Warning: {error}")
                    continue
                if params.group == "images":
                    break
            if not groups:
                text.append("No input spectra to combine")
                continue
            for members in groups:
                if params.group == "images":
                    index += 1
                ap = members[0].sh.ap if params.group == "apertures" else None
                at = index if params.group == "images" else 1
                names = {key: _odc_name(names, at, ap) for key, names in lists.items()}
                written += _odc_output(members, names, params, mtype, text)
        log = params.logfile.strip()
        if log and log != "STDOUT":
            with open(log, "a", encoding="utf-8") as handle:
                handle.write("".join(line + "\n" for line in text
                                     if not line.startswith(("Warning:", "No input"))))
        shown = text if log == "STDOUT" else [line for line in text if line.startswith(
            ("Warning:", "No input"))]
        return TaskResult(outputs=written, steps=self.plan(params), data={"text": shown})


def _odc_read(groups, name: str, aps, params, mtype: str) -> None:
    """odc_gspec on one input: its selected apertures, and then its mask's."""
    from ..core import odcombine, sarith

    path, header, apertures = _read(name)
    shown = name.removesuffix(".fits")
    members = []
    for line, aperture in enumerate(apertures, 1):
        if aperture.number not in aps:
            continue
        names = _band_names(aperture)
        sh = sarith.open_aperture(aperture, shown,
                                  _sections(len(apertures), len(names), line, 1), names[0])
        members.append(odcombine.Member(sh, line, header, aperture))
    odcombine.add_spectra(groups, members, params.group)
    bpm = str(header.get("BPM", "")).strip() if mtype != "none" else ""
    if bpm:
        _, _, masks = _read(bpm)
        odcombine.add_masks(groups, bpm, masks, params.smaskformat,
                            lambda nlines, line: _sections(nlines, 1, line, 1))


def _odc_name(names: list[str], index: int, aperture: int | None) -> str:
    """odc_imtgetim: the list's name at index, with .NNNN for the aperture."""
    if index > len(names):
        return ""
    name = names[index - 1]
    if aperture is None:
        return name
    path = Path(name)
    if path.suffix in (".fits", ".fit", ".fts", ".pl", ".imh"):
        return f"{path.with_suffix('')}.{aperture:04d}{path.suffix}"
    return f"{name}.{aperture:04d}"


def _odc_output(members, names: dict, params, mtype: str, text: list[str]) -> list[Path]:
    """One output: the template's grid, the spectra placed on it, combined, written."""
    import numpy as np

    from ..core import odcombine, sarith, spectrum
    from .imcombine import _exposure, _write_plane

    first = members[0]
    a1 = first.aperture
    beam = a1.number if a1.beam is None else a1.beam
    dtype, wave, z = odcombine.grid(members, params.first, params.w1, params.w2,
                                    params.dw, params.nw, params.log)
    nw = len(wave) if wave is not None else first.sh.n
    template = spectrum.Aperture(number=a1.number, centre=a1.centre, low=a1.low,
                                 high=a1.high, title=a1.title,
                                 pixel=np.arange(1.0, nw + 1.0), flux=np.zeros(nw),
                                 wavelength=wave, redshift=z)
    out = sarith.open_aperture(template, "template", "", "flux")
    placed = [p for p in (odcombine.place(m, out, mtype, params.smaskvalue, params.interp)
                          for m in members) if p is not None]
    if not placed:
        text.append("Warning: No images to combine")
        return []
    output = names["output"]
    target = _odc_file(output, ".fits")
    if target.exists():
        advice.fail(f"Output spectrum {output} already exists",
                    "give another name, or remove it.")
    cfg = params.model_dump()
    cfg["smasktype"] = mtype
    shown = {key: names[key] for key in ("bpmasks", "rejmasks", "nrejmasks", "expmasks")}
    for key, value in shown.items():
        cfg[key] = str(_odc_file(value, ".pl")) if value else ""
    cfg["sigmas"] = names["sigmas"].removesuffix(".fits")
    cfg["output"] = output.removesuffix(".fits")
    header = first.header
    result = odcombine.combine(placed, cfg, header)
    text += result.log

    npts = result.combined.size
    pixel = np.arange(1.0, npts + 1.0)
    span = slice(result.first, result.first + npts)
    wavelength = None if wave is None else np.asarray(wave)[span]
    combined = result.combined.astype(np.float32)
    if params.outtype != "real":
        combined = _odc_outtype(combined, params.outtype)
    made = spectrum.Aperture(number=a1.number, centre=a1.centre, low=a1.low, high=a1.high,
                             title=a1.title, pixel=pixel, flux=combined.astype(float),
                             wavelength=wavelength, redshift=z,
                             beam=beam if beam != a1.number else None)
    out_header = header.copy()
    for key in [k for k in out_header if k.startswith("IMCMB")] + ["BPM", "ICBPM"]:
        out_header.remove(key, ignore_missing=True, remove_all=True)
    out_header["OBJECT"] = a1.title
    if placed[0].member.mask is not None:
        out_header["ICBPM"] = placed[0].member.mask_label
    if len(placed) < 100:
        for i, p in enumerate(placed, 1):
            out_header[f"IMCMB{i:03d}"] = p.member.label
    out_header["NCOMBINE"] = result.nout
    if cfg["bpmasks"]:
        out_header["BPM"] = cfg["bpmasks"]
    written = [spectrum.write(target, [made], header=out_header, source=first.sh.image)]

    if names["sigmas"]:
        # icombine makes the sigma from the output before ic_scale gives that its
        # NCOMBINE and BPM.
        sigma_header = out_header.copy()
        for key in ("NCOMBINE", "BPM"):
            sigma_header.remove(key, ignore_missing=True)
        sigma_header["OBJECT"] = f"Combine sigma images for {cfg['output']}"
        sigma_target = _odc_file(names["sigmas"], ".fits")
        sigma = spectrum.Aperture(number=a1.number, centre=a1.centre, low=a1.low,
                                  high=a1.high, title=a1.title, pixel=pixel,
                                  flux=result.sigma.astype(np.float32).astype(float),
                                  wavelength=wavelength, redshift=z, beam=made.beam)
        written.append(spectrum.write(sigma_target, [sigma], header=sigma_header,
                                      source=first.sh.image))
    nimages = len(placed)
    planes = {}
    if cfg["bpmasks"]:
        planes["bpmasks"] = np.where(result.used > 0, 0, np.where(result.used == 0, 1, 2))
    if cfg["rejmasks"]:
        planes["rejmasks"] = np.where(result.used == nimages, False, ~result.keep)
    if cfg["nrejmasks"]:
        planes["nrejmasks"] = nimages - result.used
    if cfg["expmasks"]:
        planes["expmasks"], _ = _exposure(result.keep, [header] * nimages,
                                          result.scaling.weights, params.expname)
    for key, plane in planes.items():
        _write_plane(cfg[key], plane, out_header)
        written.append(Path(cfg[key]))
    return written


def _odc_file(name: str, extension: str) -> Path:
    """The file of an image named as IRAF names it: its extension, or `extension`."""
    known = (".fits", ".fit", ".fts", ".pl", ".imh")
    return Path(name if Path(name).suffix in known else name + extension)


def _odc_outtype(values, outtype: str):
    """The combined spectrum in an integer pixel type, as IMIO converts a real."""
    import numpy as np

    kinds = {"short": np.int16, "ushort": np.uint16, "integer": np.int32,
             "long": np.int32}
    return np.trunc(values).astype(kinds[outtype])


SFWHY = {
    "select": "The lines (apertures, in the image's order) and bands to fit. What the "
              "image's SFIT and SFITB say was fitted before -- in the input, or in an "
              "output that exists -- is not fitted again unless --override, so that a "
              "spectrum fitted by hand is not lost to a later run.",
    "fit": "icfit's curve, IRAF's in real: the sample, averaged by --naverage (a median "
           "when negative), fitted, and the points beyond --low-reject or "
           "--high-reject sigma rejected --niterate times, with those within --grow.",
    "write": "The output is a copy of the input with each fitted line replaced by the "
             "fit, the data (the rejected points replaced by the fit with --replace), "
             "data minus fit, or data over fit; a division by zero is 0.",
}


class _SfitParams(TaskParams):
    input: str = Field(..., description="Spectra to fit. Commas, globs and @lists expand.")
    output: str = Field("", description="An output for each input; the input itself "
                                        "to fit it in place. Not with --listonly.",
                        json_schema_extra={"positional": True})
    lines: str = Field("*", description="Lines (apertures, in order) to fit: 1,3:5.")
    bands: str = Field("1", description="Bands to fit: 1 the spectrum, 2 raw, 3 sky, "
                                        "4 sigma, as apall writes them.")
    type: Literal["data", "fit", "difference", "ratio"]
    replace: bool = Field(False, description="Replace the rejected points by the fit.")
    wavescale: bool = Field(True, description="Fit against wavelength, not pixel.")
    logscale: bool = Field(False, description="Fit the log10 of both axes.")
    override: bool = Field(False, description="Fit again what SFIT says was fitted.")
    listonly: bool = Field(False, description="Log the fits, write nothing.")
    logfiles: str = Field("logfile", description="Log files: STDOUT, names, or empty.")
    interactive: bool = Field(False, description="Fit each spectrum by hand, in the icfit "
                                                 "panel, with IRAF's keys.")
    cursor: str = Field("", description="Those keys from a file, 'x y 1 key' a line, for "
                                        "a session with no window; each spectrum reads "
                                        "on to its q.")
    sample: str = Field("*", description="Sample ranges to fit, in x: 4010:4150,4210:4300.")
    naverage: int = Field(1, description="Points averaged, or with a minus their median, "
                                         "into each fitted.")
    function: Literal["spline3", "legendre", "chebyshev", "spline1"] = Field(
        "spline3", description="The curve.")
    order: int = Field(1, ge=1, description="Its order: terms, or a spline's pieces.")
    low_reject: float
    high_reject: float
    niterate: int
    grow: float = Field(1.0, ge=0, description="Reject too what is this close in x.")


class ContinuumParams(_SfitParams):
    type: Literal["data", "fit", "difference", "ratio"] = Field(
        "ratio", description="What to write: data, fit, difference or ratio.")
    low_reject: float = Field(2.0, ge=0, description="Reject below this many sigma.")
    high_reject: float = Field(0.0, ge=0, description="Reject above this many sigma.")
    niterate: int = Field(10, ge=0, description="Rejection passes.")


class SfitParams(_SfitParams):
    type: Literal["data", "fit", "difference", "ratio"] = Field(
        "fit", description="What to write: data, fit, difference or ratio.")
    low_reject: float = Field(3.0, ge=0, description="Reject below this many sigma.")
    high_reject: float = Field(3.0, ge=0, description="Reject above this many sigma.")
    niterate: int = Field(0, ge=0, description="Rejection passes.")


class _Sfit:
    def plan(self, params) -> list[Step]:
        return [Step("select", f"lines {params.lines}, bands {params.bands}",
                     why=SFWHY["select"]),
                Step("fit", f"{params.function} order {params.order}, sample "
                            f"{params.sample}", why=SFWHY["fit"]),
                Step("write", params.type, why=SFWHY["write"])]

    def execute(self, params) -> TaskResult:
        import getpass
        import socket

        from ..core import images, sfit
        from ..core.iraffmt import cnvdate
        from ..core.provenance import _version

        ins = [str(s) for s in images.expand(params.input)] or [params.input]
        outs = [] if params.listonly else (
            [o.strip() for o in images.split_list(params.output)]
            if params.output.strip() else [])
        if not params.listonly and len(outs) != len(ins):
            advice.fail("Input and output image lists do not match",
                        "give an output for each input, or the input itself to fit it "
                        "in place.")
        ymax = bmax = 0
        for name in ins + outs:
            found = _sft_file(_sft_split(name)[0])
            if found is not None:
                apertures = _sft_apertures(found)
                ymax = max(ymax, len(apertures))
                bmax = max(bmax, len(_band_names(apertures[0])) if apertures else 0)
        ranges = {}
        for key in ("lines", "bands"):
            text = getattr(params, key)
            if "x" in text.lower():
                advice.fail("Range step (`x' notation) not implemented",
                            "list the lines, 1,3,5, or a range, 1:5.")
            ranges[key] = sfit.rg_ranges(text.replace("-", ":"), 1,
                                         ymax if key == "lines" else bmax)
        if not ranges["lines"] or not ranges["bands"]:
            advice.fail("With range specification for `lines or bands'",
                        "lines and bands are ranges of the image's lines and bands.")
        spec = {key: getattr(params, key) for key in (
            "function", "order", "sample", "naverage", "low_reject", "high_reject",
            "niterate", "grow")}
        logs = [w for w in params.logfiles.replace(",", " ").split()]
        sysid = (f"harwood {_version()} {getpass.getuser()}@{socket.gethostname()} "
                 f"{cnvdate()}")
        text: list[str] = []
        logged: list[str] = []

        def log(lines):
            logged.extend(lines)
            if "STDOUT" in logs or "STDERR" in logs:
                text.extend(lines)

        log(["", f"SFIT: {sysid}", ""])
        written: list[Path] = []
        for k, name in enumerate(ins):
            output = outs[k] if outs else name
            try:
                done = _sft_one(name, output, params, ranges, spec, text, log)
            except _SftWarning as warning:
                text.append(f"Warning: {warning}")
                if warning.written is not None:
                    written.append(warning.written)
                continue
            if done is not None:
                written.append(done)
        log(["", f"END: {sysid}", ""])
        _SFT_CURSORS.pop(params.cursor, None)
        for name in logs:
            if name not in ("STDOUT", "STDERR"):
                with open(name, "a", encoding="utf-8") as handle:
                    handle.write("".join(line + "\n" for line in logged))
        return TaskResult(outputs=written, steps=self.plan(params), data={"text": text})


class _SftWarning(Exception):
    """sft_immap's error, which t_sfit takes as a warning and goes on."""

    def __init__(self, message: str, written: Path | None = None):
        super().__init__(message)
        self.written = written


def _sft_file(name: str) -> Path | None:
    for candidate in (name, name + ".fits"):
        if Path(candidate).is_file():
            return Path(candidate)
    return None


def _sft_apertures(path: Path):
    from ..core import spectrum

    return spectrum.read(path)


def _sft_one(name, output, params, ranges, spec, text, log) -> Path | None:
    """sft_immap and sft_icfit on one input: the output, or None with --listonly."""
    import copy

    import numpy as np
    from astropy.io import fits

    from ..core import sfit, spectrum

    inroot, insect = _sft_split(name)
    outroot, outsect = _sft_split(output)
    inroot = inroot.removesuffix(".fits")
    if insect or outsect:
        raise _SftWarning(f"Sections not allowed ({inroot}{insect} --> {output})")
    inpath = _sft_file(inroot)
    if inpath is None:
        raise _SftWarning(f"Cannot access {name.removesuffix('.fits')}")
    header = fits.getheader(inpath)
    apertures = _sft_apertures(inpath)
    outpath = None if params.listonly else _sft_file(outroot)
    if params.listonly:
        out_header, out_apertures = header, apertures
    elif outpath is not None and outpath.resolve() == inpath.resolve():
        out_header, out_apertures, outpath = header, apertures, inpath
    elif outpath is None:
        outpath = Path(outroot if Path(outroot).suffix == ".fits" else outroot + ".fits")
        out_header = header.copy()
        out_apertures = copy.deepcopy(apertures)
        for aperture in out_apertures:
            for band in _band_names(aperture):
                setattr(aperture, band, np.asarray(getattr(aperture, band), np.float32)
                        .astype(float))
    else:
        out_header = fits.getheader(outpath)
        out_apertures = _sft_apertures(outpath)
        if len(out_apertures) != len(apertures) or any(
                len(a.flux) != len(b.flux) or _band_names(a) != _band_names(b)
                for a, b in zip(apertures, out_apertures, strict=True)):
            raise _SftWarning(f"{inroot} & {outroot} aren't the same size")
    ax2 = len(apertures)
    ax3 = len(_band_names(apertures[0])) if apertures else 0
    sfit_in = str(header.get("SFIT", ""))
    if "SFIT" not in out_header:
        out_header["SFIT"] = ""
    if "SFITB" not in out_header:
        out_header["SFITB"] = ""
    before = sfit.rg_ranges(sfit_in, 1, ax2) | sfit.rg_ranges(str(out_header["SFIT"]), 1, ax2)
    if params.override:
        lines = {line for line in ranges["lines"] if 1 <= line <= ax2}
    else:
        lines = {line for line in ranges["lines"] if 1 <= line <= ax2 and line not in before}
    bands = sorted(band for band in ranges["bands"] if 1 <= band <= ax3)

    def save():
        if not params.listonly:
            spectrum.write(outpath, out_apertures, header=out_header,
                           source=str(inpath))
        return None if params.listonly else outpath

    if not lines:
        raise _SftWarning(f"No lines left to fit for {inroot}", save())
    shown = str(inpath)
    current = dict(spec)
    for band in bands:
        for line in sorted(lines):
            aperture = apertures[line - 1]
            column = _band_names(aperture)[band - 1]
            y = np.asarray(getattr(aperture, column), np.float32).copy()
            n = y.size
            if params.logscale:
                y = sfit.alog(y)
            if params.wavescale and aperture.wavelength is not None:
                x = np.asarray(aperture.wavelength, np.float32)
            else:
                x = np.arange(1, n + 1, dtype=np.float32)
            if params.logscale:
                x = sfit.alog(x)
            weights = None
            if params.interactive or params.cursor:
                current, x, y, weights = _sft_session(current, x, y, params, shown, line,
                                                      band, aperture.title)
            data, curve, ic = sfit.fit(x, y, current, params.type, params.replace,
                                       weights)
            text.extend(ic.messages)
            if not params.listonly:
                target = out_apertures[line - 1]
                setattr(target, column, data.astype(float))
                out_header["SFIT"] = sfit.rg_encode(
                    {line} | sfit.rg_ranges(str(out_header["SFIT"]), 1, ax2))
                out_header["SFITB"] = sfit.rg_encode(
                    {band} | sfit.rg_ranges(str(out_header["SFITB"]), 1, ax3))
            log(sfit.power(curve, current["function"], line, shown))
    return save()


def _sft_split(name: str) -> tuple[str, str]:
    """imgimage and imgsection: the image and its section, [...] at the end."""
    if name.endswith("]") and "[" in name:
        at = name.index("[")
        return name[:at], name[at:]
    return name, ""


def _sft_session(spec, x, y, params, image, line, band, title):
    """icg_fit on one spectrum: the parameters it leaves, which the next spectrum
    starts from, and the points and weights the fit is of."""
    import numpy as np

    from ..core import icfit, icfitsession
    from ..viewer import icfit_panel

    fitspec = icfit.FitSpec(**spec)
    name = f"{image}, [{line},{band}]"
    if params.cursor:
        session = icfitsession.IcfitSession(x.astype(float), y.astype(float), fitspec,
                                            np.ones(x.size), name)
        session.run_commands("\n".join(_sft_keys(params.cursor)))
    else:
        session = icfit_panel.make(x.astype(float), y.astype(float), fitspec,
                                   title=f"{params_name(params)} · {name} · {title}",
                                   weights=np.ones(x.size))
    done = session.spec
    out = {key: getattr(done, key) for key in spec}
    return (out, np.asarray(session.x, np.float32), np.asarray(session.y, np.float32),
            np.asarray(session.fitted_wts, np.float32))


_SFT_CURSORS: dict[str, list[list[str]]] = {}


def _sft_keys(cursor: str) -> list[str]:
    """The next spectrum's keys from the cursor file: up to and with its q, as IRAF
    reads one file on from spectrum to spectrum; none once it is used up."""
    if cursor not in _SFT_CURSORS:
        if not Path(cursor).exists():
            advice.fail(f"cursor file '{cursor}' does not exist",
                        "write one key a line, or use --interactive for the window",
                        FileNotFoundError)
        chunks, chunk = [], []
        for text in Path(cursor).read_text(encoding="utf-8").splitlines():
            words = text.split()
            if not words:
                continue
            keyed = len(words) >= 4 and words[0][0] not in ":"
            line = text if keyed else f"0 0 1 {text.strip()}"
            chunk.append(line)
            if line.split()[3] == "q":
                chunks.append(chunk)
                chunk = []
        if chunk:
            chunks.append(chunk)
        _SFT_CURSORS[cursor] = chunks
    chunks = _SFT_CURSORS[cursor]
    return chunks.pop(0) if chunks else []


def params_name(params) -> str:
    return "continuum" if isinstance(params, ContinuumParams) else "sfit"


@register
class Continuum(_Sfit, Task[ContinuumParams]):
    name = "continuum"
    package = "onedspec"
    description = "Fit the continuum of spectra, and divide by it (or subtract it)."
    Params = ContinuumParams


@register
class Sfit(_Sfit, Task[SfitParams]):
    name = "sfit"
    package = "onedspec"
    description = "Fit a curve to spectra, and write the fit, the data, or both."
    Params = SfitParams


FPWHY = {
    "profiles": "Each line of --positions is a profile: its position, and if given its "
                "peak, its type (gaussian, lorentzian, voigt) and its widths; what is "
                "left out is --profile, --gfwhm and --lfwhm, and a peak left out is the "
                "data's there.",
    "fit": "Over --region, above a straight background through its ends (or as "
           "--background says), the profiles are fitted by least squares: the "
           "background held and then fitted, the positions fixed, then together, then "
           "each its own, as --fitpositions, --fitgfwhm and --fitlfwhm allow.",
    "errors": "With --sigma0 and --invgain, the noise is sqrt(sigma0^2 + invgain*data); "
              "--nerrsample fits of the model with that noise added give errors, the "
              "68th percentile of how far they fall from the fit.",
    "write": "The log gives each profile's centre, the continuum under it, its flux, "
             "equivalent width, peak and widths; --output gets the model, or the data "
             "less the model.",
}


class FitprofsParams(TaskParams):
    input: str = Field(..., description="Spectra to fit. Commas, globs and @lists expand.")
    lines: str = Field("", description="Apertures to fit (1,3-5); a leading ! all but "
                                       "those.")
    bands: str = Field("", description="Bands of the output (1 the spectrum); a leading "
                                       "! all but those -- and, as IRAF's, all but the "
                                       "apertures in --lines too.")
    region: str = Field("", description="The region to fit: two wavelengths.")
    positions: str = Field("", description="The file of profiles: position [peak "
                                           "[type [gfwhm [lfwhm]]]] a line.")
    background: str = Field("", description="The background at each end: a value, "
                                            "avg(w1,w2[,scale]) or med(w1,w2[,scale]); "
                                            "left out, the data at the region's ends.")
    profile: Literal["gaussian", "lorentzian", "voigt"] = Field(
        "gaussian", description="The profile of a line that does not say.")
    gfwhm: float = Field(20.0, description="The gaussian FWHM of a line that does not "
                                           "say.")
    lfwhm: float = Field(20.0, description="The lorentzian FWHM of a line that does not "
                                           "say.")
    fitbackground: bool = Field(True, description="Fit the background.")
    fitpositions: Literal["fixed", "single", "all"] = Field(
        "all", description="Positions: held, moved together, or each its own.")
    fitgfwhm: Literal["fixed", "single", "all"] = Field(
        "all", description="Gaussian widths: held, scaled together, or each its own.")
    fitlfwhm: Literal["fixed", "single", "all"] = Field(
        "all", description="Lorentzian widths: held, scaled together, or each its own.")
    nerrsample: int = Field(0, ge=0, description="Monte-Carlo samples for the errors "
                                                 "(fewer than 15: none).")
    sigma0: float | None = Field(None, description="The noise's constant term.")
    invgain: float | None = Field(None, description="The noise's term in the data, the "
                                                    "inverse gain.")
    components: str = Field("", description="Profiles to log and model (1,3-5).")
    verbose: bool = Field(True, description="Say it on the standard output.")
    logfile: str = Field("logfile", description="A file the log is added to; empty for "
                                                "none.")
    plotfile: str = Field("", description="IRAF's metacode plot file: not harwood's.")
    output: str = Field("", description="Spectra of the model, one per input.")
    option: Literal["fit", "difference"] = Field(
        "fit", description="The output: the model, or the data less the model.")
    clobber: bool = Field(False, description="Replace an output that exists.")
    merge: bool = Field(False, description="IRAF's merge into an existing output: not "
                                           "harwood's.")


@register
class Fitprofs(Task[FitprofsParams]):
    name = "fitprofs"
    package = "onedspec"
    description = "Fit gaussian, lorentzian and voigt profiles to lines of spectra."
    Params = FitprofsParams

    def plan(self, params: FitprofsParams) -> list[Step]:
        steps = [Step("profiles", params.positions, why=FPWHY["profiles"]),
                 Step("fit", f"region {params.region}", why=FPWHY["fit"])]
        if params.sigma0 is not None and params.invgain is not None:
            steps.append(Step("errors", f"{params.nerrsample} samples",
                              why=FPWHY["errors"]))
        return steps + [Step("write", params.output or "the log", why=FPWHY["write"])]

    def execute(self, params: FitprofsParams) -> TaskResult:
        from ..core import fitprofs, images, ranges

        ins = [str(s) for s in images.expand(params.input)] or [params.input]
        outs = [o.strip() for o in images.split_list(params.output)] \
            if params.output.strip() else []
        if len(outs) > 1 and len(outs) != len(ins):
            advice.fail("Input and output image lists do not make sense",
                        "give one output for each input.")
        if params.plotfile.strip():
            advice.fail("plotfile is IRAF's metacode, which harwood does not write",
                        "leave --plotfile empty.")
        if params.merge and params.clobber:
            advice.fail("merge is not harwood's: IRAF 2.18's merge into an existing "
                        "output ends in a segmentation violation, so there is nothing to "
                        "hold it to",
                        "write a new output, or --clobber the old one.")
        fit = [fitprofs.SINGLE if params.fitbackground else fitprofs.FIXED,
               fitprofs.FITTYPES.index(params.fitpositions) + 1, fitprofs.INDEP,
               fitprofs.FITTYPES.index(params.fitgfwhm) + 1,
               fitprofs.FITTYPES.index(params.fitlfwhm) + 1]
        sigma0, invgain = params.sigma0, params.invgain
        if sigma0 is None or invgain is None or sigma0 < 0 or invgain < 0:
            sigma0 = invgain = None
        profiles = _fp_positions(params)
        region = params.region.split()
        try:
            region = [float(region[0]), float(region[1])]
        except (IndexError, ValueError):
            advice.fail(f"region '{params.region}' is not two wavelengths",
                        "give the region to fit as --region 'w1 w2'.")
        complement = False
        texts = {}
        for key in ("lines", "bands"):
            text = getattr(params, key).strip()
            if text.startswith("!"):
                complement = True
                text = text[1:]
            texts[key] = ranges.parse(text, -ranges.RNG_MAXINT, ranges.RNG_MAXINT)
        components = ranges.parse(params.components, -ranges.RNG_MAXINT,
                                  ranges.RNG_MAXINT)
        logged: list[str] = []
        text: list[str] = []
        written: list[Path] = []
        gasdev = fitprofs.Gasdev()

        def say(lines):
            logged.extend(lines)
            if params.verbose:
                text.extend(lines)

        def warn(message):
            text.append(f"Warning: {message}")

        for k, name in enumerate(ins):
            output = outs[k] if k < len(outs) else ""
            try:
                done = _fp_one(name, output, params, texts, complement, profiles,
                               region, fit, sigma0, invgain, components, gasdev, say,
                               warn)
            except ValueError as error:
                warn(str(error))
                continue
            if done is not None:
                written.append(done)
        if params.logfile.strip():
            with open(params.logfile.strip(), "a", encoding="utf-8") as handle:
                handle.write("".join(line + "\n" for line in logged))
        return TaskResult(outputs=written, steps=self.plan(params), data={"text": text})


def _fp_positions(params):
    """The positions file as t_fitprofs reads it: x, peak, type, gfwhm, lfwhm a line,
    as many as scan, the rest from the defaults."""
    import numpy as np

    from ..core import fitprofs, iraffmt

    name = params.positions.strip()
    if not name:
        advice.fail("A 'positions' file must be specified",
                    "give --positions, a file of the lines to fit.")
    if not Path(name).is_file():
        advice.fail(f"Cannot open file ({name})", "check the name of the positions file.",
                    FileNotFoundError)
    ptype = fitprofs.PTYPES.index(params.profile) + 1
    gfwhm, lfwhm = np.float32(params.gfwhm), np.float32(params.lfwhm)
    profiles = []
    for line in Path(name).read_text(encoding="utf-8").splitlines():
        words = line.split()
        values = []
        word = ""
        for index, kind in enumerate(("r", "r", "w", "r", "r")):
            if index >= len(words):
                break
            if kind == "w":
                word = words[index]
                values.append(word)
                continue
            value = iraffmt.ctod(words[index])
            if value is None:
                break
            values.append(value)
        nscan = len(values)
        if nscan == 0:
            continue
        matched = [i + 1 for i, p in enumerate(fitprofs.PTYPES)
                   if word and p.startswith(word)]
        p = matched[0] if len(matched) == 1 else ptype
        x = np.float32(values[0])
        y = None if nscan < 2 or values[1] >= 1e300 else np.float32(values[1])
        g = np.float32(values[3]) if nscan >= 4 else gfwhm
        l = np.float32(values[4]) if nscan >= 5 else lfwhm  # noqa: E741
        if nscan <= 2:
            p, g, l = ptype, gfwhm, lfwhm  # noqa: E741
        elif nscan == 3:
            g, l = gfwhm, lfwhm  # noqa: E741
        elif nscan == 4:
            if p == fitprofs.LORENTZ:
                g, l = gfwhm, g  # noqa: E741
            else:
                l = lfwhm  # noqa: E741
        # As t_fitprofs stores them: a lorentzian's width is g, which is --gfwhm unless
        # the line has all five fields, and then its fourth.
        if p == fitprofs.GAUSS:
            profiles.append((p, x, y, g, np.float32(0)))
        elif p == fitprofs.LORENTZ:
            profiles.append((p, x, y, np.float32(0), g))
        else:
            profiles.append((p, x, y, g, l))
    if not profiles:
        advice.fail("No profiles defined", "the positions file has no lines in it.")
    return profiles


def _fp_one(name, output, params, texts, complement, profiles, region, fit, sigma0,
            invgain, components, gasdev, say, warn) -> Path | None:
    """fp_ms on one input: its selected apertures fitted, and the output written."""
    import os

    import numpy as np

    from ..core import fitprofs, iraffmt, sarith, spectrum

    shown = name.removesuffix(".fits")
    outname = output.removesuffix(".fits") if output else ""
    temp = outname
    if outname:
        target = Path(outname + ".fits")
        if target.exists():
            if not params.clobber:
                raise ValueError(f"Output spectrum {outname} already exists")
            temp = f"temp{os.getpid() % 100000}a"
    path, header, apertures = _read(name)
    nlines = len(apertures)
    bands = _band_names(apertures[0]) if apertures else []
    nbands = sum(1 for band in range(1, len(bands) + 1) if band in texts["bands"])
    if nbands == 0:
        raise ValueError("No bands selected in image")
    chosen = []
    for line, aperture in enumerate(apertures, 1):
        select = aperture.number in texts["lines"]
        if select != complement:
            chosen.append((line, aperture))
    if not chosen:
        raise ValueError(f"No apertures selected in {shown}")
    naps = len(chosen)
    made = []
    for line, aperture in chosen:
        sh = sarith.open_aperture(aperture, shown, _sections(nlines, len(bands), line, 1),
                                  "flux")
        say([f"# {iraffmt.cnvdate()} {shown} - Ap {aperture.number}: {aperture.title}"])
        x = np.asarray(aperture.wavelength if aperture.wavelength is not None
                       else np.arange(1, len(aperture.flux) + 1), np.float32)
        y = np.asarray(aperture.flux, np.float32)
        try:
            # IRAF's arithmetic, NaN and infinity included: a profile that starts with
            # no peak is 0/0, and stays NaN, as it does in IRAF.
            with np.errstate(all="ignore"):
                lines, mod = fitprofs.fit_region(sh, x, y, profiles, region, fit,
                                                 params.nerrsample, sigma0, invgain,
                                                 components, params.background, gasdev)
            say(lines)
        except ValueError as error:
            warn(str(error))
            mod = np.zeros(y.size, np.float32)
        if not outname:
            continue
        values = mod if params.option == "fit" else (y - mod).astype(np.float32)
        l = len(made) + 1  # noqa: E741
        out_bands = {}
        for band in sarith.BANDS[:nbands]:
            out_bands[band] = np.zeros(y.size) if band != "flux" else values.astype(float)
        made.append(spectrum.Aperture(
            number=aperture.number, centre=aperture.centre, low=aperture.low,
            high=aperture.high, title=aperture.title, pixel=aperture.pixel,
            wavelength=aperture.wavelength, beam=aperture.beam,
            redshift=aperture.redshift, **out_bands))
        if params.verbose:
            # The output's line, as shdr names it; with --clobber, the temporary
            # image IRAF writes before it renames it.
            section = "" if naps == 1 and nbands == 1 else (
                f"[*,{l}]" if nbands == 1 else f"[*,{l},1]")
            say([f"{shown}{sh.section}({aperture.number})  -->  "
                 f"{temp}{section}({aperture.number})"])
    if not outname:
        return None
    out_header = header.copy()
    if naps == 1:
        out_header["OBJECT"] = made[0].title
    target = Path(outname + ".fits")
    if target.exists():
        target.unlink()
    return spectrum.write(target, made, header=out_header, source=str(path))
