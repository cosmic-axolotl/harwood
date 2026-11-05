"""onedspec -- the tools that work on a spectrum once it has wavelengths.

dopcor takes it to the rest frame, specshift moves its zero point, sflip turns it
round, deredden takes the dust out, slist says what each aperture is, rspectext
reads one from text, names spells out a record list. Each is IRAF's
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
