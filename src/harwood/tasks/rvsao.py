"""rvsao -- the Smithsonian Astrophysical Observatory's radial velocity package.

RVSAO is an IRAF external package (Mink & Kurtz 1998), not part of IRAF; harwood
carries its emsao, which measures a redshift from emission lines -- the step of the
course's exercise 7.4 -- and its xcsao, which measures a velocity by
cross-correlation with templates. The computations are core/emsao.py and
core/xcsao.py, ports of RVSAO 2.8.5 held to it line for line; what is shared with
RVSAO's other tasks (the spectrum, the time of the observation, the barycentric
correction) is core/rvsao.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "continuum": "The spectrum smoothed and its continuum fit and subtracted "
                 "(contpars: spline3 of 10 pieces, clipped at 2 sigma), so that a "
                 "line is what stands above it.",
    "guess": "A first redshift from the tallest line near [O II], Hbeta, [O III] "
             "or Halpha (vel_init search), or the one given.",
    "fit": "Each line of the list where that redshift puts it, if it stands linesig "
           "times its continuum's rms above it; then one, two or three Gaussians "
           "on a straight background around it ([N II]-Halpha-[N II] and the "
           "[S II] and [O III] pairs together).",
    "combine": "The lines' redshifts weighted by their centres' errors; a line too "
               "near an end, or with too small an equivalent width, left out.",
}


class EmsaoParams(TaskParams):
    spectra: list[str] = Field(..., description="Spectra (one-dimensional, multispec "
                                                "or harwood's), a list or @file.",
                               json_schema_extra={"positional": True})
    specnum: str = Field("0", description="Apertures of a multispec ('1-3'); 0: the "
                                          "first.")
    specband: int = Field(0, description="Band of a multispec.")
    specdir: str = Field("", description="Directory of the spectra.")
    st_lambda: float | None = Field(None, description="Starting wavelength; negative: "
                                                      "that much in from the start.")
    end_lambda: float | None = Field(None, description="Ending wavelength; negative: "
                                                       "that much in from the end.")
    nsmooth: int = Field(10, description="Smoothing passes (1-2-1) for the search.")
    vel_init: Literal["search", "guess", "zguess"] = Field(
        "search", description="Initial velocity: searched for, czguess as cz (km/s), "
                              "or czguess as z.")
    czguess: float = Field(0.0, description="The initial cz (guess) or z (zguess).")
    wspan: float = Field(10.0, description="Angstroms searched around each redshifted "
                                           "line (its fraction is dropped, as RVSAO's).")
    linesig: float = Field(1.5, description="Sigmas above the continuum a line must "
                                            "stand.")
    emsearch: str = Field("emsearch.dat", description="Lines the initial velocity is "
                                                      "searched with.")
    emlines: str = Field("emlines.dat", description="Emission line list.")
    linedir: str = Field("rvsao$lib/", description="Directory of the line lists "
                                                   "(rvsao$lib/: RVSAO's own).")
    npfit: int = Field(2, description="Pixels each side of a peak.")
    esmooth: int = Field(0, description="Smoothing passes for the line fits.")
    emcombine: str = Field("emcomb.dat", description="Lines fit together (blends).")
    mincont: float = Field(0.0, description="Continuum above which equivalent widths "
                                            "are computed.")
    lsmin: float = Field(1.0, description="Minimum equivalent width, in its sigmas, of "
                                          "a line counted.")
    sigline: float = Field(0.0, description="Velocity error of a single line (0: the "
                                            "fit's).")
    disperr: float = Field(0.01, description="RMS dispersion error, Angstroms.")
    vel_corr: Literal["none", "file", "heliocentric", "barycentric", "hfile"] = Field(
        "barycentric", description="Correction to the solar system's barycentre (or the "
                                   "Sun), from the header's date, time, position and "
                                   "observatory; file/hfile: its BCV/HCV cards.")
    report_mode: Literal[1] = Field(1, description="Report format (1, the normal one).")
    logfiles: str = Field("STDOUT,emsao.log", description="Where the report goes: "
                                                          "STDOUT and files, appended.")
    c_function: Literal["spline3", "legendre", "chebyshev", "spline1"] = Field(
        "spline3", description="contpars: continuum function.")
    c_sample: str = Field("*", description="contpars: sample of the continuum fit.")
    naverage: int = Field(1, description="contpars: points averaged.")
    order: int = Field(10, ge=0, description="contpars: order (pieces of a spline).")
    s_low_reject: float = Field(2.0, description="contpars: low rejection, sigmas.")
    s_high_reject: float = Field(2.0, description="contpars: high rejection, sigmas.")
    niterate: int = Field(10, description="contpars: rejection iterations.")
    grow: float = Field(1.0, description="contpars: rejection growing radius.")


@register
class Emsao(Task[EmsaoParams]):
    """RVSAO's emsao (Smithsonian Astrophysical Observatory, an IRAF external
    package): the redshift from emission lines, held to RVSAO 2.8.5 compiled against
    IRAF 2.18 -- its log identical on 44 cases. Not ported yet: the graph and its
    cursor (displot, curmode), saving the velocity to the header, archives, the sky
    spectrum and the report modes other than 1."""

    name = "emsao"
    package = "rvsao"
    description = "Redshift from emission lines (RVSAO's emsao)."
    Params = EmsaoParams

    def plan(self, params: EmsaoParams) -> list[Step]:
        return [Step("continuum", ", ".join(params.spectra), why=WHY["continuum"]),
                Step("guess", params.vel_init, why=WHY["guess"]),
                Step("fit", params.emlines, why=WHY["fit"]),
                Step("combine", params.vel_corr, why=WHY["combine"])]

    def execute(self, params: EmsaoParams) -> TaskResult:
        from ..core import emsao, images

        values = params.model_dump()
        names = [str(s) for s in images.expand(params.spectra)] \
            if not any(n.endswith("]") for n in params.spectra) else list(params.spectra)
        session = emsao.Emsao(values)
        try:
            session.run(names)
        except emsao.RvsaoError as error:
            advice.fail(f"emsao stopped: {error}", "check the line lists and linedir")
        written = []
        logs = [w for w in session.logfiles() if w != "STDOUT"]
        for name in logs:
            with open(name, "a", encoding="utf-8") as fd:
                fd.writelines(session.log)
            written.append(Path(name))
        rows = []
        for r in session.results:
            rows.append({
                "spectrum": r.spectrum.name, "object": r.spectrum.object,
                "cz": None if r.vel == emsao.INDEFD else r.vel, "cz_error": r.err,
                "z": None if r.vel == emsao.INDEFD else r.vel / emsao.C0,
                "lines_used": r.nfit, "lines_found": len(r.lines),
                "correction": r.spectrum.hcv,
                "lines": [{"name": f.name, "rest": f.wlrest, "observed": f.wlobs,
                           "z": float(f.params[0, 8]), "eqw": float(f.params[0, 6]),
                           "weight": f.weight, "dropped": f.drop}
                          for f in r.lines]})
        data: dict = {"spectra": rows}
        if "STDOUT" in session.logfiles():
            data["text"] = session.out.splitlines()
        return TaskResult(outputs=written, steps=self.plan(params), data=data)


# ------------------------------------------------------------- xcsao

XWHY = {
    "rebin": "The spectrum and each template rebinned onto one grid even in log "
             "wavelength (ncols points) over the range they share, so that a "
             "velocity is a shift by the same number of bins everywhere.",
    "continuum": "Each one's continuum fit and taken off (contpars), its ends "
                 "tapered (bell_window), and -- if asked -- its emission or "
                 "absorption lines clipped, so that the lines alone correlate.",
    "correlate": "Their transforms filtered (low_bin..nrun: the slow wiggles of the "
                 "continuum and the pixel noise out) and multiplied: the "
                 "cross-correlation, normalised by their rms.",
    "peak": "The highest peak inside minvel..maxvel fit by a parabola: its centre is "
            "the velocity; R, its height over twice the rms of the correlation's "
            "antisymmetric part (Tonry & Davis 1979); the error 3w/8(1+R).",
}


class XcsaoParams(TaskParams):
    spectra: list[str] = Field(..., description="Spectra (one-dimensional, multispec "
                               "or a list).", json_schema_extra={"positional": True})
    templates: str = Field("", description="Templates, separated by commas; name:w1-w2 "
                           "correlates that template over w1..w2 Angstroms only.")
    tempdir: str = Field("", description="Directory of the templates. rvsao$templates is "
                         "RVSAO's own, which come with harwood.")
    list_templates: bool = Field(False, description="List the RVSAO templates that come "
                                 "with harwood, with their range and dispersion, and stop.")
    specnum: str = Field("0", description="Apertures of a multispec ('1-3'); 0: the "
                                          "spectrum.")
    specband: int = Field(0, description="Band of a multispec.")
    specdir: str = Field("", description="Directory of the spectra.")
    tempband: int = Field(1, description="Band of a multispec template.")
    st_lambda: float | None = Field(None, description="Starting wavelength to "
                                                      "correlate (empty: the overlap's).")
    end_lambda: float | None = Field(None, description="Ending wavelength (empty: the "
                                                       "overlap's).")
    s_emchop: Literal["no", "yes", "tempfile", "specfile"] = Field(
        "no", description="Clip the spectrum's emission (or absorption) lines.")
    t_emchop: Literal["no", "yes", "tempfile"] = Field(
        "no", description="Clip the template's lines.")
    s_abs_reject: float = Field(100.0, description="Spectrum: absorption clipped below "
                                                   "this many sigmas of the continuum.")
    s_em_reject: float = Field(2.0, description="Spectrum: emission clipped above this "
                                                "many sigmas.")
    t_abs_reject: float = Field(0.0, description="Template: absorption clipping, sigmas.")
    t_em_reject: float = Field(0.0, description="Template: emission clipping, sigmas.")
    bell_window: float = Field(0.05, description="Fraction tapered at each end.")
    renormalize: bool = Field(False, description="Scale the spectrum to a mean of 1000 "
                                                 "first (done anyway if all its counts "
                                                 "are under 1).")
    ncols: int = Field(2048, ge=256, le=65536, description="Points of the log grid (a "
                                                           "power of two is used).")
    interp_mode: Literal["spline3", "linear", "nearest"] = Field(
        "spline3", description="Interpolation onto the grid.")
    zeropad: Literal["no", "yes", "tempfile"] = Field(
        "no", description="Pad the transforms with zeros (twice as many points).")
    low_bin: int = Field(5, description="Filter: bin where the low cut starts.")
    top_low: int = Field(20, description="Filter: bin where it reaches one.")
    top_nrun: int = Field(125, description="Filter: bin where the high cut starts.")
    nrun: int = Field(250, description="Filter: bin where it reaches zero.")
    vel_init: Literal["zero", "guess", "zguess"] = Field(
        "zero", description="Initial velocity: none, czguess in km/s, or czguess as z.")
    czguess: float = Field(0.0, description="The initial cz (guess) or z (zguess).")
    nzpass: int = Field(0, ge=0, description="Times the template is shifted to the "
                                             "velocity found and correlated again.")
    tshift: float = Field(0.0, description="Night-to-night zero point, km/s.")
    svel_corr: Literal["none", "file", "heliocentric", "barycentric", "hfile"] = Field(
        "barycentric", description="Velocity correction of the spectrum.")
    tvel_corr: Literal["none", "file", "heliocentric", "barycentric", "hfile"] = Field(
        "file", description="Velocity correction of the templates.")
    pkmode: Literal[1, 2] = Field(1, description="Peak fit: 1, a parabola; 2, a quartic.")
    pkfrac: float = Field(0.5, description="Fraction of the peak fit (above 1: that "
                                           "many points).")
    minvel: float | None = Field(-5000.0, description="Lowest velocity looked at.")
    maxvel: float | None = Field(100000.0, description="Highest velocity looked at.")
    report_mode: Literal[1, 2] = Field(1, description="Report format: 1, the normal one; "
                                                     "2, a line per template.")
    logfiles: str = Field("STDOUT,xcsao.log", description="Where the report goes: "
                                                          "STDOUT and files, by commas.")
    c_function: Literal["spline3", "legendre", "chebyshev", "spline1"] = Field(
        "spline3", description="contpars: continuum function.")
    c_sample: str = Field("*", description="contpars: sample of the continuum fit.")
    naverage: int = Field(1, description="contpars: points averaged.")
    order: int = Field(10, ge=0, description="contpars: order (pieces of a spline).")
    s_low_reject: float = Field(2.0, description="contpars: spectrum low rejection.")
    s_high_reject: float = Field(2.0, description="contpars: spectrum high rejection.")
    t_low_reject: float = Field(2.0, description="contpars: template low rejection.")
    t_high_reject: float = Field(2.0, description="contpars: template high rejection.")
    niterate: int = Field(10, description="contpars: rejection iterations.")
    grow: float = Field(1.0, description="contpars: rejection growing radius.")


def _template_list(task, params) -> TaskResult:
    """The RVSAO templates harwood carries: name, pixels, range and dispersion (with
    log-linear ones in Angstroms at their ends), and the object."""
    from astropy.io import fits

    from ..core.rvsao import TEMPLATES

    rows, text = [], [f"{'template':<20} {'npix':>6} {'start':>10} {'end':>10} "
                      f"{'A/pix':>7}  object"]
    for path in sorted(TEMPLATES.glob("*.fits")):
        h = fits.getheader(path)
        n = int(h.get("NAXIS1", 0))
        c1 = float(h.get("CRVAL1", 0.0))
        d1 = float(h.get("CDELT1", h.get("CD1_1", 0.0)) or 0.0)
        p1 = float(h.get("CRPIX1", 1.0))
        log = int(h.get("DC-FLAG", 0)) == 1
        a = c1 + (1 - p1) * d1
        b = c1 + (n - p1) * d1
        if log:
            a, b = 10 ** a, 10 ** b
        step = (b - a) / max(n - 1, 1)
        obj = str(h.get("OBJECT", "")).strip()
        rows.append({"template": path.stem, "npix": n, "start": round(a, 2),
                     "end": round(b, 2), "dispersion": round(step, 4), "log": log,
                     "object": obj})
        text.append(f"{path.stem:<20} {n:>6} {a:>10.2f} {b:>10.2f} {step:>7.4f}"
                    f"{' log' if log else '    '}  {obj}")
    text.append(f"{len(rows)} templates in rvsao$templates/ ({TEMPLATES}); "
                "use them by name: --templates fallstars,femtemp97")
    return TaskResult(outputs=[], steps=[Step("list", f"{len(rows)} templates",
                                              why="RVSAO's templates, shipped with harwood")],
                      data={"rows": rows, "text": text})


@register
class Xcsao(Task[XcsaoParams]):
    """RVSAO's xcsao (Smithsonian Astrophysical Observatory, an IRAF external
    package): the radial velocity by cross-correlation with templates (Tonry & Davis
    1979), held to RVSAO 2.8.5 compiled against IRAF 2.18 -- its log identical on
    55 cases. Not ported yet: the graph and its cursor (displot, curmode), saving
    the velocity to the header, archives, correlate in wavelength or pixels, the
    cosine peak fit (pkmode 3) and the report modes other than 1 and 2."""

    name = "xcsao"
    package = "rvsao"
    description = "Radial velocity by cross-correlation (RVSAO's xcsao)."
    Params = XcsaoParams

    def plan(self, params: XcsaoParams) -> list[Step]:
        return [Step("rebin", f"{', '.join(params.spectra)} x {params.templates}",
                     why=XWHY["rebin"]),
                Step("continuum", f"bell {params.bell_window:g}", why=XWHY["continuum"]),
                Step("correlate", f"filter {params.low_bin} {params.top_low} "
                     f"{params.top_nrun} {params.nrun}", why=XWHY["correlate"]),
                Step("peak", f"pkfrac {params.pkfrac:g}", why=XWHY["peak"])]

    def execute(self, params: XcsaoParams) -> TaskResult:
        from ..core import images, xcsao

        if params.list_templates:
            return _template_list(self, params)
        if not params.templates.strip():
            advice.fail("xcsao needs templates to correlate with",
                        "give --templates star.fits (several by commas), and --tempdir "
                        "for their folder. RVSAO's own templates come with harwood: "
                        "--templates fallstars,femtemp97 (the list: harwood xcsao "
                        "--list-templates)")
        values = params.model_dump()
        names = [str(s) for s in images.expand(params.spectra)] \
            if not any(n.endswith("]") for n in params.spectra) else list(params.spectra)
        session = xcsao.Xcsao(values)
        try:
            session.run(names)
        except xcsao.RvsaoError as error:
            advice.fail(f"xcsao stopped: {error}", "check the templates and tempdir")
        written = []
        for name in [w for w in session.logfiles() if w != "STDOUT"]:
            with open(name, "a", encoding="utf-8") as fd:
                fd.writelines(session.log)
            written.append(Path(name))
        rows = []
        for r in session.results:
            rows.append({
                "spectrum": r.spectrum.name, "object": r.spectrum.object,
                "cz": r.xcvel, "cz_error": r.xcerr, "r": r.xcr,
                "z": r.xcvel / xcsao.C0, "correction": r.spectrum.hcv,
                "best": r.temps[r.best - 1].template.id,
                "templates": [{"template": t.template.id, "cz": t.zvel,
                               "cz_error": t.czerr, "r": t.czr, "height": t.height,
                               "peak": t.cz} for t in r.temps]})
        data: dict = {"spectra": rows}
        if "STDOUT" in session.logfiles():
            data["text"] = session.out.splitlines()
        warnings = [advice.warn(line.strip(), "check the name and the directory")
                    for line in session.err.splitlines() if line.strip()]
        return TaskResult(outputs=written, steps=self.plan(params), data=data,
                          warnings=warnings)
