"""rvsao -- the Smithsonian Astrophysical Observatory's radial velocity package.

RVSAO is an IRAF external package (Mink & Kurtz 1998), not part of IRAF; harwood
carries its emsao, which measures a redshift from emission lines -- the step of the
course's exercise 7.4. The computation is core/emsao.py, a port of RVSAO 2.8.5
held to it line for line; what is shared with RVSAO's other tasks (the spectrum,
the time of the observation, the barycentric correction) is core/rvsao.py.
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
