"""IRAF's apphot text database: the .mag file phot writes, for --format iraf.

From apphot/aplib (apwparam1.x, apwres1.x, apwres2.x) and phot/appmag.x:
the #K block of parameters, then for each group of fields its #N names, #U
units and #F formats, then one record per star -- four lines, and one per
aperture, each 80 columns with a backslash continuing the record. IRAF's
tasks (txdump, pdump, mkapfile, ...) read it, and harwood's photcal too.

The layout is IRAF's fprintf with its %Nt, "on to column N", and fields
whose widths differ from the ones #F announces (the first field of a line
gives three columns to the indent): ported as written, not as documented.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import getpass
import math
import platform
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import iraffmt

_SPEC = re.compile(r"%(?:(\d+)t|(-?)(\d*)(?:\.(\d+))?([sdfgbc])|%)")


def fprintf(template: str, *args) -> str:
    """IRAF's fprintf for the conversions apphot uses: %Nt, %-W.Ps, %-Wd,
    %-W.Pf, %-W.Pg, %b, %c and %%. An undefined real prints INDEF."""
    out = ""
    values = iter(args)
    pos = 0
    for match in _SPEC.finditer(template):
        out += template[pos:match.start()]
        pos = match.end()
        column, minus, width, decpl, conv = match.groups()
        if match.group(0) == "%%":
            out += "%"
            continue
        if column:
            out = out.ljust(int(column) - 1)
            continue
        value = next(values)
        if conv == "s":
            text = str(value)
            if decpl:
                text = text[:int(decpl)]
        elif conv == "d":
            text = str(int(value))
        elif conv == "b":
            text = "yes" if value else "no"
        elif conv == "c":
            text = str(value)
        elif value is None or (isinstance(value, float) and not math.isfinite(value)):
            text = "INDEF"
        else:
            text = iraffmt.printf(float(value), conv, int(decpl) if decpl else None,
                                  width=int(width) if width else None)
        if width:
            text = text.ljust(int(width)) if minus else text.rjust(int(width))
        out += text
    return out + template[pos:]


# --------------------------------------------------------------- the #K block


def _k(keyword: str, value, units: str, fmt: str) -> str:
    conv = {"%-23.7g": "%-23.7g", "%-23d": "%-23d", "%-23b": "%-23b"}.get(fmt, "%-23.23s")
    return fprintf(f"#K%4t%-10.10s%14t = %17t{conv}%41t%-10.10s%52t%-10s\n",
                   keyword, value, units, fmt)


def r(keyword, value, units):
    return _k(keyword, value, units, "%-23.7g")


def i(keyword, value, units):
    return _k(keyword, value, units, "%-23d")


def b(keyword, value, units):
    return _k(keyword, value, units, "%-23b")


def s(keyword, value, units):
    return _k(keyword, value if value else '""', units, "%-23s")


def parameters(task: str, p: dict) -> str:
    """ap_param for phot: who, when, then datapars, the noise, centerpars,
    fitskypars and photpars, as IRAF lists them."""
    now = time.localtime()
    out = [s("IRAF", "harwood", "version"),
           s("USER", getpass.getuser(), "name"),
           s("HOST", platform.node(), "computer"),
           s("DATE", time.strftime("%Y-%m-%d", now), "yyyy-mm-dd"),
           s("TIME", time.strftime("%H:%M:%S", now), "hh:mm:ss"),
           s("PACKAGE", "apphot", "name"),
           s("TASK", task, "name"), "#\n",
           r("SCALE", p["scale"], "units"), r("FWHMPSF", p["fwhmpsf"], "scaleunit"),
           b("EMISSION", p["emission"], "switch"), r("DATAMIN", p["datamin"], "counts"),
           r("DATAMAX", p["datamax"], "counts"), s("EXPOSURE", p["exposure"], "keyword"),
           s("AIRMASS", p["airmass"], "keyword"), s("FILTER", p["filter"], "keyword"),
           s("OBSTIME", p["obstime"], "keyword"), "#\n",
           s("NOISE", p["noise"], "model"), r("SIGMA", p["sigma"], "counts"),
           s("GAIN", p["gain"], "keyword"), r("EPADU", p["epadu"], "e-/adu"),
           s("CCDREAD", p["ccdread"], "keyword"), r("READNOISE", p["readnoise"], "e-"),
           "#\n",
           s("CALGORITHM", p["calgorithm"], "algorithm"),
           r("CBOXWIDTH", p["cbox"], "scaleunit"), r("CTHRESHOLD", p["cthreshold"], "sigma"),
           r("MINSNRATIO", p["minsnratio"], "number"), i("CMAXITER", p["cmaxiter"], "number"),
           r("MAXSHIFT", p["maxshift"], "scaleunit"), b("CLEAN", p["clean"], "switch"),
           r("RCLEAN", p["rclean"], "scaleunit"), r("RCLIP", p["rclip"], "scaleunit"),
           r("KCLEAN", p["kclean"], "sigma"), "#\n",
           s("SALGORITHM", p["salgorithm"], "algorithm"),
           r("ANNULUS", p["annulus"], "scaleunit"), r("DANNULUS", p["dannulus"], "scaleunit"),
           r("SKYVALUE", p["skyvalue"], "counts"), r("KHIST", p["khist"], "sigma"),
           r("BINSIZE", p["binsize"], "sigma"), b("SMOOTH", p["smooth"], "switch"),
           i("SMAXITER", p["smaxiter"], "number"), r("SLOCLIP", p["sloclip"], "percent"),
           r("SHICLIP", p["shiclip"], "percent"), i("SNREJECT", p["snreject"], "number"),
           r("SLOREJECT", p["sloreject"], "sigma"), r("SHIREJECT", p["shireject"], "sigma"),
           r("RGROW", p["rgrow"], "scaleunit"), "#\n",
           s("WEIGHTING", "constant", "model"), s("APERTURES", p["apertures"], "scaleunit"),
           r("ZMAG", p["zmag"], "zeropoint"), "#\n"]
    return "".join(out)


# ------------------------------------------------------------ the headers

HEADERS = (
    ("#N%4tIMAGE%24tXINIT%34tYINIT%44tID%50tCOORDS%73tLID%80t\\\n"
     "#U%4timagename%24tpixels%34tpixels%44t##%50tfilename%73t##%80t\\\n"
     "#F%4t%%-23s%24t%%-10.3f%34t%%-10.3f%44t%%-6d%50t%%-23s%73t%%-6d%80t \n"),
    ("#N%4tXCENTER%15tYCENTER%26tXSHIFT%34tYSHIFT%42tXERR%50tYERR%66tCIER%71tCERROR%80t\\\n"
     "#U%4tpixels%15tpixels%26tpixels%34tpixels%42tpixels%50tpixels%66t##%71tcerrors%80t\\\n"
     "#F%4t%%-14.3f%15t%%-11.3f%26t%%-8.3f%34t%%-8.3f%42t%%-8.3f%50t%%-15.3f%66t%%-5d"
     "%71t%%-9s%80t \n"),
    ("#N%4tMSKY%19tSTDEV%34tSSKEW%49tNSKY%56tNSREJ%66tSIER%71tSERROR%80t\\\n"
     "#U%4tcounts%19tcounts%34tcounts%49tnpix%56tnpix%66t##%71tserrors%80t\\\n"
     "#F%4t%%-18.7g%19t%%-15.7g%34t%%-15.7g%49t%%-7d%56t%%-9d%66t%%-5d%71t%%-9s%80t \n"),
    ("#N%4tITIME%19tXAIRMASS%34tIFILTER%57tOTIME%80t\\\n"
     "#U%4ttimeunit%19tnumber%34tname%57ttimeunit%80t\\\n"
     "#F%4t%%-18.7g%19t%%-15.7g%34t%%-23s%57t%%-23s%80t \n"),
    ("#N%4tRAPERT%13tSUM%27tAREA%38tFLUX%52tMAG%59tMERR%66tPIER%71tPERROR%80t\\\n"
     "#U%4tscale%13tcounts%27tpixels%38tcounts%52tmag%59tmag%66t##%71tperrors%80t\\\n"
     "#F%4t%%-12.2f%13t%%-14.7g%27t%%-11.7g%38t%%-14.7g%52t%%-7.3f%59t%%-6.3f%66t%%-5d"
     "%71t%%-9s%80t \n"),
)


RPROF_HEADERS = (
    ("#N%4tPFWHM%14tINORM%29tTINORM%66tRIER%71tRERROR%80t\\\n"
     "#U%4tScale%14tcounts%29tcounts%66t##%71trerrors%80t\\\n"
     "#F%4t%%-13.3f%14t%%-15.7f%29t%%-36.7f%66t%%-5d%71t%%-9s%80t \n"),
    ("#N%4tPRADIUS%14tINTENSITY%29tTINTENSITY%80t\\\n"
     "#U%4tscale%14tcounts%29tcounts%80t\\\n"
     "#F%4t%%-13.3f%14t%%-15.7f%29t%%-15.7f%80t \n"),
)
RERRORS = {901: "OffImage", 902: "EdgeImage", 903: "TooFewPts", 904: "BadFit"}


def _template_lines(text: str) -> str:
    return "".join(fprintf(line + "\n") for line in text.split("\n") if line)


def headers() -> str:
    return "".join(_template_lines(h) + "#\n" for h in HEADERS)


# ---------------------------------------------------------------- records

CERRORS = {101: "OffImage", 102: "EdgeImage", 103: "LowSnr", 104: "TooFewPts",
           105: "Singular", 106: "BadFit", 107: "BigShift", 108: "BadPixels"}
SERRORS = {201: "OffImage", 202: "EdgeImage", 203: "NoHist", 204: "FlatHist",
           205: "TooFewPts", 206: "Singular", 207: "BadFit", 208: "NoGraph",
           209: "NoFile", 210: "ShortFile", 211: "BadRecord", 212: "BadParams"}
PERRORS = {301: "OffImage", 302: "EdgeImage", 303: "NoSky", 304: "NoFlux",
           305: "BadPixels"}
APERT_OUTOFBOUNDS, APERT_BADDATA = 302, 305


@dataclass
class Star:
    """What phot measured of one star, as IRAF writes it."""
    xinit: float
    yinit: float
    id: int
    lid: int
    x: float
    y: float
    xshift: float
    yshift: float
    xerr: float
    yerr: float
    cier: int
    msky: float
    stdev: float
    sskew: float
    nsky: int
    nsrej: int
    sier: int
    itime: float
    xairmass: float
    ifilter: str
    otime: str
    radii: list[float]
    sums: list[float]
    areas: list[float]
    flux: list[float]
    mag: list[float]
    merr: list[float]
    pier: int = 0
    nmin: int = 0
    nmax: int = 0
    extra: dict = field(default_factory=dict)


def _aperture_ier(star: Star, k: int) -> int:
    """ap_wmres: aperture k's own code (1-based) from the star's."""
    if not math.isfinite(star.mag[k - 1]):
        if star.pier != APERT_OUTOFBOUNDS:
            return star.pier
        return APERT_OUTOFBOUNDS if k > star.nmax else 0
    return APERT_BADDATA if k >= star.nmin else 0


def _real(value) -> float:
    """What IRAF holds in a real: the value it prints from."""
    return float(np.float32(value)) if value is not None else float("nan")


def record(image: str, coords: str, star: Star, more: bool = False) -> str:
    """ap_pmag: the star's four lines, and one per aperture. IRAF keeps
    positions, the sky and magnitudes in reals and the sums and areas in
    doubles, and the last digit printed depends on which: so here too.
    With `more` lines to come, radprof's, every aperture line goes on."""
    xinit, yinit, x, y = (_real(v) for v in (star.xinit, star.yinit, star.x, star.y))
    out = fprintf("%-23.23s%24t%-10.3f%34t%-10.3f%44t%-6d%50t%-23.23s%73t%-6d%80t%c\n",
                  image, xinit, yinit, star.id, coords or "nullfile", star.lid, "\\")
    out += fprintf("%4t%-11.3f%-11.3f%-8.3f%-8.3f%-8.3f%-15.3f%-5d%-9.9s%80t%c\n",
                   x, y, _real(x - xinit), _real(y - yinit), _real(star.xerr),
                   _real(star.yerr), star.cier, CERRORS.get(star.cier, "NoError"), "\\")
    out += fprintf("%4t%-15.7g%-15.7g%-15.7g%-7d%-9d%-5d%-9.9s%80t%c\n",
                   _real(star.msky), _real(star.stdev), _real(star.sskew), star.nsky,
                   star.nsrej, star.sier, SERRORS.get(star.sier, "NoError"), "\\")
    out += fprintf("%4t%-15.7g%-15.7g%-23.23s%-23.23s%80t\\\n",
                   _real(star.itime), _real(star.xairmass), star.ifilter, star.otime)
    n = len(star.radii)
    for k in range(1, n + 1):
        end = "  " if n == 1 else "* " if k == n else "*\\"
        if more:
            end = " \\" if n == 1 else "*\\"
        ier = _aperture_ier(star, k)
        merr = star.merr[k - 1]
        out += fprintf("%4t%-9.2f%-14.7g%-11.7g%-14.7g%-7.3f%-6.3f%-5d%-9.9s%79t%2s\n",
                       _real(star.radii[k - 1]), star.sums[k - 1], star.areas[k - 1],
                       # real(sum - sky * area), for every aperture
                       _real(star.sums[k - 1] - _real(star.msky) * star.areas[k - 1])
                       if math.isfinite(star.msky) else 0.0,
                       _real(star.mag[k - 1]),
                       _real(merr) if math.isfinite(merr) and merr <= 99.999 else None,
                       ier, PERRORS.get(ier, "NoError"), end)
    return out


def rprof_record(star: Star, profile, radius: float, step: float) -> str:
    """ap_wrres: the FWHM line and the profile. IRAF's rows go by radius / step, in real,
    plus one; a star with no profile still has them, its radii and INDEF (ap_rpindef)."""
    ier = profile["ier"]
    out = fprintf("%4t%-10.3f%-15.7g%-36.7g%-5d%-9.9s%80t\\\n", _real(profile["fwhm"]),
                  _real(profile["inorm"]), _real(profile["tinorm"]), ier,
                  RERRORS.get(ier, "NoError"))
    rows = int(np.float32(np.float32(radius) / np.float32(step))) + 1
    for k in range(rows):
        end = "* " if k == rows - 1 else "*\\"
        out += fprintf("%4t%-10.3f%-15.7g%-15.7g%79t%2s\n", _real(profile["radius"][k]),
                       _real(profile["intensity"][k]), _real(profile["tintensity"][k]), end)
    return out


def _rprof_card(name: str, value) -> str:
    """ap_profcolon: a change of radprof's parameters, as the file has it when it comes."""
    return {"radius": lambda: r("RADIUS", float(value), "scaleunit"),
            "step": lambda: r("STEPSIZE", float(value), "scaleunit"),
            "order": lambda: i("ORDER", int(value), "number"),
            "kreject": lambda: r("KREJECT", float(value), "sigma"),
            "nreject": lambda: i("NREJECT", int(value), "number")}[name]()


def write_rprof(path: str | Path, params: dict, image: str, coords: str, stars: list,
                first: dict) -> Path:
    """radprof's .prf: phot's text, with its parameters as they were at the first star, and a
    block for each star, as (star, profile, the settings then, the changes since the last)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = parameters("radprof", params)
    text += "".join(_rprof_card(name, first[name]) for name in
                    ("radius", "step", "order", "kreject", "nreject")) + "#\n"
    text += headers() + "".join(_template_lines(h) + "#\n" for h in RPROF_HEADERS)
    for star, profile, settings, cards in stars:
        text += "".join(_rprof_card(name, value) for name, value in cards)
        text += record(image, coords, star, more=True)
        text += rprof_record(star, profile, settings["radius"], settings["step"])
    path.write_text(text)
    return path


def write(path: str | Path, task: str, params: dict, image: str, coords: str,
          stars: list[Star]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = parameters(task, params) + headers()
    text += "".join(record(image, coords, star) for star in stars)
    path.write_text(text)
    return path
