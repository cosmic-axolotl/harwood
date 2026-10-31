"""mkobsfile and mknobsfile: the observations file straight from the photometry.

From photcal/mkobsfile.cl and mknobsfile.cl, which are scripts: pdump takes the
fields of one aperture out of each photometry file into a temporary file, in the
order obsfile is told to read them, and obsfile does the rest. The two scripts
differ only in what they ask at the terminal, which is mkphotcors's part and not
ported: here the image sets come from a file, as `mkimsets` writes them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import obsfile as obs
from .photfiles import read

#: Where obsfile finds its fields in what `dump` writes: the image, the x and y, the
#: ifilter, then OTIME and so on, as the scripts have it; no id.
INCOLUMNS = "1,2,3,6,9,7,8,4,5,0"


def _text(value) -> str:
    """A field as pdump prints it: what is undefined is INDEF."""
    if np.ma.is_masked(value):
        return "INDEF"
    text = str(value).strip()
    if text in ("", "nan", "--"):
        return "INDEF"
    return text


def dump(photfiles: list[str], aperture: int = 1) -> str:
    """pdump: IMAGE XCENTER YCENTER MAG MERR IFILTER XAIRMASS OTIME ITIME, a star
    a line, for the aperture named."""
    lines = []
    for path in photfiles:
        table = read(path)
        mag, merr = f"MAG{aperture}", f"MERR{aperture}"
        if mag not in table.colnames:
            raise ValueError(f"The photometry file {path} has no aperture {aperture}")
        for row in table:
            lines.append(" ".join(_text(row[name]) for name in (
                "IMAGE", "XCENTER", "YCENTER", mag, merr, "IFILTER", "XAIRMASS", "OTIME",
                "ITIME")))
    return "".join(line + "\n" for line in lines)


def mkobsfile(photfiles: list[str], idfilters: str, imsets: str, observations: str,
              wrap: bool = True, obsparams: str = "", obscolumns: str = "2 3 4 5",
              minmagerr: float = 0.001, shifts: str = "", apercors: str = "",
              aperture: int = 1, tolerance: float = 5.0, allfilters: bool = False,
              verbose: bool = True) -> obs.Result:
    """The scripts' body. Raises ValueError with IRAF's message when the output
    is there already, or an input is not."""
    if Path(observations).exists():
        raise ValueError(f"The output catalog file {observations} already exists")
    for what, path in (("image set", imsets), ("obsparams", obsparams), ("shifts", shifts),
                       ("apercors", apercors)):
        if path and not Path(path).exists():
            raise ValueError(f"The {what} file {path} does not exist")
    with tempfile.TemporaryDirectory() as folder:
        data = Path(folder) / "photometry"
        data.write_text(dump(photfiles, aperture))
        result = obs.obsfile([str(data)], INCOLUMNS, idfilters, imsets, observations,
                             wrap=wrap, obsparams=obsparams,
                             obscolumns="1," + obscolumns if obsparams else "2 3 4 5",
                             minmagerr=minmagerr, shifts=shifts, apercors=apercors,
                             normtime=False, tolerance=tolerance, allfilters=allfilters,
                             verbose=verbose)
    return result
