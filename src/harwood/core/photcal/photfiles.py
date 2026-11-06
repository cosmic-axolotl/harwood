"""The photometry photcal reads: IRAF's apphot text files, or harwood's phot.

photcal's tasks were written against apphot's text database, and read its
fields by IRAF's names -- IMAGE, IFILTER, ITIME, XAIRMASS, OTIME, XCENTER,
and RAPERT, MAG and MERR per aperture. harwood's phot writes ECSV with its
own names (x, mag_3, ...); here either becomes a table with IRAF's names,
so the tasks read both the same way. From harwood's, the values keep their
full precision: IRAF's file rounds the magnitudes to three decimals before
photcal sees them, which is the only difference the two make downstream.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def is_ecsv(path: str | Path) -> bool:
    with open(path, errors="replace", encoding="utf-8") as handle:
        return handle.readline().startswith("# %ECSV")


def read(path: str | Path):
    """The file as a table with IRAF's column names, apertures numbered."""
    from astropy.table import Table

    if not is_ecsv(path):
        table = Table.read(str(path), format="ascii.daophot")
        # A file of one aperture names its fields RAPERT, MAG...; pdump still reads
        # them as mag[1], and photcal's scripts ask for that.
        if "MAG" in table.colnames and "MAG1" not in table.colnames:
            for name in ("RAPERT", "SUM", "AREA", "FLUX", "MAG", "MERR", "PIER",
                         "PERROR"):
                if name in table.colnames:
                    table.rename_column(name, f"{name}1")
        return table
    table = Table.read(str(path), format="ascii.ecsv")
    n = len(table)
    image = str(table.meta.get("harwood_image", Path(path).stem))   # as given to phot
    out = Table()
    out["IMAGE"] = [image] * n
    for name, column, default in (("IFILTER", "ifilter", "INDEF"),
                                  ("ITIME", "itime", np.nan),
                                  ("XAIRMASS", "xairmass", np.nan),
                                  ("OTIME", "otime", "INDEF")):
        out[name] = table[column] if column in table.colnames else [default] * n
    out["XCENTER"], out["YCENTER"] = table["x"], table["y"]
    if "id" in table.colnames:
        out["ID"] = table["id"]
    for k, (radius, suffix) in enumerate(_apertures(table), start=1):
        out[f"RAPERT{k}"] = [radius] * n
        out[f"MAG{k}"] = table[f"mag{suffix}"]
        out[f"MERR{k}"] = table[f"merr{suffix}"]
    return out


def _apertures(table) -> list[tuple[float, str]]:
    """The radii, smallest first, and the suffix of their columns."""
    found = []
    for name in table.colnames:
        match = re.fullmatch(r"mag_([0-9.eE+-]+)", name)
        if match:
            found.append((float(match.group(1)), f"_{match.group(1)}"))
    if found:
        return sorted(found)
    radii = table.meta.get("harwood_apertures") or [np.nan]
    return [(float(radii[0]), "")]
