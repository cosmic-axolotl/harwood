"""mkphotcors: the image sets, observing parameters, shifts and aperture
corrections mkobsfile needs, read from their files or asked for.

From photcal/mkobsfile/t_mkphotcors.x. Each of the four is a file: one that
exists is read, one that does not is asked for at the terminal, image by image
(entry.py), and written; the image sets left out are asked for and not written.
mkobsfile runs it first, on temporary files for the ones given as STDIN.

IRAF reads `obscolumns` here as it is given, "2 3 4 5", with no column for the
image, so that column 2 is taken for the image name; mkobsfile gives it so, and
says, of an obsparams file, that its filter is not in the image sets. harwood
says it too.
"""

from __future__ import annotations

from pathlib import Path

from . import entry
from .obsfile import (
    OBS_NFIELDS,
    ImTable,
    getlabels,
    read_apercors,
    read_obsparams,
    read_sets,
    read_shifts,
)


def _columns(text: str) -> list[int]:
    """t_mkphotcors's reading of obscolumns: a number per label, 0 when not one."""
    cols = [0] * OBS_NFIELDS
    for i, label in enumerate(getlabels(text)[:OBS_NFIELDS]):
        try:
            cols[i] = int(label)
        except ValueError:
            cols[i] = 0
    return cols


def mkphotcors(imsets: str, idfilters: str, obsparams: str = "", shifts: str = "",
               apercors: str = "", obscolumns: str = "2 3 4 5", verify: bool = False,
               verbose: bool = True, terminal: entry.Terminal | None = None
               ) -> tuple[ImTable, list[str], entry.Terminal]:
    """t_mkphotcors: the image table, what it says on the standard error, and the
    terminal it asked at. The files asked for are written."""
    term = terminal if terminal is not None else entry.Terminal()
    nfilters = len(getlabels(idfilters))
    notes: list[str] = []
    written = {}

    def kind(path: str) -> str:
        if not path:
            return "none"
        return "read" if Path(path).exists() else "ask"

    if kind(imsets) == "read":
        table = read_sets(imsets, nfilters, notes, verbose)
    else:
        table = entry.enter_sets(term, nfilters, verify)
        if imsets:
            written["sets"] = imsets
    if kind(obsparams) == "read":
        read_obsparams(table, obsparams, _columns(obscolumns), notes, verbose)
    elif kind(obsparams) == "ask":
        entry.enter_obsparams(term, table, verify)
        written["obs"] = obsparams
    if kind(shifts) == "read":
        read_shifts(table, shifts, notes, verbose)
    elif kind(shifts) == "ask":
        entry.enter_shifts(term, table, verify)
        written["sh"] = shifts
    if kind(apercors) == "read":
        read_apercors(table, apercors, notes, verbose)
    elif kind(apercors) == "ask":
        entry.enter_apercors(term, table, verify)
        written["ap"] = apercors
    if table.entries:
        write(table, written)
    return table, notes, term


def write(table: ImTable, paths: dict[str, str]) -> None:
    """ph_wimtable: the sets, observing parameters, shifts and corrections asked for."""
    out = {key: [] for key in paths}
    osetno = 0
    for k, item in enumerate(table.entries):
        if "sets" in out:
            if item.setno != osetno:
                out["sets"].append(f"{item.label} :" if item.setno == 1
                                   else f"\n{item.label} :")
            out["sets"].append(f"  {item.name}")
            if k == len(table.entries) - 1:
                out["sets"].append("\n")
        if "obs" in out:
            out["obs"].append(f"{item.name}  {item.ifilter}  {entry._g(item.itime)}  "
                              f"{entry._g(item.xairmass)} {entry._h(item.otime)}\n")
        if "sh" in out:
            out["sh"].append(f"{item.name}  {entry._g(item.xshift)}  "
                             f"{entry._g(item.yshift)}\n")
        if "ap" in out:
            out["ap"].append(f"{item.name}  {entry._g(item.apercor)}\n")
        osetno = item.setno
    for key, path in paths.items():
        Path(path).write_text("".join(out[key]), encoding="utf-8")
