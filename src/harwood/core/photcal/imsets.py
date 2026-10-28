"""mkimsets: which images are one field seen in each filter.

From photcal/mkimsets.cl and mkimsets/t_imgroup.x, without the editor
sessions. The images' filters (and exposure times, airmasses and times)
are read from the photometry files, as pdump prints them, or from the image
headers; the list is sorted -- by name, or by a field -- and walked in
order, an image of each filter making a set. A filter seen again before the
set is complete closes it, its missing filters written as INDEF.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .io import words
from .obsfile import getlabels

KEY = """
 ##############################################################################
 #                                                                            #
 #  This file is a temporary file containing the entire list of images from   #
 #  which  the final  list of image sets wil be  formed.  The image  name     #
 #  appears in column 1. The image filter id appears in column 2. Additional  #
 #  fields, if defined, appear in succeeding columns.                         #
 #                                                                            #
"""


@dataclass
class ImageSet:
    name: str
    images: list[str]              # one per filter, "INDEF" where missing


def text_records(path: str | Path) -> list[dict[str, str]]:
    """An apphot text database as pdump reads it: the #N lines name the
    fields, a record runs over the lines ending in a backslash, and each
    field is kept as the text it is written as."""
    names: list[str] = []
    records, current = [], []
    for line in Path(path).read_text(errors="replace").split("\n"):
        if line.startswith("#N"):
            names += line[2:].rstrip().rstrip("\\").split()
            continue
        if line.startswith("#") or not line.strip():
            continue
        body = line.rstrip()
        more = body.endswith("\\")
        current += body.rstrip("\\").split()
        if not more:
            records.append(dict(zip(names, current, strict=False)))
            current = []
    return records


def photfile_rows(paths: list[str], fields: list[str]) -> list[list[str]]:
    """pdump IMAGE,IFILTER,ITIME,XAIRMASS,OTIME,fields of apphot files."""
    names = ["IMAGE", "IFILTER", "ITIME", "XAIRMASS", "OTIME"] + fields
    return [[record.get(name, "INDEF") for name in names]
            for path in paths for record in text_records(path)]


def image_rows(paths: list[str], filter_key: str, fields: list[str]) -> list[list[str]]:
    """hselect $I,filter,fields: the same from the image headers."""
    from astropy.io import fits

    rows = []
    for path in paths:
        header = fits.getheader(path)
        row = [str(path)]
        for key in [filter_key] + fields:
            value = header.get(key)
            row.append("INDEF" if value is None else str(value).strip())
        rows.append(row)
    return rows


def sort_rows(rows: list[list[str]], sort: str, filter_name: str,
              fields: list[str]) -> list[list[str]]:
    """sort | unique, then sort on a field numerically if asked."""
    lines = sorted({"  ".join(row): row for row in rows}.items())
    rows = [row for _, row in lines]
    if not sort:
        return rows
    if sort == "image":
        column = 0
    elif sort == filter_name:
        column = 1
    elif sort in fields:
        column = fields.index(sort) + 2
    else:
        return rows

    def number(row):
        try:
            text = row[column]
            return _sexagesimal(text) if ":" in text else float(text)
        except (ValueError, IndexError):
            return float("inf")

    return sorted(rows, key=number)


def _sexagesimal(text: str) -> float:
    sign = -1.0 if text.strip().startswith("-") else 1.0
    parts = [abs(float(p)) for p in text.strip().lstrip("+-").split(":")]
    return sign * sum(p / 60.0 ** k for k, p in enumerate(parts))


def imroot(image: str) -> str:
    """ph_imroot: the name without its directory."""
    head, bracket, tail = image.partition("[")
    return os.path.basename(head) + bracket + tail


def imgroup(lines: list[str], filters: list[str],
            names: list[str] | None = None) -> tuple[list[ImageSet], list[str]]:
    """ph_group: the image sets, and what IRAF prints while making them."""
    nvalues = len(filters)
    messages: list[str] = []
    sets: list[ImageSet] = []
    names = list(names or [])
    record: list[str | None] = [None] * nvalues

    def close() -> None:
        name = names[len(sets)] if len(sets) < len(names) else f"OBS{len(sets) + 1}"
        sets.append(ImageSet(name, [r if r is not None else "INDEF" for r in record]))

    for line in lines:
        tokens = words(line)
        if not tokens or tokens[0].startswith("#"):
            continue
        image = tokens[0]
        setval = tokens[1] if len(tokens) > 1 else ""
        if nvalues == 1 and filters[0] == "":
            num = 1
        else:
            num = 0
            for i, value in enumerate(filters, start=1):
                if setval == value:
                    num = i
            if num == 0:
                messages.append(f"<Warning>: Image {image} filter id {setval}  does not "
                                f"belong to the filter set {','.join(filters)}")
                continue
        clash = record[num - 1]
        if clash is not None:
            messages.append(f"<Warning> Set {len(sets) + 1} image {image} has the same "
                            f"filter id {setval} as image {clash}")
            close()
            record = [None] * nvalues
        record[num - 1] = image
        if all(r is not None for r in record):
            close()
            record = [None] * nvalues
    if any(r is not None for r in record):
        close()
    messages.append("\n<Warning> The image set file is empty\n" if not sets else "")
    return sets, messages


def write_sets(path: str | Path, sets: list[ImageSet]) -> None:
    Path(path).write_text("".join(
        f"{s.name} :" + "".join(f"  {imroot(i)}" for i in s.images) + "\n" for s in sets))


def mkimsets(imlist: list[str], idfilters: str, imsets: str, imobsparams: str = "",
             input: str = "photfiles", filter: str = "", fields: str = "",
             sort: str = "", names: list[str] | None = None) -> tuple[list[ImageSet],
                                                                      list[str]]:
    """mkimsets with edit, rename and review off, or rename answered by
    `names` in the sets' order."""
    filters = getlabels(idfilters) or [""]
    extra = getlabels(fields)
    if input == "photfiles":
        rows = photfile_rows(imlist, extra)
        filter_name = "IFILTER"
        extra = ["ITIME", "XAIRMASS", "OTIME"] + extra
    else:
        rows = image_rows(imlist, filter, extra)
        filter_name = filter
    rows = sort_rows(rows, sort, filter_name, extra)
    lines = KEY.split("\n")[:-1] + ["  ".join(row) for row in rows]
    sets, messages = imgroup(lines, filters, names)
    write_sets(imsets, sets)
    if imobsparams:
        kept = [line for line in lines if not line.startswith(" #")]
        Path(imobsparams).write_text("\n".join(kept) + "\n")
    return sets, messages
