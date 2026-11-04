"""mkimsets: which images are one field seen in each filter.

From photcal/mkimsets.cl and mkimsets/t_imgroup.x. The images' filters (and
exposure times, airmasses and times) are read from the photometry files, as
pdump prints them, or from the image headers; the list is sorted -- by name,
or by a field -- and walked in order, an image of each filter making a set.
A filter seen again before the set is complete closes it, its missing filters
written as INDEF. With edit, the list goes to the editor first, below IRAF's
explanation, to be put in order; with rename, each set's name is asked for at
the terminal; with review, the sets go to the editor, below the explanation
and the warnings, and what is left of them without those is the sets file --
with the two blank lines IRAF's leaves at its top.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import iraffmt
from . import photfiles
from .io import words
from .obsfile import getlabels

#: photcal$mkimsets/imlist.key: what the editor shows above the image list.
KEY = """
 ##############################################################################
 #                                                                            #
 #  This file is a temporary file containing the entire list of images from   #
 #  which  the final  list of image sets wil be  formed.  The image  name     #
 #  appears in column 1. The image filter id appears in column 2. Additional  #
 #  fields, if defined, appear in succeeding columns.                         #
 #                                                                            #
 #  At the this point the user must edit the temporary image list to 1)       #
 #  remove unwanted images from the list 2) correct bad filter id values      #
 #  3) correct other bad field values, for example incorrect airmass or       #
 #  exposure time values if defined  4) reorder the image names so that the   #
 #  images in an observation set are adjacent to each other in the temporary  #
 #  image list 4) enter an image name of "INDEF" and the correct  filter id   #
 #  for any missing images.                                                   #
 #                                                                            #
 #  It is not necessary to remove images which have filter ids which are      #
 #  not in the filter id set as these will be removed by the grouping         #
 #  process.                                                                  #
 #                                                                            #
 #  Example:                                                                  #
 #                                                                            #
 #  Before editing                    After editing                           #
 #                                                                            #
 #  image2    B    STD1               image1    U    STD1                     #
 #  image3    V    STD1               image2    B    STD1                     #
 #  image5    B    STD2               image3    V    STD1                     #
 #  image6    V    STD2               INDEF     U    STD2                     #
 #  image7    V    STD3               image5    B    STD2                     #
 #  image8    B    STD3               image6    V    STD2                     #
 #  image9    U    STD3               image7    U    STD3                     #
 #  image1    U    STD1               image8    B    STD3                     #
 #                                    image9    V    STD3                     #
 #                                                                            #
 ##############################################################################
"""

#: photcal$mkimsets/imsets.key: what it shows above the warnings and the sets.
SETS_KEY = """
 ##############################################################################
 #                                                                            #
 #  This file is  a temporary file containing the  named image sets produced  #
 #  by MKIMSETS, preceded by any errors or warnings produced by the image     #
 #  grouping algorithm. The image set  name appears in column 1 followed by   #
 #  a space, colon, space and the individual member image names in filter     #
 #  order, as specified by the idfilters parameter.                           #
 #                                                                            #
 #  At this point the user may edit the temporary image set file to 1)        #
 #  correct any erroneous image set names,  keeping in mind that image set    #
 #  names for single standard star observations should match those in the     #
 #  standard star catalog, 2) remove unwanted image sets from the image set   #
 #  file, 3) correct any image sets where the grouping process did not        #
 #  function correctly. It is not necessary to remove the warning and error   #
 #  messages if present. They will be automatically deleted before the final  #
 #  image set file is written.                                                # 
 #                                                                            #
 #  Example:                                                                  #
 #                                                                            #
 #  Before editing                    After editing                           #
 #                                                                            #
 #  <Warning> Set 3 image im8 filter b is not a member of filter set U B V    # 
 #                                                                            #
 #  std1 : im1 im2 im3                std1 : im1 im2 im3                      #
 #  std22 : im4 im5 im6               std2 : im4 im5 im6                      #
 #  std3 : im7 INDEF im9              std3 : im7 im8 im9                      #
 #  m92 : im10 im11 im12                                                      #
 #                                                                            #
 ##############################################################################
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
    for line in Path(path).read_text(errors="replace", encoding="utf-8").split("\n"):
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
    rows = []
    for path in paths:
        if photfiles.is_ecsv(path):
            # harwood's phot: the same fields, printed as IRAF prints them.
            table = photfiles.read(path)
            rows += [[_text(row[name]) if name in table.colnames else "INDEF"
                      for name in names] for row in table]
        else:
            rows += [[record.get(name, "INDEF") for name in names]
                     for record in text_records(path)]
    return rows


def _text(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "INDEF" if np.isnan(value) else iraffmt.g(float(value))
    return str(value).strip() or "INDEF"


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


def imgroup(lines: list[str], filters: list[str], names: list[str] | None = None,
            terminal=None) -> tuple[list[ImageSet], list[str]]:
    """ph_group: the image sets, and what IRAF prints while making them. With a
    `terminal` (rename), each set is shown and its name asked for there."""
    nvalues = len(filters)
    messages: list[str] = []
    sets: list[ImageSet] = []
    names = list(names or [])
    record: list[str | None] = [None] * nvalues
    if terminal is not None:
        terminal.printf("\nEnter/Confirm Names for Each Image Set\n\n")

    def close() -> None:
        name = names[len(sets)] if len(sets) < len(names) else f"OBS{len(sets) + 1}"
        images = [r if r is not None else "INDEF" for r in record]
        if terminal is not None:
            # ph_swrite to the terminal, then ph_rname
            terminal.printf(f"{name} :" + "".join(f"  {imroot(i)}" for i in images) + "\n")
            terminal.printf(f"\tEnter new name for field {name} (name, <CR>=ok): ")
            got = terminal.scan()
            word = got.wrd() if got is not None else None
            name = word or name
            terminal.printf("\n")
        sets.append(ImageSet(name, images))

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
        f"{s.name} :" + "".join(f"  {imroot(i)}" for i in s.images) + "\n" for s in sets),
        encoding="utf-8")


def mkimsets(imlist: list[str], idfilters: str, imsets: str, imobsparams: str = "",
             input: str = "photfiles", filter: str = "", fields: str = "",
             sort: str = "", names: list[str] | None = None, edit: bool = False,
             rename: bool = False, review: bool = False, editor=None,
             terminal=None) -> tuple[list[ImageSet], list[str]]:
    """mkimsets.cl: the sets, and what it says. `names` answer rename in the sets'
    order without asking; with `rename` the rest are asked for at `terminal`. With
    `edit` and `review`, `editor(path)` edits the image list and the sets."""
    import tempfile

    from .entry import Terminal

    edit_file = editor if editor is not None else (lambda path: None)
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
    with tempfile.TemporaryDirectory() as folder:
        if edit:
            listing = Path(folder) / "imlist"
            listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
            edit_file(listing)
            lines = listing.read_text(encoding="utf-8").splitlines()
        asking = (terminal if terminal is not None else Terminal()) if rename else None
        sets, messages = imgroup(lines, filters, names, asking)
        write_sets(imsets, sets)
        if imobsparams:
            kept = [line for line in lines if not line.startswith(" #")]
            Path(imobsparams).write_text("\n".join(kept) + "\n", encoding="utf-8")
        if review:
            # The explanation, the warnings imgroup said, the sets: edited, and what
            # is not a warning (<) or the explanation ( #) left as the sets file.
            sheet = Path(folder) / "imsets"
            sheet.write_text(SETS_KEY + "\n".join(messages) + "\n"
                             + Path(imsets).read_text(encoding="utf-8"), encoding="utf-8")
            edit_file(sheet)
            kept = [line for line in sheet.read_text(encoding="utf-8").splitlines(True)
                    if not line.startswith(("<", " #"))]
            Path(imsets).write_text("".join(kept), encoding="utf-8")
    return sets, messages
