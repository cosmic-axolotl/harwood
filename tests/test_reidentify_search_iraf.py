"""reidentify's shift search (a nonzero search radius) against IRAF's own.

tests/data/iraf_reidentify_search holds what IRAF 2.18.1's reidentify wrote
with shift INDEF and a search radius for multispec apertures and for a long
slit, traced and not. The shift is then found as IRAF's id_shift finds it:
autoidentify near the solution there is, the second time with weaker targets
if the first finds nothing. (A radius as a fraction of the range, -0.05, gave
IRAF's lines too, but at 15 Angstroms it is a search of two minutes, so it is
not kept here; a fraction is what identify's x key searches, -1, and its
cursor test checks it.)

The same lines must be found, the same ones rejected, and the log must read
the same to within one in the last figure IRAF prints. Not to its last
character, as the other reidentify tests ask: the search's answer is a fit of
its own, and center1d stops within 0.001 pixel of wherever it starts, so a
start a hundredth of an Angstrom different leaves the fitted wavelengths of
the end lines a thousandth of an Angstrom apart.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from test_reidentify_iraf import write_night

from harwood.core import database
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data" / "iraf_reidentify_search"
REFERENCE = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))
NUMBER = re.compile(r"-?\d+\.?\d*(?:E-?\d+)?")


def _close(ours: str, theirs: str) -> bool:
    """The same line, its numbers to one in their last printed figure."""
    if NUMBER.sub("#", ours) != NUMBER.sub("#", theirs):
        return False
    for a, b in zip(NUMBER.findall(ours), NUMBER.findall(theirs), strict=True):
        figures = b.split("E")[0]
        last = 10.0 ** -len(figures.split(".")[1]) if "." in figures else 1.0
        scale = 10.0 ** int(b.split("E")[1]) if "E" in b else 1.0
        if abs(float(a) - float(b)) > 1.01 * last * scale:
            return False
    return True


@pytest.mark.parametrize("case", list(REFERENCE["cases"]))
def test_the_shift_search_finds_what_iraf_finds(tmp_path, monkeypatch, case):
    entry = REFERENCE["cases"][case]
    iraf = DATA / case / "database"
    write_night(tmp_path)
    (tmp_path / "database").mkdir()
    name = f"id{entry['reference']}"
    autoidentified = re.split(r"(?m)^(?=# )", (iraf / name).read_text(encoding="utf-8"))[1]
    (tmp_path / "database" / name).write_text(autoidentified, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = get_task("reidentify").run(reference=entry["reference"], images=entry["images"],
                                        coordlist="arc_lines.dat", logfiles="",
                                        **entry["params"])
    assert len(result.data["text"]) == len(entry["log"])
    for ours, theirs in zip(result.data["text"], entry["log"], strict=True):
        assert _close(ours, theirs), f"{ours!r} != {theirs!r}"

    for image in entry["images"].split(","):
        ours = {record.section: record for record in database.read_all(image, ".")}
        theirs = {record.section: record for record in database.read_all(image, iraf.parent)}
        assert ours.keys() == theirs.keys()
        for section, t in theirs.items():
            m = ours[section]
            assert len(m) == len(t), f"{image}{section}: a different set of lines"
            assert (m.used == t.used).all(), f"{image}{section}: other lines rejected"
            assert np.abs(m.pixel - t.pixel).max() <= 0.008
            assert np.abs(m.fitted - t.fitted).max() < 2e-3
