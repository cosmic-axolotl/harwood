"""phot and center by the cursor, against IRAF 2.18.

IRAF's interactive mode, driven by an icommands file: which keys write a
row, which star of the list it is (lid), when the centre or the sky is
reused rather than refitted, where the cursor snaps onto a list star --
tests/data/iraf_phot_interactive.json has what IRAF wrote, and harwood
writes the same rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from iraf_field import CURSOR, field

import harwood
from harwood.core import catalog

REF = json.loads((Path(__file__).parent / "data" / "iraf_phot_interactive.json")
                 .read_text())


@pytest.fixture
def night(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARWOOD_DISPLAY", "headless")
    fits.PrimaryHDU(np.asarray(field(), dtype=np.float32)).writeto("field.fits")
    catalog.write(Table({"x": [c[0] for c in CURSOR], "y": [c[1] for c in CURSOR]}),
                  tmp_path / "field.coo")
    return tmp_path


def test_phot_by_the_cursor_writes_iraf_s_rows(night):
    (night / "icur").write_text(REF["phot_commands"])
    result = harwood.run("phot", image="field.fits", coords="field.coo", icommands="icur",
                         annulus=10, dannulus=10, output=night / "f.mag")
    table = catalog.read(result.outputs[0])
    assert len(table) == len(REF["phot"])
    for row, ref in zip(table, REF["phot"], strict=True):
        assert (row["id"], row["lid"]) == (ref["id"], ref["lid"])
        assert (row["xinit"], row["yinit"]) == (ref["xinit"], ref["yinit"])
        assert (round(row["x"], 3), round(row["y"], 3)) == (ref["x"], ref["y"])
        assert row["cier"] == ref["cier"] and row["nsky"] == ref["nsky"]
        assert row["msky"] == pytest.approx(float(ref["msky"]), rel=1e-6)
        for radius, total, *_rest in ref["apertures"]:
            suffix = "" if "sum" in table.colnames else f"_{radius:g}"
            # seven figures printed, the last one a single-precision sum's
            assert row[f"sum{suffix}"] == pytest.approx(total, rel=1e-6)


def test_center_by_the_cursor_writes_iraf_s_rows(night):
    (night / "ccur").write_text(REF["center_commands"])
    result = harwood.run("center", image="field.fits", coords="field.coo",
                         icommands="ccur", output=night / "f.ctr")
    table = catalog.read(result.outputs[0])
    got = [{"id": int(r["id"]), "lid": int(r["lid"]), "xinit": float(r["xinit"]),
            "yinit": float(r["yinit"]), "x": round(float(r["x"]), 3),
            "y": round(float(r["y"]), 3), "cier": int(r["cier"])} for r in table]
    assert got == REF["center"]
