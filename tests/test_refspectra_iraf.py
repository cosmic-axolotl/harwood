"""refspectra against IRAF 2.18: fifteen runs over two nights and two
apertures -- every selection, groups, apertures and refaps, a sort on UT
with timewrap -- compared header by header and line by line."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import harwood
from harwood.core import iraffmt, ranges

REF = json.loads((Path(__file__).parent / "data" / "iraf_refspectra.json").read_text())
OBJECTS = sorted(n for n, v in REF["frames"].items() if not v[3])
ARCS = sorted(n for n, v in REF["frames"].items() if v[3])


@pytest.mark.parametrize("case", REF["cases"],
                         ids=[f"{c['select']}-{c['params']}" for c in REF["cases"]])
def test_refspectra_assigns_what_iraf_assigns(case, tmp_path, monkeypatch):
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    for name, (jd, ljd, beam, arc, ut) in REF["frames"].items():
        header = fits.Header({"JD": jd, "LJD": ljd, "BEAM-NUM": beam, "UT": ut,
                              "OBJECT": name})
        if arc:
            header["REFSPEC1"] = name
        fits.PrimaryHDU(np.zeros(16, np.float32), header).writeto(f"{name}.fits")
    params = {k: {"yes": True, "no": False}.get(v, v) for k, v in case["params"].items()}
    inputs = OBJECTS if case["select"] != "match" else OBJECTS[:3]
    refs = ARCS if case["select"] != "match" else ARCS[:3]
    result = harwood.run("refspectra", input=[f"{n}.fits" for n in inputs],
                         references=[f"{n}.fits" for n in refs], select=case["select"],
                         **params)
    for name in OBJECTS:
        header = fits.getheader(f"{name}.fits")
        assert f"{header.get('REFSPEC1')!r}|{header.get('REFSPEC2')!r}" \
            == case["headers"][name], name
    printed = [line for line in case["stdout"].splitlines() if line.strip()]
    assert result.data.get("text", []) == printed
    for complaint in case["stderr"].splitlines():
        if complaint.startswith(("[", "Input")):
            assert any(complaint in str(w) for w in result.warnings), complaint


def test_ctod_reads_decimals_as_iraf_does():
    """The digits as one integer, times a power of ten: not always the
    nearest double, and the difference decides a tie in refspectra."""
    assert iraffmt.ctod("2448647.675") == 2448647675 * (0.1 * (0.1 * 0.1))
    assert iraffmt.ctod("2448647.675") != float("2448647.675")
    assert iraffmt.ctod("2.25") == 2.2500000000000004, "225 * 0.1 * 0.1, as pow_di does"
    assert iraffmt.ctod("12:30:45") == pytest.approx(12.5125)
    assert iraffmt.ctod("1.5D2") == 150.0
    assert iraffmt.ctod("abc") is None


def test_range_lists_read_as_iraf_reads_them():
    r = ranges.parse("1,3-5,10:20:5")
    assert [x for x in range(25) if x in r] == [1, 3, 4, 5, 10, 15, 20]
    assert 7 in ranges.parse("")
    assert [x for x in range(12) if x in ranges.parse("2x3")] == [2, 5, 8, 11]
    with pytest.raises(ValueError, match="Range syntax"):
        ranges.parse("1:2:3:4")


def test_a_reference_table_gives_each_input_its_own_arcs(tmp_path, monkeypatch):
    """reftable.x: 'inputs references' per line, and a line without
    references takes the previous one's."""
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    for name, (jd, ljd, beam, arc, ut) in REF["frames"].items():
        header = fits.Header({"JD": jd, "LJD": ljd, "BEAM-NUM": beam, "UT": ut})
        if arc:
            header["REFSPEC1"] = name
        fits.PrimaryHDU(np.zeros(16, np.float32), header).writeto(f"{name}.fits")
    Path("table").write_text("obj01.fits arc03.fits\nobj03.fits\n"
                             "obj05.fits arc01.fits,arc03.fits\n")
    harwood.run("refspectra", input=["obj01.fits", "obj03.fits", "obj05.fits", "obj07.fits"],
                references=["table"], select="nearest")
    got = {n: fits.getheader(f"{n}.fits").get("REFSPEC1") for n in ("obj01", "obj03",
                                                                     "obj05", "obj07")}
    assert got == {"obj01": "arc03", "obj03": "arc03", "obj05": "arc01", "obj07": None}
