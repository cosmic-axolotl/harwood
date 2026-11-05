"""sinterp against IRAF 2.18's: each interpolant at x's inside the table, at its
points and beyond its ends, along ranges given three ways, and into a spectrum.

tests/data/iraf_sinterp holds the table, the x's and the cases (si_cases.py), and
what IRAF printed and wrote (si.json). Printed with %12.5g, the lines are held to the
character; the spectrum to listpixels' seven figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_sinterp"
GOLD = json.loads((HERE / "si.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from si_cases import CASES, POINTS, TABLE  # noqa: E402


@pytest.fixture
def start(tmp_path, monkeypatch):
    (tmp_path / "table.dat").write_text(TABLE, encoding="utf-8")
    (tmp_path / "points.dat").write_text(POINTS, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", list(CASES))
def test_sinterp_interpolates_as_iraf_does(start, case):
    result = get_task("sinterp").run(tbl_file="table.dat", input="points.dat", image=case,
                                     **CASES[case])
    assert result.data["text"] == GOLD[case]["said"]
    if CASES[case].get("make_image"):
        aperture, = spectrum.read(f"{case}.fits")
        assert aperture.wavelength == pytest.approx(GOLD[case]["world"], rel=1e-9)
        assert aperture.flux == pytest.approx(GOLD[case]["pixels"], rel=6e-7)
