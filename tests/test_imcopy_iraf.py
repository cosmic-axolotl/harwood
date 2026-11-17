"""imcopy keeps the pixel type, as IRAF 2.18's does -- each type's bits as IRAF writes
them, bytes as short, a scaled integer image as real, a section of an unsigned one still
unsigned -- and --pixtype writes another, integers truncated toward zero.

tests/data/iraf_imcopy holds the images (icp_inputs.py) and IRAF's copies' types and
pixels (run.py, icp_collect.py -> imcopy.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import harwood

HERE = Path(__file__).parent / "data" / "iraf_imcopy"
GOLD = json.loads((HERE / "imcopy.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from icp_collect import stored  # noqa: E402
from icp_inputs import write  # noqa: E402
from run import CASES  # noqa: E402


@pytest.fixture
def folder(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", list(GOLD))
def test_imcopy_keeps_the_type_as_iraf_does(folder, case):
    harwood.run("imcopy", input=[CASES[case]], output=f"c_{case}.fits")
    assert stored(folder / f"c_{case}.fits") == GOLD[case]


@pytest.mark.parametrize("source,pixtype,bitpix,first", [
    ("f32.fits", "short", 16, None), ("f32.fits", "double", -64, None),
    ("i16.fits", "real", -32, None), ("u16.fits", "int", 32, None),
    ("i32.fits", "ushort", 16, None)])
def test_pixtype_writes_the_type_asked(folder, source, pixtype, bitpix, first):
    harwood.run("imcopy", input=[source], output="out.fits", pixtype=pixtype)
    with fits.open(folder / "out.fits", memmap=False) as hdul:
        assert hdul[0].header["BITPIX"] == bitpix
        copied = np.asarray(hdul[0].data, float)
    original = np.asarray(fits.getdata(folder / source, memmap=False), float)
    expected = np.trunc(original) if pixtype in ("short", "int", "ushort") else original
    assert np.allclose(copied, expected.astype(np.float32 if pixtype == "real"
                                                  else float))
