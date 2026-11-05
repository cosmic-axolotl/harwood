"""sarith against IRAF 2.18's: what it says, and each output's apertures -- their
numbers, beams, titles, wavelengths and pixels, band by band.

tests/data/iraf_sarith holds the spectra (sa_inputs.py), the cases (sa_cases.py) and
what IRAF said and wrote (sa.json, from run.py and sa_collect.py): copies, regions
resampled and not, backwards, apertures and beams chosen, renumbered, a modulus,
every operation, with constants and spectra resampled, unmatched apertures and lines,
bands, onedspec's files, an output that exists, a merge, and two inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_sarith"
GOLD = json.loads((HERE / "sa.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from sa_cases import CASES  # noqa: E402
from sa_inputs import SPECTRA, N, data  # noqa: E402

COLUMNS = ("flux", "raw", "sky", "error")


@pytest.fixture
def start(tmp_path, monkeypatch):
    pixel = np.arange(1.0, N + 1.0)
    for name, (rows, _bands) in SPECTRA.items():
        header = fits.Header()
        header["OBJECT"] = f"star {name}"
        apertures = []
        for (ap, beam, w1, dw, title), values in zip(rows, data(name), strict=True):
            made = spectrum.Aperture(number=ap, centre=10.0 * ap, low=-3.0, high=3.0,
                                     pixel=pixel, wavelength=w1 + dw * (pixel - 1),
                                     title=title, beam=beam)
            for column, band in zip(COLUMNS, values, strict=False):
                setattr(made, column, band.astype(float))
            apertures.append(made)
        spectrum.write(tmp_path / f"{name}.fits", apertures, header=header)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(params):
    params = dict(params)
    return get_task("sarith").run(input1=params.pop("input1"), op=params.pop("op"),
                                  output="o", verbose=True, **params)


@pytest.mark.parametrize("case", list(CASES))
def test_sarith_does_what_iraf_does(start, case):
    params = dict(CASES[case])
    then = params.pop("then", None)
    said = _run(params).data["text"]
    if then is not None:
        said += ["----"] + _run(then).data["text"]
    assert said == GOLD[case]["said"]
    for name, lines in GOLD[case]["outputs"].items():
        apertures = spectrum.read(name)
        assert len(apertures) == len(lines)
        for aperture, theirs in zip(apertures, lines, strict=True):
            assert aperture.number == theirs["ap"]
            assert (aperture.beam or aperture.number) == theirs["beam"]
            assert aperture.title == theirs["title"]
            assert len(aperture.flux) == theirs["nw"]
            columns = [c for c in COLUMNS if getattr(aperture, c) is not None]
            assert len(columns) == len(theirs["bands"])
            for column, band in zip(columns, theirs["bands"], strict=True):
                assert aperture.wavelength == pytest.approx(band["world"], rel=1e-11)
                iraf_s = np.array(band["bits"], np.uint32).view(np.float32)
                mine = getattr(aperture, column).astype(np.float32)
                # To the bit, or a unit of the last place of the operands: a resampled
                # second operand's, which a difference keeps as it is.
                assert mine == pytest.approx(iraf_s, rel=2.5e-7, abs=1.5e-4,
                                             nan_ok=True), column
