"""scombine against IRAF 2.18's: its log, and each output's apertures -- numbers,
beams, titles, wavelengths and pixels -- and NCOMBINE and EXPTIME.

tests/data/iraf_scombine holds the spectra (scb_inputs.py), the cases (scb_cases.py)
and what IRAF said and wrote (scb.json, from run.py and scb_collect.py): every
grouping, the output's grid from the inputs or given, linear and logarithmic, the
first's, every rejection with and without the median, values put back, grow, every
scaling, zero offset and weight, a sample, thresholds, spectra that cover part of
the output or none of it, and the number combined.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_scombine"
GOLD = json.loads((HERE / "scb.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from scb_cases import CASES  # noqa: E402
from scb_inputs import spectra  # noqa: E402

DATE = re.compile(r"^[A-Z][a-z]{2} [ \d]\d [ \d]\d:\d\d: SCOMBINE$")


@pytest.fixture
def start(tmp_path, monkeypatch):
    spectra(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _said(lines):
    return ["<date>: SCOMBINE" if DATE.match(line) else line for line in lines]


@pytest.mark.parametrize("case", list(CASES))
def test_scombine_does_what_iraf_does(start, case):
    params = dict(CASES[case])
    said = get_task("scombine").run(output=params.pop("output", "o"), **params).data["text"]
    assert _said(said) == _said(GOLD[case]["said"])
    for name, theirs in GOLD[case]["outputs"].items():
        from astropy.io import fits

        header = fits.getheader(name)
        assert header.get("NCOMBINE") == theirs["ncombine"]
        assert header.get("EXPTIME") == pytest.approx(theirs["exptime"])
        assert header.get("OBJECT") == theirs["object"]
        apertures = spectrum.read(name)
        assert len(apertures) == len(theirs["lines"])
        for aperture, line in zip(apertures, theirs["lines"], strict=True):
            assert aperture.number == line["ap"]
            assert (aperture.beam or aperture.number) == line["beam"]
            assert aperture.title == line["title"]
            assert len(aperture.flux) == line["nw"]
            if aperture.wavelength is not None:
                assert aperture.wavelength == pytest.approx(line["world"], rel=1e-11)
            iraf_s = np.array(line["bits"], np.uint32).view(np.float32)
            mine = aperture.flux.astype(np.float32)
            assert mine.view(np.uint32).tolist() == iraf_s.view(np.uint32).tolist()
