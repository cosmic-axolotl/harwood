"""deredden against IRAF 2.18's: what it says, the DEREDDEN card and each aperture's
fluxes, over every piece of Cardelli, Clayton & Mathis's law.

tests/data/iraf_deredden holds the cases (der_cases.py), the spectra they start from
(der_inputs.py: three apertures from the far ultraviolet to the infrared, and one
shorter than the law goes), IRAF's runs (run.py) and what they left (der.json). The
card's date is the run's. IRAF writes its warnings unbuffered and the rest buffered,
so the two are held apart, each in its order.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_deredden"
GOLD = json.loads((HERE / "der.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from der_cases import CASES  # noqa: E402
from der_inputs import HEADER, N, RANGES, spectra  # noqa: E402

DATE = re.compile(r"^(\s*)\w{3} [ \d]\d [ \d]\d:\d\d ")


@pytest.fixture
def start(tmp_path, monkeypatch):
    header = fits.Header()
    for key, value in HEADER.items():
        header[key] = value
    pixel = np.arange(1.0, N + 1.0)
    for name in RANGES:
        spectrum.write(tmp_path / f"{name}.fits",
                       [spectrum.Aperture(number=ap, centre=10.0 * ap, low=-3.0, high=3.0,
                                          pixel=pixel, flux=flux.astype(float),
                                          wavelength=w1 + dw * (pixel - 1))
                        for ap, w1, dw, flux in spectra(name)], header=header)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _apart(lines):
    lines = [DATE.sub(r"\1(when) ", line) for line in lines]
    return ([line for line in lines if line.startswith("Warning:")],
            [line for line in lines if not line.startswith("Warning:")])


@pytest.mark.parametrize("case", list(CASES))
def test_deredden_corrects_as_iraf_does(start, case):
    source, chain = CASES[case]
    for number, (step, theirs) in enumerate(zip(chain, GOLD[case], strict=True), 1):
        where = f"{case}, run {number}"
        result = get_task("deredden").run(input=source, output=f"o{number}", **step)
        source = f"o{number}"
        assert _apart(result.data["text"]) == _apart(theirs["said"]), where
        written = Path(f"o{number}.fits")
        assert written.exists() == theirs["written"], where
        if not theirs["written"]:
            continue
        card = fits.getheader(written)["DEREDDEN"]
        assert DATE.sub("", card) == DATE.sub("", theirs["card"]), where
        for aperture in spectrum.read(written):
            mine = aperture.flux.astype(np.float32)
            iraf_s = np.asarray(theirs["flux"][str(aperture.number)], np.float32)
            # In real, as IRAF: all but a few to the bit, those a unit of the last place.
            assert mine == pytest.approx(iraf_s, rel=2.5e-7), where
