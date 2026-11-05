"""slist against IRAF 2.18's, line for line.

tests/data/iraf_slist holds the spectra (sl_inputs.py: apertures linear, logarithmic
and in pixels, a beam that is not its aperture's, titles of their own and not, the
header's times and place), the cases (sl_cases.py) and what IRAF printed (sl.json).
The first wavelength and the step are shdr's, in real.
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

HERE = Path(__file__).parent / "data" / "iraf_slist"
GOLD = json.loads((HERE / "sl.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from sl_cases import CASES  # noqa: E402
from sl_inputs import APERTURES, HEADER, N, spectra  # noqa: E402


@pytest.fixture
def start(tmp_path, monkeypatch):
    header = fits.Header()
    for key, value in HEADER.items():
        header[key] = value
    pixel = np.arange(1.0, N + 1.0)
    for name in APERTURES:
        spectrum.write(tmp_path / f"{name}.fits",
                       [spectrum.Aperture(number=ap, centre=10.0 * ap, low=-3.0, high=3.0,
                                          pixel=pixel, flux=flux.astype(float),
                                          wavelength=wave, title=title, beam=beam)
                        for ap, beam, _, _, _, title, flux, wave in spectra(name)],
                       header=header)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", list(CASES))
def test_slist_says_what_iraf_s_says(start, case):
    result = get_task("slist").run(**CASES[case])
    assert result.data["text"] == GOLD[case]
