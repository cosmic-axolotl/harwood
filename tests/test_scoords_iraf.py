"""scoords against IRAF 2.18's: a list of wavelengths, one with words around the
numbers, one short and one long, one that turns back, and one whose first step is
backwards, which scoords does not look at.

tests/data/iraf_scoords holds the files and the cases (sc_cases.py), and what IRAF
said and the world coordinates it left (sc.json). IRAF writes its warnings
unbuffered and the rest buffered, so the two are held apart, each in its order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_scoords"
GOLD = json.loads((HERE / "sc.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from sc_cases import CASES, FILES, N  # noqa: E402


@pytest.fixture
def start(tmp_path, monkeypatch):
    for name, text in FILES.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _apart(lines):
    return ([line for line in lines if line.startswith("Warning:")],
            [line for line in lines if not line.startswith("Warning:")])


@pytest.mark.parametrize("case", list(CASES))
def test_scoords_sets_what_iraf_sets(start, case):
    image, coords = CASES[case]
    get_task("mkspec").run(image_name=image, image_title=image, ncols=N, function=1,
                           constant=5.0)
    result = get_task("scoords").run(images=image, coords=coords, label="Wavelength",
                                     units="Angstroms")
    assert _apart(result.data["text"]) == _apart(GOLD[case]["said"])
    aperture, = spectrum.read(f"{image}.fits")
    world = np.asarray(GOLD[case]["world"])
    if aperture.wavelength is None:
        assert world == pytest.approx(np.arange(1.0, N + 1.0))
    else:
        assert aperture.wavelength == pytest.approx(world, rel=1e-13)
