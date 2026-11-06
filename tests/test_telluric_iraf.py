"""telluric and skytweak against IRAF 2.18's: what they say and warn of, and every image
in the folder after them, pixel for pixel.

tests/data/iraf_telluric holds the spectra (twk_inputs.py), the cases (twk_cases.py) and
what IRAF said and left (twk.json, from run.py and twk_collect.py): the shift and scale
found by cross-correlation and by the least RMS, or each alone, or given; the lag, the
sample, the search steps; the calibration's apertures, by number, ignored or missing;
thresholds and a calibration that goes to zero; the airmass from the header or asked;
a calibration that covers part of the spectrum; in place; four bands; two inputs with
one calibration or one each; and what IRAF refuses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_telluric"
GOLD = json.loads((HERE / "twk.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from twk_cases import CASES  # noqa: E402
from twk_inputs import spectra  # noqa: E402

ERRORS = {c for c, v in GOLD.items() if any(w.startswith("ERROR") for w in v["warned"])}


@pytest.fixture
def start(tmp_path, monkeypatch):
    spectra(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(case):
    task, params = CASES[case]
    text = get_task(task).run(**params).data["text"]
    return ([line for line in text if not line.startswith("Warning: ")],
            [line for line in text if line.startswith("Warning: ")])


def _pixels(path):
    from harwood.core.sarith import BANDS

    apertures = spectrum.read(path)
    bands = [b for b in BANDS if getattr(apertures[0], b, None) is not None]
    data = np.array([[np.asarray(getattr(a, b), np.float32) for a in apertures]
                     for b in bands], np.float32)
    shape = [s for s in data.shape[:2] if s > 1] + [data.shape[2]]
    return data.reshape(shape)


@pytest.mark.parametrize("case", [c for c in CASES if c not in ERRORS])
def test_tweak_does_what_iraf_does(start, case):
    said, warned = _run(case)
    gold = GOLD[case]
    assert said == gold["said"]
    assert warned == gold["warned"]
    names = sorted(p.stem for p in Path().glob("*.fits"))
    assert names == sorted(gold["images"])
    for name, theirs in gold["images"].items():
        mine = _pixels(name + ".fits")
        assert list(mine.shape) == theirs["shape"], name
        assert mine.view(np.uint32).ravel().tolist() == theirs["bits"], name


@pytest.mark.parametrize("case", sorted(ERRORS))
def test_tweak_refuses_what_iraf_refuses(start, case):
    line = next(w for w in GOLD[case]["warned"] if w.startswith("ERROR"))
    message = line.split('"')[1].rstrip("\a")
    with pytest.raises(ValueError, match=message):
        _run(case)
