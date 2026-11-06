"""fitprofs against IRAF 2.18.1's: what it says on the standard output and what it warns
of, the log file, and every image it writes, pixel for pixel.

tests/data/iraf_fitprofs holds the spectra and positions files (fpr_inputs.py), the
cases (fpr_cases.py) and what IRAF said and wrote (fpr.json, from run.py and
fpr_collect.py): gaussian, lorentzian and voigt profiles, alone, blended and mixed,
every constraint on positions and widths, the background fitted or held, given, an
average, a median or scaled, Monte-Carlo errors, the components to give, outputs of the
model and of the difference, existing outputs and clobber, the apertures and bands, two
inputs, the log to a file, and what IRAF warns of or refuses -- and the two places
IRAF 2.18.1 fails: an output written verbosely, which corrupts its memory, and merge.
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

HERE = Path(__file__).parent / "data" / "iraf_fitprofs"
GOLD = json.loads((HERE / "fpr.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from fpr_cases import BEFORE, CASES, CRASHES  # noqa: E402
from fpr_inputs import spectra  # noqa: E402

DATE = re.compile(r"^# [A-Z][a-z]{2} [ \d]\d [ \d]\d:\d\d ")
TEMP = re.compile(r"-->  temp\d+[a-z]")
ERRORS = {c for c, v in GOLD.items() if any(w.startswith("ERROR") for w in v["warned"])
          and c not in CRASHES}


@pytest.fixture
def start(tmp_path, monkeypatch):
    spectra(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _masked(lines):
    return [TEMP.sub("-->  temp", DATE.sub("# <date> ", line)) for line in lines]


def _run(params):
    """(what it said, what it warned of): IRAF's standard output and its warnings."""
    params = dict(params)
    params.setdefault("logfile", "")
    text = get_task("fitprofs").run(**params).data["text"]
    return ({"said": [line for line in text if not line.startswith("Warning: ")],
             "warned": [line for line in text if line.startswith("Warning: ")]})


def _pixels(path):
    from harwood.core.sarith import BANDS

    apertures = spectrum.read(path)
    bands = [b for b in BANDS if getattr(apertures[0], b, None) is not None]
    data = np.array([[np.asarray(getattr(a, b), np.float32) for a in apertures]
                     for b in bands], np.float32)
    shape = [s for s in data.shape[:2] if s > 1] + [data.shape[2]]
    return data.reshape(shape)


@pytest.mark.parametrize("case", [c for c in CASES if c not in ERRORS | {"pre_merge"}])
def test_fitprofs_does_what_iraf_does(start, case):
    if case in BEFORE:
        _run(CASES[BEFORE[case]])
    data = _run(CASES[case])
    gold = GOLD[case]
    assert _masked(data["said"]) == _masked(gold["said"])
    assert data["warned"] == [w for w in gold["warned"] if not w.startswith("ERROR")]
    if "logfile" in gold:
        assert _masked(Path("logfile").read_text(encoding="utf-8").splitlines()) == \
            _masked(gold["logfile"])
    for name, theirs in gold["images"].items():
        if theirs is None or case in CRASHES or name in ("a", "c", "one"):
            continue
        mine = _pixels(name + ".fits")
        assert list(mine.shape) == theirs["shape"], name
        assert mine.view(np.uint32).ravel().tolist() == theirs["bits"], name


@pytest.mark.parametrize("case", sorted(ERRORS))
def test_fitprofs_refuses_what_iraf_refuses(start, case):
    line = next(w for w in GOLD[case]["warned"] if w.startswith("ERROR"))
    message = line.split('"')[1].rstrip("\a")
    with pytest.raises(ValueError, match=re.escape(message)):
        _run(CASES[case])


def test_fitprofs_will_not_merge_as_iraf_cannot(start):
    _run(CASES["pre"])
    with pytest.raises(ValueError, match="segmentation violation"):
        _run(CASES["pre_merge"])
