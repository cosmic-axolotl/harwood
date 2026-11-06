"""sfit and continuum against IRAF 2.18's: what they say -- the log with the power
series of a polynomial fit, the warnings, the errors -- and every image in the folder
after them, pixel for pixel, with its SFIT and SFITB.

tests/data/iraf_sfit holds the spectra (sft_inputs.py), the cases (sft_cases.py) and
what IRAF said and left (sft.json, from run.py and sft_collect.py): every output type,
function and order, rejection with grow and without, replacing the rejected points in
each of icfit's ways (all the points, a sample, averaged), a sample, averaging and
medians, pixels for x, logs of both axes and of what has none, division by zero, the
lines and bands to fit, the fits SFIT says were made in the input and in an existing
output, override, in place, listonly, the log to a file, and what IRAF warns of or
refuses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_sfit"
GOLD = json.loads((HERE / "sft.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from sft_cases import CASES  # noqa: E402
from sft_inputs import spectra  # noqa: E402

ERRORS = {c for c, v in GOLD.items() if v["said"][:1] and v["said"][0].startswith("ERROR")}


@pytest.fixture
def start(tmp_path, monkeypatch):
    spectra(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _said(lines):
    return ["(sysid)" if line.startswith(("SFIT: ", "END: ")) else line
            for line in lines]


def _pixels(path):
    """The image as IRAF's: [band, line, pixel], dropping axes of one."""
    from astropy.io import fits

    apertures = spectrum.read(path)
    from harwood.core.sarith import BANDS

    bands = [b for b in BANDS if getattr(apertures[0], b, None) is not None]
    data = np.array([[np.asarray(getattr(a, b), np.float32) for a in apertures]
                     for b in bands], np.float32)
    shape = [s for s in data.shape[:2] if s > 1] + [data.shape[2]]
    header = fits.getheader(path)
    return data.reshape(shape), header.get("SFIT"), header.get("SFITB")


def _run(case):
    task, params = CASES[case]
    params = dict(params)
    params.setdefault("logfiles", "STDOUT")
    if not params.get("listonly"):
        params.setdefault("output", "o")
    return get_task(task).run(**params).data["text"]


@pytest.mark.parametrize("case", [c for c in CASES if c not in ERRORS])
def test_sfit_does_what_iraf_does(start, case):
    said = _run(case)
    gold = GOLD[case]
    assert _said(said) == _said(gold["said"])
    if "logfile" in gold:
        assert _said(Path("logfile").read_text(encoding="utf-8").splitlines()) == \
            _said(gold["logfile"])
    names = sorted(p.stem for p in Path().glob("*.fits"))
    assert names == sorted(gold["images"])
    for name, theirs in gold["images"].items():
        data, sfit, sfitb = _pixels(name + ".fits")
        assert list(data.shape) == theirs["shape"]
        assert data.view(np.uint32).ravel().tolist() == theirs["bits"], name
        assert (sfit, sfitb) == (theirs["SFIT"], theirs["SFITB"]), name


@pytest.mark.parametrize("case", sorted(ERRORS))
def test_sfit_refuses_what_iraf_refuses(start, case):
    message = GOLD[case]["said"][0].split('"')[1].rstrip("\a")
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        _run(case)


def test_a_session_leaves_its_parameters_to_the_next_spectrum(start):
    """The keys of the first spectrum's session change the order; the others start from
    it, and every fit is the one of that order without a session."""
    Path("keys").write_text(":order 3\nf\nq\n", encoding="utf-8")
    get_task("sfit").run(input="a", output="s", function="legendre", order=5,
                         cursor="keys", logfiles="")
    get_task("sfit").run(input="a", output="n", function="legendre", order=3,
                         logfiles="")
    assert _pixels("s.fits")[0].tolist() == _pixels("n.fits")[0].tolist()
