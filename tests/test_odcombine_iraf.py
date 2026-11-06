"""odcombine against IRAF 2.18.1's: its log, and each output's pixels, wavelengths and
header -- and the sigma and the masks beside them.

tests/data/iraf_odcombine holds the spectra and masks (odc_inputs.py), the cases
(odc_cases.py) and what IRAF said and wrote (odc.json, from run.py and
odc_collect.py). The installed odcombine never writes an output; the gabarito is its
source with the template of its temporary spectra corrected (build.sh). The cases:
every grouping, the grid from the inputs or given, rising and falling, linear and
logarithmic, the first's; every rejection, values kept with nkeep, grow; every
scaling, zero offset and weight, from statistics in each kind of section, a file or
the header -- the first image's, as IRAF reads it for all; thresholds; spectra that
reach part of the output or none of it; every kind and format of mask, and the masks
only the last spectrum of an image gets; the output's types; and the sigma, bad pixel,
rejection, number rejected and exposure masks.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from harwood.core import plio, spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_odcombine"
GOLD = json.loads((HERE / "odc.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from odc_cases import CASES  # noqa: E402
from odc_inputs import spectra  # noqa: E402

DATE = re.compile(r"[A-Z][a-z]{2} [ \d]\d [ \d]\d:\d\d")
#: harwood computes in real; IRAF's outtype double combines in double.
DOUBLE = {"outtype_double"}


@pytest.fixture
def start(tmp_path, monkeypatch):
    spectra(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _said(lines):
    return [DATE.sub("<date>", line) for line in lines]


def _bits(values, kind):
    if kind == "real":
        return np.asarray(values, np.float32).view(np.uint32).tolist()
    return np.asarray(values).astype(np.int64).tolist()


@pytest.mark.parametrize("case", [c for c in CASES if c not in DOUBLE])
def test_odcombine_does_what_iraf_does(start, case):
    from astropy.io import fits

    params = dict(CASES[case])
    said = get_task("odcombine").run(output=params.pop("output", "o"), **params).data["text"]
    gold = GOLD[case]
    assert _said(said) == _said(gold["said"])
    for name, theirs in gold["outputs"].items():
        header = fits.getheader(name)
        cards = theirs["cards"]
        assert header.get("OBJECT") == cards["OBJECT"]
        assert header.get("NCOMBINE") == cards.get("NCOMBINE")
        assert header.get("ICBPM") == cards.get("ICBPM")
        assert header.get("EXPTIME") == pytest.approx(cards.get("EXPTIME"))
        assert [header[k] for k in sorted(header) if k.startswith("IMCMB")] == theirs["imcmb"]
        (aperture,) = spectrum.read(name)
        assert aperture.number, (aperture.beam or aperture.number) == \
            tuple(int(v) for v in cards["APNUM1"].split()[:2])
        assert len(aperture.flux) == theirs["shape"][0]
        if aperture.wavelength is not None:
            # A one-dimensional spectrum even in log wavelength lists its log10.
            world = np.asarray(theirs["world"])
            world = 10 ** world if cards.get("DC-FLAG") == 1 else world
            assert aperture.wavelength == pytest.approx(world, rel=1e-11)
        if theirs["kind"] == "real":
            assert _bits(aperture.flux, "real") == theirs["bits"]
        else:
            assert aperture.flux.astype(np.int64).tolist() == theirs["values"]
    for name, theirs in gold["masks"].items():
        data, _ = plio.read(name + ".pl")
        assert list(data.shape) == theirs["shape"]
        assert np.asarray(data, np.int64).ravel().tolist() == theirs["values"]


def test_odcombine_says_it_does_not_combine_in_double(start):
    with pytest.raises(ValueError, match="outtype double"):
        get_task("odcombine").run(**CASES["outtype_double"], output="o")
