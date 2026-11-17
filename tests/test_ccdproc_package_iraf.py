"""ccdproc under ccdred's package parameters, against IRAF 2.18's: set in harwood.yaml's
params.ccdred as setinstrument writes them, read by ccdproc as IRAF's reads them from the
package. The output's pixel type from each input type and each pixeltype; the original
kept under a backup prefix or in a directory, the prefix's first name taken; a bad pixel
file of regions; the overscan fitted from a file of cursor keys, the questions answered
image by image, the fit's parameters carried to the next frames.

tests/data/iraf_ccdproc holds the cases (pkg_cases.py), the frames (pkg_inputs.py) and
what IRAF left (run_pkg.py, pkg_collect.py -> pkg.json). IRAF processed in place, with
no output list, and so does harwood here.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

import harwood

HERE = Path(__file__).parent / "data" / "iraf_ccdproc"
GOLD = json.loads((HERE / "pkg.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from pkg_cases import CASES  # noqa: E402
from pkg_inputs import write  # noqa: E402

DATE = re.compile(r"[A-Z][a-z]{2} [ \d]\d [ \d]\d:\d\d ")


def _image(path: Path) -> dict:
    with fits.open(path, do_not_scale_image_data=True) as hdul:
        header = hdul[0].header
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        raw = np.ascontiguousarray(hdu.data)
        header = hdu.header
    return {"bitpix": header["BITPIX"], "bzero": header.get("BZERO"),
            "shape": list(raw.shape), "sha": hashlib.sha256(raw.tobytes()).hexdigest()}


def _value(text):
    return {"yes": True, "no": False}.get(text, text) if isinstance(text, str) else text


@pytest.mark.parametrize("case", list(GOLD))
def test_ccdproc_under_the_package_does_what_iraf_does(tmp_path, monkeypatch, case):
    images, package, params, cursor, answers = CASES[case]
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    before = {p.name for p in tmp_path.glob("*.fits")}
    ccdred = {"logfile": "logfile", **package}
    extra = {}
    if cursor:
        (tmp_path / "cursor").write_text(cursor, encoding="utf-8")
        (tmp_path / "answers").write_text(answers, encoding="utf-8")
        ccdred["cursor"] = "cursor"
        extra["answers"] = "answers"
    (tmp_path / "harwood.yaml").write_text(yaml.safe_dump({"params": {"ccdred": ccdred}}),
                                           encoding="utf-8")
    values = {key: _value(value) for key, value in params.items()}
    result = harwood.run("ccdproc", input=images.split(","), inplace=True, format="iraf",
                                     **values, **extra)
    gold = GOLD[case]
    for name, theirs in gold["images"].items():
        assert _image(tmp_path / name) == theirs, name
    # harwood keeps the bad pixels it fixed in a .bpm.fits beside an IRAF-format image,
    # which has nowhere else to hold them; IRAF does not
    new = sorted({p.name for p in tmp_path.glob("*.fits")
                  if not p.name.endswith(".bpm.fits")} - before) + sorted(
        f"bak/{p.name}" for p in (tmp_path / "bak").glob("*"))
    assert new == gold["new"]
    assert DATE.sub("", (tmp_path / "logfile").read_text(encoding="utf-8")) == \
        gold["logfile"]
    assert result.data.get("stdout", "") == gold["said"]
