"""fitcoords and transform against IRAF 2.18's: the surface's coefficients to the last
digit, every pixel, the world system's cards and the logs.

tests/data/iraf_transform holds the cases (tr_cases.py), the inputs (tr_inputs.py: a
lamp whose lines curve along the slit, a galaxy on the same geometry, the arc trimmed
-- its database in physical pixels -- and transposed, one with a single line
identified, a 16-bit copy, a mask; identify databases written in IRAF's format) and
what IRAF's tasks left in each case (run.py, tr_collect.py -> transform.json).
transform is run on the fc databases IRAF's own fitcoords wrote, so that each task is
held to IRAF's alone; and by the image's own world system (no fits) on the galaxy
IRAF rectified, whole or a section of it, resampled to other limits, intervals and
to logarithmic ones.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import harwood
from harwood.core import images

HERE = Path(__file__).parent / "data" / "iraf_transform"
GOLD = json.loads((HERE / "transform.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from tr_cases import FITCOORDS, TRANSFORM, WCS  # noqa: E402
from tr_inputs import write  # noqa: E402

YES = {"yes": True, "no": False}


@pytest.fixture
def frames(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _said(lines: list[str]) -> list[str]:
    """The log without the system's line and the warnings IRAF's kernel adds."""
    return [line for line in lines if not line.startswith(("Community IRAF", "harwood ",
                                                            "Warning:"))]


def _fc(lines: list[str]) -> list[str]:
    return [line for line in lines if not line.startswith("#")]


def _values(params: dict) -> dict:
    return {k: YES.get(v, v) for k, v in params.items()}


def _fitcoords(images: str, params: dict) -> dict:
    params = _values(params)
    return harwood.run("fitcoords", images=images.split(","), **params).data


@pytest.mark.parametrize("case", list(FITCOORDS))
def test_fitcoords_writes_what_iraf_writes(frames, case):
    images, params = FITCOORDS[case]
    params = dict(params)
    again = params.pop("_again", None)
    data = _fitcoords(images, params)
    text = list(data.get("text", []))
    if again is not None:
        text += _fitcoords(images, again).get("text", [])
    gold = GOLD[f"fitcoords/{case}"]
    ours = {p.name: p.read_text(encoding="utf-8").splitlines()
            for p in sorted((frames / "database").glob("fc*"))}
    if case == "extension":
        # IRAF looks for database/idarc.fits; harwood takes arc.fits for arc
        assert {k: _fc(v) for k, v in ours.items()} == \
            {k: _fc(v) for k, v in GOLD["fitcoords/default"]["fc"].items()}
        return
    assert {k: _fc(v) for k, v in ours.items()} == {k: _fc(v) for k, v in gold["fc"].items()}
    assert _said(text) == _said(gold["said"])
    if "logfile" in gold:
        assert _said((frames / "logfile").read_text(encoding="utf-8").splitlines()) == \
            _said(gold["logfile"])


def _image(entry: dict) -> np.ndarray:
    raw = base64.b64decode(entry["data"])
    return np.frombuffer(raw, dtype=np.dtype(entry["dtype"])).reshape(entry["shape"])


def _wcs_case(frames, case: str) -> dict:
    """IRAF's rectified galaxy (w.fits) as the input -- or the section of it a case
    copies -- and the parameters of a transform by its world system."""
    section, params = WCS[case]
    gold = GOLD[f"transform/{case}"]["w"]
    header = fits.Header()
    for key, value in gold["cards"].items():
        header[key] = value
    fits.PrimaryHDU(_image(gold).astype(np.float32), header).writeto(frames / "w.fits")
    source = "w"
    if section:
        harwood.run("imcopy", input=[f"w.fits{section}"], output="ws.fits")
        source = "ws"
    return {"input": source, "output": "o", "fitnames": "", **params}


@pytest.mark.parametrize("case", list(TRANSFORM) + list(WCS))
def test_transform_writes_what_iraf_writes(frames, case):
    gold = GOLD[f"transform/{case}"]
    if case in WCS:
        params = _wcs_case(frames, case)
    else:
        params = TRANSFORM[case][1]
        for name, lines in gold["fc"].items():
            (frames / "database" / name).write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
    params = _values(params)
    ins = params.pop("input").split(",")
    outs = params.pop("output").split(",")
    data = harwood.run("transform", input=ins, output=outs,
                       fitnames=params.pop("fitnames"), **params).data
    assert _said(data.get("text", [])) == _said(gold["said"])
    assert _said((frames / "logfile").read_text(encoding="utf-8").splitlines()) == \
        _said(gold["logfile"])
    for name, entry in gold["images"].items():
        if name.startswith("mo"):
            with fits.open(frames / name) as hdul:
                ours, header = np.asarray(hdul[1].data), hdul[1].header
        else:
            _, header = images.read(str(frames / name))
            with fits.open(frames / name) as hdul:
                ours = np.asarray(next(h.data for h in hdul if h.data is not None))
        theirs = _image(entry)
        assert ours.shape == theirs.shape, name
        assert ours.dtype.itemsize * 8 * (-1 if ours.dtype.kind == 'f' else 1) == \
            entry["bitpix"], name
        assert np.array_equal(ours.astype(theirs.dtype.newbyteorder("=")),
                              theirs.astype(theirs.dtype.newbyteorder("=")),
                              equal_nan=True), name
        for key, value in entry["cards"].items():
            assert header.get(key) == value, (name, key)
