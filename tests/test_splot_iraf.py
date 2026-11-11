"""splot's measurements against IRAF 2.18's: what it prints on both streams and
the save_file it writes, key for key.

tests/data/iraf_splot holds the cases (sp_cases.py: cursor files and the answers to
splot's queries), the spectra (sp_inputs.py: an onedspec with blended emission lines
and an absorption, a long slit along DISPAXIS 2, a multispec of a linear, a
log-linear and a chebyshev dispersion) and what IRAF wrote (run.py, sp_collect.py
-> splot.json): the cursor position, m, e (with noise), k with each profile and
Monte-Carlo errors, d with every fit choice, the overplots, a line list and typed
positions, -, h, s, f's functions, j, x, flux conversions, pixels, velocity,
units, the colon commands, lines and nsum of a long slit, apertures of a
multispec and another image read with g.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from harwood.core import splot

HERE = Path(__file__).parent / "data" / "iraf_splot"
GOLD = json.loads((HERE / "splot.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from sp_cases import CASES  # noqa: E402
from sp_inputs import write  # noqa: E402

DATE = re.compile(r"^[A-Z][a-z]{2} [ \d]\d [ \d]\d:\d\d ")


reads = splot.cursor_reads


def _log(lines):
    return None if lines is None else [DATE.sub("DATE ", line) for line in lines]


@pytest.mark.parametrize("case", list(CASES))
def test_splot_says_what_iraf_says(tmp_path, monkeypatch, case):
    write(tmp_path)
    (tmp_path / "lines.dat").write_text((HERE / "lines.dat").read_text())
    monkeypatch.chdir(tmp_path)
    image, params, cursor, answers = CASES[case]
    params = dict(params)
    nsum = int(params.pop("nsum", 1))
    line = params.pop("line", None)
    session = splot.Splot({"save_file": "splot.log", "nsum": (nsum, nsum), "line": line,
                           **params}, iter(answers.splitlines()))
    session.run([f"{image}.fits"], reads(cursor))
    gold = GOLD[case]
    assert session.out == gold["out"]
    assert session.err == gold["err"]
    ours = session.log or None
    assert _log(ours) == _log(gold["log"])


@pytest.mark.parametrize("case", list(CASES))
def test_splot_draws_the_frames_iraf_draws(tmp_path, monkeypatch, case):
    """Each graph splot clears and draws: its window (from the metacode, held to
    the figures it keeps), its title and its x label."""
    write(tmp_path)
    (tmp_path / "lines.dat").write_text((HERE / "lines.dat").read_text())
    monkeypatch.chdir(tmp_path)
    image, params, cursor, answers = CASES[case]
    params = dict(params)
    nsum = int(params.pop("nsum", 1))
    line = params.pop("line", None)
    session = splot.Splot({"save_file": "", "nsum": (nsum, nsum), "line": line,
                           **params}, iter(answers.splitlines()))
    session.run([f"{image}.fits"], reads(cursor))
    frames = GOLD[case]["frames"]
    assert len(session.graphs) == len(frames)
    for ours, theirs in zip(session.graphs, frames, strict=True):
        # t's icfit graphs: its fit in double where IRAF's is in real (as in
        # test_icfit_keys), so their windows are held to the fifth figure
        rel = 2e-5 if ours.get("icfit") else 2e-7
        assert ours["window"] == pytest.approx(theirs["window"], rel=rel, abs=1e-7)
        assert ours["title"] in theirs["text"]
        if ours["xlabel"]:
            assert ours["xlabel"] in theirs["text"]


WRITTEN = [case for case in CASES if GOLD[case].get("written")]


def _run(tmp_path, monkeypatch, case):
    write(tmp_path)
    (tmp_path / "lines.dat").write_text((HERE / "lines.dat").read_text())
    monkeypatch.chdir(tmp_path)
    image, params, cursor, answers = CASES[case]
    params = dict(params)
    nsum = int(params.pop("nsum", 1))
    line = params.pop("line", None)
    session = splot.Splot({"save_file": "", "nsum": (nsum, nsum), "line": line,
                           **params}, iter(answers.splitlines()))
    session.run([f"{image}.fits"], reads(cursor))
    return session


def _wat(cards: dict, axis: int) -> str:
    return "".join(str(cards[k]).ljust(68) for k in sorted(cards)
                   if k.startswith(f"WAT{axis}_")).rstrip()


@pytest.mark.parametrize("case", WRITTEN)
def test_splot_writes_the_spectra_iraf_writes(tmp_path, monkeypatch, case):
    """i: the image written -- its pixels, and the dispersion p and u gave it."""
    import base64

    import numpy as np
    from astropy.io import fits

    _run(tmp_path, monkeypatch, case)
    for name, gold in GOLD[case]["written"].items():
        with fits.open(tmp_path / name) as hdul:
            ours, header = np.asarray(hdul[0].data), hdul[0].header
        theirs = np.frombuffer(base64.b64decode(gold["data"]), dtype=np.dtype(gold["dtype"]))
        theirs = theirs.reshape(gold["shape"]).astype(np.float64)
        assert list(ours.shape) == gold["shape"], name
        assert np.array_equal(ours.astype(np.float64), theirs), name
        cards = gold["cards"]
        for key in ("CRVAL1", "CDELT1", "CD1_1", "CRPIX1", "CRVAL2", "CD2_2", "DC-FLAG"):
            if key in cards and key in header:
                assert header[key] == pytest.approx(cards[key], rel=1e-12), (name, key)
        if "WAT2_001" in cards and "multispec" in str(cards["WAT2_001"]):
            # each specN as IRAF writes it; the numbers wrspect1 works out again from
            # the spectrum's ends -- a log-linear dispersion through 10^x and log10 --
            # can differ in the 14th figure, which no reading of IRAF's arithmetic
            # tried here reproduces
            ours_wat, theirs_wat = _wat(dict(header), 2), _wat(cards, 2)
            a, b = ours_wat.split(), theirs_wat.split()
            assert len(a) == len(b), name
            for u, v in zip(a, b, strict=True):
                if u != v:
                    assert float(u.strip('"')) == pytest.approx(
                        float(v.strip('"').replace("E", "e")), rel=1e-12), (name, u, v)


def test_splot_labels_what_iraf_labels(tmp_path, monkeypatch):
    """:label, :mabove and :mbelow: the texts IRAF writes, none outside the graph."""
    session = _run(tmp_path, monkeypatch, "labels")
    labels = [item[1][2] for item in session.drawn if item[0] in ("label", "mabove",
                                                                  "mbelow")]
    assert labels == ["H alpha", "6583.0", "absorption"]
    assert set(labels) <= set(GOLD["labels"]["texts"])
