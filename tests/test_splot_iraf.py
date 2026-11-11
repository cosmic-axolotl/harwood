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
        assert ours["window"] == pytest.approx(theirs["window"], rel=2e-7, abs=1e-7)
        assert ours["title"] == theirs["text"][-1]
        assert ours["xlabel"] in theirs["text"]
