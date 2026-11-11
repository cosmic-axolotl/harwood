"""specplot against IRAF 2.18's: what it prints on both streams and the log it
writes, and each frame's window, title and texts from its metacode.

tests/data/iraf_specplot holds the cases (spp_cases.py: cursor files and what the
prompts read) and what IRAF did (run.py, spp_collect.py -> specplot.json), on the
splot gabarito's spectra (a multispec of three dispersions, an onedspec, a long
slit): the automatic layout and its step, no layout, offsets and scales to a
common mean, every colon command, velocities and redshifts, units, moving,
shifting, deleting, adding and inserting spectra, labels by hand, the s and t
keys, the window, the parameters, the log transform, user labels and the logfile.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from harwood.core import specplot, splot

HERE = Path(__file__).parent / "data" / "iraf_specplot"
GOLD = json.loads((HERE / "specplot.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "iraf_splot"))
from sp_inputs import write  # noqa: E402
from spp_cases import CASES  # noqa: E402

SYSID = re.compile(r"(Community IRAF V[\d.]+|harwood \S+) \S+@\S+ \w{3} "
                   r"\d\d:\d\d:\d\d \d\d-\w{3}-\d{4}")
YES = {"yes": True, "no": False}


def _run(tmp_path, monkeypatch, case):
    write(tmp_path)
    (tmp_path / "labels.txt").write_text("first one\nsecond\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    spectra, params, cursor, answers = CASES[case]
    params = {k: YES.get(v, v) for k, v in params.items()}
    session = specplot.Specplot(params, iter(answers.splitlines()))
    session.run([f"{n}.fits" for n in spectra.split(",")], splot.cursor_reads(cursor))
    return session


def _plain(text: str) -> str:
    return SYSID.sub("SYSID", text)


@pytest.mark.parametrize("case", list(CASES))
def test_specplot_says_what_iraf_says(tmp_path, monkeypatch, case):
    session = _run(tmp_path, monkeypatch, case)
    gold = GOLD[case]
    assert _plain(session.out) == _plain(gold["out"])
    fatal = re.search(r'ERROR \(\d+, "(.*)"\)', gold["err"], re.S)
    if fatal:       # IRAF stopped: PyRAF's words around the task's error
        assert session.fatal == fatal.group(1)
        assert session.err == gold["err"][:gold["err"].index("ERROR: Error running")]
    else:
        assert session.err == gold["err"]


@pytest.mark.parametrize("case", list(CASES))
def test_specplot_draws_the_frames_iraf_draws(tmp_path, monkeypatch, case):
    session = _run(tmp_path, monkeypatch, case)
    frames = GOLD[case]["frames"]
    assert len(session.graphs) == len(frames)
    for ours, theirs in zip(session.graphs, frames, strict=True):
        assert ours["window"] == pytest.approx(theirs["window"], rel=2e-7, abs=1e-7)
        for text in (ours["title"], ours["params"], ours["xlabel"]):
            if text:
                assert text in theirs["text"], text
        for spectrum in ours["spectra"]:
            if spectrum["label"].strip():
                assert spectrum["label"] in theirs["text"]


@pytest.mark.parametrize("case", [c for c in CASES if GOLD[c].get("log")])
def test_specplot_logs_what_iraf_logs(tmp_path, monkeypatch, case):
    """The logfile: sp_vshow at the end, as IRAF appends it."""
    session = _run(tmp_path, monkeypatch, case)
    assert [_plain(line) for line in session.vshow_lines] == \
        [_plain(line) for line in GOLD[case]["log"]]
