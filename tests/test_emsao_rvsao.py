"""emsao against RVSAO 2.8.5's: what it prints and the emsao.log it writes, line for
line -- the velocities, their errors, every line found, its fit, its weight and
why it was left out.

tests/data/iraf_emsao holds the spectra (em_inputs.py), the cases (em_cases.py)
and what RVSAO did (build.sh compiles it against IRAF 2.18; run.py, em_collect.py
-> emsao.json): the NGC 660 nucleus of the course, a galaxy at z = 0.05 and one at
z = 0.3, one line, none, a log-linear dispersion, a multispec, a continuum already
subtracted, every velocity correction (with the date as yyyy-mm-dd, dd/mm/yy and
MJD, and BCV or HCV from the header), the initial guesses, and the parameters of
the search, the fit and the continuum.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from harwood.core import emsao

HERE = Path(__file__).parent / "data" / "iraf_emsao"
GOLD = json.loads((HERE / "emsao.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from em_cases import CASES  # noqa: E402
from em_inputs import write  # noqa: E402

STAMP = re.compile(r"^(IRAF|harwood) rvsao\.emsao 2\.8\.5 .*\n", re.M)


def _run(tmp_path, monkeypatch, case):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    spectra, params, cont = CASES[case]
    session = emsao.Emsao({**params, **cont, "logfiles": "STDOUT,emsao.log"})
    try:
        session.run(spectra.split(","))
    except emsao.RvsaoError as error:
        session.fatal = str(error)
    return session


def _plain(text: str) -> str:
    return STAMP.sub("STAMP\n", text)


@pytest.mark.parametrize("case", list(CASES))
def test_emsao_logs_what_rvsao_logs(tmp_path, monkeypatch, case):
    session = _run(tmp_path, monkeypatch, case)
    assert _plain("".join(session.log)) == _plain(GOLD[case]["log"])


@pytest.mark.parametrize("case", list(CASES))
def test_emsao_prints_what_rvsao_prints(tmp_path, monkeypatch, case):
    session = _run(tmp_path, monkeypatch, case)
    assert _plain(session.out) == _plain(GOLD[case]["out"])
    fatal = re.search(r'ERROR \(\d+, "(.*)"\)', GOLD[case]["err"])
    if fatal:   # RVSAO stopped: PyRAF's words around the task's error
        assert getattr(session, "fatal", None) == fatal.group(1).rstrip("\a")
    else:
        assert GOLD[case]["err"] == ""
        assert not hasattr(session, "fatal")
