"""xcsao against RVSAO 2.8.5's: what it prints and the xcsao.log it writes, line for
line -- for each template the peak, its height, R, the velocity and its error,
and which template is best.

tests/data/iraf_xcsao holds the spectra (xc_inputs.py: synthetic G and A stars and
an emission-line galaxy as templates, with the header cards a template can carry,
and stars and galaxies made from them), the cases (xc_cases.py) and what RVSAO did
(../iraf_emsao/build.sh compiles it against IRAF 2.18; run.py, xc_collect.py ->
xcsao.json): one template or several, a broadened galaxy, noise, zero padding,
limits in wavelength (the parameters', the template's, OVERLAP), the initial
velocity and its passes, a velocity window, the peak's fraction or points, the
filter, the taper, the grid's size, the interpolation, lines clipped from the
spectrum or the template, renormalisation, zeroed ends, a log-linear dispersion,
a multispec, every velocity correction, the template's own velocity, shift and
flags, the continuum's parameters, templates that do not overlap or are not there.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from harwood.core import xcsao

HERE = Path(__file__).parent / "data" / "iraf_xcsao"
GOLD = json.loads((HERE / "xcsao.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from xc_cases import CASES  # noqa: E402
from xc_inputs import write  # noqa: E402

STAMP = re.compile(r"^(IRAF|harwood) rvsao\.xcsao 2\.8\.5 .*\n", re.M)


def _run(tmp_path, monkeypatch, case):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    spectra, params, cont = CASES[case]
    session = xcsao.Xcsao({**params, **cont, "logfiles": "STDOUT,xcsao.log"})
    try:
        session.run(spectra.split(","))
    except xcsao.RvsaoError as error:
        session.fatal = str(error)
    return session


def _plain(text: str) -> str:
    return STAMP.sub("STAMP\n", text)


@pytest.mark.parametrize("case", list(CASES))
def test_xcsao_logs_what_rvsao_logs(tmp_path, monkeypatch, case):
    session = _run(tmp_path, monkeypatch, case)
    assert _plain("".join(session.log)) == _plain(GOLD[case]["log"] or "")


@pytest.mark.parametrize("case", list(CASES))
def test_xcsao_prints_what_rvsao_prints(tmp_path, monkeypatch, case):
    session = _run(tmp_path, monkeypatch, case)
    assert _plain(session.out) == _plain(GOLD[case]["out"])
    fatal = re.search(r'ERROR \(\d+, "(.*)"\)', GOLD[case]["err"])
    if fatal:   # RVSAO stopped: PyRAF's words around the task's error
        assert getattr(session, "fatal", None) == fatal.group(1).rstrip("\a")
    else:
        assert session.err == GOLD[case]["err"]
        assert not hasattr(session, "fatal")
