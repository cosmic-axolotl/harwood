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

With save_vel, the header after it: the cards vwhead and xcwhead write (the
velocity joined with emsao's, one card per template, the earlier XCSAO kept as
HISTORY, a multispec's per aperture), each as RVSAO's FITS kernel writes it, and
nothing else of the file changed.
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
#: RVSAO writes back the header it holds, with getimage's RA, DEC, HA and ST of
#: 00:00:00 and IRAF's DATE: the second aperture is then reported (and corrected)
#: for that position and today. harwood writes RVSAO's own cards only.
DIVERGE = {"save_multispec": "RVSAO's second aperture reads the RA, DEC and DATE "
                             "the first one's save wrote"}
WHEN = re.compile(r"\d\d-[A-Z][a-z]{2}-\d{4} \d\d:\d\d")
#: a template already filtered (FI-FLAG 1): cz differs from RVSAO's in its 15th
#: figure (2e-14 of it; R, the error and the log are the same) -- open
LAST_FIGURE = {"save_tfilt", "save_filtered"}
RVSAO_KEYS = re.compile(r"^(VELOCITY|Z|CZERR|ZERR|VELQUAL|BCV|VELSET|CZXC|ZXC|CZXCERR|"
                        r"CZXCR|BESTTEMP|CZPARAMS|COMMENT|HISTORY|XCSAO|APVEL\d+|"
                        r"APVXC\d+|XCTEMP\d+|XT\d+_\d+) *$")


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
    if case in DIVERGE:
        pytest.skip(DIVERGE[case])
    session = _run(tmp_path, monkeypatch, case)
    assert _plain("".join(session.log)) == _plain(GOLD[case]["log"] or "")


@pytest.mark.parametrize("case", list(CASES))
def test_xcsao_prints_what_rvsao_prints(tmp_path, monkeypatch, case):
    if case in DIVERGE:
        pytest.skip(DIVERGE[case])
    session = _run(tmp_path, monkeypatch, case)
    assert _plain(session.out) == _plain(GOLD[case]["out"])
    fatal = re.search(r'ERROR \(\d+, "(.*)"\)', GOLD[case]["err"])
    if fatal:   # RVSAO stopped: PyRAF's words around the task's error
        assert getattr(session, "fatal", None) == fatal.group(1).rstrip("\a")
    else:
        assert session.err == GOLD[case]["err"]
        assert not hasattr(session, "fatal")


def _cards(images, rvsao: bool) -> list[str]:
    return [WHEN.sub("DATE", c) for c in images
            if bool(RVSAO_KEYS.match(c[:8])) == rvsao and c.strip()]


@pytest.mark.parametrize("case", [c for c in CASES if CASES[c][1].get("save_vel")])
def test_xcsao_saves_what_rvsao_saves(tmp_path, monkeypatch, case):
    from astropy.io import fits

    write(tmp_path / "fresh")
    _run(tmp_path, monkeypatch, case)
    for name, gold in (GOLD[case].get("head") or {}).items():
        mine = [c.image for c in fits.getheader(tmp_path / name).cards]
        ours, theirs = _cards(mine, True), _cards(gold, True)
        if case in LAST_FIGURE:
            assert len(ours) == len(theirs)
            for a, b in zip(ours, theirs, strict=True):
                if a != b:
                    x, y = (float(c[10:30]) for c in (a, b))
                    assert a[:8] == b[:8] and abs(x - y) <= 1e-13 * abs(y), (a, b)
            ours = theirs
        assert ours == theirs, name
        before = [c.image for c in fits.getheader(tmp_path / "fresh" / name).cards]
        assert _cards(mine, False) == _cards(before, False), name


def test_xcsao_does_not_write_unasked(tmp_path, monkeypatch):
    from astropy.io import fits

    write(tmp_path / "fresh")
    _run(tmp_path, monkeypatch, "default")
    assert fits.getheader(tmp_path / "g2500.fits") == \
        fits.getheader(tmp_path / "fresh" / "g2500.fits")


def test_xcsao_read_only_spectrum_is_not_saved(tmp_path, monkeypatch):
    import os

    write(tmp_path)
    os.chmod(tmp_path / "g2500.fits", 0o444)
    monkeypatch.chdir(tmp_path)
    session = xcsao.Xcsao({"templates": "tg.fits", "save_vel": True, "logfiles": ""})
    session.run(["g2500.fits"])
    assert session.err == "XCSAO: cannot write to g2500.fits; not saving results\n"
    assert session.results and session.results[0].xcr > 100
