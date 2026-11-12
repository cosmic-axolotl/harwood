"""rvcorrect against IRAF 2.18's: the table it prints (every component, to its
last printed digit), the observatory lines, its warnings, the velocities it
records in its parameters and, with imupdate, in the images' headers.

tests/data/iraf_rvcorrect holds the inputs (rv_inputs.py), the cases (rv_cases.py)
and what IRAF did (run.py, rv_collect.py -> rvcorrect.json): lists with and
without the epoch and the velocity, several observatories, another solar motion,
the parameters, images with each form of date and an OBSERVAT card.

The values IRAF records are its doubles; harwood's agree to 5e-9 km/s and 5e-9
days -- IRAF's theories are good to 0.01 km/s -- because f2c's arithmetic on SPP's
real constants is reproduced as found, not from its C (core/rvcorrect.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from harwood.core import rvcorrect

HERE = Path(__file__).parent / "data" / "iraf_rvcorrect"
GOLD = json.loads((HERE / "rvcorrect.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from rv_cases import CASES  # noqa: E402
from rv_inputs import write  # noqa: E402

YES = {"yes": True, "no": False}


def _run(tmp_path, monkeypatch, case):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    params = {"observatory": "kpno", **{k: YES.get(v, v) for k, v in CASES[case].items()}}
    if params.get("epoch") == "INDEF":
        params["epoch"] = None
    session = rvcorrect.Rvcorrect(params)
    updates = []
    if "files" in params:
        session.files(params["files"].split(","))
    elif "images" in params:
        updates = session.images(params["images"].split(","))
    else:
        session.parameters()
    return session, updates


@pytest.mark.parametrize("case", list(CASES))
def test_rvcorrect_prints_what_iraf_prints(tmp_path, monkeypatch, case):
    session, _ = _run(tmp_path, monkeypatch, case)
    assert session.out == GOLD[case]["out"]
    assert session.err == GOLD[case]["err"]


@pytest.mark.parametrize("case", [c for c in CASES if GOLD[c]["params"]])
def test_rvcorrect_records_what_iraf_records(tmp_path, monkeypatch, case):
    session, _ = _run(tmp_path, monkeypatch, case)
    hjd, vhelio, vlsr = session.last
    gold = {k: float(v) for k, v in GOLD[case]["params"].items()}
    assert hjd == pytest.approx(gold["hjd"], abs=5e-9)
    assert vhelio == pytest.approx(gold["vhelio"], abs=5e-9)
    assert vlsr == pytest.approx(gold["vlsr"], abs=5e-9)


@pytest.mark.parametrize("case", [c for c in CASES if GOLD[c]["cards"]])
def test_rvcorrect_updates_the_images_as_iraf(tmp_path, monkeypatch, case):
    _, updates = _run(tmp_path, monkeypatch, case)
    assert [u["image"] for u in updates] == sorted(GOLD[case]["cards"])
    for u in updates:
        cards = GOLD[case]["cards"][u["image"]]
        for key in ("HJD", "VHELIO", "VLSR"):
            assert u[key] == pytest.approx(float(cards[key][10:].split("/")[0]), abs=5e-9)
        assert u["VSUN"] == cards["VSUN"][10:].strip().strip("'").rstrip()
