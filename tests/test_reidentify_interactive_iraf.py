"""reidentify, interactive, against IRAF 2.18.

tests/data/iraf_reidentify_interactive holds what IRAF's reidentify printed and
wrote with interactive yes -- every question answered yes, as PyRAF answers a
query with nothing on its input -- and YES, carrying arc3's identified aperture
to its others and to arc3b: before each fit's line goes to the log, identify's
keys on that spectrum (a line deleted, the fit redone; j and :image refused, as
reidentify refuses them), from one cursor file the spectra share. With yes the
line is said before and after, with YES after; the current feature is the one
the last look left, carried with the reference. What is printed, the log and
every record must be IRAF's.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_identify_keys import _same
from test_reidentify_iraf import write_night

from harwood.core import database
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data" / "iraf_reidentify_interactive"


def _start(tmp_path, monkeypatch, mode):
    write_night(tmp_path)
    (tmp_path / "database").mkdir()
    autoidentified = re.split(r"(?m)^(?=# )", (DATA / f"{mode}.db").read_text(
        encoding="utf-8"))[1]
    (tmp_path / "database" / "idarc3").write_text(autoidentified, encoding="utf-8")
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("mode", ["yes", "YES"])
def test_each_fit_is_looked_at_as_iraf_shows_it(tmp_path, monkeypatch, mode):
    _start(tmp_path, monkeypatch, mode)
    result = get_task("reidentify").run(reference="arc3", images="arc3,arc3b",
                                        coordlist="arc_lines.dat", logfiles="",
                                        interactive=mode, nlost=40,
                                        gcommands=str(DATA / f"{mode}.cur"))
    ours = result.data["stdout"].splitlines()
    theirs = (DATA / f"{mode}.out").read_text(encoding="utf-8").splitlines()
    assert len(ours) == len(theirs)
    for mine, iraf_s in zip(ours, theirs, strict=True):
        assert _same(mine, iraf_s), (mine, iraf_s)
    assert result.data["text"] == (DATA / f"{mode}.log").read_text(
        encoding="utf-8").splitlines()

    theirs_dir = tmp_path / "iraf"
    (theirs_dir / "database").mkdir(parents=True)
    for image, name in (("arc3", f"{mode}.db"), ("arc3b", f"{mode}.dbb")):
        shutil.copy(DATA / name, theirs_dir / "database" / f"id{image}")
        ours_r = {r.section: r for r in database.read_all(image, tmp_path)}
        theirs_r = {r.section: r for r in database.read_all(image, theirs_dir)}
        assert ours_r.keys() == theirs_r.keys()
        for section, t in theirs_r.items():
            m = ours_r[section]
            assert len(m) == len(t), f"{image}{section}: a different set of lines"
            assert (m.used == t.used).all()
            assert np.abs(m.fitted - t.fitted).max() < 5e-5


def test_no_looks_at_none_and_no_to_all_stops_asking(tmp_path, monkeypatch):
    """The answers t_reidentify.x reads (PyRAF cannot be given them with the keys):
    yes looks, no says the line and goes on, NO stops asking."""
    _start(tmp_path, monkeypatch, "yes")
    answers = tmp_path / "answers"
    answers.write_text("yes\nno\nNO\n", encoding="utf-8")
    cursor = tmp_path / "cursor"
    cursor.write_text("4100 0 1 d\n0 0 1 q\n", encoding="utf-8")
    result = get_task("reidentify").run(reference="arc3", images="arc3,arc3b",
                                        coordlist="arc_lines.dat", logfiles="",
                                        interactive="yes", nlost=40,
                                        gcommands=str(cursor), answers=str(answers))
    said = [line.split()[0:3] for line in result.data["stdout"].splitlines()
            if "Ap" in line]
    assert said == [["arc3", "-", "Ap"]] * 3 + [["arc3b", "-", "Ap"]], \
        "Ap 1 before and after its look, Ap 3 before no, arc3b's first before NO"
    assert "61/102" in result.data["text"][2], "the line deleted in the look is gone"
