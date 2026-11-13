"""fitcoords by hand -- IRAF's igsfit graph -- against IRAF 2.18's, from files of cursor
reads and answers: everything printed (c, the colon commands, the prompts, the bells),
the fc records, the deletion lists and the logfile to the byte, and every graph drawn:
its window, viewport, labels and title, and each mark, plus or cross, where IRAF's
metacode puts it -- the ones d and u draw over a graph too. The graph of the plot log
is the last state's.

tests/data/iraf_igsfit holds the sessions (ig_cases.py), and what IRAF left in each
(run.py, ig_collect.py -> igsfit.json); the inputs are tests/data/iraf_transform's.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

import harwood
from harwood.core import gtools, igsfit

HERE = Path(__file__).parent / "data" / "iraf_igsfit"
GOLD = json.loads((HERE / "igsfit.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).parent / "data" / "iraf_transform"))
from ig_cases import CASES  # noqa: E402
from tr_inputs import write  # noqa: E402

BANNER = re.compile(r"^(Community IRAF|harwood) .*$", re.M)
DATE = re.compile(r"^# .*$", re.M)
GKI = 32767


@pytest.fixture
def frames(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(frames, monkeypatch, case):
    images, params, cursor, answers = CASES[case]
    (frames / "cursor").write_text(cursor, encoding="utf-8")
    (frames / "answers").write_text(answers, encoding="utf-8")
    sessions = []
    made = igsfit.Session.__init__

    def keep(self, *args, **kwargs):
        made(self, *args, **kwargs)
        sessions.append(self)

    monkeypatch.setattr(igsfit.Session, "__init__", keep)
    values = {k: {"yes": True, "no": False}.get(v, v) for k, v in params.items()}
    data = harwood.run("fitcoords", images=images.split(","), cursor="cursor",
                       answers="answers", **values).data
    return data, sessions


def _gki(value, a, b, v1, v2, log):
    if log:
        value, a, b = gtools.elog(value), gtools.elog(a), gtools.elog(b)
    return (v1 + (value - a) / (b - a) * (v2 - v1)) * GKI


#: Half a mark of gt's size 2 on stdvdm, in GKI: GIO clips a mark the viewport's edge
#: cuts, so only whole ones are compared.
HALF = (371 / 2, 482 / 2)


def _whole(gx, gy, vx1, vx2, vy1, vy2):
    return (vx1 * GKI + HALF[0] - 1 <= gx <= vx2 * GKI - HALF[0] + 1
            and vy1 * GKI + HALF[1] - 1 <= gy <= vy2 * GKI - HALF[1] + 1)


def _compare(ours: igsfit.Frame, gold: dict):
    logs = ours.logs
    x1, x2, y1, y2, vx1, vx2, vy1, vy2 = gold["window"]
    for mine, theirs in zip(ours.window, (x1, x2, y1, y2), strict=True):
        assert mine == pytest.approx(theirs, rel=2e-6, abs=6e-8)   # the json keeps 7 decimals
    assert ours.viewport == pytest.approx((vx2 - vx1, vy2 - vy1), abs=1e-4)
    labels = [ours.xlabel, ours.ylabel, *ours.title]
    assert gold["text"][-len(labels):] == labels
    drawn = [(k, x, y) for k, x, y in ours.marks] + [(k, x, y) for k, x, y, _ in ours.over]
    # ours, not the json's 7 decimals, which are too few for a zoomed residual axis
    x1, x2, y1, y2 = ours.window
    inside = [(k, _gki(x, x1, x2, vx1, vx2, logs[0]), _gki(y, y1, y2, vy1, vy2, logs[1]))
              for k, x, y in drawn]
    inside = [m for m in inside if _whole(m[1], m[2], vx1, vx2, vy1, vy2)]
    theirs = [m for m in gold["marks"] if _whole(m[1], m[2], vx1, vx2, vy1, vy2)]
    assert [k for k, _, _ in inside] == [m[0] for m in theirs]
    for (_, gx, gy), (_, ix, iy) in zip(inside, theirs, strict=True):
        assert abs(gx - ix) <= 2 and abs(gy - iy) <= 2


@pytest.mark.parametrize("case", list(CASES))
def test_the_session_prints_and_writes_what_iraf_does(frames, monkeypatch, case):
    data, _ = _run(frames, monkeypatch, case)
    gold = GOLD[case]
    assert BANNER.sub("", data["stdout"]) == gold["said"]
    ours = {p.name: DATE.sub("", p.read_text(encoding="utf-8"))
            for p in sorted((frames / "database").glob("fc*"))}
    assert ours == gold["fc"]
    assert {p.name: p.read_text(encoding="utf-8")
            for p in sorted(frames.glob("*.db"))} == gold["deletions"]
    assert BANNER.sub("", (frames / "logfile").read_text(encoding="utf-8")) == \
        gold["logfile"]


@pytest.mark.parametrize("case", list(CASES))
def test_the_graphs_are_iraf_s(frames, monkeypatch, case):
    _, sessions = _run(frames, monkeypatch, case)
    gold = GOLD[case]
    drawn = [f for s in sessions for f in s.frames]
    assert len(drawn) == len(gold["frames"])
    for ours, theirs in zip(drawn, gold["frames"], strict=True):
        _compare(ours, theirs)
    assert len(sessions) == len(gold["plotfile"])
    for session, theirs in zip(sessions, gold["plotfile"], strict=True):
        _compare(session.final(), theirs)


def test_the_panel_is_the_session(frames, monkeypatch):
    """The window's keys are the session's: the delete_point case typed into the panel
    leaves IRAF's graph and fit; a fit with no degrees of freedom is said, and the one
    before it kept (IRAF would end the task)."""
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.core import fitcoords as fc
    from harwood.tasks.longslit import _header_of
    from harwood.viewer._qt import application
    from harwood.viewer.fitcoords_panel import FitcoordsPanel

    application()
    coords = fc.get_coords(["arc"], frames / "database", _header_of)
    session = igsfit.Session(coords, igsfit.Common(), "arc")
    session.graph()
    panel = FitcoordsPanel(session)
    ended = []
    panel.finished.connect(lambda: ended.append(True))
    panel._cursor = (135.8, 0.0010)
    assert panel.key("d") and session.state == "d"
    assert "Delete 'p'oint" in panel.status.text()
    assert panel.key("p") and not session.state
    assert int((session.pts[fc.W] == 0).sum()) == 1
    assert panel.key("f") and panel.key("r")
    gold = GOLD["delete_point"]
    assert len(session.frames) == len(gold["frames"])
    for ours, theirs in zip(session.frames, gold["frames"], strict=True):
        _compare(ours, theirs)
    rms = session.rms
    panel.command.setText(":xorder 20")
    panel._run_command()
    panel.command.setText(":yorder 20")
    panel._run_command()
    assert panel.key("f")
    assert "No degrees of freedom" in panel.status.text()
    assert session.rms == rms and not session.done
    panel.command.setText(":show")
    panel._run_command()
    assert "xorder 20" in panel.status.text()
    assert panel.key("q")
    assert ended == [True]
