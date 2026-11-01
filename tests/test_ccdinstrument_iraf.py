"""ccdinstrument against IRAF 2.18, every character of a session and the files it leaves.

tests/data/iraf_ccdinstrument holds what IRAF wrote (run.py, ccdinst_cases.py) when fed
these lines.
"""

import json
import sys
from pathlib import Path

import pytest

from harwood.core import ccdinstsession, images

HERE = Path(__file__).parent / "data" / "iraf_ccdinstrument"
GOLD = json.loads((HERE / "ccdinstrument.json").read_text())
sys.path.insert(0, str(HERE))
from ccdinst_cases import CASES  # noqa: E402
from ccdinst_inputs import write  # noqa: E402


def session_of(name):
    """What the task does, with the lines given and everything in one text."""
    kw = CASES[name]
    pending = kw.get("stdin", "").split("\n")
    # a last line without its newline is still a line; after a newline there is no more
    if pending and pending[-1] == "":
        pending.pop()
    lines = iter(pending)
    out = []
    session = ccdinstsession.Session(
        kw.get("instrument", "inst.dat"), "subs", kw.get("parameters", "basic"),
        lambda: next(lines, None), out.append, out.append,
    )  # fmt: skip
    for spec in images.expand(kw["images"]):
        label = str(spec)
        header = ccdinstsession.open_image(label)
        if header is None:
            out.append(f"Warning: Cannot open image ({label})\n")
            continue
        session.image, session.header = label, header
        if kw.get("edit", True):
            session.edit(label, header)
        else:
            session.show()
        if session.ncmd == "quit":
            break
    session.finish()
    return "".join(out)


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path)
    Path("o2.dat").write_text("exptime OTHER\nsubset  'OF ME' 'a b'\n")


@pytest.mark.parametrize("case", list(GOLD))
def test_a_session_types_what_iraf_types(work, case):
    expected = GOLD[case]
    before = {p.name: p.read_bytes() for p in Path().iterdir()}
    text = session_of(case)
    if expected["error"]:
        # the null key, where IRAF ends and the session here goes on
        assert "null key string" in expected["error"]
        assert text.startswith(expected["text"])
        assert "no image type to translate" in text
        return
    assert text == expected["text"]
    files = {
        p.name: p.read_text(errors="replace")
        for p in Path().iterdir()
        if before.get(p.name) != p.read_bytes()
    }
    assert files == expected["files"]


def test_the_session_s_commands_from_the_terminal(work, monkeypatch, capsys):
    import io

    from harwood.core.registry import get_task

    monkeypatch.setattr("sys.stdin", io.StringIO("exptime EXPTIME\nq\nyes\n"))
    get_task("ccdinstrument").run(images=["a.fits"], instrument="inst.dat")
    assert "exptime   EXPTIME" in capsys.readouterr().out
    assert "exptime" in Path("inst.dat").read_text()
    assert "EXPTIME" in Path("inst.dat").read_text()
