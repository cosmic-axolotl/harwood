"""curfit against IRAF 2.18, line by line in both precisions.

tests/data/iraf_curfit holds what IRAF printed (run.py, curfit_cases.py): every function and
weighting on lists and images, and in sessions driven by a cursor file what only a session
sets there, the sample, the binning and the rejection, here options.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_curfit"
GOLD = json.loads((HERE / "curfit.json").read_text())
sys.path.insert(0, str(HERE))
from curfit_cases import LISTS  # noqa: E402
from curfit_images import write  # noqa: E402
from curfit_keys_cases import KEYS as KEY_CASES  # noqa: E402

KEYS = json.loads((HERE / "curfit_keys.json").read_text())
IMAGES = {"im2": "fits.fits", "im3": "cube.fits"}


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, text in LISTS.items():
        Path(name).write_text(text)
    write(tmp_path)


def mine(params, **more):
    name = params["input"]
    for iraf_name, file in IMAGES.items():
        name = name.replace(iraf_name, file)
    args = {k: v for k, v in params.items() if k not in ("input", "cmds")}
    return get_task("curfit").run(input=[name], **args, **more)


def clean(lines):
    """What is not the fit: the time, and an image's name in IRAF's format."""
    out = []
    for line in lines:
        if line.startswith(("# Sat", "# Sun", "# Mon", "# Tue", "# Wed", "# Thu", "# Fri")):
            continue
        for iraf_name, file in IMAGES.items():
            line = line.replace(f"# {file}", f"# {iraf_name}")
        out.append(line.replace("# sample =  ", "# sample = "))
    return out


@pytest.mark.parametrize("case", list(GOLD["cases"]))
def test_curfit_prints_what_iraf_prints(work, case):
    expected = GOLD["cases"][case]
    if expected["error"]:
        with pytest.raises(Exception) as caught:
            mine(expected["params"])
        message = expected["error"].split('"')[1].rstrip("\x07")
        assert message in str(caught.value)
        return
    result = mine(expected["params"])
    assert clean(result.stdout) == clean(expected["stdout"])
    told = [w for w in expected["stderr"] if w]
    assert [w.message for w in result.warnings] == told


@pytest.mark.parametrize("case", list(GOLD["sessions"]))
def test_a_session_s_settings_as_options(work, case):
    expected = GOLD["sessions"][case]
    result = mine(expected["params"], **expected["params"]["cmds"])
    assert clean(result.stdout) == clean(expected["stdout"])


@pytest.mark.parametrize("case", list(KEYS))
def test_a_session_with_keys_prints_what_iraf_prints(work, case):
    cursor, kw = KEY_CASES[case]
    Path("cursor").write_text(cursor)
    result = get_task("curfit").run(input=["wide.txt"], cursor="cursor", **kw)
    assert clean(result.stdout) == clean(KEYS[case]["stdout"])


def test_the_sessions_rejected_something():
    """Otherwise the sessions above would not hold the rejection to anything."""
    rejected = {
        case: next(line for line in gold["stdout"] if line.startswith("# nrejected"))
        for case, gold in GOLD["sessions"].items()
    }
    assert sum(not line.endswith(" 0") for line in rejected.values()) >= 8


def _press(monkeypatch, reads):
    """The window, its keys pressed where a user would have the cursor."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.viewer import dock
    from harwood.viewer._qt import application

    application()
    seen = []

    def show(panel, title, size=(0, 0)):
        seen.append(title)
        for x, y, key in reads:
            panel._cursor = (x, y)
            assert panel.key(key)
        assert panel.session.done
        return panel

    monkeypatch.setattr(dock, "show", show)
    return seen


@pytest.mark.parametrize("reads", [
    [(0.0, 0.0, "q")],
    [(0.0, 0.0, "f"), (0.0, 0.0, "q")],
])
def test_the_window_fits_as_the_cursor_file_does(work, monkeypatch, reads):
    Path("cursor").write_text("".join(f"{x} {y} 1 {key}\n" for x, y, key in reads))
    expected = get_task("curfit").run(input=["wide.txt"], cursor="cursor")
    seen = _press(monkeypatch, reads)
    result = get_task("curfit").run(input=["wide.txt"], interactive=True)
    assert seen and "curfit" in seen[0]
    assert clean(result.stdout) == clean(expected.stdout)


def test_a_point_deleted_in_the_window_is_out_of_the_fit(work, monkeypatch):
    rows = [line.split() for line in Path("wide.txt").read_text().splitlines() if line.strip()]
    x, y = float(rows[len(rows) // 2][0]), float(rows[len(rows) // 2][1])
    reads = [(x, y, "d"), (x, y, "f"), (x, y, "q")]
    Path("cursor").write_text("".join(f"{a} {b} 1 {key}\n" for a, b, key in reads))
    expected = get_task("curfit").run(input=["wide.txt"], cursor="cursor")
    plain = get_task("curfit").run(input=["wide.txt"], cursor=_quit())
    _press(monkeypatch, reads)
    result = get_task("curfit").run(input=["wide.txt"], interactive=True)
    assert clean(result.stdout) == clean(expected.stdout)
    assert clean(result.stdout) != clean(plain.stdout)


def _quit():
    Path("quit").write_text("0 0 1 q\n")
    return "quit"
