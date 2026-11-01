"""curfit against IRAF 2.18, line by line in both precisions.

tests/data/iraf_curfit holds what IRAF printed (run.py, curfit_cases.py): every function and
weighting on lists and images, and in sessions driven by a cursor file what only a session
sets there, the sample, the binning and the rejection, here options.
"""

import json
import sys
from pathlib import Path

import pytest

from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_curfit"
GOLD = json.loads((HERE / "curfit.json").read_text())
sys.path.insert(0, str(HERE))
from curfit_cases import LISTS  # noqa: E402
from curfit_images import write  # noqa: E402

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


def test_the_sessions_rejected_something():
    """Otherwise the sessions above would not hold the rejection to anything."""
    rejected = {
        case: next(line for line in gold["stdout"] if line.startswith("# nrejected"))
        for case, gold in GOLD["sessions"].items()
    }
    assert sum(not line.endswith(" 0") for line in rejected.values()) >= 8
