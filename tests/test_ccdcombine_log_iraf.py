"""ccdred's combine and its presets with the package's logfile and verbose, against IRAF
2.18's: the IMCOMBINE block -- each rejection's parameters, thresholds, the columns of
what differs among the images (exposure, mode, median, mean, scale, zero, weight), a
section in the names -- in the logfile and, with verbose, on the standard output; the
date (cnvdate's) apart.

tests/data/iraf_ccdcombine holds the frames (make.py), the cases (cases.py) and what
IRAF wrote (run_log.py -> combine_log.json).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

import harwood

HERE = Path(__file__).parent / "data" / "iraf_ccdcombine"
GOLD = json.loads((HERE / "combine_log.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from make import write  # noqa: E402

DATE = re.compile(r"[A-Z][a-z]{2} [ \d]\d [ \d]\d:\d\d")


def _value(key, value):
    if value in ("yes", "no"):
        return value == "yes"
    if key in ("lthreshold", "hthreshold") and value == "INDEF":
        return None
    return value


@pytest.mark.parametrize("case", list(GOLD))
def test_the_combine_log_is_iraf_s(tmp_path, monkeypatch, case):
    gold = GOLD[case]
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    images = gold["images"]
    from harwood.core.images import split_list

    names = (Path("all.lis").read_text(encoding="utf-8").split() if images == "@all.lis"
             else [n.strip() for n in split_list(images)])
    params = {key: _value(key, value) for key, value in gold["params"].items()}
    params.pop("images", None)
    task = gold["task"]
    defaults = {} if task != "combine" else {"reject": "none"}
    result = harwood.run(task, input=names, output="out", logfile="logfile",
                         verbose=True, **{"ccdtype": "", "subsets": False,
                                          "process": False, **defaults, **params})
    mine = DATE.sub("", (tmp_path / "logfile").read_text(encoding="utf-8"))
    assert mine == DATE.sub("", gold["logfile"])
    assert DATE.sub("", result.data["stdout"]) == DATE.sub("", gold["printed"])
