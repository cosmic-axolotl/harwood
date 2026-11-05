"""names against IRAF 2.18's, line for line: record lists in order and out of it,
steps, from 0, open, too many and bad; roots with .imh, .pl, .fits and a section; an
appended section; no records; and only the images that exist.

tests/data/iraf_names holds the cases (nm_cases.py) and what IRAF printed (nm.json),
with sp.0002 and sp.0005 made for the check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_names"
GOLD = json.loads((HERE / "nm.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from nm_cases import CASES  # noqa: E402


@pytest.fixture
def start(tmp_path, monkeypatch):
    for number in (2, 5):
        fits.writeto(tmp_path / f"sp.{number:04d}.fits", np.zeros(10, np.float32))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", list(CASES))
def test_names_says_what_iraf_s_says(start, case):
    theirs = GOLD[case]
    if theirs and theirs[0].startswith("ERROR"):
        message = theirs[0].split('"')[1].rstrip("\x07")
        with pytest.raises(ValueError, match=message):
            get_task("names").run(**CASES[case])
        return
    assert get_task("names").run(**CASES[case]).data["text"] == theirs
