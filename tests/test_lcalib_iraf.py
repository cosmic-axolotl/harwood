"""lcalib against IRAF 2.18's, line for line on both streams: a star's bands,
magnitudes, F-nu and F-lambda (another fnuzero), the extinction table, the stars of a
directory and a blackbody.

tests/data/iraf_lcalib holds the cases (lc_cases.py) and what IRAF printed (lc.json),
from the calibration tables harwood carries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_lcalib"
GOLD = json.loads((HERE / "lc.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from lc_cases import CASES  # noqa: E402


@pytest.mark.parametrize("case", list(CASES))
def test_lcalib_lists_what_iraf_s_lists(case):
    result = get_task("lcalib").run(**CASES[case])
    assert result.data["stdout"] == GOLD[case]["stdout"]
    assert result.data["errors"] == GOLD[case]["stderr"]
