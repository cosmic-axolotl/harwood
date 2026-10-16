"""xyxymatch against IRAF's own output, byte for byte.

tests/data/iraf_xyxymatch holds coordinate lists (a field of 25 stars;
the same shifted with two stars lost and two strangers added, rotated and
scaled, flipped, with close pairs; a field of 120 stars and a shifted,
slightly rotated copy) and what IRAF 2.18.1's xyxymatch wrote for ten sets
of parameters: tolerance matching with and without the transformation,
triangles on few and on more stars than nmatch (the path that fits a
transformation to the matched triangles and then matches by tolerance),
a separation that merges close pairs, a ratio limit, and refpoints whose
ill-placed tie points leave only themselves matched. The files harwood
writes must be IRAF's, character for character.
"""

import json
import shutil
from pathlib import Path

import pytest

from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data" / "iraf_xyxymatch"
CASES = json.loads((DATA / "cases.json").read_text())


@pytest.fixture
def lists(tmp_path, monkeypatch):
    for path in DATA.iterdir():
        shutil.copy(path, tmp_path / path.name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", list(CASES))
def test_xyxymatch_writes_what_iraf_writes(lists, case):
    params = dict(CASES[case])
    params["input"], params["reference"] = Path(params["input"]), Path(params["reference"])
    if "refpoints" in params:
        params["refpoints"] = Path(params["refpoints"])
    for name in ("xref", "yref", "xin", "yin"):
        if name in params:
            params[name] = float(params[name])
    get_task("xyxymatch").run(output=Path("ours.out"), **params)
    assert Path("ours.out").read_text() == (DATA / f"{case}.out").read_text()
