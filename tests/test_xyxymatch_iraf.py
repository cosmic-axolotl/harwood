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
CASES = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))


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
    assert Path("ours.out").read_text(encoding="utf-8") == (DATA / f"{case}.out").read_text(
            encoding="utf-8")


def test_a_harwood_catalogue_is_matched_by_its_x_and_y(tmp_path, monkeypatch):
    """daofind's .coo puts its id first. Counting columns, as IRAF does,
    matched the ids against the x's and reported a shift of zero; and with
    both lists under nmatch, the summary printed the initial transformation
    as though it had been found."""
    from harwood.core import sample
    from harwood.docsite import examples

    monkeypatch.chdir(tmp_path)
    examples.prepare("reduced", tmp_path)
    for name in ("obj001", "obj002"):
        get_task("daofind").run(image=f"red/{name}.fits", output=tmp_path / f"{name}.coo",
                                fwhmpsf=3.0)
    result = get_task("xyxymatch").run(input=tmp_path / "obj002.coo",
                                       reference=tmp_path / "obj001.coo",
                                       output=tmp_path / "pairs.txt", tolerance=2.0)
    assert result.data["pairs"] >= 15
    dx, dy = sample.DRIFT[1]
    assert abs(result.data["xshift"] + dx) < 0.1 and abs(result.data["yshift"] + dy) < 0.1
