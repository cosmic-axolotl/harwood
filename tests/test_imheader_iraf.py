"""imheader against IRAF 2.18's: the short line (name[NX,NY][type]: title) and, with
--longheader, the header a card a line as IRAF prints it -- for every pixel type, a
one-dimensional image, an untitled one, an extension and a real header's cards.
IRAF's four lines about its own storage (bad pixels, storage mode, dates, pixel file)
are left out; harwood's short form is its table, so only the long form is compared.

tests/data/iraf_imheader holds the images (imh_inputs.py) and what IRAF printed
(run.py -> imheader.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import harwood

HERE = Path(__file__).parent / "data" / "iraf_imheader"
GOLD = json.loads((HERE / "imheader.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from imh_inputs import write  # noqa: E402

#: IRAF's lines about an .imh image's storage, which harwood does not print.
STORAGE = ("No bad pixels", "Line storage mode", "Created ", "Pixel file ")


@pytest.mark.parametrize("name", sorted({k.rsplit(" ", 1)[0] for k in GOLD}))
def test_the_long_header_is_iraf_s(tmp_path, monkeypatch, name):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = harwood.run("imheader", input=[name], longheader=True)
    theirs = [line for line in GOLD[f"{name} long"] if not line.startswith(STORAGE)]
    assert result.stdout == theirs
    assert theirs[0] == GOLD[f"{name} short"][0]
