"""hedit, its expressions and IRAF's %g, against IRAF 2.18.

tests/data/iraf_hedit.json holds what IRAF's hedit printed and the cards it
left for 52 edits of one header, the value of 39 expressions as hedit
evaluated them, and 33 numbers as IRAF's %g writes them. The cards are
compared as text, column for column -- with one exception, on purpose:
where IRAF writes text without quotes into a numeric or logical card (yes,
abc), which no FITS reader accepts, harwood writes the valid card.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import harwood
from harwood.core import evexpr, iraffmt

REF = json.loads((Path(__file__).parent / "data" / "iraf_hedit.json").read_text())
NUMBER = re.compile(r"^-?(\d+\.?\d*|\.\d+)(E-?\d+)?$")


def _base(path: Path) -> None:
    from astropy.io import fits

    header = fits.Header([fits.Card.fromstring(card) for card in REF["base"]])
    fits.PrimaryHDU(np.zeros((3, 4), np.float32), header=header).writeto(path)


def _cards(path: Path) -> list[str]:
    from astropy.io import fits

    skip = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "HISTORY", ""}
    return [card.image.rstrip() for card in fits.getheader(path).cards
            if card.keyword not in skip]


def _valid(card: str, base: dict[str, str]) -> str:
    """The card harwood writes where IRAF's is not valid FITS."""
    key, value = card[:8], card[10:].strip()
    if value.startswith("'") or NUMBER.match(value) or value in ("T", "F"):
        return card
    was_logical = base.get(key, "")[10:].strip() in ("T", "F")
    if was_logical and value in ("yes", "no"):
        return f"{key}= {'T' if value == 'yes' else 'F':>20}"
    return f"{key}= '{value:<8}'"


@pytest.mark.parametrize("case", REF["cases"], ids=[c["name"] for c in REF["cases"]])
def test_hedit_matches_iraf(case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = f"{case['name']}.fits"
    _base(tmp_path / name)
    result = harwood.run("hedit", images=[name], fields=[case["fields"]],
                         value=case["value"], **case["params"])
    printed = [line for line in result.data.get("text", [])
               if not line.startswith("Warning")]
    assert printed == case["stdout"]
    base = {card[:8]: card for card in REF["base"]}
    assert _cards(tmp_path / name) == [_valid(card, base) for card in case["cards"]]
    # addonly on a keyword that is there. (IRAF's other complaint, that a
    # wildcard caught i_naxis and it cannot be deleted, does not arise: a
    # wildcard here matches the header's own keywords only.)
    for complaint in case["stderr"]:
        if "already exists" in complaint:
            assert any(complaint in str(w) for w in result.warnings)


@pytest.mark.parametrize("expr", list(REF["expressions"]))
def test_expressions_evaluate_as_in_iraf(expr):
    def getop(name):
        if name.upper() == "EXPTIME":
            return evexpr.Operand("int", 15)
        raise evexpr.ExprError(name)

    assert evexpr.encode(evexpr.evaluate(expr, getop)) == REF["expressions"][expr].strip('"')


@pytest.mark.parametrize("text", list(REF["g_format"]))
def test_reals_print_as_iraf_prints_them(text):
    value = float(np.float32(float(text)))
    assert iraffmt.g(value) == REF["g_format"][text]


def test_lexnum_classifies_numbers_as_iraf():
    assert evexpr.lexnum("17b") == (evexpr.OCT, 3)
    assert evexpr.lexnum("1fx") == (evexpr.HEX, 3)
    assert evexpr.lexnum("12:30:45") == (evexpr.REAL, 8)
    assert evexpr.lexnum("-42") == (evexpr.DEC, 3)
    assert evexpr.lexnum("1.5e-7") == (evexpr.REAL, 6)
    assert evexpr.lexnum(".") == (evexpr.NON, 0)
    assert evexpr.lexnum("INDEF") == (evexpr.REAL, 5)
    assert evexpr.lexnum("abc")[0] == evexpr.NON


def test_expression_errors_are_iraf_errors():
    with pytest.raises(evexpr.ExprError, match="incompatible types"):
        evexpr.evaluate("('a' + 1)")
    with pytest.raises(evexpr.ExprError, match="unknown function"):
        evexpr.evaluate("(frobnicate(1))")
    with pytest.raises(evexpr.ExprError, match="requires 2 arguments"):
        evexpr.evaluate("(mod(1))")


def test_verify_asks_and_takes_the_answer(tmp_path, monkeypatch):
    """verify: Enter keeps the proposed value, n skips, other text is the
    value to use; then 'update image ?' decides whether anything is written."""
    import io

    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    _base(tmp_path / "v.fits")
    monkeypatch.setattr("sys.stdin", io.StringIO("25\nyes\n"))
    harwood.run("hedit", images=["v.fits"], fields=["EXPTIME"], value="20", verify=True)
    assert fits.getheader(tmp_path / "v.fits")["EXPTIME"] == 25
    monkeypatch.setattr("sys.stdin", io.StringIO("\nno\n"))
    harwood.run("hedit", images=["v.fits"], fields=["EXPTIME"], value="30", verify=True)
    assert fits.getheader(tmp_path / "v.fits")["EXPTIME"] == 25


def test_history_keeps_what_was_there(tmp_path, monkeypatch):
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    _base(tmp_path / "h.fits")
    harwood.run("hedit", images=["h.fits"], fields=["AIRMASS"], value="(AIRMASS + 0.1)")
    history = list(fits.getheader(tmp_path / "h.fits")["HISTORY"])
    assert any("AIRMASS = 1.334567, was 1.234567" in line for line in history)


@pytest.mark.parametrize("run", list(REF["hselect"]["runs"]))
def test_hselect_prints_what_iraf_prints(run, tmp_path, monkeypatch):
    """Tab-separated, the card's own text, a string with blanks in double
    quotes, a field the header lacks left out -- the lines that go into an
    @list."""
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    for name, cards in REF["hselect"]["images"].items():
        header = fits.Header([fits.Card.fromstring(card) for card in cards])
        fits.PrimaryHDU(np.zeros((3, 4), np.float32), header=header).writeto(name)
    fields, expr = REF["hselect"]["runs"][run]
    result = harwood.run("hselect", input=["base.fits", "add_str.fits"], fields=[fields],
                         expr=expr)
    assert "".join(line + "\n" for line in result.stdout) == REF["hselect"]["cases"][run]
