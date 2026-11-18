"""RVSAO's templates ship with harwood: rvsao$templates/ and a bare name both find
them, and xcsao against one gives what RVSAO 2.8.5 gives (fallstars against itself:
R 1.712E8, h 0.982, CZ 830.300 -- the template's own velocity -- in the RVSAO run of
2026-11-18)."""

from __future__ import annotations

import shutil

import harwood
from harwood.core import rvsao


def test_the_templates_are_in_the_package():
    names = {p.name for p in rvsao.TEMPLATES.glob("*.fits")}
    assert {"fallstars.fits", "femtemp97.fits", "eatemp.fits", "A1.fits"} <= names
    assert len(names) >= 90
    assert (rvsao.TEMPLATES / "Readme").exists()


def test_a_template_is_found_by_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bundled = str(rvsao.TEMPLATES / "fallstars")
    assert rvsao.template_file("rvsao$templates/fallstars") == bundled
    assert rvsao.template_file("fallstars") == str(rvsao.TEMPLATES / "fallstars.fits")
    (tmp_path / "fallstars.fits").write_bytes(b"")
    assert rvsao.template_file("fallstars") == "fallstars", "the working directory first"


def test_xcsao_with_a_shipped_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copy(rvsao.TEMPLATES / "fallstars.fits", tmp_path / "estrela.fits")
    result = harwood.run("xcsao", spectra=["estrela.fits"], templates="fallstars")
    text = "\n".join(result.data.get("text", [])) + result.data.get("stdout", "")
    assert "CZ: 830.300 +/-   0.000" in text
    assert "R: 1.712E8" in text


def test_the_list_of_templates():
    result = harwood.run("xcsao", spectra=["x"], list_templates=True)
    names = {row["template"] for row in result.data["rows"]}
    assert {"fallstars", "femtemp97"} <= names
    fall = next(r for r in result.data["rows"] if r["template"] == "fallstars")
    assert fall["log"] and abs(fall["start"] - 3700.0) < 1.0
