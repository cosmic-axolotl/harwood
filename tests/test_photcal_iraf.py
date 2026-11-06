"""fitparams, invertfit and evalfit against IRAF's own output.

The data are the Skinakas night (PG2331+055 and its two comparison stars,
five times through the night, and nine program stars in LSI+61 235), as the
automatico pipeline left them: observations.obsfile, catalogo.mkcatalog and
configuration.mkconfig. config2.cfg exercises the rest of the language --
a set equation, a constant, a delta, a derivative, weight equations with
limits, error equations -- and obs2.txt has an INDEF magnitude, a standard
missing from the catalog and a program star with an undefined magnitude.
IRAF 2.18 ran each case (run.cl, run2.cl, run3.cl, run4.py); its files are here.
config3.cfg is a system evalfit can evaluate, with the catalog on the left.

The parameters files and the logs agree to the last digit IRAF prints; the
calibrated magnitudes to the seventh decimal, but for one star in one case:
inverted with no catalog to start from, its last step is taken where the
rms is 1e-12, all rounding, and it lands two units of the seventh decimal
away (1.9e-6 in 15.16, two units in the last place of a float).
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from harwood.core.photcal import apfile, config, entry, evalfit, fit, fitsession, invert
from harwood.core.photcal import mkobsfile as mkobs
from harwood.core.photcal import obsfile as obs

DATA = Path(__file__).parent / "data" / "iraf_photcal"
CATALOG = "catalogo.mkcatalog"

FITS = {
    "1": ("observations.obsfile", "configuration.mkconfig", dict(weighting="uniform")),
    "B": ("observations.obsfile", "configuration.mkconfig",
          dict(weighting="photometric", addscatter=True)),
    "C": ("observations.obsfile", "configuration.mkconfig",
          dict(weighting="photometric", addscatter=False)),
    "D": ("observations.obsfile", "configuration.mkconfig",
          dict(weighting="uniform", nreject=3, low_reject=1.5, high_reject=1.5)),
    "E": ("observations.obsfile", "config2.cfg", dict(weighting="equations")),
    "F": ("observations.obsfile", "config2.cfg", dict(weighting="photometric")),
    "G": ("obs2.txt", "configuration.mkconfig", dict(weighting="uniform")),
    "3": ("observations.obsfile", "config3.cfg", dict(weighting="uniform")),
}

F6 = "%-10s %14.7f %14.7f %14.7f %14.7f %14.7f %14.7f\\n"
INVERTS = {
    "inv1": ("observations.obsfile", "configuration.mkconfig", "fit1.txt",
             dict(catalogs=[CATALOG])),
    "invB": ("observations.obsfile", "configuration.mkconfig", "fitB.txt",
             dict(catalogs=[CATALOG])),
    "invE1": ("observations.obsfile", "config2.cfg", "fitE.txt",
              dict(catalogs=[CATALOG], errors="equations")),
    "invE2": ("observations.obsfile", "config2.cfg", "fitE.txt",
              dict(catalogs=[CATALOG], print_vars=["BV", "XV", "mV"], objects="standards")),
    "invG": ("obs2.txt", "configuration.mkconfig", "fitG.txt",
             dict(catalogs=[CATALOG], objects="program")),
    "invG2": ("obs2.txt", "configuration.mkconfig", "fitG.txt", dict(errors="undefined")),
    "invP1": ("observations.obsfile", "configuration.mkconfig", "fit1.txt",
              dict(catalogs=[CATALOG], format=F6)),
    "invPE1": ("observations.obsfile", "config2.cfg", "fitE.txt",
               dict(catalogs=[CATALOG], errors="equations", format=F6)),
    "invPE2": ("observations.obsfile", "config2.cfg", "fitE.txt",
               dict(catalogs=[CATALOG], print_vars=["BV", "XV", "mV"], objects="standards",
                    format="%-10s" + " %14.7f" * 9 + "\\n")),
    "invPG2": ("obs2.txt", "configuration.mkconfig", "fitG.txt",
               dict(errors="undefined", format="%-10s %14.7f %14.7f\\n")),
}


F9 = "%-10s" + " %14.7f" * 9 + "\\n"
F6E = "%-10s" + " %14.7f" * 6 + "\\n"
EVALS = {
    "1": ("observations.obsfile", "config3.cfg", "fit3.txt", dict(catalogs=[CATALOG])),
    "P": ("observations.obsfile", "config3.cfg", "fit3.txt",
          dict(catalogs=[CATALOG], format=F9)),
    "Pe": ("observations.obsfile", "config3.cfg", "fit3.txt",
           dict(catalogs=[CATALOG], errors="equations", format=F9)),
    "Pu": ("observations.obsfile", "config3.cfg", "fit3.txt",
           dict(catalogs=[CATALOG], errors="undefined", format=F6E)),
    "prog": ("observations.obsfile", "config3.cfg", "fit3.txt",
             dict(catalogs=[CATALOG], objects="program", format=F6E)),
    "std": ("observations.obsfile", "config3.cfg", "fit3.txt",
            dict(catalogs=[CATALOG], objects="standards", format=F9)),
    "nocat": ("observations.obsfile", "config3.cfg", "fit3.txt", dict(format=F6E)),
    "print": ("observations.obsfile", "config3.cfg", "fit3.txt",
              dict(catalogs=[CATALOG], objects="standards", errors="undefined",
                   print_vars=["CI", "XV", "V", "nosuch", "VFIT", "v1", "error(mV)"])),
    "G": ("obs2.txt", "config3.cfg", "fit3.txt", dict(catalogs=[CATALOG], format=F9)),
    "Ge": ("obs2.txt", "config3.cfg", "fit3.txt",
           dict(catalogs=[CATALOG], errors="equations", format=F9)),
    "auto": ("observations.obsfile", "configuration.mkconfig", "fit1.txt",
             dict(catalogs=[CATALOG], format=F6E)),
    "autoprog": ("observations.obsfile", "configuration.mkconfig", "fit1.txt",
                 dict(catalogs=[CATALOG], objects="program")),
    "autonocat": ("observations.obsfile", "configuration.mkconfig", "fit1.txt", dict()),
    "E": ("observations.obsfile", "config2.cfg", "fitE.txt",
          dict(catalogs=[CATALOG], errors="equations", format=F6E)),
    "Eo": ("observations.obsfile", "config2.cfg", "fitE.txt",
           dict(catalogs=[CATALOG], print_vars=["BV"], objects="standards",
                format="%-10s" + " %14.7f" * 7 + "\\n")),
    "badfmt": ("observations.obsfile", "config3.cfg", "fit3.txt",
               dict(catalogs=[CATALOG], format="%-10s %7.3f\\n")),
    "missing": ("observations.obsfile", "config3.cfg", "fit3noB.txt",
                dict(catalogs=[CATALOG])),
    "stat": ("observations.obsfile", "config3.cfg", "fit3stat.txt",
             dict(catalogs=[CATALOG])),
    "noobs": ("nosuchobs.txt,observations.obsfile", "config3.cfg", "fit3.txt",
              dict(catalogs=[CATALOG])),
}


def _stamped(line: str) -> bool:
    """A time stamp: # Wed 15:49:56 21-Oct-2026, or #WED ... in the log."""
    return line.startswith("#") and line[1:].strip()[:3].upper() in (
        "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(
            encoding="utf-8").splitlines() if not _stamped(line)]


@pytest.fixture
def here(tmp_path, monkeypatch):
    for item in DATA.iterdir():
        if item.is_file():
            (tmp_path / item.name).write_bytes(item.read_bytes())
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", FITS)
def test_fitparams_writes_what_iraf_writes(here, case):
    obs, cfgfile, kw = FITS[case]
    fit.fitparams(config.parse(cfgfile), [obs], [CATALOG], "out.fit", logfile="out.log",
                  log_fit=True, log_results=True, **kw)
    assert _lines(here / "out.fit") == _lines(DATA / f"fit{case}.txt")
    assert _lines(here / "out.log") == _lines(DATA / f"fit{case}.log")


@pytest.mark.parametrize("case", INVERTS)
def test_invertfit_writes_what_iraf_writes(here, case):
    obs, cfgfile, parameters, kw = INVERTS[case]
    invert.invertfit(config.parse(cfgfile), [obs], parameters, "out.cal", **kw)
    ours = _lines(here / "out.cal")[4:]          # after the file names
    iraf = _lines(DATA / f"{case}.txt")[4:]
    if case != "invPG2":
        assert ours == iraf
        return
    assert len(ours) == len(iraf)
    for a, b in zip(ours, iraf, strict=True):
        if a.startswith("#") or not a.strip():
            assert a == b
            continue
        x, y = a.split(), b.split()
        assert x[0] == y[0]
        def numbers(words):
            return [float("nan") if v == "INDEF" else float(v) for v in words]

        np.testing.assert_allclose(numbers(x[1:]), numbers(y[1:]), atol=2.5e-6, rtol=0)
    assert sum(a != b for a, b in zip(ours, iraf, strict=True)) <= 1


@pytest.mark.parametrize("case", EVALS)
def test_evalfit_writes_what_iraf_writes(here, case):
    obs, cfgfile, parameters, kw = EVALS[case]
    done = evalfit.evalfit(config.parse(cfgfile), obs.split(","), parameters, "out.cal", **kw)
    assert _lines(here / "out.cal") == _lines(DATA / f"ev{case}.txt")
    said = (DATA / f"ev{case}.err").read_text(encoding="utf-8")
    assert done.warnings == said.splitlines()


def test_evalfit_on_the_automatico_equations(here):
    """The automatico's equations have the catalog on the right: a standard gets
    the instrumental magnitude predicted, and a program star has none."""
    cfg = config.parse("configuration.mkconfig")
    done = evalfit.evalfit(cfg, ["observations.obsfile"], "fit1.txt", "out.cal",
                           catalogs=[CATALOG])
    assert done.stars == 24 and done.undefined == 9


def test_evalfit_appends(here):
    cfg = config.parse("config3.cfg")
    evalfit.evalfit(cfg, ["observations.obsfile"], "fit3.txt", "out.cal")
    once = (here / "out.cal").read_text(encoding="utf-8")
    evalfit.evalfit(cfg, ["observations.obsfile"], "fit3.txt", "out.cal", append=True)
    assert (here / "out.cal").read_text(encoding="utf-8") == once + once


def test_evalfit_without_the_parameters_writes_nothing(here):
    """IRAF warns and leaves an empty file behind, which looks like a result."""
    assert (DATA / "evnofile.err").read_text(
            encoding="utf-8").startswith("Warning: Cannot open file")
    assert (DATA / "evnofile.txt").read_text(encoding="utf-8") == ""
    with pytest.raises(evalfit.EvalError, match="Cannot open file"):
        evalfit.evalfit(config.parse("config3.cfg"), ["observations.obsfile"],
                        "nosuchfile.txt", "out.cal", catalogs=[CATALOG])
    assert not (here / "out.cal").exists()


def test_an_infinite_error_is_indef_rather_than_0_00E309(here):
    """The one difference in the numbers: the first star's error takes the
    square root out of its domain, IRAF's error is infinite, and it prints it
    as a number nobody can read back."""
    cfg = config.parse("config5.cfg")
    format = "%-10s %14.7f %14.7f\\n"
    evalfit.evalfit(cfg, ["observations.obsfile"], "fit5.txt", "out.cal", format=format)
    ours, iraf = _lines(here / "out.cal"), _lines(DATA / "evdom.txt")
    assert len(ours) == len(iraf)
    different = [(a, b) for a, b in zip(ours, iraf, strict=True) if a != b]
    assert len(different) == 1
    a, b = different[0]
    assert a.split()[:2] == b.split()[:2] == ["1", "0.0632433"]
    assert a.split()[2] == "INDEF" and b.split()[2] == "0.000000000E309"


def test_no_catalog_section_means_no_id(here):
    """IRAF stops with Out of memory (evnoid.err), so this is t_evalfit.x's
    meaning and not its output: column 1 is a variable, every star has a
    residual, and the equation's value is the right side."""
    assert "Out of memory" in (DATA / "evnoid.err").read_text(encoding="utf-8")
    cfg = config.parse("config4.cfg")
    done = evalfit.evalfit(cfg, ["obs4.txt"], "fit3.txt", "out.cal",
                           format="%14.7f %14.7f %14.7f\\n")
    rows = [line.split() for line in _lines(here / "out.cal")
            if line.strip() and not line.startswith("#")]
    assert done.stars == 5 and done.undefined == 0
    assert "#\t1\tmV" in _lines(here / "out.cal")
    assert len(rows) == 5 and all(len(row) == 3 for row in rows)
    fit_value, _, resid = (float(v) for v in rows[0])
    assert fit_value + resid == pytest.approx(-10.714, abs=2e-6)
    assert rows[3][2] == "INDEF", "mV undefined: a value from mB, but nothing to compare it to"


KEYS = DATA / "apkeys"


def _unwritten(lines, ours):
    """IRAF's log with the columns no star has reached yet in a `mags` row
    taken from ours: after the graphs IRAF prints the memory it was given there,
    which changes with the keys. Only where ours is zero, so that a number that
    is wrong still shows."""
    out, inside = [], False
    for line, mine in zip(lines, ours, strict=True):
        if line.startswith("    mags"):
            inside = True
        elif line.startswith("   cmags"):
            inside = False
        if inside:
            for a in range(10, len(mine), 9):
                if mine[a:a + 9] == "   0.0000":
                    line = line[:a] + mine[a:a + 9] + line[a + 9:]
        out.append(line)
    return out


KEYCASES = sorted(path.stem for path in KEYS.glob("*.cur"))


@pytest.mark.parametrize("case", KEYCASES)
def test_mkapfile_by_the_keys_says_and_writes_what_iraf_does(tmp_path, monkeypatch, case):
    """Each case is a cursor file IRAF 2.18 ran (apkeys/run_iraf.py): what it
    printed key by key, the corrections, the best magnitudes and the log."""
    from harwood.core.photcal import apsession

    monkeypatch.chdir(APFILE)
    images = apfile.read_photfiles(PHOTFILES, 20, 0.10)
    session = apsession.ApSession(images, 20)
    said = session.run_commands((KEYS / f"{case}.cur").read_text(encoding="utf-8"))
    apercors, mags, log = tmp_path / "a", tmp_path / "m", tmp_path / "l"
    _, more = session.finish(str(apercors), str(mags), str(log))
    assert said + more == (KEYS / f"{case}.out").read_text(
            encoding="utf-8").replace("\a", "").splitlines()
    assert apercors.read_text(encoding="utf-8") == (KEYS / f"{case}.apercors").read_text(
            encoding="utf-8")
    assert mags.read_text(encoding="utf-8") == (KEYS / f"{case}.mkapfile").read_text(
            encoding="utf-8")
    written = ("NEW LOGFILE", "APFILE")
    ours = [line for line in _lines(log) if not line.startswith(written)]
    theirs = [line for line in _lines(KEYS / f"{case}.log") if not line.startswith(written)]
    assert ours == _unwritten(theirs, ours)


GRAPHS = Path(__file__).parent / "data" / "iraf_photcal" / "apgraphs"
GRAPHCASES = sorted(path.stem for path in GRAPHS.glob("*.cur"))
#: A metacode unit is 1/32767 of the screen, and IRAF rounds where it falls.
GKI_UNITS = 2


def _on_screen(frame, x, y):
    """Data coordinates as metacode ones, through the frame's own window, on
    GIO's elog scale where the frame's axis is logarithmic."""
    from harwood.core.gtools import elog

    wx1, wx2, wy1, wy2, sx1, sx2, sy1, sy2 = frame["window"]
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xlog, ylog = frame.get("log", (False, False))
    if xlog:
        x, wx1, wx2 = np.array([elog(v) for v in x]), elog(wx1), elog(wx2)
    if ylog:
        y, wy1, wy2 = np.array([elog(v) for v in y]), elog(wy1), elog(wy2)
    return np.c_[(sx1 + (x - wx1) / (wx2 - wx1) * (sx2 - sx1)) * 32767,
                 (sy1 + (y - wy1) / (wy2 - wy1) * (sy2 - sy1)) * 32767]


def _unmatched(points, others):
    """Those of `points` with nothing of `others` where they are."""
    points, others = np.reshape(points, (-1, 2)), np.reshape(others, (-1, 2))
    if not len(others):
        return points
    return np.array([p for p in points if np.abs(others - p).max(axis=1).min() > GKI_UNITS]
                    ).reshape(-1, 2)


def _graphs_of(case):
    """Every graph the session draws for a case's cursor file, in order."""
    from harwood.core.photcal import apsession

    class Recording(apsession.ApSession):
        # A key that draws says the status after it; w says it and draws nothing.
        drawn = None
        by_w = False

        def key(self, key, *args, **kwargs):
            self.by_w = key == "w"
            return super().key(key, *args, **kwargs)

        def status(self):
            if self.drawn is not None and not self.by_w:
                self.drawn.append(self.graph())
            return super().status()

    options = yaml.safe_load((GRAPHS / "CASES.yaml").read_text(encoding="utf-8"))[case]
    images = apfile.read_photfiles(PHOTFILES, 20, options.get("maglim", 0.10))
    session = Recording(images, 20, smallap=options.get("smallap", 1),
                        largeap=options.get("largeap", 0))
    session.drawn = [session.graph()]
    session.run_commands((GRAPHS / f"{case}.cur").read_text(encoding="utf-8"))
    return session.drawn


@pytest.mark.parametrize("case", GRAPHCASES)
def test_mkapfile_draws_what_iraf_draws(monkeypatch, request, case):
    """IRAF's own metacode (apgraphs/run_iraf.py): each frame's window, title,
    labels, marks and lines, to the resolution of the metacode."""
    if case == "nofit" and sys.platform == "win32":
        # Its title's Ro comes out 2.252 on Windows, and 2.256 as IRAF's: a fit in single
        # precision on Windows' libm, which is not glibc's. Not understood yet: it still runs,
        # rather than being loosened, and strictly, so that it says when it is fixed.
        request.applymarker(pytest.mark.xfail(
            strict=True, reason="Ro is 2.252 against IRAF's 2.256 on Windows' libm"))
    monkeypatch.chdir(APFILE)
    frames = json.loads((GRAPHS / f"{case}.json").read_text(encoding="utf-8"))
    graphs = _graphs_of(case)
    assert len(graphs) == len(frames)
    for number, (graph, frame) in enumerate(zip(graphs, frames, strict=True)):
        where = f"frame {number}"
        width = abs(frame["window"][1] - frame["window"][0])
        height = abs(frame["window"][3] - frame["window"][2])
        assert graph.window[:2] == pytest.approx(frame["window"][:2], abs=1e-4 * width), where
        assert graph.window[2:] == pytest.approx(frame["window"][2:4],
                                                 abs=1e-4 * height), where
        assert {graph.xlabel, graph.ylabel} <= set(frame["text"]), where
        assert graph.title == [text for text in frame["text"]
                               if text.startswith(("Image:", "Dashed"))], where
        for theirs, ours in ((frame["plus"], graph.kept), (frame["cross"], graph.dropped)):
            mine = _on_screen(frame, *ours)
            assert len(mine) == len(theirs), where
            assert not len(_unmatched(mine, theirs)), where
            assert not len(_unmatched(theirs, mine)), where
        left, right, bottom, top = (np.array(frame["window"][4:]) * 32767).tolist()
        for style in (1, 2):
            mine = [_on_screen(frame, x, y) for x, y, dashed in graph.lines
                    if dashed == (style == 2)]
            mine = np.vstack(mine) if mine else np.zeros((0, 2))
            mine = mine[np.isfinite(mine).all(axis=1)]
            inside = ((mine[:, 0] >= left - 1) & (mine[:, 0] <= right + 1)
                      & (mine[:, 1] >= bottom - 1) & (mine[:, 1] <= top + 1))
            theirs = [p for s, points in frame["lines"] if s == style for p in points]
            assert not len(_unmatched(mine[inside], theirs)), where
            # All but where IRAF cut a line at the edge of the plot.
            extra = _unmatched(theirs, mine[inside])
            assert all(min(abs(x - left), abs(x - right), abs(y - bottom), abs(y - top))
                       <= GKI_UNITS for x, y in extra), where


def _run_mkapfile(tmp_path, **params):
    from harwood.tasks.photcal import Mkapfile, MkapfileParams

    return Mkapfile().execute(MkapfileParams(
        photfiles=PHOTFILES, naperts=20, apercors=tmp_path / "a", magfile=str(tmp_path / "m"),
        logfile=str(tmp_path / "l"), **params))


def test_mkapfile_gcommands_is_the_session_iraf_ran(tmp_path, monkeypatch):
    monkeypatch.chdir(APFILE)
    result = _run_mkapfile(tmp_path, gcommands=str(KEYS / "del.cur"))
    assert result.data["text"] == (KEYS / "del.out").read_text(encoding="utf-8").splitlines()
    assert (tmp_path / "a").read_text(encoding="utf-8") == (KEYS / "del.apercors").read_text(
            encoding="utf-8")
    assert (tmp_path / "m").read_text(encoding="utf-8") == (KEYS / "del.mkapfile").read_text(
            encoding="utf-8")
    assert not result.warnings


def test_mkapfile_says_when_q_left_the_fit_out_of_date(tmp_path, monkeypatch):
    monkeypatch.chdir(APFILE)
    result = _run_mkapfile(tmp_path, gcommands=str(KEYS / "stale.cur"))
    assert (tmp_path / "a").read_text(encoding="utf-8") == (KEYS / "stale.apercors").read_text(
            encoding="utf-8")
    assert any("press f before q" in warning.fix for warning in result.warnings)


def test_mkapfile_keys_need_a_screen_or_a_file(tmp_path, monkeypatch):
    from harwood.core import display

    monkeypatch.chdir(APFILE)
    monkeypatch.setattr(display, "_autodetect", lambda: "none")
    with pytest.raises(ValueError) as caught:
        _run_mkapfile(tmp_path, interactive=True)
    assert "--gcommands" in caught.value.fix
    with pytest.raises(FileNotFoundError):
        _run_mkapfile(tmp_path, gcommands="nosuch.cur")
    assert not (tmp_path / "a").exists()


OBSKEYS = DATA / "obskeys"
SKINAKAS_SETS = str((DATA / "obsfile") / "imsets.mkimsets")
OBSKEYCASES = {
    "mA": {}, "mB": dict(aperture=5), "mC": dict(allfilters=True),
    "mD": dict(apercors=str((DATA / "obsfile") / "apercors.all")),
    "mE": dict(shifts=str(OBSKEYS / "shifts.txt")), "mF": dict(minmagerr=0.02),
    "mG": dict(tolerance=0.0), "mH": dict(wrap=False),
    "mI": dict(obsparams=str(OBSKEYS / "obspars.txt")), "mJ": {},
    "nA": {}, "nB": dict(aperture=11, allfilters=True),
    "nI": dict(obsparams=str(OBSKEYS / "obspars.txt"),
               apercors=str((DATA / "obsfile") / "apercors.all"))}


@pytest.mark.parametrize("case", OBSKEYCASES)
def test_mkobsfile_writes_and_says_what_iraf_does(tmp_path, monkeypatch, case):
    """mA to mJ ran mkobsfile, nA to nI mknobsfile (obskeys/run_iraf.py): the same
    script but for mkphotcors, whose warning on mI's obsparams -- it reads column 2
    as the image -- is said too."""
    from harwood.tasks.photcal import Mknobsfile, Mkobsfile

    monkeypatch.chdir(APFILE)
    task = Mknobsfile() if case.startswith("n") else Mkobsfile()
    out = tmp_path / case
    settings = dict(tolerance=8.0) | OBSKEYCASES[case]
    result = task.execute(task.Params(
        photfiles=PHOTFILES, idfilters="B,V", imsets=SKINAKAS_SETS, observations=out,
        **settings))
    assert out.read_text(encoding="utf-8") == (OBSKEYS / case).read_text(encoding="utf-8")
    assert obs.format_path(out).read_text(
            encoding="utf-8") == (OBSKEYS / f"f{case}.dat").read_text(
            encoding="utf-8")
    if case != "mJ":        # verbose=no, which harwood's task has no use for
        written = [(set_["name"], set_["stars"]) for set_ in result.data["sets"]]
        assert obs.summary(case, written) == (OBSKEYS / f"{case}.out").read_text(
                encoding="utf-8")
    said = (OBSKEYS / f"{case}.err").read_text(encoding="utf-8").splitlines()
    assert [w.message.replace(f"{OBSKEYS}{os.sep}", "") for w in result.warnings] == said


@pytest.mark.parametrize("case, settings", [
    ("mS1", dict(imsets="STDIN")),
    ("mS2", dict(apercors="STDIN", verify=True)),
    ("mS3", dict(imsets="STDIN", shifts="STDIN"))])
def test_mkobsfile_asks_what_iraf_asks(tmp_path, monkeypatch, case, settings):
    """The image sets and the corrections typed at the terminal (NAME.ans), through
    mkphotcors into temporary files, as mkobsfile.cl does; after the sets the input
    has ended, so mS3's shifts get nothing. IRAF names its pdump file, tmp$..., in
    the warnings; harwood its own."""
    monkeypatch.chdir(APFILE)
    out = tmp_path / case
    settings = dict(imsets=SKINAKAS_SETS) | settings
    typed = iter((OBSKEYS / f"{case}.ans").read_text(encoding="utf-8").splitlines())
    result = mkobs.mkobsfile(PHOTFILES, "B,V", observations=str(out), tolerance=8.0,
                             terminal=entry.Terminal(typed), **settings)
    assert out.read_text(encoding="utf-8") == (OBSKEYS / case).read_text(encoding="utf-8")
    assert obs.format_path(out).read_text(encoding="utf-8") == (
        OBSKEYS / f"f{case}.dat").read_text(encoding="utf-8")
    assert result.typed + obs.summary(case, result.written) == (
        OBSKEYS / f"{case}.out").read_text(encoding="utf-8")

    def tmp(text):
        return re.sub(r"File: \S+ Image", "File: (pdump) Image", text)

    assert tmp("".join(n + "\n" for n in result.notes)) == tmp(
        (OBSKEYS / f"{case}.err").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case, files", [
    ("mP1", dict(imsets="sets", obsparams="obs", shifts="sh", apercors="ap")),
    ("mP2", dict(imsets=SKINAKAS_SETS, obsparams="obs", verify=True))])
def test_mkphotcors_asks_and_writes_what_iraf_does(tmp_path, monkeypatch, case, files):
    """mkphotcors alone: the files that are not there asked for and written."""
    from harwood.tasks.photcal import Mkphotcors

    monkeypatch.chdir(tmp_path)
    names = {key: value if value == SKINAKAS_SETS else f"{case}.{value}"
             for key, value in files.items() if key != "verify"}
    result = Mkphotcors().execute(Mkphotcors.Params(
        idfilters="B,V", answers=str(OBSKEYS / f"{case}.ans"),
        verify=files.get("verify", False), **names))
    assert result.data["typed"] == (OBSKEYS / f"{case}.out").read_text(encoding="utf-8")
    assert [w.message for w in result.warnings] == (OBSKEYS / f"{case}.err").read_text(
        encoding="utf-8").splitlines()
    for path in names.values():
        if path != SKINAKAS_SETS:
            assert (tmp_path / path).read_text(encoding="utf-8") == (
                OBSKEYS / path).read_text(encoding="utf-8"), path


def test_a_file_of_one_aperture_is_read_as_pdump_reads_it():
    """qphot's file of one aperture names its fields MAG, MERR, RAPERT; IRAF's pdump
    reads them as mag[1], which is what mknobsfile.cl asks for. The file and pdump's
    lines are IRAF's own (oneaperture/), qphot on the M92 frame of the course."""
    from harwood.core.photcal import mkobsfile

    folder = DATA / "oneaperture"
    mine = mkobsfile.dump([str(folder / "m92010s.mag.1")]).split("\n")
    theirs = (folder / "pdump.txt").read_text(encoding="utf-8").split("\n")

    def values(line):
        out = []
        for word in line.split():
            try:
                out.append(float(word))
            except ValueError:
                out.append(word)
        return out

    # pdump echoes the file's text (19.000); obsfile reads the numbers either way.
    assert [values(line) for line in mine if line] == \
        [values(line) for line in theirs if line]


def test_mkobsfile_refuses_what_iraf_refuses(tmp_path, monkeypatch):
    from harwood.tasks.photcal import Mkobsfile

    monkeypatch.chdir(APFILE)
    task = Mkobsfile()

    def run(**settings):
        settings = dict(photfiles=PHOTFILES, idfilters="B,V", imsets=SKINAKAS_SETS,
                        observations=tmp_path / "out") | settings
        return task.execute(task.Params(**settings))

    with pytest.raises(ValueError, match="does not exist"):
        run(imsets="nosuch")
    with pytest.raises(ValueError, match="does not exist"):
        run(apercors="nosuch")
    with pytest.raises(ValueError, match="no aperture 99"):
        run(aperture=99)
    assert not (tmp_path / "out").exists()
    run()
    with pytest.raises(ValueError, match="already exists"):
        run()


FITKEYS = DATA / "fitkeys"
FITCASES = sorted(path.stem for path in FITKEYS.glob("*.cur"))


def _said(text: str) -> list[str]:
    """What IRAF printed less the terminal's bell."""
    text = text.replace("\a", "")
    return [line for line in text.splitlines() if not _stamped(line)]


def _named_by_memory(lines: list[str]) -> list[str]:
    """`:const 3`: IRAF names the parameter with whatever was in memory."""
    return [re.sub(r"^\(.*\) changed to", "changed to", line) for line in lines]


def _written(path: Path) -> list[str]:
    """A parameters file or log without its time stamps."""
    if not path.exists():
        return []
    return [line for line in _lines(path) if not line.startswith("# ")]


def _fit_session(settings, output, session=fitsession.FitSession, **more):
    settings = dict(settings)
    cfgfile = settings.pop("config", "configuration.mkconfig")
    if cfgfile == "cfgplot.cfg":
        cfgfile = str(FITKEYS / cfgfile)
    observations = settings.pop("observations", None)
    observations = str(FITKEYS / observations) if observations else "observations.obsfile"
    return session(config.parse(cfgfile), [observations], [CATALOG], output, **settings,
                   **more)


@pytest.mark.parametrize("case", FITCASES)
def test_fitparams_by_the_keys_says_and_writes_what_iraf_does(here, case):
    """Each case is a cursor file and the answers to the prompts, as IRAF 2.18
    ran them (fitkeys/run_iraf.py): what it printed on both streams, the
    parameters file, the log, and what a colon command wrote to a file."""
    settings = yaml.safe_load((FITKEYS / "CASES.yaml").read_text(encoding="utf-8"))[case]
    session = _fit_session(settings, "out.fit", logfile="out.log", log_fit=True,
                           log_results=True)
    session.run_commands((FITKEYS / f"{case}.cur").read_text(encoding="utf-8"),
                         (FITKEYS / f"{case}.ans").read_text(encoding="utf-8"))
    assert _named_by_memory(_said(session.stdout)) == _named_by_memory(
        _said((FITKEYS / f"{case}.out").read_text(encoding="utf-8")))
    assert session.stderr == (FITKEYS / f"{case}.err").read_text(encoding="utf-8")
    assert _written(here / "out.fit") == _written(FITKEYS / f"{case}.fit")
    assert _written(here / "out.log") == _written(FITKEYS / f"{case}.log")
    if (FITKEYS / f"{case}.keep").exists():
        assert _lines(here / "keep.txt") == _lines(FITKEYS / f"{case}.keep")


@pytest.mark.parametrize("case", FITCASES)
def test_fitparams_draws_what_iraf_draws(here, case):
    """IRAF's own metacode (fitkeys/run_iraf.py): each frame's window, title and
    labels, its pluses, crosses, diamonds and boxes. A frame drawn over with o, in
    IRAF, is the frame before it with more on it."""
    class Recording(fitsession.FitSession):
        drawn: list = []

        def graph(self):
            graph = super().graph()
            self.drawn.append(graph)
            return graph

    Recording.drawn = []
    settings = yaml.safe_load((FITKEYS / "CASES.yaml").read_text(encoding="utf-8"))[case]
    session = _fit_session(settings, "out.fit", session=Recording)
    session.run_commands((FITKEYS / f"{case}.cur").read_text(encoding="utf-8"),
                         (FITKEYS / f"{case}.ans").read_text(encoding="utf-8"))
    drawn = []
    for graph in Recording.drawn:
        if graph.overplot and drawn:
            drawn[-1].append(graph)
        else:
            drawn.append([graph])
    frames = json.loads((FITKEYS / f"{case}.json").read_text(encoding="utf-8"))
    assert len(drawn) == len(frames)
    for number, (graphs, frame) in enumerate(zip(drawn, frames, strict=True)):
        where = f"frame {number}"
        graph = graphs[0]
        width = abs(frame["window"][1] - frame["window"][0])
        height = abs(frame["window"][3] - frame["window"][2])
        assert graph.window[:2] == pytest.approx(frame["window"][:2], abs=1e-4 * width), where
        assert graph.window[2:] == pytest.approx(frame["window"][2:4],
                                                 abs=1e-4 * height), where
        said = {graph.xlabel, graph.ylabel, *graph.title, *graph.params} - {""}
        assert said <= set(frame["text"]), where
        assert bool(graph.xlabel) == (graph.xlabel in frame["text"]), where
        for kind in ("plus", "cross", "diamond", "box"):
            mine = np.vstack([np.zeros((0, 2))] + [_on_screen(frame, *getattr(g, part))
                                                    for g in graphs
                              for part, drawn_as in (("kept", g.mark), ("deleted", "cross"),
                                                     ("rejected", "diamond"), ("fit", "box"))
                              if drawn_as == kind]).reshape(-1, 2)
            mine, theirs = _whole(frame, mine, graph), _whole(frame, frame[kind], graph)
            assert len(mine) == len(theirs), (where, kind)
            assert not len(_unmatched(mine, theirs)), (where, kind)
            assert not len(_unmatched(theirs, mine)), (where, kind)


def _whole(frame, points, graph):
    """The marks GIO draws whole: those a mark's width inside the viewport, where
    a window that leaves points out clips them (size 2 is 371 metacode units, and
    the diamonds and boxes are 3)."""
    points = np.reshape(np.asarray(points, dtype=float), (-1, 2))
    sx1, sx2, sy1, sy2 = np.array(frame["window"][4:8]) * 32767
    margin = 371 / 4 * max(3.0, *graph.size) + GKI_UNITS
    inside = ((points[:, 0] > sx1 + margin) & (points[:, 0] < sx2 - margin)
              & (points[:, 1] > sy1 + margin) & (points[:, 1] < sy2 - margin))
    return points[inside]


def _run_fitparams(tmp_path, case=None, **params):
    from harwood.tasks.photcal import Fitparams, FitparamsParams

    if case:
        params |= dict(gcommands=str(FITKEYS / f"{case}.cur"),
                       answers=str(FITKEYS / f"{case}.ans"))
    return Fitparams().execute(FitparamsParams(
        observations=["observations.obsfile"], catalogs=[CATALOG],
        config="configuration.mkconfig", parameters=tmp_path / "out.fit", **params))


def test_fitparams_gcommands_is_the_session_iraf_ran(here):
    result = _run_fitparams(here, "del")
    assert _written(here / "out.fit") == _written(FITKEYS / "del.fit")
    assert result.data["VFIT"]["points"] == 14
    assert "Do you want to save fit results ?" in result.data["text"][-1]


def test_fitparams_says_which_equations_were_not_saved(here):
    result = _run_fitparams(here, "nosave")
    assert not (here / "out.fit").exists()
    assert any("VFIT, BFIT" in str(warning) for warning in result.warnings)
    assert "keys" in [step.action for step in result.steps]


def test_fitparams_keys_need_a_screen_or_a_file(here, monkeypatch):
    from harwood.core import display

    monkeypatch.setattr(display, "_autodetect", lambda: "none")
    with pytest.raises(ValueError) as caught:
        _run_fitparams(here, interactive=True)
    assert "--gcommands" in caught.value.fix
    with pytest.raises(FileNotFoundError):
        _run_fitparams(here, gcommands="nosuch.cur")
    with pytest.raises(FileNotFoundError):
        _run_fitparams(here, gcommands=str(FITKEYS / "del.cur"), answers="nosuch.ans")
    assert not (here / "out.fit").exists()


def test_a_square_is_what_iraf_computes_not_what_powf_does():
    """Found by the fit above: numpy's ** on a float32 scalar left its variance
    off in the last place."""
    from harwood.core.nlfit import f32, square

    values = np.random.default_rng(1).uniform(-1, 1, 5000).astype(f32)
    assert all(square(v) == v * v for v in values)


def _cli(*args):
    from click.testing import CliRunner

    from harwood.cli.main import app

    return CliRunner().invoke(app, list(args))


def test_the_evalfit_task_runs_and_counts(here):
    result = _cli("evalfit", "observations.obsfile", "config3.cfg", "fit3.txt", "out.cal",
                  "--catalogs", CATALOG, "--format", F9)
    assert result.exit_code == 0, result.output
    assert _lines(here / "out.cal") == _lines(DATA / "evP.txt")
    assert "stars: 24" in result.output and "came to" not in result.output


def test_the_evalfit_task_says_when_the_equations_want_invertfit(here):
    result = _cli("evalfit", "observations.obsfile", "configuration.mkconfig", "fit1.txt",
                  "out.cal", "--catalogs", CATALOG)
    said = " ".join(result.output.split())
    assert result.exit_code == 0
    assert "9 of 24 stars" in said and "VFIT, BFIT" in said and "invertfit" in said


def test_the_evalfit_task_stops_without_its_parameters(here):
    result = _cli("evalfit", "observations.obsfile", "config3.cfg", "nosuch.txt", "out.cal")
    assert result.exit_code == 1 and "Cannot open file" in result.output
    assert "fitparams" in result.output and not (here / "out.cal").exists()


def test_the_automatico_night_is_reproduced(here):
    """The files the automatico pipeline left: fitfile.fitparams and
    observations.magcalibrada are fit1.txt and inv1.txt."""
    cfg = config.parse("configuration.mkconfig")
    results = fit.fitparams(cfg, ["observations.obsfile"], [CATALOG], "out.fit")
    vfit = results[0]
    assert vfit.eq.name == "VFIT"
    from harwood.core import iraffmt

    written = [iraffmt.g(float(v)) for v in vfit.params]
    assert written == ["-21.90591", "0.0445506", "0.2056334"]
    assert vfit.niter == 4 and len(vfit.y) == 15


# ---------------------------------------------------------------- mkapfile

APFILE = DATA / "apfile"
PHOTFILES = (APFILE / "all.lis").read_text(encoding="utf-8").split()
APCASES = {name.removesuffix(".fits.mag.1"): ([name], dict(smallap=1, largeap=11))
           for name in PHOTFILES}
APCASES |= {
    "M": (PHOTFILES, {}),
    "P5": (PHOTFILES, dict(smallap=2, largeap=15, maglim=0.05, nparams=5)),
    "P4": (PHOTFILES, dict(smallap=3, nparams=4, params=(1.5, 0.1, 0.5, 0.9, 0.1))),
}


@pytest.mark.parametrize("case", APCASES)
def test_mkapfile_writes_what_iraf_writes(tmp_path, monkeypatch, case):
    """Each Skinakas image alone, as the automatico pipeline runs it (in a
    fresh IRAF process each: the fit keeps a static between runs), and all
    twelve together three ways (run_iraf.py made them)."""
    files, kw = APCASES[case]
    kw = dict(kw)
    maglim = kw.pop("maglim", 0.10)
    monkeypatch.chdir(APFILE)
    images = apfile.read_photfiles(files, 20, maglim)
    log = f"{case}.mkapfile.log" if len(files) == 1 else f"{case}.log"
    out = {name: tmp_path / name for name in (f"{case}.apercors", f"{case}.mkapfile", log)}
    apfile.apfile(images, 20, str(out[f"{case}.apercors"]),
                  magfile=str(out[f"{case}.mkapfile"]), logfile=str(out[log]), **kw)
    for name, path in out.items():
        ours = [line.replace(f"{tmp_path}{os.sep}", "") for line in path.read_text(
                encoding="utf-8").splitlines()]
        iraf = (APFILE / name).read_text(encoding="utf-8").splitlines()
        assert ours[1:] == iraf[1:], name                 # after NEW LOGFILE ENTRY AT
        assert ours[0] == iraf[0] or ours[0].startswith("NEW LOGFILE ENTRY AT")


@pytest.mark.parametrize("case, obsparams, obscolumns", [
    ("O", "obs1.dat", "2 3 4 5"), ("O2", "obs2.dat", "0 0 2 0")])
def test_mkapfile_takes_the_observing_parameters_iraf_takes(tmp_path, monkeypatch, case,
                                                           obsparams, obscolumns):
    """obsparams in place of the photometry's filter, exposure, airmass and time: a
    file of all four, with an image that is not there and an airmass left INDEF, and
    one of the airmass alone. What is said and every file written must be IRAF's --
    but for one thing. With no column for the time, IRAF's ph_aobsdata leaves the
    time it reads into uninitialised, and writes what was there (0:00:00.0 here) as
    the image's time; its help says a 0 column is a place holder, "to correct only
    the airmass", and harwood keeps the photometry's time."""
    from harwood.tasks.photcal import Mkapfile, MkapfileParams

    monkeypatch.chdir(APFILE)
    out = {name: tmp_path / name for name in (f"{case}.apercors", f"{case}.mkapfile",
                                              f"{case}.log")}
    result = Mkapfile().execute(MkapfileParams(
        photfiles=PHOTFILES, naperts=20, apercors=out[f"{case}.apercors"],
        magfile=str(out[f"{case}.mkapfile"]), logfile=str(out[f"{case}.log"]),
        obsparams=obsparams, obscolumns=obscolumns))
    assert result.data.get("text", []) == (APFILE / f"{case}.out").read_text(
        encoding="utf-8").splitlines()
    given = {line.split()[0] for line in (APFILE / obsparams).read_text(
        encoding="utf-8").splitlines()}
    for name, path in out.items():
        ours = [line.replace(f"{tmp_path}{os.sep}", "") for line in path.read_text(
                encoding="utf-8").splitlines()]
        iraf = (APFILE / name).read_text(encoding="utf-8").splitlines()
        if "4" not in obscolumns.split() and name.endswith(".mkapfile"):
            # The time, column 5 of the magnitudes: IRAF's uninitialised zero
            # against the photometry's time.
            for k, line in enumerate(iraf):
                words = line.split()
                if words and words[0] in given:
                    assert words[4] == "0:00:00.0" and ours[k].split()[4] != "0:00:00.0"
                    iraf[k] = line.replace(" 0:00:00.0", ours[k].split()[4].rjust(10))
        assert ours[1:] == iraf[1:], name
        assert ours[0] == iraf[0] or ours[0].startswith("NEW LOGFILE ENTRY AT")


def test_mkapfile_asks_for_the_observing_parameters_as_iraf_does(tmp_path, monkeypatch):
    """obsparams STDIN: typed at the terminal (OS.ans), with verify. IRAF's verify
    reads three fields -- the time is pargr'd, not gargr'd -- so a time typed there
    is not taken, here as there."""
    from harwood.tasks.photcal import Mkapfile, MkapfileParams

    monkeypatch.chdir(APFILE)
    out = {name: tmp_path / name for name in ("OS.apercors", "OS.mkapfile", "OS.log")}
    result = Mkapfile().execute(MkapfileParams(
        photfiles=PHOTFILES, naperts=20, apercors=out["OS.apercors"],
        magfile=str(out["OS.mkapfile"]), logfile=str(out["OS.log"]), obsparams="STDIN",
        verify=True, answers="OS.ans"))
    assert result.data["text"] == (APFILE / "OS.out").read_text(encoding="utf-8").splitlines()
    for name, path in out.items():
        ours = [line.replace(f"{tmp_path}{os.sep}", "") for line in path.read_text(
                encoding="utf-8").splitlines()]
        iraf = (APFILE / name).read_text(encoding="utf-8").splitlines()
        assert ours[1:] == iraf[1:], name


def test_mkapfile_as_a_task(tmp_path, monkeypatch):
    from harwood import run

    monkeypatch.chdir(APFILE)
    result = run("mkapfile", workdir=tmp_path, photfiles=PHOTFILES[:2], naperts=20,
                 apercors=tmp_path / "ap", largeap=11)
    assert result.data["converged"]
    assert set(result.data["images"]) == {"data0097cr.fits", "data0098cr.fits"}
    assert (tmp_path / "ap").read_text(encoding="utf-8").startswith(
        "# The aperture correction from apertures 1 to 11 in magnitudes")


# ------------------------------------------------ mkcatalog at the terminal

TYPEDCAT = DATA / "mkcatalog"
TYPEDCASES = {"C1": {}, "C2": dict(verify=True), "C3": {}, "C4": dict(review=True),
            "C5": dict(verify=True), "C6": dict(edit=True), "NT": {}, "NM": {}}
TYPEDBASE = {"C3": "catC1", "C4": "catC3", "C5": "catC1", "C6": "catC1"}


@pytest.mark.parametrize("case", TYPEDCASES)
def test_mkcatalog_asks_and_writes_what_iraf_does(tmp_path, monkeypatch, case):
    """mkcatalog typed at the terminal (mkcatalog/run_iraf_typed.py): a new catalog's
    columns, with verify; records added to an old one, with verify, a record cut short
    by the end of the input not written; the old records reviewed; the catalog edited
    at the end (the stand-in vi); a title that is not the name, and a file mkcatalog did
    not make. What is asked and said, and the catalog and its format file, are IRAF's."""
    import json
    import subprocess
    import sys

    from harwood.core import editor
    from harwood.tasks.photcal import Mkcatalog

    monkeypatch.chdir(tmp_path)
    name = f"cat{case}"
    if case in TYPEDBASE:
        base = TYPEDBASE[case]
        Path(name).write_text((TYPEDCAT / base).read_text(encoding="utf-8").replace(
            f"CATALOG: {base}", f"CATALOG: {name}"), encoding="utf-8")
    if case == "NT":
        Path(name).write_text((TYPEDCAT / "catC1").read_text(encoding="utf-8").replace(
            "CATALOG: catC1", "CATALOG: other"), encoding="utf-8")
    if case == "NM":
        Path(name).write_text("PG2331 12.5 0.02\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([[["replace", "S2  ", "S2x "]]]), encoding="utf-8")

    def stand_in(path):
        subprocess.run([sys.executable, str(MKIMSETS / "bin" / "vi"), str(path)],
                       env=dict(os.environ, FAKE_EDIT_PLAN=str(plan)), check=True)
        return True

    monkeypatch.setattr(editor, "edit", stand_in)
    answers = TYPEDCAT / f"{case}.ans"
    empty = tmp_path / "empty"
    empty.write_text("", encoding="utf-8")
    params = Mkcatalog.Params(catalog=name, answers=str(answers if answers.exists()
                                                         else empty), **TYPEDCASES[case])
    if case in ("NT", "NM"):
        with pytest.raises(ValueError) as caught:
            Mkcatalog().execute(params)
        said = " ".join((TYPEDCAT / f"{case}.err").read_text(encoding="utf-8").split())
        assert str(caught.value).split(" -- ")[0] in said
        return
    result = Mkcatalog().execute(params)
    assert result.data["typed"] == (TYPEDCAT / f"{case}.out").read_text(encoding="utf-8")
    assert (TYPEDCAT / f"{case}.err").read_text(encoding="utf-8") == ""
    assert Path(name).read_text(encoding="utf-8") == (TYPEDCAT / name).read_text(
        encoding="utf-8")
    if (TYPEDCAT / f"f{name}.dat").exists():
        assert Path(f"f{name}.dat").read_text(encoding="utf-8") == (
            TYPEDCAT / f"f{name}.dat").read_text(encoding="utf-8")


# ----------------------------------------------------------------- obsfile

OBSFILE = DATA / "obsfile"


def _obs_cases():
    import importlib.util

    spec = importlib.util.spec_from_file_location("obsfile_cases", OBSFILE / "cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES


OBSCASES = _obs_cases()


@pytest.mark.parametrize("case", OBSCASES)
def test_obsfile_writes_and_says_what_iraf_does(tmp_path, monkeypatch, case):
    """A is the automatico's own run; the others take the other paths --
    no ids, matching by order, any filter, the corrections, the shifts,
    obsparams, and every kind of bad line in a sets file; S1 to S5 type the
    sets, the shifts, the corrections and the observing parameters at the
    terminal, with and without their verify prompts, as IRAF asks for them."""
    sets, cols, kw = OBSCASES[case]
    kw = dict(kw)
    monkeypatch.chdir(OBSFILE)
    files = (OBSFILE / "mkapfile.lista").read_text(encoding="utf-8").split()
    out = tmp_path / f"obs{case}"
    if "answers" in kw:
        from harwood.core.photcal import entry

        typed = (OBSFILE / kw.pop("answers")).read_text(encoding="utf-8").splitlines()
        kw["terminal"] = entry.Terminal(iter(typed))
    result = obs.obsfile(files, cols, "B,V", sets, str(out), **kw)
    assert out.read_text(encoding="utf-8") == (OBSFILE / f"obs{case}").read_text(
            encoding="utf-8")
    assert obs.format_path(out).read_text(
            encoding="utf-8") == (OBSFILE / f"fobs{case}.dat").read_text(
            encoding="utf-8")
    assert result.typed + obs.summary(f"obs{case}", result.written) == (
        OBSFILE / f"obs{case}.out").read_text(encoding="utf-8")
    assert "".join(n + "\n" for n in result.notes) == (OBSFILE / f"obs{case}.err").read_text(
            encoding="utf-8")


def test_the_automatico_s_observations_are_what_fitparams_was_given():
    """Case A is observations.obsfile, the file the fit above starts from."""
    assert (OBSFILE / "obsA").read_text(
            encoding="utf-8") == (DATA / "observations.obsfile").read_text(
            encoding="utf-8")


# ----------------------------------------------------- harwood's own phot

NATIVE = DATA / "native"


def _native_chain(tmp_path, monkeypatch, rounded: bool):
    """harwood's phot output for the twelve images (as harwood phot wrote
    it, with the automatico's settings) through mkapfile, obsfile,
    fitparams and invertfit. `rounded` rounds positions, magnitudes and
    errors to the three decimals IRAF's .mag file keeps."""
    from harwood.core import iraffmt
    from harwood.core.photcal import photfiles

    if rounded:
        original = photfiles.read

        def read(path):
            table = original(path)
            for name in table.colnames:
                if name.startswith(("MAG", "MERR", "XCENTER", "YCENTER")):
                    table[name] = [float(iraffmt.printf(float(v), "f", 3))
                                   if np.isfinite(v) else np.nan for v in table[name]]
            return table

        monkeypatch.setattr(photfiles, "read", read)
    files = []
    for name in (NATIVE / "mkapfile.lista").read_text(encoding="utf-8").split():
        root = name.removesuffix(".mkapfile")
        out = tmp_path / name
        apfile.apfile(apfile.read_photfiles([str(NATIVE / f"{root}.fits.mag")], 20), 20,
                      str(tmp_path / f"{root}.apercors"), 1, 11, magfile=str(out),
                      names=True)
        files.append(str(out))
    obs.obsfile(files, "1,6,7,2,3,4,5,8,9,11", "B,V", str(OBSFILE / "imsets.mkimsets"),
                str(tmp_path / "obs"), tolerance=8.0)
    cfg = config.parse(str(DATA / "configuration.mkconfig"))
    fit.fitparams(cfg, [str(tmp_path / "obs")], [str(DATA / CATALOG)], str(tmp_path / "fit"))
    invert.invertfit(cfg, [str(tmp_path / "obs")], str(tmp_path / "fit"),
                     str(tmp_path / "cal"), catalogs=[str(DATA / CATALOG)])
    return tmp_path


def _body(path):
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")]


def test_harwood_s_phot_rounded_as_iraf_writes_it_gives_the_automatico_s_night(
        tmp_path, monkeypatch):
    """The whole chain in harwood, from its own phot: with the numbers
    rounded where IRAF's file rounds them, every file the automatico wrote
    comes out the same -- the observations, the coefficients and the
    calibrated magnitudes."""
    out = _native_chain(tmp_path, monkeypatch, rounded=True)
    assert (out / "obs").read_text(
            encoding="utf-8") == (DATA / "observations.obsfile").read_text(
            encoding="utf-8")
    assert _body(out / "fit") == _body(NATIVE / "fitfile.fitparams")
    assert _body(out / "cal") == _body(NATIVE / "observations.magcalibrada")


def test_harwood_s_phot_at_full_precision_calibrates_the_night_too(tmp_path, monkeypatch):
    """At full precision the chain differs from the automatico's by what
    IRAF's rounding did. Its errors of 0.001-0.002 mag in the small
    apertures, rounded to 0.001, change a star's weight up to twofold, and
    with three stars an image's aperture correction moves by up to 0.12 mag
    (stan0058, against its own error of 0.16). No offset overall."""
    out = _native_chain(tmp_path, monkeypatch, rounded=False)
    ours = [line.split() for line in _body(out / "cal") if line.strip()]
    iraf = [line.split() for line in _body(NATIVE / "observations.magcalibrada")
            if line.strip()]
    assert [r[0] for r in ours] == [r[0] for r in iraf]
    dv = np.array([float(a[1]) - float(b[1]) for a, b in zip(ours, iraf, strict=True)])
    db = np.array([float(a[4]) - float(b[4]) for a, b in zip(ours, iraf, strict=True)])
    assert abs(dv.mean()) < 0.01 and abs(db.mean()) < 0.01
    assert np.abs(dv).max() < 0.11 and np.abs(db).max() < 0.035


def test_an_iraf_coo_edited_by_hand_still_reads_as_iraf_s():
    """The automatico rewrote its .coo.1 files with names in the ID column
    and its own spacing: the #N names still name the columns, the #K IRAF
    line still says 1-based, and .1 is IRAF's version number."""
    from harwood.core import catalog

    path = NATIVE / "stan0051cr.fits.coo.1"
    table = catalog.read(path)
    assert list(table["ID"]) == ["1s", "2s", "3s"]
    x, _ = catalog.coordinates(table, path)
    assert x[0] == pytest.approx(471.056)


# ------------------------------------------------------------- the language


def test_expressions_are_kept_as_written_without_blanks():
    cfg = config.parse_text("""
catalog
V 2
observations
m 3
X 4
transformation
fit a = 1.0, b = -.5
E : m = V + a * ( V - 1 ) + b*X + log10 (X)
""")
    eq = cfg.equations[0]
    assert eq.fit.text == "V+a*(V-1)+b*X+log10(X)"
    assert [p.name for p in eq.params] == ["a", "b"]
    assert eq.values.tolist() == [1.0, np.float32(-0.5)]


def test_unary_minus_binds_tighter_than_power_and_power_goes_left():
    cfg = config.parse_text("""
observations
x 1
transformation
fit p = 0.0
E1 : x = p + -x**2
E2 : x = p + 2**3**2
""")
    row = np.float32([3.0])
    assert config.evaluate(cfg.equations[0].fit.code, row, [0.0]) == 9.0     # (-3)**2
    assert config.evaluate(cfg.equations[1].fit.code, row, [0.0]) == 64.0    # (2**3)**2


def test_keywords_abbreviate_and_an_undefined_value_stops_evaluation():
    cfg = config.parse_text("""
obs
x 1
trans
fit p = 1.0
E : x = p * log(x)
""")
    eq = cfg.equations[0]
    assert np.isnan(config.evaluate(eq.fit.code, np.float32([-1.0]), eq.values))
    assert np.isnan(config.evaluate(eq.fit.code, np.float32([np.nan]), eq.values))


def test_a_syntax_error_is_reported_as_iraf_reports_it():
    with pytest.raises(config.ConfigError) as error:
        config.parse_text("observations\nx 1\ntransformation\nE : x = = 2\n")
    assert error.value.messages == ["** Error near line 4: E : x = = 2\n"
                                    "Cannot continue parsing at '='"]
    with pytest.raises(config.ConfigError) as error:
        config.parse_text("observations\nx 1\ntransformation\nE : x = y\n")
    assert "Undefined identifier in expression [y]" in str(error.value)


def test_the_catalog_column_one_is_the_id():
    with pytest.raises(config.ConfigError, match="reserved for matching name"):
        config.parse_text("catalog\nV 1\nobservations\nm 3\ntransformation\nE : m = V\n")


def test_the_time_stamp_is_iraf_s_whatever_the_locale():
    """A Portuguese desktop (as Qt leaves the process) must not turn
    'Wed ... Oct' into 'qua ... out' in a file IRAF reads."""
    import locale
    import time

    from harwood.core import iraffmt

    saved = locale.setlocale(locale.LC_TIME)
    try:
        locale.setlocale(locale.LC_TIME, "pt_BR.UTF-8")
    except locale.Error:
        pytest.skip("no pt_BR locale here")
    try:
        when = time.mktime((2026, 10, 21, 15, 49, 56, 0, 0, -1))
        assert time.strftime("%a", time.localtime(when)) != "Wed"
        assert iraffmt.cnvtime(when) == "Wed 15:49:56 21-Oct-2026"
    finally:
        locale.setlocale(locale.LC_TIME, saved)


def test_obsfile_as_a_task_reads_an_at_list(tmp_path, monkeypatch):
    from harwood import run

    monkeypatch.chdir(OBSFILE)
    result = run("obsfile", workdir=tmp_path, photfiles=["@mkapfile.lista"],
                 incolumns=OBSCASES["A"][1], idfilters="B,V", imsets="imsets.mkimsets",
                 observations=tmp_path / "obs", tolerance=8.0)
    assert (tmp_path / "obs").read_text(encoding="utf-8") == (OBSFILE / "obsA").read_text(
            encoding="utf-8")
    assert [s["stars"] for s in result.data["sets"]] == [9, 3, 3, 3, 3, 3]
    assert result.data["notes"] == []


# ---------------------------------------------------------------- mkimsets

MKIMSETS = DATA / "mkimsets"
IMCASES = {"A": ("all.lis", "B,V", dict(imobsparams="obsparA")),
           "B": ("all.lis", "V,B", {}),
           "C": ("../mkimsets/part.lista", "B,V,R", {}),
           "D": ("all.lis", "B", dict(sort="XAIRMASS"))}


@pytest.mark.parametrize("case", IMCASES)
def test_mkimsets_groups_as_iraf_does(tmp_path, monkeypatch, case):
    """Sets of B and V, the other way round, a third filter never seen
    (sets closed early, INDEF), and one filter sorted on the airmass."""
    from harwood.core.photcal import imsets

    listing, filters, kw = IMCASES[case]
    monkeypatch.chdir(APFILE)
    files = (APFILE / listing).read_text(encoding="utf-8").split()
    kw = {k: str(tmp_path / v) for k, v in kw.items() if k == "imobsparams"} | {
        k: v for k, v in kw.items() if k != "imobsparams"}
    _, messages = imsets.mkimsets(files, filters, str(tmp_path / "sets"), **kw)
    assert (tmp_path / "sets").read_text(
            encoding="utf-8") == (MKIMSETS / f"sets{case}").read_text(
            encoding="utf-8")
    assert "\n".join(messages) + "\n" == (MKIMSETS / f"sets{case}.out").read_text(
            encoding="utf-8")
    if case == "A":
        assert (tmp_path / "obsparA").read_text(
                encoding="utf-8") == (MKIMSETS / "obsparA").read_text(
                encoding="utf-8")


EDITCASES = {"R1": dict(rename=True), "E1": dict(edit=True, imobsparams="obsE1"),
             "V1": dict(review=True),
             "EV": dict(edit=True, rename=True, review=True, imobsparams="obsEV")}


@pytest.mark.parametrize("case", EDITCASES)
def test_mkimsets_edits_and_asks_as_iraf_does(tmp_path, monkeypatch, case):
    """mkimsets with its editor sessions and its questions (run_iraf_edit.py): the
    same edits made to the image list and to the sets as IRAF's editor made them --
    bin/vi, a stand-in, made both -- and the names typed. What the editor is shown,
    what is asked, and every file written must be IRAF's."""
    import json
    import subprocess
    import sys

    from harwood.core.photcal import entry, imsets

    settings = dict(EDITCASES[case])
    if "imobsparams" in settings:
        settings["imobsparams"] = str(tmp_path / settings["imobsparams"])
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(json.loads((MKIMSETS / "plans.json").read_text(
        encoding="utf-8"))[case]), encoding="utf-8")
    env = dict(os.environ, FAKE_EDIT_PLAN=str(plan))

    def editor(path):
        subprocess.run([sys.executable, str(MKIMSETS / "bin" / "vi"), str(path)], env=env,
                       check=True)

    answers = MKIMSETS / f"{case}.ans"
    term = entry.Terminal(iter(answers.read_text(encoding="utf-8").splitlines()
                               if answers.exists() else []))
    monkeypatch.chdir(APFILE)
    files = (APFILE / "all.lis").read_text(encoding="utf-8").split()
    _, messages = imsets.mkimsets(files, "B,V", str(tmp_path / "sets"), editor=editor,
                                  terminal=term, **settings)
    assert (tmp_path / "sets").read_text(encoding="utf-8") == (
        MKIMSETS / f"sets{case}").read_text(encoding="utf-8")
    for n in range(2):
        if (MKIMSETS / f"{case}.seen{n}").exists():
            assert (tmp_path / f"plan.json.seen{n}").read_text(encoding="utf-8") == (
                MKIMSETS / f"{case}.seen{n}").read_text(encoding="utf-8"), f"editor {n}"
    if settings.get("rename"):
        assert term.text == (MKIMSETS / f"{case}.term").read_text(encoding="utf-8")
    if case == "E1":
        assert "\n".join(messages) + "\n" == (MKIMSETS / "setsE1.out").read_text(
            encoding="utf-8")
    if "imobsparams" in settings:
        assert Path(settings["imobsparams"]).read_text(encoding="utf-8") == (
            MKIMSETS / Path(settings["imobsparams"]).name).read_text(encoding="utf-8")


def test_mkimsets_as_a_task_asks_for_the_names(tmp_path, monkeypatch):
    from harwood import run

    monkeypatch.chdir(APFILE)
    result = run("mkimsets", workdir=tmp_path, imlist=["@all.lis"], idfilters="B,V",
                 imsets=tmp_path / "sets", rename=True, answers=str(MKIMSETS / "R1.ans"))
    assert [s["name"] for s in result.data["sets"]] == ["LSI", "OBS2", "PG2", "PG3",
                                                        "OBS5", "OBS6"]


def test_mkimsets_names_the_sets_from_the_headers(tmp_path, monkeypatch):
    from astropy.io import fits

    from harwood import run

    monkeypatch.chdir(tmp_path)
    names = []
    for k, (obj, band) in enumerate([("PG 2331+055", "B"), ("PG 2331+055", "V"),
                                     ("SA113", "B"), ("SA113", "V")]):
        hdu = fits.PrimaryHDU()
        hdu.header["OBJECT"], hdu.header["FILTER"] = obj, band
        hdu.writeto(f"im{k}.fits")
        names.append(f"im{k}.fits")
    result = run("mkimsets", imlist=names, idfilters="B,V", imsets=tmp_path / "sets",
                 input="images", name_keyword="OBJECT")
    assert (tmp_path / "sets").read_text(encoding="utf-8") == (
        "PG2331+055 :  im0.fits  im1.fits\nSA113 :  im2.fits  im3.fits\n")
    assert not result.warnings


# --------------------------------------------------------------- chkconfig

CHKCONFIG = DATA / "chkconfig"
CHKCASES = sorted(p.stem for p in CHKCONFIG.glob("*.cfg"))


@pytest.mark.parametrize("case", CHKCASES)
@pytest.mark.parametrize("verbose", [False, True])
def test_chkconfig_says_what_iraf_says(case, verbose):
    """The skinakas and lang configurations, one for each catalog of the
    standards directory as mkconfig puts it together (landolt ... ukirt),
    and e*, f*: every message the parser has, each where IRAF places it --
    on the line after an equation when it had to read on, on the line past
    the end for what pr_exit finds -- syntax errors at the end of the file,
    a file without its last newline, an empty one."""
    from harwood.core.photcal import check

    report = check.check(CHKCONFIG / f"{case}.cfg", verbose)
    assert report.text == (CHKCONFIG / f"{case}.{'v' if verbose else 'q'}").read_text(
            encoding="utf-8")
    err = CHKCONFIG / f"{case}.err"
    assert report.stderr == (err.read_text(encoding="utf-8") if err.exists() else "")


def test_a_last_comment_without_its_newline_does_not_hang():
    """IRAF loops for ever on a file ending in '#' with no newline: the
    comment sends the lexer to the last character, which is the '#'."""
    from harwood.core.photcal import check

    text = "catalog\nV 2\nobservations\nmV 3\ntransformation\nfit v1 = 25.0\n" \
           "VFIT : mV = V + v1 #"
    report = check.check("t.cfg", text=text)
    assert report.nerrors == 1 and report.stderr == "syntax error\n"


def test_chkconfig_as_a_task():
    from harwood import run

    result = run("chkconfig", config=CHKCONFIG / "skinakas.cfg", verbose=True)
    assert "\n".join(result.stdout) + "\n" == (CHKCONFIG / "skinakas.v").read_text(
            encoding="utf-8")
    assert result.data == {"errors": 0, "warnings": 0} and not result.warnings
    result = run("chkconfig", config=CHKCONFIG / "e1.cfg")
    assert result.data == {"errors": 3, "warnings": 2} and result.warnings


# --------------------------------------------------------------- mkcatalog

MKCATALOG = DATA / "mkcatalog"
CATCASES = {
    "c1": (["ID", "V", "error(V)", "B", "R", "error(R)"], [15, 10, 8, 10, 12, 10],
           [["PG2331+055", "15.182", "0.0057", "15.116", "14.0", "0.01"], [],
            ["92_336", "8.046", "INDEF", "9.031"],
            ["AVERYLONGIDENTIFIERNAME", "1.23456789012345", "0.1", "2", "3", "4"]]),
    "c3": (["NAME", "V", "error(V)", "UB"], [15, 10, 10, 5], [["x1", "1", "2", "3"]]),
}


@pytest.mark.parametrize("case", CATCASES)
def test_mkcatalog_writes_what_iraf_writes(tmp_path, monkeypatch, case):
    """IRAF's catalog task was given c*.cols at its prompts, which made the
    header (c*.hdr) and the format file, then c*.recs, which added the
    records (c*): a blank id is a blank line, a missing value INDEF, a long
    one cut to its column."""
    from harwood.core.photcal import catalog

    names, widths, records = CATCASES[case]
    monkeypatch.chdir(tmp_path)
    made = catalog.mkcatalog(case, names, widths)
    assert Path(case).read_text(encoding="utf-8") == (MKCATALOG / f"{case}.hdr").read_text(
            encoding="utf-8")
    assert Path(f"f{case}.dat").read_text(
            encoding="utf-8") == (MKCATALOG / f"f{case}.dat").read_text(
            encoding="utf-8")
    assert made.created and made.format_file == f"f{case}.dat"
    added = catalog.mkcatalog(case, records=records)
    assert Path(case).read_text(encoding="utf-8") == (MKCATALOG / case).read_text(
            encoding="utf-8")
    assert not added.created and added.nrecords == sum(map(bool, records))
    assert catalog.read_header(case) == (case, names, widths)


def test_a_record_longer_than_a_line_goes_on_continuation_lines():
    """Fourteen columns of 77 do not fit in IRAF's 1022 characters. IRAF's
    own header (c4.hdr) has the eleventh name blank -- its buffer of names is
    reallocated under a pointer it keeps -- so that is what is compared;
    harwood names the column."""
    from harwood.core.photcal import catalog

    names = ["ID"] + [f"C{i}" for i in range(2, 15)]
    widths = [30] + [77] * 13
    iraf = names.copy()
    iraf[10] = ""
    assert catalog.header("c4", iraf, widths) == (MKCATALOG / "c4.hdr").read_text(
            encoding="utf-8")
    record = catalog.data_record(["star"] + [f"{i}.25" for i in range(2, 15)], widths)
    first, second = record.split("\n")[:2]
    assert len(first) == 982 and second.startswith("* 14.25")


def test_mkcatalog_refuses_what_it_cannot_write(tmp_path, monkeypatch):
    from harwood.core.photcal import catalog

    monkeypatch.chdir(tmp_path)
    for names, widths, match in [(["ID"], None, "no data columns"),
                                 (["ID", "U_B"], None, "not a legal column name: U_B"),
                                 (["ID", "V"], [15], "2 columns but 1 widths"),
                                 (["ID", "V"], [15, 90], "width must be")]:
        with pytest.raises(ValueError, match=match):
            catalog.mkcatalog("cat", names, widths)
    catalog.mkcatalog("cat", ["ID", "V"])
    with pytest.raises(ValueError, match="3 values for 2 columns"):
        catalog.mkcatalog("cat", records=[["a", "1", "2"]])
    with pytest.raises(ValueError, match="already has the columns ID, V"):
        catalog.mkcatalog("cat", ["ID", "B"])
    Path("other").write_text("x 1 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="was not created with mkcatalog"):
        catalog.mkcatalog("other", records=[["a"]])


def test_mkcatalog_as_a_task(tmp_path, monkeypatch):
    from harwood import run

    monkeypatch.chdir(tmp_path)
    Path("recs").write_text("# id V error(V) B error(B)\n"
                            "PG2331+055  15.182 0.0057 15.116 0.0042\n92_336 8.046\n\n",
                                    encoding="utf-8")
    result = run("mkcatalog", catalog="stds", columns=["ID", "V", "error(V)", "B",
                                                       "error(B)"], records="recs")
    assert result.data["records"] == 2 and result.data["created"]
    assert Path("stds").read_text(encoding="utf-8").endswith(
        "  92_336           8.046       INDEF       INDEF       INDEF     \n")
    result = run("mkcatalog", catalog="stds", records="recs")
    assert not result.data["created"]
    assert Path("stds").read_text(encoding="utf-8").count("PG2331+055") == 2


# ---------------------------------------------------------------- mkconfig

MKCONFIG = DATA / "mkconfig"


def test_mkconfig_puts_the_sections_together_as_iraf_does(tmp_path, monkeypatch):
    """m1: Landolt's format file and template from the standards directory,
    the observations' format file obsfile wrote; m2: a catalog's format file
    from mkcatalog (fc1.dat) and a template of one's own (tmine.dat)."""
    from harwood.core.photcal import mkconfig

    monkeypatch.chdir(tmp_path)
    for name in ("fc1.dat", "tmine.dat"):
        Path(name).write_bytes((MKCONFIG / name).read_bytes())
    Path("fobsA.dat").write_bytes((OBSFILE / "fobsA.dat").read_bytes())
    found = mkconfig.mkconfig("m1", "landolt", "obsA", "landolt")
    assert Path("m1").read_bytes() == (MKCONFIG / "m1").read_bytes()
    assert Path(found.catalog).as_posix().endswith("data/iraf/catalogs/flandolt.dat")
    mkconfig.mkconfig("m2", "c1", "obsA", "mine")
    assert Path("m2").read_bytes() == (MKCONFIG / "m2").read_bytes()
    assert mkconfig.mkconfig("m3", template="m1") is None
    assert Path("m3").read_bytes() == Path("m1").read_bytes()
    with pytest.raises(FileExistsError, match="The configuration file m1 already exists"):
        mkconfig.mkconfig("m1", "landolt", "obsA", "landolt")
    with pytest.raises(FileNotFoundError, match="nosuch in current or standards"):
        mkconfig.mkconfig("m4", "nosuch", "obsA", "landolt")
    with pytest.raises(FileNotFoundError, match="nosuch in current directory"):
        mkconfig.mkconfig("m5", "landolt", "nosuch", "landolt")


TYPEDCFG = {"T1": ("STDIN", "obsA", "mine", False), "T2": ("STDIN", "obsA", "mine", True),
            "T3": ("c1", "STDIN", "mine", True), "T4": ("c1", "obsA", "STDIN", False),
            "T5": ("c1", "obsA", "STDIN", True), "T6": ("STDIN", "STDIN", "STDIN", False),
            "T7": ("c1", "obsA", "STDIN", False), "T8": ("c1", "obsA", "STDIN", False)}


@pytest.mark.parametrize("case", TYPEDCFG)
def test_mkconfig_asks_and_writes_what_iraf_does(tmp_path, monkeypatch, case):
    """mkconfig with sections typed (mkconfig/run_iraf_typed.py, IRAF at a terminal,
    <EOF> a Ctrl-D): the catalog's columns with and without verify (T1, T2), the
    observations' with verify (T3), the equations with the help, every error, a
    continued line and a blank one (T4), with verify -- the label cut, the value
    kept (T5) --, all three typed with lines broken at 78 (T6), the 160 characters
    of parameters overflowed (T7), and what is a number (T8). What is asked and said,
    the help without its standout marks, and the file, are IRAF's; then the check."""
    from harwood import run

    monkeypatch.chdir(tmp_path)
    for name in ("fc1.dat", "tmine.dat"):
        Path(name).write_bytes((MKCONFIG / name).read_bytes())
    Path("fobsA.dat").write_bytes((OBSFILE / "fobsA.dat").read_bytes())
    catalog, observations, transform, verify = TYPEDCFG[case]
    result = run("mkconfig", config=f"{case}.cfg", catalog=catalog,
                 observations=observations, transform=transform, verify=verify,
                 answers=str(MKCONFIG / f"{case}.ans"))
    out = (MKCONFIG / f"{case}.out").read_text(encoding="utf-8")
    out = out.replace("\x1b[7m", "").replace("\x1b[m", "")
    cut = out.index("\n\n** Beginning")
    assert result.data["typed"] == out[:cut]
    assert Path(f"{case}.cfg").read_text(encoding="utf-8") == (
        MKCONFIG / f"{case}.cfg").read_text(encoding="utf-8")
    from harwood.core.photcal import check

    # mkconfig.cl's two print ("") reach PyRAF's Stdout wherever its buffer was
    # flushed in chkconfig's output: two blank lines more, somewhere in it
    report = check.check(f"{case}.cfg").text
    assert [line for line in out[cut:].split("\n") if line] == [
        line for line in report.split("\n") if line]
    assert out[cut:].count("\n") == report.count("\n") + 2


def test_mkconfig_typed_goes_on_after_an_end_at_the_keyboard():
    """At a terminal Ctrl-D ends one list and the next is still asked for; from a file
    the end of the input ends them all, as in IRAF, and <EOF> stands for Ctrl-D."""
    from harwood.core.photcal import entry, mkconfig

    typed = iter(["V 2", None, "mV 3", None, None])
    term = entry.Terminal(read=lambda _: next(typed), sticky=False)
    assert mkconfig.ask_columns(term, "catalog") == "\ncatalog\n\nV  2\n"
    assert mkconfig.ask_columns(term, "observation") == "\nobservation\n\nmV  3\n"
    term = entry.Terminal(iter(["V 2"]))
    mkconfig.ask_columns(term, "catalog")
    assert mkconfig.ask_columns(term, "observation") == "\nobservation\n\n"


def test_mkconfig_as_a_task_checks_what_it_wrote(tmp_path, monkeypatch):
    from harwood import run

    monkeypatch.chdir(tmp_path)
    Path("fobsA.dat").write_bytes((OBSFILE / "fobsA.dat").read_bytes())
    Path("tbv.dat").write_text("transformation\n\nfit v1=0, v2=0.17, v3=0\n"
                               "VFIT : mV = V + v1 + v2 * XV + v3 * BV\n", encoding="utf-8")
    result = run("mkconfig", config="bv.cfg", catalog="landolt", observations="obsA",
                 transform="bv")
    assert result.data == {"errors": 0, "warnings": 0} and not result.warnings
    result = run("mkconfig", config="ubvri.cfg", catalog="landolt",
                 observations="obsA", transform="landolt")
    assert result.data["errors"] == 9 and result.warnings


def test_the_fit_panel_draws_and_takes_what_gtools_has(here):
    """w and its key, :/ commands and :. ones at the panel: the view is the
    window gtools gives, flipped, on elog, and nearest is nearest in it."""
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from harwood.viewer import _qt
    from harwood.viewer.fit_panel import FitPanel

    _qt.application()
    session = _fit_session({}, "out.fit")
    panel = FitPanel(session)
    box = panel.plot.getViewBox()
    xs, ys = session._axis(0), session._axis(1)
    panel._cursor = (float(xs[4]), float(ys[4]))
    assert panel.key("w") and session.pending == "window"
    assert panel.key("z") and not session.pending
    (x0, x1), _ = box.viewRange()
    assert x0 < xs[4] < x1 and x1 - x0 == pytest.approx(
        (session.window[1] - session.window[0]), rel=1e-6)
    panel.command.setText(":/xflip yes")
    panel._run_command()
    assert not box.xInverted(), "as in IRAF, a format waits for the next graph"
    assert panel.key("r") and box.xInverted() and session.window[0] > session.window[1]
    assert panel.key("c")
    assert f"5 {session.obs.names[4]} x=" in session.stdout.splitlines()[-1]
    panel.command.setText(":/mark box")
    panel._run_command()
    panel.command.setText(":/redraw")
    panel._run_command()
    assert session.graph().mark == "box"
    panel.command.setText(":/xtransform log")
    panel._run_command()
    panel.key("k")
    ticks = panel.plot.getAxis("bottom")._tickLevels
    assert ticks and [label for _, label in ticks[0]][-2:] == ["10", "100"]
    panel.command.setText(":.snap graph.png")
    panel._run_command()
    assert (here / "graph.png").stat().st_size > 0
    panel.command.setText(":.zero")
    panel._run_command()
    assert "cursor-mode command harwood does not have" in panel.text.toPlainText()
    assert ":.zero" not in session.stdout, "cursor mode is the panel's, not the task's"
