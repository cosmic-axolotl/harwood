"""fitparams and invertfit against IRAF's own output.

The data are the Skinakas night (PG2331+055 and its two comparison stars,
five times through the night, and nine program stars in LSI+61 235), as the
automatico pipeline left them: observations.obsfile, catalogo.mkcatalog and
configuration.mkconfig. config2.cfg exercises the rest of the language --
a set equation, a constant, a delta, a derivative, weight equations with
limits, error equations -- and obs2.txt has an INDEF magnitude, a standard
missing from the catalog and a program star with an undefined magnitude.
IRAF 2.18 ran each case (run.cl, run2.cl, run3.cl); its files are here.

The parameters files and the logs agree to the last digit IRAF prints; the
calibrated magnitudes to the seventh decimal, but for one star in one case:
inverted with no catalog to start from, its last step is taken where the
rms is 1e-12, all rounding, and it lands two units of the seventh decimal
away (1.9e-6 in 15.16, two units in the last place of a float).
"""

from pathlib import Path

import numpy as np
import pytest

from harwood.core.photcal import apfile, config, fit, invert
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


def _stamped(line: str) -> bool:
    """A time stamp: # Wed 15:49:56 21-Oct-2026, or #WED ... in the log."""
    return line.startswith("#") and line[1:].strip()[:3].upper() in (
        "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if not _stamped(line)]


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
PHOTFILES = (APFILE / "all.lis").read_text().split()
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
        ours = [line.replace(f"{tmp_path}/", "") for line in path.read_text().splitlines()]
        iraf = (APFILE / name).read_text().splitlines()
        assert ours[1:] == iraf[1:], name                 # after NEW LOGFILE ENTRY AT
        assert ours[0] == iraf[0] or ours[0].startswith("NEW LOGFILE ENTRY AT")


def test_mkapfile_as_a_task(tmp_path, monkeypatch):
    from harwood import run

    monkeypatch.chdir(APFILE)
    result = run("mkapfile", photfiles=PHOTFILES[:2], naperts=20,
                 apercors=tmp_path / "ap", largeap=11)
    assert result.data["converged"]
    assert set(result.data["images"]) == {"data0097cr.fits", "data0098cr.fits"}
    assert (tmp_path / "ap").read_text().startswith(
        "# The aperture correction from apertures 1 to 11 in magnitudes")


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
    obsparams, and every kind of bad line in a sets file."""
    sets, cols, kw = OBSCASES[case]
    monkeypatch.chdir(OBSFILE)
    files = (OBSFILE / "mkapfile.lista").read_text().split()
    out = tmp_path / f"obs{case}"
    result = obs.obsfile(files, cols, "B,V", sets, str(out), **kw)
    assert out.read_text() == (OBSFILE / f"obs{case}").read_text()
    assert obs.format_path(out).read_text() == (OBSFILE / f"fobs{case}.dat").read_text()
    assert obs.summary(f"obs{case}", result.written) == (
        OBSFILE / f"obs{case}.out").read_text()
    assert "".join(n + "\n" for n in result.notes) == (OBSFILE / f"obs{case}.err").read_text()


def test_the_automatico_s_observations_are_what_fitparams_was_given():
    """Case A is observations.obsfile, the file the fit above starts from."""
    assert (OBSFILE / "obsA").read_text() == (DATA / "observations.obsfile").read_text()


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
    for name in (NATIVE / "mkapfile.lista").read_text().split():
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
    return [line for line in Path(path).read_text().splitlines()
            if not line.startswith("#")]


def test_harwood_s_phot_rounded_as_iraf_writes_it_gives_the_automatico_s_night(
        tmp_path, monkeypatch):
    """The whole chain in harwood, from its own phot: with the numbers
    rounded where IRAF's file rounds them, every file the automatico wrote
    comes out the same -- the observations, the coefficients and the
    calibrated magnitudes."""
    out = _native_chain(tmp_path, monkeypatch, rounded=True)
    assert (out / "obs").read_text() == (DATA / "observations.obsfile").read_text()
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
    result = run("obsfile", photfiles=["@mkapfile.lista"], incolumns=OBSCASES["A"][1],
                 idfilters="B,V", imsets="imsets.mkimsets",
                 observations=tmp_path / "obs", tolerance=8.0)
    assert (tmp_path / "obs").read_text() == (OBSFILE / "obsA").read_text()
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
    files = (APFILE / listing).read_text().split()
    kw = {k: str(tmp_path / v) for k, v in kw.items() if k == "imobsparams"} | {
        k: v for k, v in kw.items() if k != "imobsparams"}
    _, messages = imsets.mkimsets(files, filters, str(tmp_path / "sets"), **kw)
    assert (tmp_path / "sets").read_text() == (MKIMSETS / f"sets{case}").read_text()
    assert "\n".join(messages) + "\n" == (MKIMSETS / f"sets{case}.out").read_text()
    if case == "A":
        assert (tmp_path / "obsparA").read_text() == (MKIMSETS / "obsparA").read_text()


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
    assert (tmp_path / "sets").read_text() == (
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
    assert report.text == (CHKCONFIG / f"{case}.{'v' if verbose else 'q'}").read_text()
    err = CHKCONFIG / f"{case}.err"
    assert report.stderr == (err.read_text() if err.exists() else "")


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
    assert "\n".join(result.stdout) + "\n" == (CHKCONFIG / "skinakas.v").read_text()
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
    assert Path(case).read_text() == (MKCATALOG / f"{case}.hdr").read_text()
    assert Path(f"f{case}.dat").read_text() == (MKCATALOG / f"f{case}.dat").read_text()
    assert made.created and made.format_file == f"f{case}.dat"
    added = catalog.mkcatalog(case, records=records)
    assert Path(case).read_text() == (MKCATALOG / case).read_text()
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
    assert catalog.header("c4", iraf, widths) == (MKCATALOG / "c4.hdr").read_text()
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
    Path("other").write_text("x 1 2\n")
    with pytest.raises(ValueError, match="was not created with mkcatalog"):
        catalog.mkcatalog("other", records=[["a"]])


def test_mkcatalog_as_a_task(tmp_path, monkeypatch):
    from harwood import run

    monkeypatch.chdir(tmp_path)
    Path("recs").write_text("# id V error(V) B error(B)\n"
                            "PG2331+055  15.182 0.0057 15.116 0.0042\n92_336 8.046\n\n")
    result = run("mkcatalog", catalog="stds", columns=["ID", "V", "error(V)", "B",
                                                       "error(B)"], records="recs")
    assert result.data["records"] == 2 and result.data["created"]
    assert Path("stds").read_text().endswith(
        "  92_336           8.046       INDEF       INDEF       INDEF     \n")
    result = run("mkcatalog", catalog="stds", records="recs")
    assert not result.data["created"]
    assert Path("stds").read_text().count("PG2331+055") == 2


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
    assert found.catalog.endswith("data/iraf/catalogs/flandolt.dat")
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


def test_mkconfig_as_a_task_checks_what_it_wrote(tmp_path, monkeypatch):
    from harwood import run

    monkeypatch.chdir(tmp_path)
    Path("fobsA.dat").write_bytes((OBSFILE / "fobsA.dat").read_bytes())
    Path("tbv.dat").write_text("transformation\n\nfit v1=0, v2=0.17, v3=0\n"
                               "VFIT : mV = V + v1 + v2 * XV + v3 * BV\n")
    result = run("mkconfig", config="bv.cfg", catalog="landolt", observations="obsA",
                 transform="bv")
    assert result.data == {"errors": 0, "warnings": 0} and not result.warnings
    result = run("mkconfig", config="ubvri.cfg", catalog="landolt",
                 observations="obsA", transform="landolt")
    assert result.data["errors"] == 9 and result.warnings
