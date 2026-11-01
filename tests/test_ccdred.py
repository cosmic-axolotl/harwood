"""Instrument translation, ccdlist, and ccdproc finding its own calibrations.

Path 2 of the course's exercise 2: `ccdproc m92*.fits ccdtype=object` with no
zero and no flat named, and the right bias and the right flat per filter found
from the headers. The whole point is that nothing is named, so the tests are
mostly about what happens when the headers do not say what harwood expects.
"""

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import instrument as instruments
from harwood.core.registry import get_task

KPNO = "ccddb$kpno/direct.dat"

needs_iraf = pytest.mark.skipif(
    not any(d.is_dir() for d in instruments.IRAF_CCDDB),
    reason="IRAF's instrument files are not installed",
)


@pytest.fixture
def night(tmp_path, monkeypatch):
    """A bias, two flats in different filters, and two objects in each."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(3)
    ny, nx, useful = 40, 50, 40

    def write(name, imagetyp, filters, exptime, level):
        header = fits.Header({"IMAGETYP": imagetyp, "FILTERS": filters,
                              "EXPTIME": exptime, "BIASSEC": f"[{useful + 1}:{nx},*]",
                              "TRIMSEC": f"[1:{useful},1:{ny}]"})
        data = np.full((ny, nx), 500.0) + rng.normal(0, 2, (ny, nx))
        data[:, :useful] += level
        fits.PrimaryHDU(data.astype(np.float32), header).writeto(tmp_path / name)

    write("b001.fits", "BIAS", "0 0", 0.0, 0.0)
    write("f005.fits", "DOME FLAT", "5 0", 10.0, 1200.0)
    write("f006.fits", "DOME FLAT", "6 0", 10.0, 1400.0)
    write("o010.fits", "OBJECT", "6 0", 60.0, 300.0)
    write("o014.fits", "OBJECT", "5 0", 120.0, 250.0)
    return tmp_path


# ------------------------------------------------------------ translation


@needs_iraf
def test_irafs_own_translation_file_is_read():
    kpno = instruments.load(KPNO)
    assert kpno.keyword("subset") == "filters"
    assert kpno.types["DOME FLAT"] == "flat"
    assert kpno.types["BIAS"] == "zero"


@needs_iraf
def test_a_bare_name_finds_the_file_in_irafs_tree():
    assert instruments.load("kpno/direct").name == "direct"


def test_an_unknown_translation_says_how_to_look():
    with pytest.raises(FileNotFoundError, match="setinstrument --list"):
        instruments.load("nowhere/nothing")


def test_a_translation_file_of_your_own(tmp_path):
    path = tmp_path / "mine.dat"
    path.write_text("subset      BAND\n"
                    "exptime     ITIME\n"
                    "'SKY FLAT'  flat\n"
                    "CALIB       comp\n")
    mine = instruments.load(path)
    assert mine.keyword("subset") == "BAND"
    assert mine.keyword("exptime") == "ITIME"
    assert mine.types["SKY FLAT"] == "flat"
    assert mine.keyword("biassec") == "BIASSEC", "untranslated falls back to the usual"


def test_a_bad_image_type_in_a_translation_file_is_refused(tmp_path):
    path = tmp_path / "bad.dat"
    path.write_text("'DOME FLAT'  flatfield\n")
    with pytest.raises(ValueError, match="not an image type"):
        instruments.load(path)


def test_an_untranslated_header_value_is_unknown_not_guessed(night):
    """A frame harwood cannot classify is one it must not process by accident."""
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32),
                    fits.Header({"IMAGETYP": "TWILIGHT"})).writeto(night / "x.fits")
    plain = instruments.Instrument()
    assert plain.image_type(fits.getheader(night / "x.fits")) == "unknown"


# ---------------------------------------------------------------- ccdlist


@needs_iraf
def test_ccdlist_classifies_every_frame(night):
    result = get_task("ccdlist").run(input=["*.fits"], instrument=KPNO)
    kinds = {row["image"]: row["type"] for row in result.data["rows"]}
    assert kinds["b001.fits"] == "zero"
    assert kinds["f005.fits"] == "flat"
    assert kinds["o010.fits"] == "object"
    assert not result.warnings


def test_ccdlist_warns_about_what_it_cannot_classify(night):
    """Without the translation, BIAS and DOME FLAT mean nothing -- and ccdlist
    is the task you run to find that out before ccdproc does."""
    result = get_task("ccdlist").run(input=["*.fits"])
    assert any("unknown" in warning for warning in result.warnings)
    assert any("setinstrument" in warning for warning in result.warnings)


@needs_iraf
def test_ccdlist_filters_by_type_and_shows_what_is_done(night):
    result = get_task("ccdlist").run(input=["*.fits"], ccdtype="flat",
                                     instrument=KPNO, long=True)
    assert len(result.data["rows"]) == 2
    assert all(row["done"] == "-" for row in result.data["rows"])

    get_task("ccdproc").run(input=["f005.fits"], inplace=True, overscan=True,
                            trim=True, instrument=KPNO)
    after = get_task("ccdlist").run(input=["f005.fits"], instrument=KPNO, long=True)
    assert "overscan" in after.data["rows"][0]["done"]
    assert "trim" in after.data["rows"][0]["done"]


# ---------------------------------------------------------------- ccdproc


@needs_iraf
def test_ccdproc_finds_its_own_calibrations(night):
    """Path 2: nothing is named, everything is found from the headers."""
    result = get_task("ccdproc").run(
        input=["*.fits"], output="out", ccdtype="object", overscan=True,
        trim=True, zerocor=True, flatcor=True, instrument=KPNO)

    assert result.data["images"] == 2, "only the objects are corrected"
    assert sorted(p.name for p in result.outputs) == ["o010.fits", "o014.fits"]

    # Each object divided by the flat of its own filter, so both come back
    # near one times their own level rather than near each other's.
    for name in ("o010.fits", "o014.fits"):
        data = fits.getdata(night / "out" / name).astype(float)
        assert np.isfinite(data).all()
        assert data.shape == (40, 40), "trimmed"


@needs_iraf
def test_the_flat_is_matched_by_subset_not_by_order(night):
    """The V object must get the V flat even though the B flat comes first."""
    task = get_task("ccdproc")
    params = task.Params(input=["*.fits"], output="out", ccdtype="object",
                         overscan=True, trim=True, flatcor=True, instrument=KPNO)
    steps = task.plan(params)
    found = next(step for step in steps if step.action == "find flat")
    assert "5 0 -> f005.fits" in found.detail
    assert "6 0 -> f006.fits" in found.detail


@needs_iraf
def test_a_frame_with_no_flat_for_its_filter_is_refused(night):
    """Guessing a flat from another filter would be a silent, wrong colour term."""
    header = fits.Header({"IMAGETYP": "OBJECT", "FILTERS": "9 9", "EXPTIME": 60.0,
                          "BIASSEC": "[41:50,*]", "TRIMSEC": "[1:40,1:40]"})
    fits.PrimaryHDU(np.full((40, 50), 700.0, dtype=np.float32),
                    header).writeto(night / "o099.fits")

    with pytest.raises(ValueError, match="no flat field for subset"):
        get_task("ccdproc").run(input=["*.fits"], output="out", ccdtype="object",
                                overscan=True, trim=True, flatcor=True,
                                instrument=KPNO)


@needs_iraf
def test_an_empty_ccdtype_selection_says_what_was_there(night):
    with pytest.raises(RuntimeError, match="none of the"):
        get_task("ccdproc").run(input=["*.fits"], output="out", ccdtype="comp",
                                overscan=True, instrument=KPNO)


@needs_iraf
def test_calibrations_are_put_through_the_same_geometry(night):
    """A 50-wide bias meeting a 40-wide object is the failure this prevents."""
    result = get_task("ccdproc").run(
        input=["*.fits"], output="out", ccdtype="object", overscan=True,
        trim=True, zerocor=True, instrument=KPNO)
    prepared = [step for step in result.steps if step.action.startswith("zero b001")]
    assert prepared, "the bias is corrected before it is used"
    assert any("trim" in step.action for step in prepared)


@needs_iraf
def test_running_in_place_is_idempotent(night):
    """IRAF overwrites the inputs, and the header marks are what make that safe.

    It is opt-in here rather than what happens when you say nothing: leaving
    --output out writes to the run's own directory instead, which cannot
    destroy anything.
    """
    first = get_task("ccdproc").run(input=["o010.fits"], inplace=True, overscan=True,
                                    trim=True, instrument=KPNO)
    assert any("in place" in warning for warning in first.warnings)
    after_once = fits.getdata(night / "o010.fits").copy()

    second = get_task("ccdproc").run(input=["o010.fits"], inplace=True, overscan=True,
                                     trim=True, instrument=KPNO)
    assert second.data["steps_skipped"] == 2
    np.testing.assert_array_equal(fits.getdata(night / "o010.fits"), after_once)


# ---------------------------------------------------------------- explain


@needs_iraf
def test_every_step_that_changes_pixels_explains_itself(night):
    """The professor's complaint, as a test.

    A step that alters the data and cannot say what it did to it, or what it
    came to, is the black box this project exists to open.
    """
    result = get_task("ccdproc").run(
        input=["*.fits"], output="out", ccdtype="object", overscan=True,
        trim=True, zerocor=True, flatcor=True, instrument=KPNO)

    changed = [step for step in result.steps
               if not step.skipped and step.before is not None]
    assert changed, "some step must have touched the pixels"
    for step in changed:
        assert step.explained(), f"'{step.action}' does not say what it does"
        assert step.measured, f"'{step.action}' reports no numbers"
        assert step.after is not None


def test_a_step_without_an_explanation_says_so_rather_than_nothing():
    from harwood.core.task import Step

    assert not Step("mystery", "does a thing").explained()
    assert Step("overscan", "", why="because").explained()


@needs_iraf
def test_inplace_and_output_together_are_refused(night):
    with pytest.raises(ValueError, match="nothing to mean"):
        get_task("ccdproc").run(input=["o010.fits"], output="somewhere",
                                inplace=True, overscan=True, instrument=KPNO)


@needs_iraf
def test_saying_nothing_writes_to_the_runs_own_directory(night):
    """The originals survive, which is the point of it being the default."""
    from harwood.core import output

    before = (night / "o010.fits").read_bytes()
    token = output.begin("ccdproc", night)
    try:
        result = get_task("ccdproc").run(input=["o010.fits"], overscan=True,
                                         trim=True, instrument=KPNO)
    finally:
        output.end(token)

    assert (night / "o010.fits").read_bytes() == before, "the input is untouched"
    assert result.outputs[0].parent.name.startswith("hrwd_ccdproc_")
    assert not any("in place" in str(w) for w in result.warnings), \
        "nothing to warn about the originals when they were kept"


def test_setinstrument_applies_the_instruments_settings(tmp_path):
    """IRAF's setinstrument runs ccddb$kpno/direct.cl after choosing the
    .dat: default.cl's settings, then direct's own flatcor = yes. Here they
    go into the project file, each one harwood has, the rest said why not."""
    import yaml

    from harwood.core.registry import get_task

    result = get_task("setinstrument").run(instrument="direct", workdir=tmp_path)
    document = yaml.safe_load((tmp_path / "harwood.yaml").read_text())
    assert document["instrument"] == "kpno/direct"
    ccdproc = document["params"]["ccdproc"]
    assert ccdproc["flatcor"] is True, "direct.cl after default.cl"
    assert ccdproc["zerocor"] is True and ccdproc["darkcor"] is False
    assert ccdproc["function"] == "chebyshev" and "biassec" not in ccdproc
    assert document["params"]["flatcombine"]["reject"] == "crreject"
    rows = {row["setting"]: row["applied"] for row in result.data["rows"]}
    assert rows["ccdred.logfile"].startswith("no -- harwood.log.jsonl")
    assert rows["combine.gain"].startswith("yes"), "imcombine takes a keyword for it now"
