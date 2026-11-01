"""The subsets file and the running log: a reduction's own bookkeeping.

Both exist because a night has more than one filter and more than one frame,
and because the person reading the result six months later is not the person
who ran it.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import logfile, subsets
from harwood.core.registry import get_task


@pytest.fixture
def night(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # A translation of its own, so these tests exercise the whole chain
    # without needing IRAF's files installed.
    (tmp_path / "mine.dat").write_text("subset        FILTERS\n"
                                       "'DOME FLAT'   flat\n"
                                       "BIAS          zero\n"
                                       "OBJECT        object\n")
    (tmp_path / "harwood.yaml").write_text("instrument: mine.dat\nparams: {}\n")

    rng = np.random.default_rng(8)
    ny, nx, useful = 30, 40, 30

    def write(name, imagetyp, filters, level):
        header = fits.Header({"IMAGETYP": imagetyp, "FILTERS": filters,
                              "EXPTIME": 10.0, "BIASSEC": f"[{useful + 1}:{nx},*]",
                              "TRIMSEC": f"[1:{useful},1:{ny}]"})
        data = np.full((ny, nx), 400.0) + rng.normal(0, 2, (ny, nx))
        data[:, :useful] += level
        fits.PrimaryHDU(data.astype(np.float32), header).writeto(tmp_path / name)

    for index in (1, 2, 3):
        write(f"z{index}.fits", "BIAS", "0 0", 0.0)
    write("f5a.fits", "DOME FLAT", "5 0", 1000.0)
    write("f5b.fits", "DOME FLAT", "5 0", 1010.0)
    write("f6a.fits", "DOME FLAT", "6 0", 1300.0)
    write("f6b.fits", "DOME FLAT", "6 0", 1290.0)
    write("o1.fits", "OBJECT", "6 0", 200.0)
    return tmp_path


# --------------------------------------------------------------- subsets


def test_a_subset_gets_a_short_name_derived_from_the_header(night):
    """IRAF's ccdsubset: the first word of the header value."""
    assert subsets.suggest("6 0", set()) == "6"
    assert subsets.suggest("V", set()) == "V"


def test_a_clashing_name_gets_a_number(night):
    """The first word again, then 1, 2... -- '6 0' and '6 1' are 6 and 61."""
    assert subsets.suggest("6 1", {"6"}) == "61"
    assert subsets.suggest("6 2", {"6", "61"}) == "62"


def test_a_name_goes_into_file_names_with_only_safe_characters(night):
    subsets.record(["B+V filter"])
    assert subsets.read() == {"B+V filter": "B+V"}
    assert subsets.name_for("B+V filter") == "B_V"


def test_the_file_is_written_the_first_time_a_subset_is_seen(night):
    subsets.record(["5 0", "6 0"])
    text = (night / "subsets").read_text()
    assert text.startswith("#"), "it explains what it is for"
    assert "'5 0'" in text and "'6 0'" in text
    assert subsets.read() == {"5 0": "5", "6 0": "6"}


def test_editing_the_name_is_the_whole_point(night):
    subsets.record(["6 0"])
    (night / "subsets").write_text("'6 0'   V\n")
    assert subsets.name_for("6 0") == "V"


def test_a_malformed_line_says_what_the_columns_are(night):
    (night / "subsets").write_text("just-one-column\n")
    with pytest.raises(ValueError, match="left column"):
        subsets.read()


def test_flatcombine_splits_by_filter_and_names_the_outputs(night):
    """`flatcombine *.fits Flat` over a night gives FlatB and FlatV: combining
    two filters together produces a flat for neither."""
    (night / "subsets").write_text("'5 0'  B\n'6 0'  V\n")

    result = get_task("flatcombine").run(input=["*.fits"], output="Flat.fits",
                                         gain=2.0, rdnoise=6.0)
    names = sorted(p.name for p in result.outputs)
    assert names == ["FlatB.fits", "FlatV.fits"]

    # Each holds its own filter's level, not a blend of the two.
    assert float(np.median(fits.getdata(night / "FlatB.fits"))) < \
        float(np.median(fits.getdata(night / "FlatV.fits")))


def test_zerocombine_picks_the_bias_frames_out_of_a_whole_night(night):
    result = get_task("zerocombine").run(input=["*.fits"], output="Zero.fits",
                                         nlow=0, nhigh=1)
    assert result.data["images"] == 3, "the flats and the object are not biases"


def test_minmax_that_would_leave_nothing_is_refused(night):
    """Dropping the lowest and highest of one frame leaves none, and every pixel
    gets the blank value -- a frame full of zeros that looks like a frame."""
    with pytest.raises(ValueError, match="leaving nothing"):
        get_task("imcombine").run(input=["o1.fits"], output="x.fits",
                                  reject="minmax", nlow=1, nhigh=1)


def test_minmax_that_would_undercut_nkeep_is_refused(night):
    with pytest.raises(ValueError, match="--nkeep"):
        get_task("imcombine").run(input=["z1.fits", "z2.fits", "z3.fits"],
                                  output="x.fits", reject="minmax",
                                  nlow=1, nhigh=1, nkeep=2)


# --------------------------------------------------------------- logfile


def test_the_log_grows_a_line_at_a_time(night):
    get_task("ccdproc").run(input=["o1.fits"], output="out", overscan=True,
                            trim=True, logfile="run.log")
    text = (night / "run.log").read_text()
    assert "ccdproc  o1.fits" in text
    assert "overscan" in text and "trim" in text
    assert "finished" in text


def test_a_second_run_appends_rather_than_replaces(night):
    get_task("ccdproc").run(input=["z1.fits"], output="out", overscan=True,
                            logfile="run.log")
    get_task("ccdproc").run(input=["z2.fits"], output="out", overscan=True,
                            logfile="run.log")
    text = (night / "run.log").read_text()
    assert "z1.fits" in text and "z2.fits" in text


def test_an_empty_logfile_name_keeps_none(night):
    get_task("ccdproc").run(input=["o1.fits"], output="out", overscan=True,
                            logfile="")
    assert not list(night.glob("*.log"))


def test_a_bare_name_goes_to_the_reduction_not_the_subdirectory(night):
    """Run from a subdirectory, a task still writes to the night's log."""
    deep = night / "raw"
    deep.mkdir()
    assert logfile.resolve("night.log", night) == night / "night.log"
    assert logfile.resolve("/tmp/absolute.log", night).is_absolute()


def test_the_log_records_which_flat_each_frame_got(night):
    """The thing you go to the log to find out.

    The real sequence: correct the flats, combine them per filter, then use
    them. A flat named explicitly is taken as given -- harwood only prepares
    the calibrations it found itself -- so combining untrimmed flats and then
    naming one is caught, as it should be.
    """
    (night / "subsets").write_text("'5 0'  B\n'6 0'  V\n")

    get_task("ccdproc").run(input=["f5*.fits", "f6*.fits"], output="%f%tf%",
                            overscan=True, trim=True, logfile="run.log")
    get_task("flatcombine").run(input=["tf*.fits"], output="Flat.fits",
                                gain=2.0, rdnoise=6.0)
    get_task("ccdproc").run(input=["o1.fits"], output="out", overscan=True,
                            trim=True, flatcor=True, flat="FlatV.fits",
                            logfile="run.log")

    text = (night / "run.log").read_text()
    assert "FlatV.fits" in text
    assert "ccdproc  o1.fits" in text


def test_a_named_calibration_is_prepared_as_well(night):
    """A raw flat named against a trimmed frame goes through the frame's steps, in memory,
    as IRAF's ccdproc does in place; the file stays as it was."""
    get_task("flatcombine").run(input=["f6*.fits"], output="Raw.fits",
                                gain=2.0, rdnoise=6.0, subsets=False)
    before = Path("Raw.fits").read_bytes()
    result = get_task("ccdproc").run(input=["o1.fits"], output="out", overscan=True,
                                     trim=True, flatcor=True, flat="Raw.fits")
    assert Path("Raw.fits").read_bytes() == before
    assert any(step.action.startswith("flat Raw.fits") for step in result.steps)
