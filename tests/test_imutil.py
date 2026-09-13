"""The small tasks a reduction leans on, and the templates that name its files.

Most of this exists because exercise 2 from the course needed it and harwood
could not do it: `%m%tr%` output names, `imarith @list`, `imstat fields=`,
`sections`, `imdelete`.
"""

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import images
from harwood.core.registry import get_task


@pytest.fixture
def frames(tmp_path, monkeypatch):
    """Three named images, so templates have something to rename."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(11)
    for index in (1, 2, 3):
        header = fits.Header({"IMAGETYP": "object", "EXPTIME": 30.0 * index,
                              "FILTERS": "V", "OBJECT": f"field {index}"})
        fits.PrimaryHDU((100.0 * index + rng.normal(0, 3, (20, 30))).astype(np.float32),
                        header).writeto(tmp_path / f"m9200{index}.fits")
    return tmp_path


# ------------------------------------------------------------- templates


def test_substitution_renames_the_way_iraf_does():
    assert str(images.substitute("m92001.fits", "%m%tr%")) == "tr92001.fits"
    assert str(images.substitute("dir/m92001.fits", "%m%n%")) == "dir/n92001.fits"


def test_a_substitution_that_matches_nothing_says_so():
    with pytest.raises(ValueError, match="nothing to replace"):
        images.substitute("obj.fits", "%zz%tr%")


def test_a_malformed_template_explains_the_syntax():
    with pytest.raises(ValueError, match="%old%new%"):
        images.substitute("m92001.fits", "%m%")


def test_outputs_match_inputs_one_for_one(frames):
    specs = images.expand("m9200*.fits")
    assert [p.name for p in images.outputs_for(specs, "%m%tr%")] == \
        ["tr92001.fits", "tr92002.fits", "tr92003.fits"]
    assert [p.name for p in images.outputs_for(specs, "a.fits,b.fits,c.fits")] == \
        ["a.fits", "b.fits", "c.fits"]
    assert [str(p) for p in images.outputs_for(specs, "out")][0] == "out/m92001.fits"


def test_a_count_mismatch_is_refused_rather_than_guessed(frames):
    """Writing three inputs to one output destroys two of them silently."""
    specs = images.expand("m9200*.fits")
    with pytest.raises(ValueError, match="3 inputs but 2 outputs"):
        images.outputs_for(specs, "a.fits,b.fits")


# ---------------------------------------------------------------- tasks


def test_sections_expands_a_substitution_template(frames):
    result = get_task("sections").run(template="%m%tr%9200*.fits")
    assert [row["name"] for row in result.data["rows"]] == \
        ["tr92001.fits", "tr92002.fits", "tr92003.fits"]


def test_imarith_pairs_a_list_with_a_list(frames):
    get_task("imarith").run(operand1="m92001.fits,m92002.fits", op="-",
                            operand2="m92003.fits", result="%m%d%")
    first = fits.getdata(frames / "d92001.fits")
    assert float(np.median(first)) == pytest.approx(100.0 - 300.0, abs=2.0)


def test_imarith_reads_an_at_list(frames):
    (frames / "all.lis").write_text("m92001.fits\nm92002.fits\n")
    result = get_task("imarith").run(operand1="@all.lis", op="/", operand2="2",
                                     result="%m%h%")
    assert len(result.outputs) == 2
    assert float(np.median(fits.getdata(frames / "h92001.fits"))) == \
        pytest.approx(50.0, abs=2.0)


def test_imarith_refuses_mismatched_operand_lists(frames):
    with pytest.raises(ValueError, match="must be a single image"):
        get_task("imarith").run(operand1="m92001.fits,m92002.fits", op="-",
                                operand2="m92001.fits,m92002.fits,m92003.fits",
                                result="%m%x%")


def test_imstatistics_reports_only_the_fields_asked_for(frames):
    result = get_task("imstatistics").run(input=["m92001.fits"],
                                          fields="image,mean,stddev")
    assert list(result.data["rows"][0]) == ["image", "mean", "stddev"]


def test_an_unknown_field_lists_the_ones_that_exist(frames):
    with pytest.raises(ValueError, match="Choose from"):
        get_task("imstatistics").run(input=["m92001.fits"], fields="image,sparkle")


def test_the_mode_finds_the_peak_of_the_distribution():
    from harwood.tasks.imutil import mode

    rng = np.random.default_rng(5)
    values = rng.normal(1300.0, 20.0, 200_000)
    assert mode(values) == pytest.approx(1300.0, abs=3.0)


def test_the_mode_is_stable_across_sensible_bin_widths():
    """On a clean distribution it should not move with the bin width.

    On real data it does, which is why the default is a tenth of a sigma
    rather than a fixed number of bins -- see the docstring, and the KPNO
    flat where the two choices differ by seventeen counts.
    """
    from harwood.tasks.imutil import mode

    rng = np.random.default_rng(6)
    values = rng.normal(1300.0, 20.0, 200_000)
    estimates = [mode(values, width) for width in (0.5, 1.0, 2.0, 4.0)]
    assert max(estimates) - min(estimates) < 4.0


def test_imstatistics_clips_and_ignores_limits(frames):
    data = np.full((10, 10), 100.0)
    data[0, 0] = 9999.0
    fits.PrimaryHDU(data.astype(np.float32)).writeto(frames / "spike.fits")

    plain = get_task("imstatistics").run(input=["spike.fits"], fields="mean")
    clipped = get_task("imstatistics").run(input=["spike.fits"], fields="mean",
                                           nclip=3)
    limited = get_task("imstatistics").run(input=["spike.fits"], fields="mean",
                                           upper=500.0)
    assert plain.data["rows"][0]["mean"] > 190
    assert clipped.data["rows"][0]["mean"] == pytest.approx(100.0)
    assert limited.data["rows"][0]["mean"] == pytest.approx(100.0)


def test_imcopy_takes_a_section(frames):
    get_task("imcopy").run(input=["m92001.fits[5:14,1:10]"], output="cut.fits")
    assert fits.getdata(frames / "cut.fits").shape == (10, 10)


def test_imdelete_refuses_until_told_twice(frames):
    with pytest.raises(RuntimeError, match="--no-verify"):
        get_task("imdelete").run(input=["m9200*.fits"])
    assert (frames / "m92001.fits").exists(), "nothing may go before confirmation"

    result = get_task("imdelete").run(input=["m9200*.fits"], verify=False)
    assert result.data["deleted"] == 3
    assert not (frames / "m92001.fits").exists()


def test_imheader_finds_the_filter_under_either_keyword(frames):
    """KPNO wrote FILTERS; most of the world writes FILTER. Looking for one of
    them leaves the column blank on half the data there is."""
    result = get_task("imheader").run(input=["m92001.fits"])
    assert result.data["rows"][0]["filter"] == "V"


# --------------------------------------------------------------- implot


def test_implot_cuts_lines_and_columns_where_it_says():
    from harwood.viewer._qt import application
    from harwood.viewer.implot_panel import ImplotPanel

    application()
    data = np.zeros((40, 60))
    data[9, :] = 7.0                      # image line 10, 1-based
    data[:, 19] = 3.0                     # image column 20, 1-based

    panel = ImplotPanel(data, "test")
    panel.plot_lines(10, 10)
    assert panel.last[1].mean() == pytest.approx(7.0, abs=0.2)
    panel.plot_columns(20, 20)
    assert panel.last[1].mean() == pytest.approx(3.0, abs=0.2)


def test_implot_colon_commands_average_a_range():
    from harwood.viewer._qt import application
    from harwood.viewer.implot_panel import ImplotPanel

    application()
    data = np.tile(np.arange(40.0)[:, None], (1, 60))
    panel = ImplotPanel(data, "test")

    panel.command.setText(":l 11 20")
    panel._run_command()
    assert "lines 11 to 20" in panel.last[2]
    assert panel.last[1].mean() == pytest.approx(np.arange(10, 20).mean())


def test_implot_says_what_is_wrong_with_a_bad_command():
    from harwood.viewer._qt import application
    from harwood.viewer.implot_panel import ImplotPanel

    application()
    panel = ImplotPanel(np.zeros((10, 10)), "test")
    panel.command.setText(":l banana")
    panel._run_command()
    assert "expected numbers" in panel.status.text()

    panel.command.setText(":z 1 2")
    panel._run_command()
    assert ":l for lines" in panel.status.text()
