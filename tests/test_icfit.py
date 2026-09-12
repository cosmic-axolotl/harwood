"""icfit: the shared fit, checked without a screen.

If these numbers are right, the overscan in ccdproc, the trace in aptrace and
the dispersion in identify are all fitting correctly, because they are the same
code.
"""

import numpy as np
import pytest

from harwood.core import icfit


def test_a_straight_line_is_recovered_exactly():
    x = np.arange(1.0, 101.0)
    result = icfit.fit(x, 3.0 + 0.5 * x, icfit.FitSpec(function="legendre", order=2))
    assert result.rms == pytest.approx(0.0, abs=1e-9)
    assert result(np.array([50.0]))[0] == pytest.approx(28.0)


def test_order_counts_terms_not_degree():
    """IRAF's `order` is the number of terms; numpy's `deg` is one less.

    Getting this backwards fits a straight line when the user asked for a
    constant, which looks plausible and is wrong.
    """
    x = np.linspace(0, 10, 50)
    flat = icfit.fit(x, 5.0 + 0.0 * x, icfit.FitSpec(order=1))
    assert flat.rms == pytest.approx(0.0, abs=1e-9)
    sloped = icfit.fit(x, 2.0 * x, icfit.FitSpec(order=1))
    assert sloped.rms > 1.0, "order 1 is a constant and cannot follow a slope"
    assert icfit.fit(x, 2.0 * x, icfit.FitSpec(order=2)).rms == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("function", icfit.FUNCTIONS)
def test_every_function_follows_a_smooth_curve(function):
    x = np.linspace(1, 200, 400)
    y = 100 + 20 * np.sin(x / 60)
    result = icfit.fit(x, y, icfit.FitSpec(function=function, order=6))
    assert result.rms < 1.0, f"{function} did not follow the curve"


def test_rejection_removes_the_outlier_and_says_how_many():
    x = np.arange(1.0, 61.0)
    y = 10.0 + 0.1 * x
    y[20] = 500.0
    result = icfit.fit(x, y, icfit.FitSpec(order=2, niterate=3,
                                           low_reject=3, high_reject=3))
    assert not result.mask[20]
    assert result.nrejected >= 1
    assert result.rms < 0.01


def test_grow_takes_the_neighbours_too():
    x = np.arange(1.0, 61.0)
    y = 10.0 + np.zeros_like(x)
    y[30] = 90.0
    tight = icfit.fit(x, y, icfit.FitSpec(order=1, niterate=2))
    wide = icfit.fit(x, y, icfit.FitSpec(order=1, niterate=2, grow=2.0))
    assert wide.nrejected > tight.nrejected


def test_sample_limits_the_fit_to_the_named_ranges():
    x = np.arange(1.0, 201.0)
    y = np.where(x < 100, 5.0, 500.0)
    result = icfit.fit(x, y, icfit.FitSpec(order=1, sample="1:99"))
    assert result(np.array([50.0]))[0] == pytest.approx(5.0)
    assert result.mask[150] == False          # outside the sample, never used


def test_sample_accepts_several_ranges():
    x = np.arange(1.0, 101.0)
    keep = icfit.parse_sample("1:10,90:100", x)
    assert keep.sum() == 21


def test_a_sample_that_selects_nothing_says_the_range_of_x():
    with pytest.raises(ValueError, match="1 to 100"):
        icfit.parse_sample("500:600", np.arange(1.0, 101.0))


def test_an_unreadable_sample_explains_the_syntax():
    with pytest.raises(ValueError, match="first:last"):
        icfit.parse_sample("1-10", np.arange(10.0))


def test_naverage_bins_and_negative_naverage_takes_the_median():
    x = np.arange(1.0, 101.0)
    y = np.full_like(x, 10.0)
    y[::10] = 1000.0                           # a spike every ten points
    binned = icfit.fit(x, y, icfit.FitSpec(order=1, naverage=10))
    median = icfit.fit(x, y, icfit.FitSpec(order=1, naverage=-10))
    assert median(np.array([50.0]))[0] == pytest.approx(10.0, abs=0.5)
    assert binned(np.array([50.0]))[0] > 50.0  # the mean is dragged by the spikes


def test_a_deleted_point_never_comes_back():
    x = np.arange(1.0, 41.0)
    y = 10.0 + 0.0 * x
    deleted = np.zeros(x.shape, dtype=bool)
    deleted[5] = True
    result = icfit.fit(x, y, icfit.FitSpec(order=1, niterate=3), deleted=deleted)
    assert result.deleted[5]
    assert not result.mask[5]
    assert result.nrejected == 0               # deleted is not the same as rejected


def test_too_high_an_order_says_what_to_do():
    x = np.arange(1.0, 5.0)
    result = icfit.fit(x, x, icfit.FitSpec(order=20))
    assert result.model is None
    assert "Lower the order" in result.message


def test_the_summary_reads_like_iraf_status_line():
    x = np.arange(1.0, 51.0)
    y = 2.0 * x
    text = icfit.fit(x, y, icfit.FitSpec(order=2)).summary()
    assert "legendre order 2" in text and "rms" in text


# ------------------------------------------------- the interactive half
# apply_command is shared by the keyboard, scripts and recipes, so the same
# string has to mean the same thing in all three.

def test_colon_commands_change_what_they_say_they_change():
    from harwood.viewer.icfit_panel import apply_command

    spec = icfit.FitSpec()
    assert apply_command(spec, "order 5").order == 5
    assert apply_command(spec, "function chebyshev").function == "chebyshev"
    assert apply_command(spec, "sample 20:500").sample == "20:500"
    assert apply_command(spec, "naverage -4").naverage == -4
    assert apply_command(spec, "niterate 3").niterate == 3
    assert apply_command(spec, "grow 1.5").grow == 1.5


def test_iraf_short_names_work_too():
    from harwood.viewer.icfit_panel import apply_command

    assert apply_command(icfit.FitSpec(), "low 2").low_reject == 2.0
    assert apply_command(icfit.FitSpec(), "high 4").high_reject == 4.0


def test_a_bad_colon_command_explains_itself():
    from harwood.viewer.icfit_panel import apply_command

    with pytest.raises(ValueError, match="whole number"):
        apply_command(icfit.FitSpec(), "order banana")
    with pytest.raises(ValueError, match="one of"):
        apply_command(icfit.FitSpec(), "function julia")
    with pytest.raises(ValueError, match="no parameter"):
        apply_command(icfit.FitSpec(), "colour red")


def test_applying_a_command_never_mutates_the_original():
    from harwood.viewer.icfit_panel import apply_command

    spec = icfit.FitSpec(order=2)
    apply_command(spec, "order 9")
    assert spec.order == 2
