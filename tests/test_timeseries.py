"""lombscargle and bls, held to the definition by hand and to periods put there beforehand."""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from harwood.core import catalog
from harwood.core.registry import get_task


def nights(rng, count=30, per_night=12):
    """Thirty nights with a gap by day, as a telescope has them."""
    times = [night + 0.1 + np.sort(rng.uniform(0, 0.3, per_night)) for night in range(count)]
    return np.concatenate(times)


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def curve(t, y, dy=None, name="c.txt"):
    columns = {"time": t, "mag": y}
    if dy is not None:
        columns["err"] = dy
    Table(columns).write(name, format="ascii.commented_header")
    return name


def test_the_power_is_the_fit_done_by_hand(work):
    """The generalised periodogram, by least squares at each frequency: 1 - chi2 / chi2 of the
    mean, the normalisation of Zechmeister & Kuerster."""
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 40, 80))
    dy = rng.uniform(0.01, 0.05, 80)
    y = 15 + 0.2 * np.sin(2 * np.pi * t / 3.3) + rng.normal(0, dy)
    done = get_task("lombscargle").run(input=curve(t, y, dy), time="time", value="mag",
                                       error="err", output="p.ecsv")
    table = catalog.read("p.ecsv")
    weights = 1 / dy ** 2
    mean = (weights * y).sum() / weights.sum()
    whole = (weights * (y - mean) ** 2).sum()
    for at in np.linspace(0, len(table) - 1, 25).astype(int):
        frequency = float(table["frequency"][at])
        design = np.column_stack([np.ones_like(t), np.sin(2 * np.pi * frequency * t),
                                  np.cos(2 * np.pi * frequency * t)])
        fit = np.linalg.lstsq(design * np.sqrt(weights)[:, None], y * np.sqrt(weights),
                              rcond=None)
        left = (weights * (y - design @ fit[0]) ** 2).sum()
        assert float(table["power"][at]) == pytest.approx(1 - left / whole, abs=1e-8)
    assert done.data["period"] == pytest.approx(3.3, rel=0.01)
    assert done.data["fap"] < 1e-10


def test_a_period_in_gapped_nights_with_its_alias_named(work):
    rng = np.random.default_rng(2)
    t = nights(rng)
    y = 0.15 * np.sin(2 * np.pi * t / 0.37) + rng.normal(0, 0.02, t.size)
    done = get_task("lombscargle").run(input=curve(t, y), time="1", value="2", peaks=6)
    assert done.data["period"] == pytest.approx(0.37, rel=0.002)
    best = done.data["rows"][0]
    assert best["amplitude"] == pytest.approx(0.15, rel=0.05)
    named = [row for row in done.data["rows"] if row["alias"]]
    assert named, "1 cycle per day from the best one"
    assert any(abs(abs(row["frequency"] - best["frequency"]) - 1.0) < 0.05 for row in named)
    assert any("cannot tell" in warning.message for warning in done.warnings)


def test_noise_is_said_to_be_noise(work):
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 30, 100))
    done = get_task("lombscargle").run(input=curve(t, rng.normal(0, 1, 100)))
    assert done.data["fap"] > 0.01
    assert any("not a detection" in warning.fix for warning in done.warnings)


def test_a_curve_that_is_no_sinusoid_wants_harmonics(work):
    rng = np.random.default_rng(4)
    t = nights(rng, 25, 20)
    phase = np.mod(t / 0.55, 1.0)
    y = 0.5 * np.where(phase < 0.15, phase / 0.15, (1 - phase) / 0.85)
    y = y + rng.normal(0, 0.01, t.size)
    one = get_task("lombscargle").run(input=curve(t, y), minperiod=0.1, maxperiod=2.0)
    many = get_task("lombscargle").run(input="c.txt", harmonics=4, minperiod=0.1,
                                       maxperiod=2.0)
    assert many.data["period"] == pytest.approx(0.55, rel=0.002)
    assert many.data["rows"][0]["power"] > one.data["rows"][0]["power"]
    assert np.isnan(many.data["fap"])


def test_a_period_longer_than_the_data_is_said(work):
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 3, 60))
    y = 0.01 * t ** 2 + rng.normal(0, 0.001, 60)
    done = get_task("lombscargle").run(input=curve(t, y))
    assert any("more than half the span" in warning.message for warning in done.warnings)


def test_what_it_writes(work):
    rng = np.random.default_rng(6)
    t = np.sort(rng.uniform(0, 30, 200))
    y = np.sin(2 * np.pi * t / 2.0) + rng.normal(0, 0.1, 200)
    get_task("lombscargle").run(input=curve(t, y), output="p.ecsv", graph="p.svg",
                                folded="f.ecsv")
    folded = catalog.read("f.ecsv")
    assert folded.meta["period"] == pytest.approx(2.0, rel=0.01)
    assert np.all(np.diff(np.asarray(folded["phase"], float)) >= 0)
    assert Path("p.svg").read_text().lstrip().startswith("<svg")


def test_indef_is_left_out_and_said(work):
    rows = "\n".join(f"{t:.3f} {np.sin(t):.4f}" for t in np.linspace(0, 50, 100))
    with open("c.txt", "w") as file:
        file.write(rows + "\n51.0 INDEF\n52.0 INDEF\n")
    done = get_task("lombscargle").run(input="c.txt")
    assert any("2 measures" in warning.message for warning in done.warnings)
    assert done.data["period"] == pytest.approx(2 * np.pi, rel=0.01)


@pytest.mark.parametrize(
    "options, match",
    [(dict(time="9"), "column 9"), (dict(value="nope"), "no column 'nope'"),
     (dict(minperiod=5.0, maxperiod=1.0), "no periods")],
)
def test_it_says_what_it_needs(work, options, match):
    curve(np.arange(20.0), np.arange(20.0))
    with pytest.raises(ValueError, match=match):
        get_task("lombscargle").run(input="c.txt", **options)


def test_errors_of_zero_and_too_few_points_are_refused(work):
    curve(np.arange(8.0), np.ones(8), np.zeros(8))
    with pytest.raises(ValueError, match="not above zero"):
        get_task("lombscargle").run(input="c.txt", error="3")
    curve(np.arange(3.0), np.arange(3.0), name="few.txt")
    with pytest.raises(ValueError, match="3 measures"):
        get_task("lombscargle").run(input="few.txt")


# ------------------------------------------------------------------------ box least squares


def transit(t, period, epoch, duration, depth, odd=None):
    phase = np.mod(t - epoch + period / 2, period) - period / 2
    inside = np.abs(phase) < duration / 2
    y = np.ones_like(t)
    y[inside] -= depth
    if odd is not None:
        number = np.round((t - epoch) / period).astype(int)
        y[inside & (number % 2 == 1)] = 1 - odd
    return y


def test_a_transit_put_there_is_found(work):
    rng = np.random.default_rng(7)
    t = nights(rng, 40, 30)
    y = transit(t, 2.31, 0.2, 0.12, 0.01)
    y = y + rng.normal(0, 0.002, t.size)
    done = get_task("bls").run(input=curve(t, y, np.full(t.size, 0.002)), error="3",
                               duration="0.06,0.12,0.24", minperiod=0.5, maxperiod=8.0)
    assert done.data["period"] == pytest.approx(2.31, rel=0.002)
    assert done.data["duration"] == pytest.approx(0.12)
    assert done.data["depth"] == pytest.approx(0.01, abs=4 * done.data["depth_err"])
    offset = np.mod(done.data["epoch"] - 0.2 + 2.31 / 2, 2.31) - 2.31 / 2
    assert abs(offset) < 0.03
    assert done.data["odd_even_sigma"] < 3
    assert not any("binary" in warning.message for warning in done.warnings)


def test_magnitudes_are_a_rise(work):
    rng = np.random.default_rng(8)
    t = nights(rng, 40, 30)
    flux = transit(t, 2.31, 0.2, 0.12, 0.02) + rng.normal(0, 0.002, t.size)
    done = get_task("bls").run(input=curve(t, 15 - 2.5 * np.log10(flux)), magnitudes=True,
                               duration="0.12", minperiod=0.5, maxperiod=8.0)
    assert done.data["period"] == pytest.approx(2.31, rel=0.002)
    assert done.data["depth"] == pytest.approx(0.02, rel=0.1)


def test_a_binary_at_half_its_period_is_told(work):
    rng = np.random.default_rng(9)
    t = nights(rng, 60, 30)
    y = transit(t, 1.3, 0.2, 0.1, 0.03, odd=0.02) + rng.normal(0, 0.002, t.size)
    done = get_task("bls").run(input=curve(t, y, np.full(t.size, 0.002)), error="3",
                               duration="0.1", minperiod=0.7, maxperiod=2.0)
    assert done.data["period"] == pytest.approx(1.3, rel=0.002)
    assert done.data["odd_even_sigma"] > 3
    told = [warning for warning in done.warnings if "odd and even" in warning.message]
    assert told and "--minperiod" in told[0].fix


@pytest.mark.parametrize(
    "options, match",
    [(dict(duration="a"), "--duration"), (dict(duration="-1"), "duration -1"),
     (dict(duration="2", minperiod=1.0), "--minperiod 1"),
     (dict(duration="0.1", minperiod=9.0, maxperiod=1.0), "no periods")],
)
def test_bls_says_what_it_needs(work, options, match):
    curve(np.arange(40.0), np.ones(40))
    with pytest.raises(ValueError, match=match):
        get_task("bls").run(input="c.txt", **options)
