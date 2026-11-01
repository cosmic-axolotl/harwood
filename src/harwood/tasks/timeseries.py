"""Periods from a light curve: the Lomb-Scargle periodogram (VanderPlas 2018, Zechmeister &
Kuerster 2009) and box least squares (Kovacs, Zucker & Mazeh 2002). IRAF has pdm, phase
dispersion minimization, from before either. The arithmetic is astropy.timeseries, which
everyone uses; what is here is reading a light curve from a table of photometry, and saying
what a peak is worth: whether the sampling makes it, whether there is a longer period it could
be half of.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "lomb": "A sinusoid is fitted at each frequency, at the times there are, with each point "
            "weighted by its error, and the power is how much of the scatter it explains. It "
            "needs no even sampling, which a night at a telescope never gives.",
    "bls": "A box is fitted at each period, duration and phase: a transit or an eclipse is "
           "flat and short, and a sinusoid spreads its power over many harmonics.",
    "alias": "A peak can be the sampling's and not the star's: a nightly gap puts a false "
             "peak one cycle per day from the true one, and the data cannot tell which.",
    "odd": "An eclipsing binary of equal stars looks like a transit at half its period. If "
           "the odd and even dips differ, the period is twice this.",
}


class CurveParams(TaskParams):
    input: Path = Field(..., description="The light curve, a table.",
                        json_schema_extra={"positional": True})
    time: str = Field("1", description="The column of the times: a name, or its number from "
                                       "1.")
    value: str = Field("2", description="The column of the measures.")
    error: str = Field("", description="The column of their errors. Left out, all the same.")
    output: Path | None = Field(None, description="The periodogram, a table.")
    graph: Path | None = Field(None, description="The periodogram, as an SVG.")
    folded: Path | None = Field(None, description="The measures folded at the best period, a "
                                                  "table of phase and value.")
    minperiod: float | None = Field(None, gt=0, description="The shortest period to try.")
    maxperiod: float | None = Field(None, gt=0, description="The longest. Left out, the whole "
                                                            "span of the times.")
    peaks: int = Field(5, ge=1, description="How many peaks to list.")


class LombscargleParams(CurveParams):
    harmonics: int = Field(1, ge=1, description="Terms of the series at each frequency: more "
                                                "for a curve that is no sinusoid, as an RR "
                                                "Lyrae.")
    oversample: int = Field(10, ge=1, description="Frequencies to each width of a peak.")
    fap: float = Field(0.01, gt=0, lt=1, description="Below this false alarm probability a "
                                                     "peak is called a detection.")


class BlsParams(CurveParams):
    duration: str = Field(..., description="The durations of the dip to try, in the times' "
                                           "units, with commas.")
    magnitudes: bool = Field(False, description="The measures are magnitudes: a dip is a "
                                                "rise.")
    oversample: int = Field(10, ge=1, description="Phases to each duration.")
    transits: int = Field(3, ge=1, description="The fewest dips a period has to have in the "
                                               "span.")


def _column(table, which: str, option: str):
    import numpy as np

    if which.isdigit():
        number = int(which)
        if not 1 <= number <= len(table.colnames):
            advice.fail(f"{option} is column {number} of a table with {len(table.colnames)}",
                        f"its columns are {', '.join(table.colnames)}")
        which = table.colnames[number - 1]
    elif which not in table.colnames:
        advice.fail(f"no column '{which}' for {option}",
                    f"its columns are {', '.join(table.colnames)}, or give a number from 1")
    column = table[which]
    if np.ma.isMaskedArray(column):
        column = column.filled(np.nan)
    values = []
    for value in column:
        # INDEF, as IRAF's tables have it
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.array(values)


def read_curve(params: CurveParams):
    """Times in order, with what has no measure left out. How many were, as well."""
    import numpy as np

    from ..core import catalog

    table = catalog.read(params.input)
    t = _column(table, params.time, "--time")
    y = _column(table, params.value, "--value")
    dy = _column(table, params.error, "--error") if params.error else None
    good = np.isfinite(t) & np.isfinite(y)
    if dy is not None:
        good &= np.isfinite(dy)
        if (dy[good] <= 0).any():
            advice.fail(f"{int((dy[good] <= 0).sum())} errors in {params.input} are not above "
                        "zero", "a point with no error has infinite weight; leave out "
                                "--error, or remove them")
    order = np.argsort(t[good])
    if order.size < 5:
        advice.fail(f"{order.size} measures in {params.input}",
                    "a period needs more than that")
    t, y = t[good][order], y[good][order]
    dy = None if dy is None else dy[good][order]
    if t[-1] <= t[0]:
        advice.fail(f"the times in {params.input} are all the same", "check --time")
    return t, y, dy, int((~good).sum())


def distinct_peaks(frequency, power, count: int, width: float):
    """The indices of the highest local maxima, no two within a peak's width."""
    import numpy as np

    # The ends too: a trend climbs to the lowest frequency and has no top before it.
    higher = np.r_[True, power[1:] > power[:-1]] & np.r_[power[:-1] >= power[1:], True]
    inner = np.flatnonzero(higher)
    order = inner[np.argsort(-power[inner])]
    kept: list[int] = []
    for at in order:
        if all(abs(frequency[at] - frequency[other]) > width for other in kept):
            kept.append(int(at))
            if len(kept) == count:
                break
    return kept


def window_peaks(t, width: float, top: float, oversample: int = 10):
    """The frequencies where the sampling alone has power: LS of a constant, with no mean."""
    import numpy as np
    from astropy.timeseries import LombScargle

    frequency = np.arange(width / 2, top, width / oversample)
    power = LombScargle(t, np.ones_like(t), fit_mean=False, center_data=False).power(
        frequency, method="fast", assume_regular_frequency=True, normalization="psd")
    power = power * 2 / t.size
    found = distinct_peaks(frequency, power, 5, 2 * width)
    return [(float(frequency[at]), float(power[at])) for at in found if power[at] > 0.3]


def alias_of(found, best, windows, width: float) -> str:
    """Which of the sampling's peaks puts a peak where it is, against the best one."""
    for spacing, _ in windows:
        for side in (best - spacing, best + spacing, spacing - best):
            if abs(found - side) < width:
                return f"{spacing:.5g}"
    return ""


def _write_outputs(params, frequency, power, t, y, best_period) -> list[Path]:
    import numpy as np
    from astropy.table import Table

    from ..core import catalog, graph

    outputs = []
    if params.output:
        table = Table({"frequency": frequency, "period": 1.0 / frequency, "power": power})
        outputs.append(catalog.write(table, params.output))
    if params.graph:
        plot = graph.Graph(f"{params.input}", "frequency", "power")
        outputs.append(plot.line(frequency, power).write(params.graph))
    if params.folded:
        phase = np.mod(t - t[0], best_period) / best_period
        order = np.argsort(phase)
        table = Table({"phase": phase[order], "value": y[order]})
        outputs.append(catalog.write(table, params.folded, period=float(best_period),
                                     epoch=float(t[0])))
    return outputs


def _range(params, span: float, shortest_gap: float) -> tuple[float, float]:
    low = 1.0 / (params.maxperiod or span)
    high = 1.0 / params.minperiod if params.minperiod else 0.5 / shortest_gap
    if high <= low:
        advice.fail(f"no periods between {1 / high:g} and {1 / low:g}",
                    "--minperiod has to be below --maxperiod")
    return low, high


@register
class Lombscargle(Task[LombscargleParams]):
    """The Lomb-Scargle periodogram, with how likely each peak is to be noise and whether
    the sampling could have made it."""

    name = "lombscargle"
    package = "harwood"
    description = "Find the periods in a light curve, and say what each peak is worth."
    Params = LombscargleParams

    def plan(self, params: LombscargleParams) -> list[Step]:
        return [Step("periodogram", f"{params.input}, {params.harmonics} term"
                     + ("s" if params.harmonics > 1 else ""), why=WHY["lomb"]),
                Step("aliases", "against the sampling's own peaks", why=WHY["alias"])]

    def execute(self, params: LombscargleParams) -> TaskResult:
        import numpy as np
        from astropy.timeseries import LombScargle

        t, y, dy, dropped = read_curve(params)
        span = float(t[-1] - t[0])
        gap = float(np.median(np.diff(t)))
        if gap <= 0:
            gap = span / t.size
        low, high = _range(params, span, gap)
        model = LombScargle(t, y, dy, nterms=params.harmonics)
        width = 1.0 / span
        step = width / params.oversample
        count = int((high - low) / step) + 1
        frequency = low + step * np.arange(count)
        power = model.power(frequency)
        found = distinct_peaks(frequency, power, params.peaks, width)
        if not found:
            found = [int(np.argmax(power))]
        faps = _false_alarms(model, power, found, params, low, high)
        windows = window_peaks(t, width, high)
        best = float(frequency[found[0]])
        rows = []
        for at, fap in zip(found, faps, strict=True):
            value = float(frequency[at])
            row = {"period": float(1.0 / value), "frequency": value, "power": float(power[at]),
                   "fap": fap, "amplitude": _amplitude(model, value, params.harmonics),
                   "alias": ("" if at == found[0] else alias_of(value, best, windows, width))}
            rows.append(row)
        outputs = _write_outputs(params, frequency, power, t, y, 1.0 / best)
        steps = self.plan(params)
        steps[0].measured = {"measures": t.size, "span": f"{span:.6g}",
                             "frequencies": frequency.size,
                             "best period": f"{1 / best:.6g}",
                             "false alarm probability": f"{faps[0]:.2g}"}
        steps[1].measured = {"sampling's peaks": ", ".join(f"{f:.5g}" for f, _ in windows)
                             or "none"}
        return TaskResult(outputs=outputs, steps=steps,
                          warnings=_warned(params, rows, faps, t, span, gap, dropped, windows),
                          data={"rows": rows, "period": float(1 / best), "fap": faps[0]})


def _false_alarms(model, power, found, params, low: float, high: float) -> list[float]:
    """Baluev's bound, which is for one term. With more, the series has no such thing, and the
    bootstrap, which does, would take a thousand periodograms."""
    if params.harmonics > 1:
        return [float("nan")] * len(found)
    return [float(model.false_alarm_probability(
        power[at], minimum_frequency=low, maximum_frequency=high,
        samples_per_peak=params.oversample)) for at in found]


def _amplitude(model, frequency: float, harmonics: int) -> float:
    """The half range of the fitted series, over a phase."""
    import numpy as np

    phase = np.linspace(0, 1 / frequency, 200 * harmonics, endpoint=False)
    fitted = model.model(phase, frequency)
    return float((fitted.max() - fitted.min()) / 2)


def _warned(params, rows, faps, t, span, gap, dropped, windows) -> list:
    import numpy as np

    warnings = []
    best = rows[0]
    if dropped:
        warnings.append(advice.note(f"{dropped} measures have no value and were left out.",
                                    "a star lost in some frames is often what that is."))
    if np.isfinite(faps[0]) and faps[0] > params.fap:
        warnings.append(advice.warn(
            f"the highest peak, at {best['period']:.5g}, has a false alarm probability of "
            f"{faps[0]:.2g}: noise makes one that high.",
            "this is not a detection; more measures, or a longer span, are what would help."))
    if span / best["period"] < 2:
        warnings.append(advice.warn(
            f"the best period, {best['period']:.5g}, is more than half the span, {span:.5g}.",
            "it may be a trend; fewer than two cycles cannot show that it repeats."))
    aliased = [row for row in rows[1:] if row["alias"] and row["power"] > 0.5 * best["power"]]
    if aliased:
        warnings.append(advice.warn(
            f"the peak at {aliased[0]['period']:.5g} is {aliased[0]['alias']} from the "
            "best in frequency, which is a peak of the sampling itself: the data cannot tell "
            "which of the two is the star's.",
            "folding at each with --folded may show one is wrong; otherwise it takes other "
            "nights, or another site."))
    return warnings


@register
class Bls(Task[BlsParams]):
    """Box least squares: the period, depth and duration of a transit or an eclipse, and
    whether it is half of a longer period."""

    name = "bls"
    package = "harwood"
    description = "Find a transit or an eclipse in a light curve, its period and depth."
    Params = BlsParams

    def plan(self, params: BlsParams) -> list[Step]:
        return [Step("periodogram", f"{params.input}, durations {params.duration}",
                     why=WHY["bls"]),
                Step("odd and even", "the dips of each, against each other", why=WHY["odd"])]

    def execute(self, params: BlsParams) -> TaskResult:
        import numpy as np
        from astropy.timeseries import BoxLeastSquares

        t, y, dy, dropped = read_curve(params)
        if params.magnitudes:
            level = float(np.median(y))
            flux = 10.0 ** (-0.4 * (y - level))
            dy = None if dy is None else dy * flux * 0.4 * np.log(10.0)
            y = flux
        try:
            durations = sorted(float(word) for word in params.duration.split(","))
        except ValueError:
            advice.fail(f"--duration {params.duration}", "give numbers, with commas")
        if durations[0] <= 0:
            advice.fail(f"a dip of duration {durations[0]:g}", "give durations above zero")
        span = float(t[-1] - t[0])
        low = params.minperiod or 2.0 * durations[-1]
        if low <= durations[-1]:
            advice.fail(f"--minperiod {low:g} and a dip of duration {durations[-1]:g}",
                        "a period has to be longer than its dip")
        top = params.maxperiod or span / max(params.transits - 1, 1)
        if top <= low:
            advice.fail(f"no periods between {low:g} and {top:g}",
                        f"a span of {span:g} with --transits {params.transits} reaches that "
                        "far: lower one, or lower --minperiod")
        model = BoxLeastSquares(t, y, dy)
        periods = model.autoperiod(np.array(durations), minimum_period=low,
                                   maximum_period=top, minimum_n_transit=params.transits)
        found = model.power(periods, np.array(durations), oversample=params.oversample,
                            objective="snr" if dy is not None else "likelihood")
        power = np.asarray(found.power)
        at = int(np.argmax(power))
        period = float(found.period[at])
        duration = float(found.duration[at])
        epoch = float(found.transit_time[at])
        stats = model.compute_stats(period, duration, epoch)
        frequency = 1.0 / periods[::-1]
        peaks = distinct_peaks(frequency, power[::-1], params.peaks, 1.0 / span)
        rows = [{"period": float(1 / frequency[i]), "power": float(power[::-1][i])}
                for i in peaks]
        depth = float(stats["depth"][0])
        depth_err = float(stats["depth"][1])
        odd, odd_err = (float(v) for v in stats["depth_odd"])
        even, even_err = (float(v) for v in stats["depth_even"])
        sigma = abs(odd - even) / max(np.hypot(odd_err, even_err), 1e-30)
        seen = int(np.count_nonzero(np.asarray(stats["per_transit_count"]) > 0))
        outputs = _write_outputs(params, frequency, power[::-1], t, y, period)
        steps = self.plan(params)
        steps[0].measured = {"measures": t.size, "best period": f"{period:.6g}",
                             "depth": f"{depth:.4g} +- {depth_err:.2g}",
                             "duration": f"{duration:.4g}", "epoch": f"{epoch:.8g}",
                             "dips seen": seen}
        steps[1].measured = {"odd": f"{odd:.4g} +- {odd_err:.2g}",
                             "even": f"{even:.4g} +- {even_err:.2g}",
                             "they differ by": f"{sigma:.1f} sigma"}
        warnings = []
        if dropped:
            warnings.append(advice.note(f"{dropped} measures have no value and were left out.",
                                        "a star lost in some frames is often what that is."))
        if sigma > 3:
            warnings.append(advice.warn(
                f"the odd and even dips differ by {sigma:.1f} sigma ({odd:.4g} and "
                f"{even:.4g}).",
                f"this is an eclipsing binary, and the period is {2 * period:.6g}: try "
                f"--minperiod {1.5 * period:.6g}."))
        if seen < params.transits:
            warnings.append(advice.warn(
                f"only {seen} dips of this period have a measure in them.",
                "a period from so few is not established; it takes more of the span."))
        return TaskResult(outputs=outputs, steps=steps, warnings=warnings,
                          data={"rows": rows, "period": period, "duration": duration,
                                "epoch": epoch, "depth": depth, "depth_err": depth_err,
                                "odd_even_sigma": float(sigma), "dips": seen})
