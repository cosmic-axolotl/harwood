"""Finding spectra on a frame, following them, and adding them up.

A long-slit or fibre frame holds one or more spectra running along the
dispersion axis. Three things have to happen before there is a one-dimensional
spectrum, and IRAF gives each of them a task:

``apfind``    where the spectra are, across the slit
``aptrace``   where each one sits at every point *along* the dispersion, which
              is never a straight line -- the camera has distortion and the
              spectrograph flexes as the telescope moves
``apsum``     add up the pixels inside the aperture, taking the background off

The arithmetic is here and the interaction is not, so the numbers a script gets
and the numbers a person gets by hand are produced by the same code.

Positions across the slit are 1-based, as everywhere else a user can see.
"""

from __future__ import annotations

import numpy as np

from . import icfit


def profile(data: np.ndarray, dispaxis: int = 2, centre: int | None = None,
            nsum: int | None = None) -> np.ndarray:
    """The spatial profile: the frame collapsed along the dispersion.

    Adding along the dispersion is what makes a faint spectrum findable -- a
    thousand rows of a few counts each become a peak you can centroid. `nsum`
    limits how much is added, which matters when the spectrum is tilted enough
    that adding the whole frame smears it.
    """
    data = np.asarray(data, dtype=float)
    along = 0 if dispaxis == 2 else 1
    length = data.shape[along]
    if nsum is None or nsum >= length:
        section = data
    else:
        middle = length // 2 if centre is None else int(centre) - 1
        low = max(0, min(middle - nsum // 2, length - nsum))
        section = data[low:low + nsum, :] if along == 0 else data[:, low:low + nsum]
    return section.sum(axis=along)


def find(spatial: np.ndarray, nfind: int = 1, minsep: float = 5.0,
         threshold: float = 0.0, width: float = 5.0) -> list[dict]:
    """The brightest `nfind` peaks in the spatial profile, centred.

    Returns dictionaries rather than Aperture objects so this stays free of the
    format: an aperture is a thing that gets written to a file, a peak is a
    measurement.
    """
    values = np.asarray(spatial, dtype=float)
    if values.size < 3:
        return []

    background = np.median(values)
    noise = 1.4826 * np.median(np.abs(values - background))
    limit = background + threshold * (noise if noise > 0 else 1.0)

    half = max(1, int(round(width / 2)))
    candidates = []
    for index in range(1, values.size - 1):
        if values[index] < limit:
            continue
        window = values[max(0, index - half):index + half + 1]
        if values[index] < window.max():
            continue
        low, high = max(0, index - half), min(values.size, index + half + 1)
        weights = np.clip(values[low:high] - background, 0, None)
        if weights.sum() <= 0:
            continue
        positions = np.arange(low, high)
        centre = float((weights * positions).sum() / weights.sum())
        candidates.append({"centre": centre + 1.0,            # 1-based
                           "peak": float(values[index] - background)})

    candidates.sort(key=lambda item: -item["peak"])
    kept: list[dict] = []
    for candidate in candidates:
        if all(abs(candidate["centre"] - other["centre"]) >= minsep
               for other in kept):
            kept.append(candidate)
        if len(kept) >= nfind:
            break
    kept.sort(key=lambda item: item["centre"])
    return kept


def edges(spatial: np.ndarray, centre: float, fraction: float = 0.05,
          limit: float = 30.0) -> tuple[float, float]:
    """Where the profile falls to `fraction` of its peak, either side of centre.

    IRAF's `apresize` does this. It is a better default than a fixed width
    because a fibre and a long slit want very different apertures, and the
    profile knows which it is looking at.
    """
    values = np.asarray(spatial, dtype=float)
    background = float(np.median(values))
    index = int(round(centre)) - 1
    index = max(0, min(index, values.size - 1))
    peak = values[index] - background
    if peak <= 0:
        return -limit / 2, limit / 2

    target = background + fraction * peak
    low = index
    while low > 0 and values[low] > target and index - low < limit:
        low -= 1
    high = index
    while high < values.size - 1 and values[high] > target and high - index < limit:
        high += 1
    return float(low + 1 - centre), float(high + 1 - centre)


def trace(data: np.ndarray, centre: float, dispaxis: int = 2, step: int = 10,
          nsum: int = 10, width: float = 5.0,
          spec: icfit.FitSpec | None = None) -> icfit.Fit:
    """Follow the spectrum's position along the dispersion, and fit it.

    Walks outwards from the middle in steps, re-centring each time on the
    previous position. Walking outwards rather than from one end matters: the
    spectrum is brightest and best defined in the middle, and a trace that
    loses its way at a faint end has already found the bright part correctly.

    The fit is an icfit, so `:order 3` and `d` to drop a bad point mean here
    exactly what they mean in the overscan.
    """
    data = np.asarray(data, dtype=float)
    spec = spec or icfit.FitSpec(function="legendre", order=3, niterate=2,
                                 low_reject=3.0, high_reject=3.0)
    length = data.shape[0] if dispaxis == 2 else data.shape[1]
    middle = length // 2

    positions, centres = [], []
    for direction in (1, -1):
        current = centre
        for offset in range(0, length // 2, max(1, step)):
            index = middle + direction * offset
            if not 0 <= index < length:
                break
            cut = _cut(data, index, nsum, dispaxis)
            found = _recentre(cut, current, width)
            if found is None:
                break                     # lost it; the fit extrapolates the rest
            current = found
            positions.append(index + 1.0)
            centres.append(found)

    if len(positions) < 4:
        return icfit.Fit(spec, np.array(positions), np.array(centres),
                         np.zeros(len(positions), bool),
                         np.zeros(len(positions), bool), None,
                         "the spectrum could not be followed; is the dispersion "
                         "axis right, and is there a spectrum at that position?")

    order = np.argsort(positions)
    return icfit.fit(np.array(positions)[order], np.array(centres)[order], spec)


def _cut(data, index: int, nsum: int, dispaxis: int) -> np.ndarray:
    """A few rows or columns added together, around one point of the dispersion."""
    length = data.shape[0] if dispaxis == 2 else data.shape[1]
    low = max(0, min(index - nsum // 2, length - max(1, nsum)))
    high = low + max(1, nsum)
    return (data[low:high, :].sum(axis=0) if dispaxis == 2
            else data[:, low:high].sum(axis=1))


def _recentre(cut: np.ndarray, centre: float, width: float) -> float | None:
    """Centroid within a box, or None when there is nothing left to centre on."""
    half = max(1, int(round(width)))
    index = int(round(centre)) - 1
    low, high = max(0, index - half), min(cut.size, index + half + 1)
    if high - low < 3:
        return None
    window = cut[low:high]
    weights = np.clip(window - np.median(cut), 0, None)
    if weights.sum() <= 0:
        return None
    positions = np.arange(low, high)
    return float((weights * positions).sum() / weights.sum()) + 1.0


def extract(data: np.ndarray, centre_at, low: float, high: float,
            dispaxis: int = 2, background: tuple[float, float] | None = None,
            bwidth: float = 10.0, gain: float = 1.0, rdnoise: float = 0.0,
            weights: str = "none") -> dict:
    """Add up the aperture at every point along the dispersion.

    `centre_at` is a callable giving the aperture centre for a 1-based position
    along the dispersion -- the trace. `low` and `high` are the aperture edges
    relative to that centre.

    `weights='variance'` is optimal extraction (Horne 1986): each pixel is
    weighted by the profile and divided by its variance, which recovers
    signal-to-noise a plain sum throws away on a faint spectrum -- at the cost
    of assuming the profile is what the bright part of the spectrum says it is.
    """
    data = np.asarray(data, dtype=float)
    length = data.shape[0] if dispaxis == 2 else data.shape[1]
    across = data.shape[1] if dispaxis == 2 else data.shape[0]
    positions = np.arange(1.0, length + 1.0)
    centres = np.asarray(centre_at(positions), dtype=float)

    flux = np.zeros(length)
    raw = np.zeros(length)
    sky = np.zeros(length)
    error = np.zeros(length)
    axis = np.arange(across) + 1.0

    for index in range(length):
        row = data[index, :] if dispaxis == 2 else data[:, index]
        centre = centres[index]
        inside = (axis >= centre + low) & (axis <= centre + high)
        if not inside.any():
            continue

        level, scatter = _background(row, axis, centre, low, high, background,
                                     bwidth)
        sky[index] = level
        clean = row - level
        raw[index] = float(row[inside].sum())

        if weights == "variance":
            flux[index], error[index] = _optimal(clean[inside], level, gain,
                                                 rdnoise)
        else:
            total = float(clean[inside].sum())
            counts = max(total, 0.0) * gain
            area = float(inside.sum())
            variance = counts + area * (level * gain + rdnoise ** 2)
            flux[index] = total
            error[index] = float(np.sqrt(variance) / gain)
        if scatter is not None:
            error[index] = float(np.hypot(error[index],
                                          scatter * np.sqrt(float(inside.sum()))))

    return {"pixel": positions, "flux": flux, "error": error, "sky": sky,
            "raw": raw, "centre": centres}


def _background(row, axis, centre, low, high, background, bwidth):
    """The sky level under the aperture, and the uncertainty in it."""
    if background is None:
        return 0.0, None
    inner, outer = background
    left = (axis >= centre + low - outer) & (axis <= centre + low - inner)
    right = (axis >= centre + high + inner) & (axis <= centre + high + outer)
    chosen = row[left | right]
    if chosen.size < 3:
        return 0.0, None
    level = float(np.median(chosen))
    scatter = float(np.std(chosen, ddof=1)) / np.sqrt(chosen.size)
    return level, scatter


def _optimal(values, sky_level, gain, rdnoise):
    """Horne 1986, with the profile taken from the data at this point.

    The profile is normalised so the weights sum to one; each pixel then
    contributes in proportion to how much signal it holds and in inverse
    proportion to its variance. On a bright spectrum it agrees with a plain
    sum; on a faint one it is worth a magnitude.
    """
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        return float(total), float(np.sqrt(abs(total) / max(gain, 1e-9)))

    shape = np.clip(values, 0, None)
    shape = shape / shape.sum()
    variance = (np.clip(values, 0, None) + sky_level) / max(gain, 1e-9) \
        + (rdnoise / max(gain, 1e-9)) ** 2
    variance = np.where(variance > 0, variance, 1e-9)

    weight = shape ** 2 / variance
    flux = float((shape * values / variance).sum() / weight.sum())
    error = float(np.sqrt(1.0 / weight.sum()))
    return flux, error
