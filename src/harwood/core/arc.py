"""Arc lamps: finding the lines, and working out what wavelength each one is.

This is the half of `identify` that is arithmetic rather than interaction. It
finds emission peaks in a one-dimensional arc spectrum, matches them against a
line list, and hands the pairs to `icfit` for the dispersion solution.

The matching problem is the one `xyxymatch` solves, in one dimension: a list of
positions and a list of known values related by a transformation nobody wrote
down. IRAF makes it interactive -- mark five lines by hand, fit, then let the
task find the rest -- because with a dense list like thorium's three thousand
entries a blind search is ambiguous. harwood does both: a search over plausible
(crval, cdelt) when you can say roughly where the spectrum sits, and marking by
hand in the panel when you cannot.

Wavelengths are air wavelengths in Angstroms, as every IRAF line list is,
unless the file says otherwise in its `# units` line.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Where IRAF keeps its line lists, when IRAF is installed alongside.
IRAF_LINELISTS = (
    # Lower case on purpose: `iraf` is the name IRAF itself sets.
    Path(os.environ.get("iraf", "/usr/lib/iraf")) / "noao" / "lib" / "linelists",  # noqa: SIM112
    Path("/usr/share/iraf/noao/lib/linelists"),
)


@dataclass
class Peak:
    """One emission line found in the arc."""

    pixel: float          #: 1-based, along the dispersion axis
    flux: float
    width: float = float("nan")


@dataclass
class LineMatch:
    """Observed peaks paired with catalogue wavelengths."""

    pixel: np.ndarray
    wavelength: np.ndarray
    crval: float = float("nan")
    cdelt: float = float("nan")
    score: float = 0.0
    message: str = ""

    def __len__(self) -> int:
        return int(self.pixel.size)

    def __bool__(self) -> bool:
        return bool(len(self) >= 2)


def resolve_linelist(name: str | Path) -> Path:
    """Find a line list, accepting IRAF's ``linelists$thorium.dat`` syntax.

    Section 4 wants the calibration data shipped with harwood; until the
    licences of IRAF's own lists have been checked, an installed IRAF is used
    when it is there, which is the common case for someone migrating.
    """
    text = str(name)
    if "$" in text:
        prefix, _, filename = text.partition("$")
        if prefix != "linelists":
            raise ValueError(f"unknown IRAF directory prefix '{prefix}$'")
        for directory in IRAF_LINELISTS:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"'{text}' needs IRAF's line lists. Looked in "
            f"{', '.join(str(d) for d in IRAF_LINELISTS)}. Give a path instead, "
            "or install iraf-noao."
        )
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"line list '{path}' does not exist")
    return path


def read_linelist(name: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read an IRAF line list: one wavelength per line, with an optional label."""
    path = resolve_linelist(name)
    wavelengths, labels = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        try:
            wavelengths.append(float(parts[0]))
        except ValueError:
            continue
        labels.append(parts[1].strip() if len(parts) > 1 else "")
    if not wavelengths:
        raise ValueError(f"'{path}' holds no wavelengths")
    order = np.argsort(wavelengths)
    return np.array(wavelengths)[order], [labels[i] for i in order]


def positions(peaks, limit: int | None = None, brightest: bool = False) -> np.ndarray:
    """Pixel positions out of either a list of `Peak` or a plain array.

    Every function here takes both, because a caller who already has positions
    should not have to wrap them. Writing the test for `if peaks and
    isinstance(...)` inline was how this became a bug: a numpy array of more
    than one element raises on `if`.
    """
    if isinstance(peaks, np.ndarray):
        values = np.asarray(peaks, dtype=float)
    elif len(peaks) and isinstance(peaks[0], Peak):
        items = list(peaks)
        if brightest:
            items = sorted(items, key=lambda p: -p.flux)
        if limit is not None:
            items = items[:limit]
        return np.sort(np.array([p.pixel for p in items], dtype=float))
    else:
        values = np.asarray(list(peaks), dtype=float)
    if limit is not None:
        values = values[:limit]
    return np.sort(values)


def collapse(data: np.ndarray, dispaxis: int = 2, line: int | None = None,
             nsum: int = 10) -> np.ndarray:
    """Take a one-dimensional cut out of a two-dimensional arc.

    IRAF's `identify` does this itself rather than requiring an extraction
    first: `line` is the position across the slit and `nsum` how many rows or
    columns to add together. The default is the middle of the slit, which for
    an arc is as good as anywhere -- the lamp fills it.
    """
    data = np.asarray(data, dtype=float)
    if dispaxis == 2:                      # dispersion runs down the rows
        across = data.shape[1]
        centre = across // 2 if line is None else int(line) - 1
        low = max(0, centre - nsum // 2)
        return data[:, low:low + max(1, nsum)].sum(axis=1)
    across = data.shape[0]
    centre = across // 2 if line is None else int(line) - 1
    low = max(0, centre - nsum // 2)
    return data[low:low + max(1, nsum), :].sum(axis=0)


def find_peaks(flux, fwidth: float = 4.0, threshold: float = 10.0,
               minsep: float = 2.0, absorption: bool = False) -> list[Peak]:
    """Find lines and centre each one.

    `fwidth` is the expected full width of a line in pixels -- IRAF's own
    parameter name -- and sets the window used to centroid. `threshold` is in
    units of the local noise, not counts, so it means the same thing on a
    bright arc and a faint one.

    With `absorption`, lines going the other way are found instead: the same
    detection against the same continuum, on the flipped residual. An arc has
    emission and a star has absorption, and the arithmetic of finding them
    does not care which.
    """
    flux = np.asarray(flux, dtype=float)
    if flux.size < 5:
        return []

    # A rough continuum, so a sloping background does not tilt the threshold.
    window = max(21, int(20 * fwidth) | 1)
    continuum = _running_median(flux, window)
    residual = continuum - flux if absorption else flux - continuum
    noise = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    if noise <= 0:
        noise = np.std(residual) or 1.0
    limit = threshold * noise

    half = max(1, int(round(fwidth / 2)))
    peaks: list[Peak] = []
    for index in range(half, flux.size - half):
        value = residual[index]
        if value < limit:
            continue
        window_values = residual[index - half:index + half + 1]
        if value < window_values.max():
            continue
        # Centroid over the line, above the local continuum.
        low, high = index - half, index + half + 1
        weights = np.clip(residual[low:high], 0, None)
        if weights.sum() <= 0:
            continue
        positions = np.arange(low, high)
        centre = float((weights * positions).sum() / weights.sum())
        width = float(np.sqrt((weights * (positions - centre) ** 2).sum()
                              / weights.sum())) * 2.3548
        peaks.append(Peak(centre + 1.0, float(value), width))     # 1-based

    # Two maxima closer than minsep are one blended line, not two.
    peaks.sort(key=lambda p: -p.flux)
    kept: list[Peak] = []
    for peak in peaks:
        if all(abs(peak.pixel - other.pixel) >= minsep for other in kept):
            kept.append(peak)
    kept.sort(key=lambda p: p.pixel)
    return kept


def _running_median(values: np.ndarray, window: int) -> np.ndarray:
    """Median filter, wide enough to follow the continuum and ignore lines."""
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    strides = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(strides, axis=-1)[: values.size]


def autoidentify(peaks, catalogue, crval: float, cdelt: float, npixels: int,
                 crval_search: float = 0.0, cdelt_search: float = 0.0,
                 steps: int = 41, tolerance: float = 1.0,
                 crpix: float | None = None, sigma: float = 0.3) -> LineMatch:
    """Match peaks to catalogue lines by searching over (crval, cdelt).

    `crval` is the wavelength at the middle of the spectrum and `cdelt` the
    dispersion in Angstroms per pixel -- both approximate. The search widens
    them by `crval_search` and a fraction `cdelt_search`.

    How candidates are scored matters more than the search itself. Counting
    peaks that land within a tolerance of *some* catalogue line is useless
    against a list as dense as thorium's three thousand entries: at a
    dispersion of one Angstrom per pixel there is a line within a pixel of
    everywhere, so every dispersion scores perfectly and the wrong one wins on
    a coin toss. The score here is a sum of narrow Gaussians, `sigma` pixels
    wide -- about a fifth of a line width -- so it rewards *exact* coincidence.
    A wrong dispersion scatters the distances uniformly and scores near zero.
    """
    pixels = positions(peaks)
    catalogue = np.asarray(catalogue, dtype=float)
    if pixels.size < 2 or catalogue.size < 2:
        return LineMatch(np.array([]), np.array([]))

    centre = (npixels + 1) / 2 if crpix is None else float(crpix)
    crvals = (np.linspace(crval - crval_search, crval + crval_search, steps)
              if crval_search > 0 else np.array([crval]))
    cdelts = (np.linspace(cdelt * (1 - cdelt_search), cdelt * (1 + cdelt_search), steps)
              if cdelt_search > 0 else np.array([cdelt]))

    best = (-1.0, crval, cdelt)
    for value in crvals:
        for step in cdelts:
            if step == 0:
                continue
            predicted = value + (pixels - centre) * step
            # Distance to the nearest catalogue line, in pixels.
            distance = np.abs(predicted[:, None]
                              - catalogue[None, :]).min(axis=1) / abs(step)
            score = float(np.exp(-0.5 * (distance / sigma) ** 2).sum())
            if score > best[0]:
                best = (score, float(value), float(step))

    score, value, step = best
    predicted = value + (pixels - centre) * step
    index = np.abs(predicted[:, None] - catalogue[None, :]).argmin(axis=1)
    distance = np.abs(predicted - catalogue[index])
    keep = distance <= tolerance * abs(step)

    # One catalogue line must not be claimed by two peaks: the closer wins.
    chosen: dict[int, int] = {}
    for position in np.flatnonzero(keep):
        line = int(index[position])
        if line not in chosen or distance[position] < distance[chosen[line]]:
            chosen[line] = int(position)
    order = sorted(chosen.values())
    return LineMatch(pixels[order], catalogue[index[order]], value, step,
                     round(score, 3))


def search_dispersion(peaks, catalogue, window: tuple[float, float],
                      cdelt_range: tuple[float, float], npixels: int,
                      nbrightest: int = 40, crpix: float | None = None,
                      steps: int = 1200, coincidence: float = 0.3,
                      margin: float = 1.25) -> LineMatch:
    """Try to find the dispersion with no prior, and refuse when it cannot.

    For each candidate dispersion, every peak paired with every catalogue line
    implies one central wavelength; the true dispersion is the one where many
    pairs agree, so it shows as a spike in the histogram of implied central
    wavelengths. That much works.

    What does not work is trusting the winner. Against a list as dense as
    thorium's -- three thousand lines, four hundred of them inside a few
    hundred Angstroms -- several quite different dispersions score nearly the
    same, and each of them will happily fit twenty-five "identified" lines to
    a tenth of a pixel with a cubic. The fit looks excellent and the answer is
    invented.

    So the search reports a solution only when the best candidate beats every
    candidate at a materially different dispersion by a factor of `margin`.
    Otherwise it returns nothing, with a message saying so. This is why IRAF's
    identify asks you to mark five lines by hand: it is not a limitation of the
    software, it is the shape of the problem.
    """
    pixels = positions(peaks, limit=nbrightest, brightest=True)
    catalogue = np.asarray(catalogue, dtype=float)
    inside = catalogue[(catalogue >= window[0]) & (catalogue <= window[1])]
    if pixels.size < 4 or inside.size < 4:
        return LineMatch(np.array([]), np.array([]), score=0.0)

    centre = (npixels + 1) / 2 if crpix is None else float(crpix)
    low, high = min(abs(v) for v in cdelt_range), max(abs(v) for v in cdelt_range)
    span = window[1] - window[0]

    candidates = []
    for magnitude in np.linspace(low, high, steps):
        for sign in (1.0, -1.0):
            cdelt = sign * magnitude
            implied = (inside[None, :] - ((pixels - centre) * cdelt)[:, None]).ravel()
            implied = implied[(implied >= window[0]) & (implied <= window[1])]
            if implied.size < 8:
                continue
            bins = max(16, int(span / (coincidence * magnitude)))
            counts, edges = np.histogram(implied, bins=bins, range=window)
            peak_bin = int(counts.argmax())
            candidates.append((int(counts[peak_bin]), cdelt,
                               float(0.5 * (edges[peak_bin] + edges[peak_bin + 1]))))
    if not candidates:
        return LineMatch(np.array([]), np.array([]), score=0.0)

    candidates.sort(key=lambda item: -item[0])
    best_score, best_cdelt, best_crval = candidates[0]

    rival = next((item for item in candidates
                  if abs(abs(item[1]) - abs(best_cdelt)) > 0.05 * abs(best_cdelt)), None)
    if rival is not None and best_score < margin * rival[0]:
        return LineMatch(
            np.array([]), np.array([]), best_crval, best_cdelt, float(best_score),
            message=(
                f"the search is ambiguous: {abs(best_cdelt):.4f} per pixel scores "
                f"{best_score} and {abs(rival[1]):.4f} scores {rival[0]}, and a cubic "
                "will fit either to a fraction of a pixel. Give --crval and --cdelt, "
                "or mark a few lines by hand with --interactive."
            ))

    return autoidentify(peaks, catalogue, best_crval, best_cdelt, npixels,
                        crpix=crpix, tolerance=1.5)


def refine(pixel, wavelength, catalogue, spec, tolerance: float = 2.0,
           labels: list[str] | None = None):
    """Fit the dispersion, then look for more catalogue lines it now explains.

    This is IRAF's `l` key: once a handful of lines give a solution, the rest
    of the list can be placed. Each pass adds the lines the current solution
    predicts, and refits.
    """
    from . import icfit

    pixel = np.asarray(pixel, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    catalogue = np.asarray(catalogue, dtype=float)

    fit = icfit.fit(pixel, wavelength, spec)
    return fit


def add_from_list(peaks, fit, catalogue, tolerance: float = 2.0):
    """Every peak the current solution puts within `tolerance` px of a line.

    Returns (pixel, wavelength) arrays -- the input to the next fit.
    """
    pixels = positions(peaks)
    catalogue = np.asarray(catalogue, dtype=float)
    if fit.model is None or pixels.size == 0:
        return np.array([]), np.array([])

    predicted = fit(pixels)
    index = np.abs(predicted[:, None] - catalogue[None, :]).argmin(axis=1)
    # Convert the tolerance from pixels to Angstroms using the local slope.
    slope = np.gradient(fit(np.arange(1.0, pixels.max() + 2.0)))
    scale = float(np.median(np.abs(slope))) or 1.0
    distance = np.abs(predicted - catalogue[index])
    keep = distance <= tolerance * scale

    chosen: dict[int, int] = {}
    for position in np.flatnonzero(keep):
        line = int(index[position])
        if line not in chosen or distance[position] < distance[chosen[line]]:
            chosen[line] = int(position)
    order = sorted(chosen.values())
    return pixels[order], catalogue[index[order]]
