"""Arc lamps: finding the lines, and working out what wavelength each one is.

Where IRAF's line lists are, and the quick peak finder the interactive
panels draw their marks with. The identification itself -- identify's
centring, its line-list matching and autoidentify's search -- is IRAF's,
ported in core/identify.py.

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
        # More than half the spectrum sits exactly on its continuum: there is
        # no noise to measure against, and any line at all is significant.
        # (The standard deviation used to stand in here, but without noise it
        # is made of the lines themselves -- with six strong lines it rose
        # above their own height and none was found.)
        noise = 1e-6 * float(np.abs(residual).max()) or 1.0
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
