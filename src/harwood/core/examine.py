"""imexamine without a window: one key, one position, one answer.

Section 8.2 keeps the analysis out of the viewer so that the same numbers come
out whether a key was pressed over the image, a line was read from a cursor
file, or a test asked. This module is that one place. The window draws what it
returns; `imexamine --imagecur` prints it; neither computes anything itself.

The keys are IRAF's, because section 3 rates muscle memory above names::

    r   radial profile and FWHM          j   gaussian fit along the line
    a   aperture photometry and FWHM     k   gaussian fit along the column
    m   statistics of a box              h   histogram of a box
    l   plot the line                    x   coordinates and value
    c   plot the column                  z   the pixel values around the cursor

Positions in and out are 1-based (x, y), as everywhere a user can see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import pixels

#: What each key does. The window lists these on `?`; the help page prints them.
KEYS: dict[str, str] = {
    "r": "radial profile and fit at the cursor; FWHM, peak and sky",
    "a": "aperture photometry at the cursor, with the FWHM",
    "m": "statistics of a box around the cursor",
    "l": "plot the line through the cursor",
    "c": "plot the column through the cursor",
    "j": "fit a gaussian along the line through the cursor",
    "k": "fit a gaussian along the column through the cursor",
    "h": "histogram of a box around the cursor",
    "x": "print the cursor position and the pixel value there",
    "z": "print the 5x5 pixel values around the cursor",
}


@dataclass
class Settings:
    """The rimexam parameters, plus the box the statistics keys use.

    `radius` is how far the profile reaches; the photometry aperture is half
    of it, and the sky annulus starts `buffer` beyond it and is `width` wide.
    """

    radius: float = 12.0
    buffer: float = 5.0
    width: float = 5.0
    fittype: str = "gaussian"
    magzero: float = 25.0
    box: int = 11
    nbins: int = 40


@dataclass
class Result:
    """What one key produced."""

    key: str
    x: float
    y: float
    line: str                                   #: one line for the log
    values: dict[str, Any] = field(default_factory=dict)
    #: What to draw, for whoever can draw: kind, x, y, and optionally a fit.
    plot: dict[str, Any] | None = field(default=None, repr=False)
    ok: bool = True


def examine(data: np.ndarray, key: str, x: float, y: float,
            settings: Settings | None = None) -> Result:
    """Answer one key at one 1-based position."""
    from ..viewer import analysis

    s = settings or Settings()
    data = np.asarray(data, dtype=float)
    if key not in KEYS:
        return Result(key, x, y, f"[{key}] is not an imexamine key; ? lists them",
                      ok=False)
    if not _inside(data, x, y):
        return Result(key, x, y, f"[{key}] x={x:.2f} y={y:.2f} is outside the "
                      f"{data.shape[1]}x{data.shape[0]} image", ok=False)

    if key in ("r", "a"):
        profile = analysis.radial_profile(data, x, y, radius=s.radius, buffer=s.buffer,
                                          width=s.width, fittype=s.fittype)
        plot = {"kind": "profile", "title": "radial profile",
                "x": profile.radii, "y": profile.values - profile.sky,
                "xlabel": "radius (pixels)", "ylabel": "counts above sky"}
        if profile.converged:
            plot.update(fit_x=profile.fit_radii, fit_y=profile.fit - profile.sky)
        if key == "r":
            if not profile.converged:
                return Result(key, x, y, f"[r] x={x:.2f} y={y:.2f}  {profile.message}",
                              plot=plot, ok=False)
            values = {"x": profile.x, "y": profile.y, "fwhm": profile.fwhm,
                      "peak": profile.peak, "sky": profile.sky}
            return Result(key, profile.x, profile.y,
                          f"[r] x={profile.x:.2f} y={profile.y:.2f}  "
                          f"FWHM={profile.fwhm:.2f} px  peak={profile.peak:.0f}  "
                          f"sky={profile.sky:.1f}", values, plot)

        phot = analysis.quick_phot(data, x, y, aperture=s.radius / 2,
                                   annulus=s.radius + s.buffer, dannulus=s.width,
                                   zmag=s.magzero)
        values = {**phot, "fwhm": profile.fwhm if profile.converged else float("nan")}
        return Result(key, phot["x"], phot["y"],
                      f"[a] x={phot['x']:.2f} y={phot['y']:.2f}  r={s.radius / 2:g}  "
                      f"mag={phot['mag']:.3f}  flux={phot['flux']:.1f}  "
                      f"sky={phot['sky']:.2f}  FWHM={values['fwhm']:.2f}", values, plot)

    if key == "m":
        stats = analysis.statistics(data, x, y, size=s.box)
        return Result(key, x, y, f"[m] x={x:.2f} y={y:.2f}  {s.box}x{s.box}  "
                      + "  ".join(f"{k}={v:.4g}" for k, v in stats.items()), stats)

    if key in ("l", "c"):
        values, number = _cut(data, x, y, key)
        axis = np.arange(1, values.size + 1)
        what = f"line {number}" if key == "l" else f"column {number}"
        stats = {"min": float(values.min()), "max": float(values.max()),
                 "median": float(np.median(values))}
        return Result(key, x, y, f"[{key}] {what}  min={stats['min']:.1f}  "
                      f"max={stats['max']:.1f}  median={stats['median']:.1f}", stats,
                      {"kind": "cut", "title": what, "x": axis, "y": values,
                       "xlabel": "x (pixels)" if key == "l" else "y (pixels)",
                       "ylabel": "counts"})

    if key in ("j", "k"):
        return _gaussian_cut(data, x, y, key, s)

    if key == "h":
        box = _box(data, x, y, s.box)
        counts, edges = np.histogram(box[np.isfinite(box)], bins=s.nbins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        mode = float(centres[counts.argmax()])
        return Result(key, x, y, f"[h] x={x:.2f} y={y:.2f}  {s.box}x{s.box}  "
                      f"{box.size} pixels  mode={mode:.4g}  range "
                      f"{edges[0]:.4g} to {edges[-1]:.4g}",
                      {"mode": mode, "npix": int(box.size)},
                      {"kind": "histogram", "title": f"histogram {s.box}x{s.box}",
                       "x": edges, "y": counts, "xlabel": "value", "ylabel": "pixels"})

    if key == "x":
        value = _value(data, x, y)
        return Result(key, x, y, f"[x] x={x:.2f} y={y:.2f}  value={value:.6g}",
                      {"value": value})

    # z: the pixels themselves, rows printed top down so they read as on screen.
    col, row = int(round(x)), int(round(y))
    columns = [c for c in range(col - 2, col + 3) if 1 <= c <= data.shape[1]]
    rows = [r for r in range(row + 2, row - 3, -1) if 1 <= r <= data.shape[0]]
    lines = ["      " + "".join(f"{c:>10d}" for c in columns)]
    for r in rows:
        lines.append(f"{r:>6d}" + "".join(f"{data[r - 1, c - 1]:>10.1f}" for c in columns))
    return Result(key, x, y, f"[z] around x={col} y={row}  value={_value(data, x, y):.6g}\n"
                  + "\n".join(lines),
                  {"columns": columns, "rows": rows})


def read_cursor(path) -> list[tuple[float, float, int, str]]:
    """Read an IRAF image-cursor file: `x y [frame] key` per line.

    The frame is optional and defaults to 1; IRAF writes it as the WCS
    number, which for a displayed image is the frame. Blank lines and
    `#` comments are skipped, and a colon command is refused by name rather
    than silently ignored.
    """
    from pathlib import Path

    from . import advice

    entries = []
    for number, text in enumerate(Path(path).read_text().splitlines(), 1):
        text = text.split("#", 1)[0].strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) not in (3, 4):
            advice.fail(f"{path}:{number}: '{text}' is not 'x y [frame] key'",
                        "write one position per line, e.g. '612.0 540.0 1 r'")
        try:
            x, y = float(parts[0]), float(parts[1])
            frame = int(parts[2]) if len(parts) == 4 else 1
        except ValueError:
            advice.fail(f"{path}:{number}: '{text}' does not start with two numbers",
                        "the first two columns are the 1-based x and y")
        key = parts[-1]
        if key.startswith(":"):
            advice.fail(f"{path}:{number}: colon commands are not read from a "
                        "cursor file", "set the parameter with --radius, --fittype "
                        "and the like instead")
        entries.append((x, y, frame, key))
    return entries


# ------------------------------------------------------------------ helpers


def _inside(data: np.ndarray, x: float, y: float) -> bool:
    return 0.5 <= x < data.shape[1] + 0.5 and 0.5 <= y < data.shape[0] + 0.5


def _value(data: np.ndarray, x: float, y: float) -> float:
    return float(data[int(round(y)) - 1, int(round(x)) - 1])


def _cut(data: np.ndarray, x: float, y: float, key: str) -> tuple[np.ndarray, int]:
    if key in ("l", "j"):
        row = int(np.clip(round(y), 1, data.shape[0]))
        return data[row - 1, :], row
    col = int(np.clip(round(x), 1, data.shape[1]))
    return data[:, col - 1], col


def _box(data: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    row, col = pixels.to_numpy(x, y)
    half = size // 2
    r0, r1 = max(0, int(round(row)) - half), min(data.shape[0], int(round(row)) + half + 1)
    c0, c1 = max(0, int(round(col)) - half), min(data.shape[1], int(round(col)) + half + 1)
    return data[r0:r1, c0:c1]


def _gaussian_cut(data: np.ndarray, x: float, y: float, key: str, s: Settings) -> Result:
    """The j and k keys: a gaussian plus a constant, across a slit or a star."""
    values, number = _cut(data, x, y, key)
    centre = x if key == "j" else y
    reach = max(3, int(np.ceil(s.radius)))
    lo, hi = max(1, int(round(centre)) - reach), min(values.size, int(round(centre)) + reach)
    axis = np.arange(lo, hi + 1, dtype=float)
    segment = values[lo - 1:hi]
    what = f"line {number}" if key == "j" else f"column {number}"
    plot = {"kind": "cut", "title": f"{what}, gaussian fit", "x": axis, "y": segment,
            "xlabel": "x (pixels)" if key == "j" else "y (pixels)", "ylabel": "counts"}

    background = float(np.median(np.concatenate([segment[:3], segment[-3:]])))
    peak = float(segment.max() - background)
    try:
        from scipy.optimize import curve_fit

        def model(t, a, mu, sigma, b):
            return a * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) + b

        guess = (peak, axis[segment.argmax()], 2.0, background)
        popt, _ = curve_fit(model, axis, segment, p0=guess, maxfev=4000)
    except Exception:
        return Result(key, x, y, f"[{key}] {what}  the fit did not converge -- "
                      "is there a feature here?", plot=plot, ok=False)

    amplitude, mu, sigma, base = (float(v) for v in popt)
    fwhm = 2.3548 * abs(sigma)
    if not lo <= mu <= hi or amplitude <= 0:
        return Result(key, x, y, f"[{key}] {what}  no peak inside +-{reach} px "
                      "of the cursor", plot=plot, ok=False)
    grid = np.linspace(axis[0], axis[-1], 200)
    plot.update(fit_x=grid, fit_y=model(grid, *popt))
    return Result(key, x, y, f"[{key}] {what}  centre={mu:.2f}  FWHM={fwhm:.2f} px  "
                  f"peak={amplitude:.1f}  background={base:.1f}",
                  {"centre": mu, "fwhm": fwhm, "peak": amplitude, "background": base},
                  plot)
