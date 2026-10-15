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
    c   plot the column                  z   the 10x10 pixel values around it

The parameters keep IRAF's names, defaults and meanings: `radius` is the
aperture (rimexam's radius = 5), the profile is drawn out to `rplot` = 8, the
fit is a Moffat, `iterations` = 3 resizes the aperture of r and a to three
times the direct FWHM, m measures `ncstat` x `nlstat` = 5x5, and h a 21x21 box in
512 bins. IRAF keeps these in psets -- rimexam, jimexam, himexam -- and
harwood in one flat set, so where two psets share a name (`rplot`, `width`)
the rimexam meaning is the parameter and the j/k one is fixed at jimexam's
default.

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
    "z": "print the 10x10 pixel values around the cursor",
}


#: jimexam's rplot and width: how far either side of the cursor the j and k
#: keys fit. Fixed, because the flat parameter set gives those names to rimexam.
JK_RPLOT = 10.0


@dataclass
class Settings:
    """The parameters behind the keys, with IRAF's names and defaults.

    `radius` is the aperture: summed for the flux, used for centring, and
    fitted. The sky annulus starts `buffer` beyond it and is `width` wide.
    The radial profile is drawn out to `rplot`.
    """

    radius: float = 5.0            #: rimexam
    buffer: float = 5.0            #: rimexam
    width: float = 5.0             #: rimexam
    rplot: float = 8.0             #: rimexam
    fittype: str = "moffat"        #: rimexam
    beta: float | None = None      #: rimexam; None is IRAF's INDEF, fitted
    iterations: int = 3            #: rimexam; r and a
    magzero: float = 25.0          #: rimexam
    ncstat: int = 5                #: imexamine, the m key's box
    nlstat: int = 5
    ncolumns: int = 21             #: himexam, the h key's box
    nlines: int = 21
    nbins: int = 512               #: himexam
    naverage: int = 5              #: jimexam, lines or columns averaged
    sigma: float = 1.0             #: jimexam, the first guess of the j/k fit


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
        # One routine answers both keys in IRAF (ie_rimexam), so both iterate
        # the aperture; the manual says only a does, the code says otherwise.
        profile = analysis.radial_profile(
            data, x, y, radius=s.radius, buffer=s.buffer, width=s.width,
            rplot=s.rplot, fittype=s.fittype, beta=s.beta, iterations=s.iterations)
        plot = {"kind": "profile", "title": "radial profile",
                "x": profile.radii, "y": profile.values - profile.sky,
                "xlabel": "radius (pixels)", "ylabel": "counts above sky"}
        if profile.converged:
            plot.update(fit_x=profile.fit_radii, fit_y=profile.fit - profile.sky)
        if not profile.converged:
            return Result(key, profile.x, profile.y,
                          f"[{key}] x={profile.x:.2f} y={profile.y:.2f}  {profile.message}",
                          {"x": profile.x, "y": profile.y, "radius": profile.radius},
                          plot=plot, ok=False)
        values = {"x": profile.x, "y": profile.y, "radius": profile.radius,
                  "mag": profile.magnitude(s.magzero), "flux": profile.flux,
                  "sky": profile.sky, "peak": profile.peak, "e": profile.ellipticity,
                  "pa": profile.pa, "beta": profile.beta,
                  "enclosed": profile.enclosed_fwhm, "fwhm": profile.fwhm,
                  "direct": profile.direct_fwhm}
        beta = f"  beta={profile.beta:.2f}" if s.fittype == "moffat" else ""
        label = "MFWHM" if s.fittype == "moffat" else "GFWHM"
        pa = "INDEF" if not np.isfinite(profile.pa) else f"{profile.pa:.0f}"
        return Result(key, profile.x, profile.y,
                      f"[{key}] x={profile.x:.2f} y={profile.y:.2f}  "
                      f"r={profile.radius:.2f}  mag={values['mag']:.3f}  "
                      f"flux={profile.flux:.1f}  sky={profile.sky:.2f}  "
                      f"peak={profile.peak:.1f}  e={profile.ellipticity:.2f}  "
                      f"pa={pa}{beta}  enclosed={profile.enclosed_fwhm:.2f}  "
                      f"{label}={profile.fwhm:.2f}  direct={profile.direct_fwhm:.2f}",
                      values, plot)

    if key == "m":
        stats = analysis.statistics(data, x, y, ncols=s.ncstat, nlines=s.nlstat)
        return Result(key, x, y, f"[m] x={x:.2f} y={y:.2f}  {s.ncstat}x{s.nlstat}  "
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
        box = _box(data, x, y, s.ncolumns, s.nlines)
        finite = box[np.isfinite(box)]
        bins = s.nbins
        # himexam's autoscale: integer data never get more bins than values,
        # or most bins are empty by construction and the plot is a comb.
        span = float(finite.max() - finite.min()) if finite.size else 0.0
        if finite.size and np.all(finite == np.round(finite)) and span + 1 < bins:
            bins = int(span) + 1
        counts, edges = np.histogram(finite, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        mode = float(centres[counts.argmax()])
        size = f"{s.ncolumns}x{s.nlines}"
        return Result(key, x, y, f"[h] x={x:.2f} y={y:.2f}  {size}  "
                      f"{box.size} pixels  {bins} bins  mode={mode:.4g}  range "
                      f"{edges[0]:.4g} to {edges[-1]:.4g}",
                      {"mode": mode, "npix": int(box.size), "nbins": bins},
                      {"kind": "histogram", "title": f"histogram {size}",
                       "x": edges, "y": counts, "xlabel": "value", "ylabel": "pixels"})

    if key == "x":
        value = _value(data, x, y)
        return Result(key, x, y, f"[x] x={x:.2f} y={y:.2f}  value={value:.6g}",
                      {"value": value})

    # z: IRAF's 10x10 grid, rows printed top down so they read as on screen.
    col, row = int(round(x)), int(round(y))
    columns = [c for c in range(col - 4, col + 6) if 1 <= c <= data.shape[1]]
    rows = [r for r in range(row + 5, row - 5, -1) if 1 <= r <= data.shape[0]]
    lines = ["      " + "".join(f"{c:>10d}" for c in columns)]
    for r in rows:
        lines.append(f"{r:>6d}" + "".join(f"{data[r - 1, c - 1]:>10.1f}" for c in columns))
    return Result(key, x, y, f"[z] around x={col} y={row}  value={_value(data, x, y):.6g}\n"
                  + "\n".join(lines),
                  {"columns": columns, "rows": rows})


def read_cursor(path, defkey: str = "a") -> list[tuple[float, float, int, str]]:
    """Read an IRAF image-cursor file: `x y [frame] [key]` per line.

    The frame defaults to 1; IRAF writes it as the WCS number, which for a
    displayed image is the frame. With no key the line gets `defkey` -- as in
    IRAF, which is what makes a bare list of positions from another program a
    valid cursor file. Blank lines and `#` comments are skipped, and a colon
    command is refused by name rather than silently ignored.
    """
    from pathlib import Path

    from . import advice

    entries = []
    for number, text in enumerate(Path(path).read_text().splitlines(), 1):
        text = text.split("#", 1)[0].strip()
        if not text:
            continue
        parts = text.split()
        if not 2 <= len(parts) <= 4:
            advice.fail(f"{path}:{number}: '{text}' is not 'x y [frame] [key]'",
                        "write one position per line, e.g. '612.0 540.0 1 r'")
        # No key at all, or a third column that is a number (the frame): the
        # line gets defkey. A third column that is not a number is the key.
        if len(parts) == 2 or (len(parts) == 3 and parts[2].lstrip("+-").isdigit()):
            parts.append(defkey)
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


def _cut(data: np.ndarray, x: float, y: float, key: str,
         naverage: int = 1) -> tuple[np.ndarray, int]:
    """The line (l, j) or column (c, k) through the cursor, averaged over
    `naverage` of them centred on it."""
    along_line = key in ("l", "j")
    size = data.shape[0] if along_line else data.shape[1]
    centre = int(np.clip(round(y if along_line else x), 1, size))
    lo = max(1, centre - (naverage - 1) // 2)
    hi = min(size, lo + naverage - 1)
    if along_line:
        return data[lo - 1:hi, :].mean(axis=0), centre
    return data[:, lo - 1:hi].mean(axis=1), centre


def _box(data: np.ndarray, x: float, y: float, ncols: int, nlines: int) -> np.ndarray:
    row, col = pixels.to_numpy(x, y)
    r0 = max(0, int(round(row)) - (nlines - 1) // 2)
    c0 = max(0, int(round(col)) - (ncols - 1) // 2)
    return data[r0:min(data.shape[0], r0 + nlines), c0:min(data.shape[1], c0 + ncols)]


def _gaussian_cut(data: np.ndarray, x: float, y: float, key: str, s: Settings) -> Result:
    """The j and k keys: a gaussian plus a constant, across a slit or a star."""
    values, number = _cut(data, x, y, key, s.naverage)
    centre = x if key == "j" else y
    reach = int(np.ceil(JK_RPLOT))
    lo, hi = max(1, int(round(centre)) - reach), min(values.size, int(round(centre)) + reach)
    axis = np.arange(lo, hi + 1, dtype=float)
    segment = values[lo - 1:hi]
    what = f"line {number}" if key == "j" else f"column {number}"
    if s.naverage > 1:
        what += f" ({s.naverage} averaged)"
    plot = {"kind": "cut", "title": f"{what}, gaussian fit", "x": axis, "y": segment,
            "xlabel": "x (pixels)" if key == "j" else "y (pixels)", "ylabel": "counts"}

    background = float(np.median(np.concatenate([segment[:3], segment[-3:]])))
    peak = float(segment.max() - background)
    try:
        from scipy.optimize import curve_fit

        def model(t, a, mu, sigma, b):
            return a * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) + b

        guess = (peak, axis[segment.argmax()], s.sigma, background)
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
    return Result(key, x, y, f"[{key}] {what}  centre={mu:.2f}  peak={amplitude:.1f}  "
                  f"sigma={abs(sigma):.2f}  FWHM={fwhm:.2f}  background={base:.1f}",
                  {"centre": mu, "peak": amplitude, "sigma": abs(sigma), "fwhm": fwhm,
                   "background": base}, plot)
