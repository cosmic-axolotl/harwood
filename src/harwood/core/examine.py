"""imexamine without a window: one key, one position, one answer.

Section 8.2 keeps the analysis out of the viewer so that the same numbers come
out whether a key was pressed over the image, a line was read from a cursor
file, or a test asked. This module is that one place. The window draws what it
returns; `imexamine --imagecur` prints it; neither computes anything itself.

The keys are IRAF's, because section 3 rates muscle memory above names::

    r   radial profile and FWHM          j   gaussian fit along the line
    a   aperture photometry and FWHM     k   gaussian fit along the column
    m   statistics of a box              h   histogram of a box
    l   plot the line                    e   contour plot of a box
    c   plot the column                  s   surface plot of a box
    x   coordinates and value            z   the 11x11 pixel values around it

and the colon commands (`colon`) set the parameters between keys.

The parameters keep IRAF's names, defaults and meanings: `radius` is the
aperture (rimexam's radius = 5), the profile is drawn out to `rplot` = 8, the
fit is a Moffat, `iterations` = 3 resizes the aperture of r and a to three
times the direct FWHM, m measures `ncstat` x `nlstat` = 5x5, and h a 21x21 box in
512 bins. IRAF keeps these in psets -- rimexam, jimexam, kimexam, himexam,
eimexam... -- and harwood in one set (see `Settings`), where a name two
psets share keeps the rimexam (or jimexam) meaning and the others carry
their pset's initial.

Positions in and out are 1-based (x, y), as everywhere a user can see.
"""

from __future__ import annotations

import math
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
    "e": "contour plot of a box around the cursor",
    "s": "surface plot of a box around the cursor",
    "x": "print the cursor position and the pixel value there",
    "z": "print the 11x11 pixel values around the cursor",
    "b": "box coordinates of two cursor positions, c1 c2 l1 l2",
    "y": "set the origin x then measures from",
    "t": "write a section around the cursor as a new image",
    "u": "plot a vector centred on a position, to a second one",
    "v": "plot a vector between two cursor positions",
    ",": "quick photometry, with one fit and no iterating",
    ".": "quick radial profile and fit",
    "o": "overplot the next graph on this one",
    "f": "redraw the last graph",
    "w": "start or stop writing to the logfile",
    "g": "go to the graph's cursor: here both are live",
    "i": "go to the image's cursor: here both are live",
    "d": "load another image into a frame",
}

#: The keys that draw a graph: a colon command that names a parameter
#: several psets share (width, rplot, naverage, ncolumns...) changes the one
#: of the last of these, as IRAF's does.
GRAPH_KEYS = "cehjklrsuv."


@dataclass
class Settings:
    """The parameters behind the keys, with IRAF's names and defaults.

    IRAF keeps them in psets -- rimexam, jimexam, kimexam, limexam,
    cimexam, himexam, eimexam, simexam -- and some names live in several.
    Here they are one set: the rimexam ones, the m key's and the j key's
    keep IRAF's plain names (they are the task's parameters); the others
    carry their pset's initial (k, l, c, e, s). A colon command uses the
    plain name and finds the field the way IRAF finds the pset.

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
    center: bool = True            #: rimexam: recentre before measuring
    background: bool = True        #: rimexam: subtract the annulus' median
    ncstat: int = 5                #: imexamine, the m key's box
    nlstat: int = 5
    ncolumns: int = 21             #: himexam, the h key's box
    nlines: int = 21
    nbins: int = 512               #: himexam
    z1: float | None = None        #: himexam; None is INDEF, the data's limits
    z2: float | None = None
    autoscale: bool = True         #: himexam, for integer pixel types
    top_closed: bool = False       #: himexam
    naverage: int = 5              #: jimexam, lines averaged
    sigma: float = 1.0             #: jimexam, the first guess of the fit
    jwidth: float = 10.0           #: jimexam width
    jrplot: float = 10.0           #: jimexam rplot
    jcenter: bool = True
    jbackground: bool = True
    jxorder: int = 0               #: 0 median, 1 constant, 2 linear background
    knaverage: int = 5             #: kimexam, columns averaged
    ksigma: float = 1.0
    kwidth: float = 10.0
    krplot: float = 10.0
    kcenter: bool = True
    kbackground: bool = True
    kxorder: int = 0
    lnaverage: int = 1             #: limexam
    cnaverage: int = 1             #: cimexam
    encolumns: int = 21            #: eimexam
    enlines: int = 21
    efloor: float | None = None
    eceiling: float | None = None
    zero: float = 0.0              #: eimexam
    ncontours: int = 5             #: eimexam
    interval: float = 0.0          #: eimexam
    sncolumns: int = 21            #: simexam
    snlines: int = 21
    sfloor: float | None = None
    sceiling: float | None = None
    angh: float = -33.0            #: simexam
    angv: float = 25.0             #: simexam
    vnaverage: int = 1             #: vimexam, the strip's width
    boundary: str = "constant"     #: vimexam
    constant: float = 0.0          #: vimexam
    defkey: str = "a"              #: imexamine, for cursor lines without a key
    output: str = ""               #: imexamine, the root of what t writes
    ncoutput: int = 101            #: imexamine, t's section
    nloutput: int = 101
    xorigin: float = 0.0           #: y
    yorigin: float = 0.0
    keeplog: bool = True           #: w
    #: The last key, and the last key that drew a graph (IRAF's gtype).
    lastkey: str = ""
    graph: str = ""
    #: Where the image's name and header are, for t and the log.
    image: str = ""
    header: Any = None
    logname: str = ""
    #: A position waiting for its second, for b, u and v; and o's asking to overplot.
    first: tuple | None = None
    overplot: bool = False
    redraw: bool = False           #: a colon command has changed what the graph shows
    last_second: tuple | None = None
    #: The last position, for f. And the last of IE_LASTKEY's keys, for , 's labels.
    last_position: tuple | None = None
    labelkey: str = ""
    last_data: Any = None


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


def native(data: np.ndarray, header) -> np.ndarray:
    """The pixels with the type IRAF would see: an integer FITS image (BITPIX
    > 0, unscaled) as integers, which is what himexam's autoscale goes by.
    harwood reads every image as floating point; this undoes that for the
    keys that care."""
    try:
        bitpix = int(header.get("BITPIX", -32))
        bscale = float(header.get("BSCALE", 1.0))
        bzero = float(header.get("BZERO", 0.0))
    except (AttributeError, TypeError, ValueError):
        return data
    if bitpix > 0 and bscale == 1.0 and bzero == int(bzero) and np.isfinite(data).all():
        return np.asarray(data).astype(np.int64)
    return data


def examine(data: np.ndarray, key: str, x: float, y: float,
            settings: Settings | None = None) -> Result:
    """Answer one key at one 1-based position.

    The settings remember the key, and the last key that drew a graph, for
    the colon commands that follow (see `colon`)."""
    s = settings if settings is not None else Settings()
    if key == "f":
        if not s.graph or s.last_position is None:
            return Result(key, x, y, "[f] no graph to redraw", ok=False)
        x, y = s.last_position
        key = s.graph
        if key in ("u", "v"):
            return _second(data, key, x, y, *s.last_second, s)
    if key in ("b", "u", "v"):
        family = "b" if key == "b" else "uv"
        if s.first is None or s.first[0] != family:
            s.first = (family, x, y)
            return Result(key, x, y, "again:")
        second = (x, y)
        _, x, y = s.first
        s.first = None
        s.last_second = second
        result = _second(data, key, x, y, *second, s)
    else:
        s.first = None
        result = _answer(data, key, x, y, s)
        if result.plot is not None and key in GRAPH_KEYS and key != ".":
            _overplot(result.plot, s)
    if key in KEYS:
        s.lastkey = key
        if key in ("?", "a", "b", "d", "m", "w", "x", "y", "z", ","):
            s.labelkey = key
        if key in GRAPH_KEYS:
            s.graph = key
            s.last_position = (x, y)
    return result


def _second(data, key, x1, y1, x2, y2, s: Settings) -> Result:
    """b, u and v, once they have their two positions."""
    from . import examkeys

    if key == "b":
        line = examkeys.box(x1, y1, x2, y2)
        c1, c2, l1, l2 = (int(word) for word in line.split())
        return Result(key, x1, y1, line, {"c1": c1, "c2": c2, "l1": l1, "l2": l2})
    axis, values, ends = examkeys.vector(data, key, x1, y1, x2, y2, s.vnaverage, s.boundary,
                                         s.constant)
    title = (f"Vector {ends[0]:.1f},{ends[1]:.1f} to {ends[2]:.1f},{ends[3]:.1f} "
             f"naverage: {s.vnaverage}")
    plot = {"kind": "cut", "title": title, "x": axis.astype(float), "y": values.astype(float),
            "xlabel": "vector distance (pixels)", "ylabel": "pixel value"}
    _overplot(plot, s)
    return Result(key, x1, y1, f"[{key}] {title}", {"npts": int(values.size)}, plot)


def _overplot(plot, s: Settings) -> None:
    if s.overplot:
        plot["append"] = True
        s.overplot = False


def _answer(data: np.ndarray, key: str, x: float, y: float, s: Settings) -> Result:
    from ..viewer import analysis

    raw = np.asarray(data)
    data = np.asarray(data, dtype=float)
    if key not in KEYS:
        return Result(key, x, y, f"[{key}] is not an imexamine key; ? lists them",
                      ok=False)
    from . import examkeys

    if key == "y":
        s.xorigin, s.yorigin = float(np.float32(x)), float(np.float32(y))
        return Result(key, x, y, examkeys.origin(x, y), {"xorigin": s.xorigin,
                                                         "yorigin": s.yorigin})
    if key == "o":
        s.overplot = True
        return Result(key, x, y, "[o] the next graph goes over this one")
    if key in ("g", "i"):
        return Result(key, x, y, f"[{key}] the image and the graph take the cursor at once "
                                 "here, so there is nothing to change to")
    if key == "w":
        if not s.logname:
            return Result(key, x, y, "No log file defined")
        s.keeplog = not s.keeplog
        return Result(key, x, y, f"Log file {s.logname} open" if s.keeplog
                      else f"Logfile {s.logname} closed")
    if key == "d":
        return Result(key, x, y, "[d] loads an image in the window; in a cursor file name "
                                 "the images on the command line", ok=False)
    if not _inside(data, x, y):
        return Result(key, x, y, f"[{key}] x={x:.2f} y={y:.2f} is outside the "
                      f"{data.shape[1]}x{data.shape[0]} image", ok=False)

    if key in ("r", "a"):
        # One routine answers both keys in IRAF (ie_rimexam), so both iterate
        # the aperture; the manual says only a does, the code says otherwise.
        profile = analysis.radial_profile(
            data, x, y, radius=s.radius, buffer=s.buffer, width=s.width,
            rplot=s.rplot, fittype=s.fittype, beta=s.beta, iterations=s.iterations,
            recentre=s.center, background=s.background,
            integer=np.issubdtype(raw.dtype, np.integer))
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
        return _statistics(raw, x, y, s)

    if key in ("l", "c"):
        return _plot_cut(raw, x, y, key, s)

    if key in ("j", "k"):
        return _gaussian_cut(raw, x, y, key, s)

    if key == "h":
        return _histogram(raw, x, y, s)

    if key == "e":
        return _contours(raw, x, y, s)

    if key == "s":
        return _surface(raw, x, y, s)

    if key == "x":
        value = _value(data, x, y)
        return Result(key, x, y, examkeys.position(data, x, y, s.xorigin, s.yorigin),
                      {"value": value})

    if key == "t":
        path, line = examkeys.write_section(data, s.header, x, y, s.ncoutput, s.nloutput,
                                            s.output, s.image)
        return Result(key, x, y, line, {"output": str(path) if path else ""},
                      ok=path is not None)

    if key in (",", "."):
        got = examkeys.quick(raw, x, y, s)
        if "error" in got:
            return Result(key, x, y, f"Warning: {got['error']}", ok=False)
        label = examkeys.GLABEL if s.fittype == "gaussian" else examkeys.MLABEL
        lines = [] if s.labelkey == "," else [label]
        if got["warning"]:
            lines.insert(0, got["warning"])
        lines.append(got["line"])
        plot = None
        if key == ".":
            plot = {"kind": "profile", "title": "quick radial profile", "x": got["plot_x"],
                    "y": got["plot_y"], "xlabel": "radius (pixels)",
                    "ylabel": "counts above sky"}
            if got["grid"] is not None:
                plot.update(fit_x=got["grid"], fit_y=got["fit"])
            _overplot(plot, s)
        values = {name: got[name] for name in ("x", "y", "mag", "flux", "sky", "n", "rmom",
                                               "e", "pa", "peak", "fwhm", "beta")}
        return Result(key, got["x"], got["y"], "\n".join(lines), values, plot,
                      ok=got["fwhm"] is not None)

    return _print_box(raw, x, y)


def _gdata(data: np.ndarray, x1: int, x2: int, y1: int, y2: int):
    """ie_gdata: the limits in order and clipped to the image, or None when
    the box misses it altogether ("Pixels out of bounds")."""
    ny, nx = data.shape
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 < 1 or x1 > nx or y2 < 1 or y1 > ny:
        return None
    x1, x2, y1, y2 = max(1, x1), min(nx, x2), max(1, y1), min(ny, y2)
    return x1, x2, y1, y2, data[y1 - 1:y2, x1 - 1:x2]


def _box_limits(x: float, y: float, ncols: int, nlines: int):
    """x - (n - 1) / 2 + 0.5 .. x + n / 2 + 0.5, truncated: IRAF's box of
    n pixels around a cursor, one more to the right when n is even."""
    f32 = np.float32
    return (int(f32(f32(f32(x) - f32((ncols - 1) // 2)) + f32(0.5))),
            int(f32(f32(f32(x) + f32(ncols // 2)) + f32(0.5))),
            int(f32(f32(f32(y) - f32((nlines - 1) // 2)) + f32(0.5))),
            int(f32(f32(f32(y) + f32(nlines // 2)) + f32(0.5))))


def _statistics(data: np.ndarray, x: float, y: float, s: Settings) -> Result:
    """ie_statistics: the m key -- statistics of an ncstat x nlstat box."""
    got = _gdata(data, *_box_limits(x, y, s.ncstat, s.nlstat))
    if got is None:
        return Result("m", x, y, "[m] pixels out of bounds", ok=False)
    x1, x2, y1, y2, box = got
    values = np.sort(box.astype(np.float32).astype(np.float64).ravel())
    npts = values.size
    total = float(np.cumsum(values)[-1])
    mean = total / npts
    if npts > 1:
        spread = (float(np.cumsum(values ** 2)[-1]) - (total / npts) * total) / (npts - 1)
        std = float(np.sqrt(spread)) if spread >= 0 else 0.0
    else:
        std = float("nan")
    median = (float(values[npts // 2 - 1] + values[npts // 2]) / 2 if npts % 2 == 0
              else float(values[npts // 2]))
    section = f"[{x1}:{x2},{y1}:{y2}]"
    stats = {"section": section, "npix": npts, "mean": mean, "median": median,
             "stddev": std, "min": float(values[0]), "max": float(values[-1])}
    return Result("m", x, y, f"[m] {section:>20s} {npts:8d} {mean:8.4g} {median:8.4g} "
                  f"{std:8.4g} {values[0]:8.4g} {values[-1]:8.4g}", stats)


def _plot_cut(data: np.ndarray, x: float, y: float, key: str, s: Settings) -> Result:
    """ie_limexam and ie_cimexam: the l and c keys, a line or a column, the
    mean of limexam's (cimexam's) naverage of them."""
    f32 = np.float32
    along_line = key == "l"
    n = s.lnaverage if along_line else s.cnaverage
    at = f32(y if along_line else x)
    first = int(f32(f32(at - f32((n - 1) // 2)) + f32(0.5)))
    last = int(f32(f32(at + f32(n // 2)) + f32(0.5)))
    if along_line:
        got = _gdata(data, 1, data.shape[1], first, last)
    else:
        got = _gdata(data, first, last, 1, data.shape[0])
    if got is None:
        return Result(key, x, y, f"[{key}] pixels out of bounds", ok=False)
    x1, x2, y1, y2, block = got
    block = block.astype(f32)
    rows = block if along_line else block.T
    total = rows[0].copy()
    for row in rows[1:]:
        total = (total + row).astype(f32)
    values = (total / f32(rows.shape[0])).astype(float)
    lo, hi = (y1, y2) if along_line else (x1, x2)
    axis = np.arange(1, values.size + 1)
    span = f"{lo}" if lo == hi else f"{lo}-{hi}"
    what = f"line {span}" if along_line else f"column {span}"
    stats = {"min": float(values.min()), "max": float(values.max()),
             "median": float(np.median(values))}
    return Result(key, x, y, f"[{key}] {what}  min={stats['min']:.1f}  "
                  f"max={stats['max']:.1f}  median={stats['median']:.1f}", stats,
                  {"kind": "cut", "title": what, "x": axis, "y": values,
                   "xlabel": "x (pixels)" if along_line else "y (pixels)",
                   "ylabel": "counts"})


def _histogram(data: np.ndarray, x: float, y: float, s: Settings) -> Result:
    """ie_himexam: the h key, with himexam's defaults (z1, z2 the data's
    limits; autoscale for integer pixel types; top bin half open)."""
    f32 = np.float32
    got = _gdata(data, *_box_limits(x, y, s.ncolumns, s.nlines))
    if got is None:
        return Result("h", x, y, "[h] pixels out of bounds", ok=False)
    x1, x2, y1, y2, box = got
    values = box.astype(f32).ravel()
    z1 = values.min() if s.z1 is None else f32(s.z1)
    z2 = values.max() if s.z2 is None else f32(s.z2)
    if z1 > z2:
        z1, z2 = z2, z1
    nbins = s.nbins
    if s.autoscale and np.issubdtype(data.dtype, np.integer):
        nlevels = _nint(z2) - _nint(z1)
        nwide = max(1, _nint(f32(f32(nlevels) / f32(nbins))))
        nbins = max(1, _nint(f32(f32(nlevels) / f32(nwide))))
        z2 = f32(_nint(z1) + nbins * nwide)
    section = f"[{x1}:{x2},{y1}:{y2}]"
    if z1 == z2:
        return Result("h", x, y, f"[h] {section} has no data range", ok=False)
    # ahgmr into nbins + 1 bins: the extra one holds the pixels equal to z2.
    dz = f32(f32(nbins) / f32(z2 - z1))
    counts = np.zeros(nbins + 1, dtype=int)
    for v in values:
        if z1 <= v <= z2:
            counts[int(f32(f32(v - z1) * dz)) if abs(dz - 1) >= 2 * 1.192e-7
                   else int(f32(v - z1))] += 1
    if s.top_closed:
        counts[nbins - 1] += counts[nbins]
    counts = counts[:nbins]
    width = f32(f32(z2 - z1) / f32(nbins))
    edges = (z1 + width * np.arange(nbins + 1, dtype=f32)).astype(float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mode = float(centres[counts.argmax()])
    return Result("h", x, y, f"[h] {section}  {values.size} pixels  {nbins} bins  "
                  f"z1={float(z1):g}  z2={float(z2):g}  mode={mode:.4g}",
                  {"mode": mode, "npix": int(values.size), "nbins": nbins,
                   "z1": float(z1), "z2": float(z2), "counts": counts.tolist()},
                  {"kind": "histogram", "title": f"histogram {section}",
                   "x": edges, "y": counts, "xlabel": "value", "ylabel": "pixels"})


def _print_box(data: np.ndarray, x: float, y: float) -> Result:
    """ie_print: the z key -- the 11 x 11 pixel values around the cursor,
    printed as IRAF prints them, top line first."""
    f32 = np.float32
    got = _gdata(data, int(f32(f32(x) - f32(4.5))), int(f32(f32(x) + f32(5.5))),
                 int(f32(f32(y) - f32(4.5))), int(f32(f32(y) + f32(5.5))))
    if got is None:
        return Result("z", x, y, "[z] pixels out of bounds", ok=False)
    x1, x2, y1, y2, box = got
    columns, rows = list(range(x1, x2 + 1)), list(range(y2, y1 - 1, -1))
    lines = ["    " + "".join(f" {c:4d} " for c in columns)]
    for r in rows:
        lines.append(f"{r:4d}" + "".join(f" {float(f32(box[r - y1, c - x1])):5g}"
                                          for c in columns))
    return Result("z", x, y, f"[z] around x={x:.2f} y={y:.2f}\n" + "\n".join(lines),
                  {"columns": columns, "rows": rows})


def read_cursor(path, defkey: str = "a") -> list[tuple[float, float, int, str]]:
    """Read an IRAF image-cursor file: `x y [frame] [key]` per line.

    The frame defaults to 1; IRAF writes it as the WCS number, which for a
    displayed image is the frame. With no key the line gets `defkey` -- as in
    IRAF, which is what makes a bare list of positions from another program a
    valid cursor file. A colon command is the rest of the line from the
    colon, `50 50 1 :radius 7`, and comes back as the key; `:defkey` changes
    the key of the lines after it, as it does in IRAF. Blank lines and `#`
    comments are skipped.
    """
    from pathlib import Path

    from . import advice

    entries = []
    for number, text in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        text = text.split("#", 1)[0].strip()
        if not text:
            continue
        command = ""
        if ":" in text:
            text, command = text.split(":", 1)
            command = ":" + command.strip()
        parts = text.split()
        if not 2 <= len(parts) <= (3 if command else 4):
            advice.fail(f"{path}:{number}: '{text}' is not 'x y [frame] [key]'",
                        "write one position per line, e.g. '612.0 540.0 1 r', or a "
                        "colon command after the frame, e.g. '612 540 1 :radius 7'")
        if command:
            parts.append(command)
        # No key at all, or a third column that is a number (the frame): the
        # line gets defkey. A third column that is not a number is the key.
        elif len(parts) == 2 or (len(parts) == 3 and parts[2].lstrip("+-").isdigit()):
            parts.append(defkey)
        try:
            x, y = float(parts[0]), float(parts[1])
            frame = int(parts[2]) if len(parts) == 4 else 1
        except ValueError:
            advice.fail(f"{path}:{number}: '{text}' does not start with two numbers",
                        "the first two columns are the 1-based x and y")
        key = parts[-1]
        words = key.lstrip(":").split()
        if key.startswith(":") and len(words) == 2 and "defkey".startswith(words[0]) \
                and len(words[0]) >= 3:
            defkey = words[1]
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


def _amedr(values: np.ndarray) -> float:
    """amedr: the ((n + 1) / 2)th smallest; the first value for one or two."""
    n = values.size
    if n <= 2:
        return float(values[0])
    return float(np.sort(values)[(n + 1) // 2 - 1])


class _Marquardt:
    """iejimexam.x's mr_solve: one Levenberg-Marquardt step per call, with
    its state between calls (f2c keeps SPP locals static)."""

    def __init__(self, xs, ys, flags):
        self.xs, self.ys, self.flags = xs, ys, flags

    def _eval(self, params):
        """mr_eval: curvature matrix, gradient and chi-squared."""
        nfit = len(self.flags)
        a = np.zeros((nfit, nfit), dtype=np.float32)
        delta = np.zeros(nfit, dtype=np.float32)
        chisq = np.float32(0)
        for x, y in zip(self.xs, self.ys, strict=True):
            ymod, dydp = _derivs(x, params)
            dy = np.float32(y - ymod)
            for j in range(nfit):
                dj = dydp[self.flags[j]]
                delta[j] = np.float32(delta[j] + np.float32(dy * dj))
                for k in range(j + 1):
                    a[j, k] = np.float32(a[j, k] + np.float32(dj * dydp[self.flags[k]]))
            chisq = np.float32(chisq + np.float32(dy * dy))
        for j in range(1, nfit):
            for k in range(j):
                a[k, j] = a[j, k]
        return a, delta, chisq

    def step(self, params, mr, chisq):
        from . import llsq

        f32 = np.float32
        nfit = len(self.flags)
        if mr < 0:
            self.new = params.copy()
            self.a2, self.delta2, chisq = self._eval(self.new)
            mr = f32(0.001)
        a1 = self.a2.copy()
        for i in range(nfit):
            a1[i, i] = f32(self.a2[i, i] * f32(f32(1) + mr))
        flat = a1.flatten(order="F")
        delta1 = self.delta2.copy()
        llsq.hfti(flat, nfit, nfit, nfit, delta1, nfit, 1, 1e-10)
        for i in range(nfit):
            self.new[self.flags[i]] = f32(params[self.flags[i]] + delta1[i])
        a1, delta1, chisq1 = self._eval(self.new)
        if chisq1 < chisq:
            mr = max(f32(1.192e-7), f32(f32(0.1) * mr))
            chisq = chisq1
            self.a2, self.delta2 = a1, delta1
            params[:] = self.new
        else:
            mr = f32(f32(10) * mr)
        return mr, chisq


def _derivs(x, a):
    """iejimexam.x's derivs: a1 + a2 x + a3 exp(-((x - a4) / a5)^2 / 2)."""
    f32 = np.float32
    arg = f32(f32(x - a[3]) / a[4])
    ex = f32(np.exp(-float(f32(f32(arg * arg) / f32(2))))) if abs(arg) < 7 else f32(0)
    fac = f32(f32(a[2] * ex) * arg)
    y = f32(f32(a[0] + f32(a[1] * x)) + f32(a[2] * ex))
    return y, (f32(1), f32(x), ex, f32(fac / a[4]), f32(f32(fac * arg) / a[4]))


def _gfit(xs, ys, fit, flags):
    """ie_gfit: Marquardt steps until chi-squared has stopped falling by
    more than one three times running, then a last step without damping."""
    f32 = np.float32
    solver = _Marquardt(xs, ys, flags)
    chi2 = f32(9.9e36)
    chi1 = f32(0)
    mr = f32(-1)
    calm = 0
    while calm != 3:
        mr, chi1 = solver.step(fit, mr, chi1)
        calm = 0 if f32(chi2 - chi1) > 1 else calm + 1
        chi2 = chi1
    solver.step(fit, f32(0), chi1)
    fit[4] = abs(fit[4])


def _gaussian_cut(data: np.ndarray, x: float, y: float, key: str, s: Settings) -> Result:
    """ie_jimexam: the j and k keys -- a gaussian plus a constant fitted
    along the line (j) or column (k) through the cursor, averaged over
    `naverage` of them, by IRAF's own Marquardt steps."""
    f32 = np.float32
    ny_image, nx_image = data.shape
    along_line = key == "j"
    xc, yc = (f32(x), f32(y)) if along_line else (f32(y), f32(x))
    # jimexam for j, kimexam for k: two psets with the same names.
    if along_line:
        sigma, navg, rplot, width = f32(s.sigma), s.naverage, f32(s.jrplot), f32(s.jwidth)
        centre_fit, background_fit, order = s.jcenter, s.jbackground, s.jxorder
    else:
        sigma, navg, rplot, width = f32(s.ksigma), s.knaverage, f32(s.krplot), f32(s.kwidth)
        centre_fit, background_fit, order = s.kcenter, s.kbackground, s.kxorder
    reach = max(rplot, f32(f32(f32(8) * sigma) + (width if background_fit else f32(0))))
    x1, x2 = int(f32(xc - reach)), int(f32(xc + reach))
    y1 = _nint(yc) - (navg - 1) // 2
    y2 = _nint(yc) + navg // 2
    length, across = (nx_image, ny_image) if along_line else (ny_image, nx_image)
    if x2 < 1 or x1 > length or y2 < 1 or y1 > across:
        return Result(key, x, y, f"[{key}] the cut is off the image", ok=False)
    x1, x2 = max(1, x1), min(length, x2)
    y1, y2 = max(1, y1), min(across, y2)
    if along_line:
        block = data[y1 - 1:y2, x1 - 1:x2].astype(f32)
        total = block[0].copy()
        for row in block[1:]:
            total = (total + row).astype(f32)
    else:
        block = data[x1 - 1:x2, y1 - 1:y2].astype(f32)
        total = np.array([np.cumsum(row, dtype=f32)[-1] for row in block], dtype=f32)
    nx, nav = x2 - x1 + 1, y2 - y1 + 1
    ys = (total / f32(nav)).astype(f32)
    what = (f"Lines {y1}-{y2}" if along_line else f"Columns {y1}-{y2}")

    # The background: the median of the cut beyond four sigmas.
    bkg = f32(0)
    if background_fit:
        offsets = np.abs(np.array([f32(f32(xc - x1) - i) for i in range(nx)], dtype=f32))
        outside = ys[offsets > f32(4) * sigma]
        bkg = f32(_amedr(outside)) if outside.size else f32(0)

    xs = np.arange(x1, x2 + 1, dtype=f32)
    k0 = min(nx - 1, max(0, _nint(f32(xc - x1))))
    dr = f32(abs(f32(xc - f32(xc + sigma))))
    fit = np.array([bkg, f32(0), f32(ys[k0] - bkg), xc, dr], dtype=f32)
    flags = [2]                                 # the amplitude, then
    if centre_fit:
        flags.append(3)                         # the centre,
        _gfit(xs, ys, fit, flags)
    flags.append(4)                             # the width,
    _gfit(xs, ys, fit, flags)
    if background_fit and order in (1, 2):      # and the background terms
        flags += [0] if order == 1 else [0, 1]
        _gfit(xs, ys, fit, flags)

    j0 = max(0, int(f32(f32(xc - x1) - rplot)))
    j1 = min(nx - 1, _nint(f32(f32(xc - x1) + rplot)))
    plot = {"kind": "cut", "title": f"{what}, gaussian fit",
            "x": xs[j0:j1 + 1].astype(float), "y": ys[j0:j1 + 1].astype(float),
            "xlabel": "x (pixels)" if along_line else "y (pixels)", "ylabel": "counts"}
    background, centre, peak, spread = (float(fit[0] + fit[1] * fit[3]), float(fit[3]),
                                        float(fit[2]), float(fit[4]))
    fwhm = float(f32(f32(2.35482) * fit[4]))
    if not np.isfinite(np.array([centre, peak, spread])).all():
        return Result(key, x, y, f"[{key}] {what}  the fit did not converge -- "
                      "is there a feature here?", plot=plot, ok=False)
    grid = np.linspace(plot["x"][0], plot["x"][-1], 200)
    arg = (grid - centre) / spread
    plot.update(fit_x=grid, fit_y=np.where(np.abs(arg) < 7, fit[0] + fit[1] * grid
                                           + peak * np.exp(-arg ** 2 / 2),
                                           fit[0] + fit[1] * grid))
    return Result(key, x, y, f"[{key}] {what}: center={centre:7g} peak={peak:7g} "
                  f"sigma={spread:7.4g} fwhm={fwhm:7.4g} bkg={background:7g}",
                  {"centre": centre, "peak": peak, "sigma": spread, "fwhm": fwhm,
                   "background": background}, plot)


def _nint(value) -> int:
    """SPP's nint: round half away from zero."""
    v = float(value)
    return int(np.floor(v + 0.5)) if v >= 0 else -int(np.floor(0.5 - v))


def _contour_levels(z: np.ndarray, floor, ceiling, finc):
    """CLGEN, NCAR's conrec as IRAF ships it: the levels from floor up to
    ceiling. finc < 0 asks for that many levels, each step cut to one
    significant figure; finc > 0 is the step; 0 lets conrec aim for 16."""
    f32 = np.float32
    glo, ha = f32(floor), f32(ceiling)
    if ha < glo:
        glo, ha = ha, glo
    if ha == glo:
        return [float(glo)], 0.0
    fanc = f32(finc)
    if fanc <= 0:
        crat = max(f32(1), -fanc) if fanc < 0 else f32(16)
        fanc = f32((ha - glo) / crat)
        power = f32(10.0 ** (int(math.log10(float(fanc)) + 5000.0) - 5000))
        fanc = f32(f32(np.trunc(f32(fanc / power))) * power)
    levels = []
    for k in range(40):
        level = f32(glo + f32(f32(k) * fanc))
        if level > ha:
            break
        levels.append(float(level))
    return levels, float(fanc)


def _contours(data: np.ndarray, x: float, y: float, s: Settings) -> Result:
    """ie_eimexam: the e key -- contours of an ncolumns x nlines box, at
    the levels conrec picks between floor and ceiling (the box's extremes
    when INDEF), shifted by zero."""
    f32 = np.float32
    got = _gdata(data, *_box_limits(x, y, s.encolumns, s.enlines))
    if got is None:
        return Result("e", x, y, "[e] pixels out of bounds", ok=False)
    x1, x2, y1, y2, box = got
    box = box.astype(f32)
    if not np.isfinite(box).all():
        return Result("e", x, y, f"[e] [{x1}:{x2},{y1}:{y2}] holds blank (NaN) pixels, "
                      "which have no contour -- fix them first (fixpix), or look "
                      "elsewhere", ok=False)
    floor = box.min() if s.efloor is None else f32(s.efloor)
    ceiling = box.max() if s.eceiling is None else f32(s.eceiling)
    zero = f32(s.zero)
    if abs(zero) > 1.192e-7:
        box = (box - zero).astype(f32)
        floor, ceiling = f32(floor - zero), f32(ceiling - zero)
    # conrec scales automatically on a zero limit; IRAF steps around it.
    floor = f32(1.192e-7) if floor == 0 else floor
    ceiling = f32(1.192e-7) if ceiling == 0 else ceiling
    finc = (-abs(s.ncontours) if s.ncontours > 0
            else (s.interval if s.interval > 0 else 0.0))
    levels, step = _contour_levels(box, floor, ceiling, finc)
    section = f"[{x1}:{x2},{y1}:{y2}]"
    return Result("e", x, y, f"[e] {section}  contoured from {levels[0]:g} to "
                  f"{levels[-1]:g}, interval = {step:g}",
                  {"section": section, "levels": levels, "interval": step},
                  {"kind": "contour", "title": f"contours {section}",
                   "x": np.arange(x1, x2 + 1), "y": np.arange(y1, y2 + 1),
                   "z": box.astype(float), "levels": levels,
                   "xlabel": "column", "ylabel": "line"})


def _surface(data: np.ndarray, x: float, y: float, s: Settings) -> Result:
    """ie_simexam: the s key -- the box seen from simexam's angh and angv,
    clipped to floor and ceiling when they are given."""
    got = _gdata(data, *_box_limits(x, y, s.sncolumns, s.snlines))
    if got is None:
        return Result("s", x, y, "[s] pixels out of bounds", ok=False)
    x1, x2, y1, y2, box = got
    box = box.astype(np.float32).astype(float)
    if s.sfloor is not None:
        box = np.maximum(box, s.sfloor)
    if s.sceiling is not None:
        box = np.minimum(box, s.sceiling)
    section = f"[{x1}:{x2},{y1}:{y2}]"
    return Result("s", x, y, f"[s] {section}  surface, angh={s.angh:g} angv={s.angv:g}",
                  {"section": section},
                  {"kind": "surface", "title": f"surface {section}",
                   "x": np.arange(x1, x2 + 1), "y": np.arange(y1, y2 + 1), "z": box,
                   "angh": s.angh, "angv": s.angv, "xlabel": "", "ylabel": ""})


# ------------------------------------------------------------ colon commands

#: Every colon command IRAF's imexamine knows (iecolon.x), for the prefix
#: matching strdic does: an abbreviation works when it is unambiguous.
COMMANDS = tuple(  # as IRAF's CMDS reads
    "angh angv background banner boundary box buffer ceiling center constant "  # noqa: SIM905
    "dashpat defkey eparam fill floor interval label logfile logx logy magzero "
    "majrx majry marker minrx minry naverage ncolumns ncontours ncstat nhi nlines "
    "nlstat pointmode radius round rplot select szmarker ticklabels title width x "
    "xlabel xorder y ylabel yorder zero unlearn autoredraw nbins z1 z2 autoscale "
    "top_closed allframes wcs xformat yformat fitplot sigma axes fittype beta "
    "iterations output ncoutput nloutput".split())

#: Only how IRAF draws its graphs: harwood's window draws its own.
_STYLING = set(
    "banner box dashpat fill label logx logy majrx majry marker minrx minry nhi "  # noqa: SIM905
    "pointmode round szmarker ticklabels title xlabel ylabel x y axes fitplot "
    "autoredraw".split())

#: Not in harwood: what to do instead.
_ELSEWHERE = {
    "eparam": "change the task's parameters with --editpar",
    "logfile": "name the log with the task's --logfile",
    "select": "use n and p to move between frames",
    "allframes": "harwood loads each image into the frame it is given",
    "wcs": "harwood's positions are logical pixels; the window shows the WCS",
    "xformat": "harwood's positions are logical pixels",
    "yformat": "harwood's positions are logical pixels",
    "yorder": "a surface fit to the annulus is not in harwood; the median is used",
}


def _field(name: str, graph: str) -> tuple[str | None, str]:
    """The Settings field a command sets, and the pset it lives in -- or None
    when, as IRAF says, the parameter does not apply to the current graph."""
    always = {
        "angh": "simexam", "angv": "simexam", "buffer": "rimexam", "radius": "rimexam",
        "magzero": "rimexam", "fittype": "rimexam", "beta": "rimexam",
        "iterations": "rimexam", "sigma": "jimexam", "ncstat": "imexamine",
        "nlstat": "imexamine", "nbins": "himexam", "z1": "himexam", "z2": "himexam",
        "autoscale": "himexam", "top_closed": "himexam", "zero": "eimexam",
        "ncontours": "eimexam", "interval": "eimexam", "defkey": "imexamine",
        "boundary": "vimexam", "constant": "vimexam", "output": "imexamine",
        "ncoutput": "imexamine", "nloutput": "imexamine",
    }
    if name in always:
        return name, always[name]
    prefix = {"r": "", "j": "j", "k": "k", "l": "l", "c": "c", "e": "e", "s": "s",
              "h": "", "u": "v", "v": "v", ".": ""}
    psets = {"r": "rimexam", "j": "jimexam", "k": "kimexam", "l": "limexam",
             "c": "cimexam", "e": "eimexam", "s": "simexam", "h": "himexam",
             "u": "vimexam", "v": "vimexam", ".": "rimexam"}
    applies = {"width": "rjk.", "rplot": "rjk.", "center": "rjk.", "background": "rjk.",
               "xorder": "jk", "naverage": "jklcuv", "ncolumns": "esh", "nlines": "esh",
               "floor": "es", "ceiling": "es"}
    if not graph or graph not in applies.get(name, ""):
        return None, ""
    if graph == "j" and name in ("naverage",):
        return "naverage", "jimexam"
    return prefix[graph] + name, psets[graph]


def colon(settings: Settings, command: str) -> str:
    """A colon command, as imexamine takes it at the cursor (iecolon.x):
    `:radius 7` sets, `:radius` shows. The parameters several psets share
    -- width, rplot, naverage, ncolumns, floor... -- belong to the pset of
    the last graph drawn (after a, to rimexam's), and elsewhere IRAF says
    the parameter does not apply to the current graph. Returns what IRAF
    would print, or raises ValueError (with a fix) for what harwood refuses."""
    from . import advice

    words = command.strip().lstrip(":").split()
    if not words:
        advice.fail("an empty colon command", "type a parameter name after the colon, "
                    "e.g. ':radius 7'")
    word, args = words[0], words[1:]
    matches = [c for c in COMMANDS if c.startswith(word)]
    if word in COMMANDS:
        matches = [word]
    if len(matches) != 1:
        advice.fail(f"':{word}' is an unrecognized or ambiguous command",
                    "imexamine's colon commands are " + " ".join(COMMANDS))
    name = matches[0]
    if name in _STYLING:
        advice.fail(f"':{name}' sets how IRAF draws its graphs",
                    "harwood's window draws its own; nothing to set")
    if name in _ELSEWHERE:
        advice.fail(f"':{name}' is not available in harwood", _ELSEWHERE[name])

    graph = settings.graph
    if settings.labelkey in ("a", ",") and name in ("background", "center", "naverage",
                                                    "rplot", "xorder", "width"):
        graph = "r"
    if name == "unlearn":
        pset = {"r": "rimexam", "j": "jimexam", "k": "kimexam", "l": "limexam",
                "c": "cimexam", "e": "eimexam", "s": "simexam", "h": "himexam"}.get(graph)
        if pset is None:
            return "Parameter does not apply to current graph"
        fresh = Settings()
        for field_name in _PSET_FIELDS[pset]:
            setattr(settings, field_name, getattr(fresh, field_name))
        return f"{pset} unlearned"
    if name == "xorder" and graph == "r":
        advice.fail("':xorder' of rimexam fits a surface to the annulus, which "
                    "harwood does not", "the annulus' median is used (xorder 0)")
    found, _ = _field(name, graph)
    if found is None:
        return "Parameter does not apply to current graph"
    field_name = found
    current = getattr(settings, field_name)
    if not args:
        shown = ("INDEF" if current is None
                 else ("yes" if current else "no") if isinstance(current, bool)
                 else f"{current:g}" if isinstance(current, float) else current)
        return f"{name} {shown}"
    parsed = _parse(name, args[0], current)
    setattr(settings, field_name, parsed)
    settings.redraw = True
    return f"{name} {args[0]}"


def _parse(name: str, text: str, current):
    from . import advice

    kinds = {"fittype": str, "defkey": str, "beta": "real?", "z1": "real?",
             "z2": "real?", "floor": "real?", "ceiling": "real?"}
    kind = kinds.get(name) or type(current)
    try:
        if kind is bool:
            if text.lower() not in ("yes", "no", "y", "n"):
                raise ValueError
            return text.lower().startswith("y")
        if kind == "real?":
            return None if text.upper() == "INDEF" else float(text)
        if kind is int:
            return int(text)
        if kind is float:
            return float(text)
    except ValueError:
        advice.fail(f"':{name} {text}': '{text}' is not a value for {name}",
                    f"give {name} a {'yes/no' if kind is bool else 'number'}")
    if name == "fittype" and text not in ("gaussian", "moffat"):
        advice.fail(f"':fittype {text}' is not a profile", "Profile fit types are "
                    "|gaussian|moffat|")
    return text


#: Which fields each pset owns, for :unlearn.
_PSET_FIELDS = {
    "rimexam": ["radius", "buffer", "width", "rplot", "fittype", "beta", "iterations",
                "magzero", "center", "background"],
    "jimexam": ["naverage", "sigma", "jwidth", "jrplot", "jcenter", "jbackground",
                "jxorder"],
    "kimexam": ["knaverage", "ksigma", "kwidth", "krplot", "kcenter", "kbackground",
                "kxorder"],
    "limexam": ["lnaverage"], "cimexam": ["cnaverage"],
    "himexam": ["ncolumns", "nlines", "nbins", "z1", "z2", "autoscale", "top_closed"],
    "eimexam": ["encolumns", "enlines", "efloor", "eceiling", "zero", "ncontours",
                "interval"],
    "simexam": ["sncolumns", "snlines", "sfloor", "sceiling", "angh", "angv"],
}
