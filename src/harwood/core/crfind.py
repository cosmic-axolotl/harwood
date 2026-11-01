"""IRAF's cosmicrays: candidates, their flux ratios, what training and the graph do to them,
and the passes that replace them, in real, in IRAF's order. core.cosmic has L.A.Cosmic, the
default.

A pixel is a candidate when it is the highest of its window and stands `threshold` above the
mean of it, leaving out it and the next highest. A plane through the border of the window comes
off, and its ratio is 100 times that mean over the pixel: a cosmic ray stands alone, and has a
low ratio, and the peak of a star a high one, so `fluxratio` tells them apart, as far as they
can be told. The replacement is the mean of the four nearest, but for the next highest.

From iraf-community 2.18: noao/imred/crutil/src/{t_cosmicrays,crfind,crlist,crexamine}.x, and
the plane from math/gsurfit. See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from . import gio, iraffmt
from .jit import jit

REAL = np.float32
MAX_REAL = 3.4028234663852886e38
NO, YES, ALWAYSNO, ALWAYSYES = 0, 1, 3, 4
RMAX = 3.0  # CR_RMAX: how near a training position has to be
BUFFER = 0.03  # GT_XBUF and GT_YBUF

HELP = [
    ("a", "all the candidates, or the training objects"),
    ("d", "replace the candidate nearest the cursor"),
    ("e", "replace those in a region: e at two corners"),
    ("q", "quit, and go on to replace what is marked"),
    ("r", "redraw"),
    ("s", "the pixels around the nearest candidate"),
    ("t", "set the flux ratio threshold at the cursor"),
    ("u", "do not replace the nearest"),
    ("v", "do not replace those in a region: v at two corners"),
    ("space", "print the column and line of the nearest"),
]


def _grid(window: int) -> tuple[np.ndarray, np.ndarray]:
    """t_cosmicrays' surface order: the border first, round it, and then the inside."""
    xs, ys = [], []
    for i in range(1, window + 1):
        xs.append(i)
        ys.append(1)
    for i in range(2, window + 1):
        xs.append(window)
        ys.append(i)
    for i in range(window - 1, 0, -1):
        xs.append(i)
        ys.append(window)
    for i in range(window - 1, 1, -1):
        xs.append(1)
        ys.append(i)
    for i in range(2, window):
        for j in range(2, window):
            xs.append(j)
            ys.append(i)
    return np.array(xs, REAL), np.array(ys, REAL)


@jit
def _factor(matrix, factored):
    """gschofac, three by three."""
    real = np.float32
    for n in range(3):
        for j in range(3):
            factored[n, j] = matrix[n, j]
    limit = real(1000.0) / real(3.4028234663852886e38)
    for n in range(3):
        if real(real(factored[n, 0] + matrix[n, 0]) - matrix[n, 0]) <= limit:
            for j in range(3):
                factored[n, j] = real(0.0)
            continue
        factored[n, 0] = real(real(1.0) / factored[n, 0])
        imax = min(2, 2 - n)
        jmax = imax
        for i in range(1, imax + 1):
            ratio = real(factored[n, i] * factored[n, 0])
            for j in range(jmax):
                factored[n + i, j] = real(factored[n + i, j]
                                          - real(factored[n, j + i] * ratio))
            jmax -= 1
            factored[n, i] = ratio


@jit
def _solve(factored, vector, coeff):
    """gschoslv."""
    real = np.float32
    for i in range(3):
        coeff[i] = vector[i]
    for n in range(3):
        for j in range(1, min(2, 2 - n) + 1):
            coeff[j + n] = real(coeff[j + n] - real(factored[n, j] * coeff[n]))
    for n in range(2, -1, -1):
        coeff[n] = real(coeff[n] * factored[n, 0])
        for j in range(1, min(2, 2 - n) + 1):
            coeff[n] = real(coeff[n] - real(factored[n, j] * coeff[j + n]))


@jit
def _fit(x, y, z, w, npts, matrix, factored, coeff):
    """gsfit of the plane: gszero, gsacpts, gssolve."""
    real = np.float32
    matrix[:, :] = real(0.0)
    vector = np.zeros(3, real)
    for i in range(npts):
        # l = 1, k = 1
        bw = real(w[i] * real(1.0))
        vector[0] = real(vector[0] + real(bw * z[i]))
        matrix[0, 0] = real(matrix[0, 0] + bw)
        matrix[0, 1] = real(matrix[0, 1] + real(bw * x[i]))
        matrix[0, 2] = real(matrix[0, 2] + real(bw * y[i]))
    for i in range(npts):
        # l = 1, k = 2
        bw = real(w[i] * x[i])
        vector[1] = real(vector[1] + real(bw * z[i]))
        matrix[1, 0] = real(matrix[1, 0] + real(bw * x[i]))
        matrix[1, 1] = real(matrix[1, 1] + real(bw * y[i]))
    for i in range(npts):
        # l = 2, k = 1
        bw = real(w[i] * y[i])
        vector[2] = real(vector[2] + real(bw * z[i]))
        matrix[2, 0] = real(matrix[2, 0] + real(bw * y[i]))
    _factor(matrix, factored)
    _solve(factored, vector, coeff)


@jit
def _refit(x, y, z, w, npts, factored, coeff):
    """gsrefit: the same factors, the vector of a new z."""
    real = np.float32
    vector = np.zeros(3, real)
    for i in range(npts):
        wz = real(w[i] * z[i])
        vector[0] = real(vector[0] + wz)
        vector[1] = real(vector[1] + real(wz * x[i]))
        vector[2] = real(vector[2] + real(wz * y[i]))
    _solve(factored, vector, coeff)


@jit
def _order(window):
    """Which place of the image's window goes where in the surface's order, 0-based
    (row, column) for each, as crfind's copies have it; and for 5 the place of the centre."""
    n = window * window
    rows = np.zeros(n, np.int64)
    cols = np.zeros(n, np.int64)
    k = 0
    for j in range(window):
        rows[k], cols[k] = 0, j
        k += 1
    for i in range(1, window):
        rows[k], cols[k] = i, window - 1
        k += 1
    for j in range(window - 2, -1, -1):
        rows[k], cols[k] = window - 1, j
        k += 1
    for i in range(window - 2, 0, -1):
        rows[k], cols[k] = i, 0
        k += 1
    for i in range(1, window - 1):
        for j in range(1, window - 1):
            rows[k], cols[k] = i, j
            k += 1
    if window == 7:
        # z[16] = f[i4] in cr_find7, the one place it is not on the border: g[i4] is left out
        rows[15], cols[15] = 5, 3
    return rows, cols


@jit
def _find_row(block, n, first_col, line, threshold, window, x, y, sf1, coeff, matrix2, factor2,
              w, out_col, out_line, out_flux, out_ratio, out_replace, start):
    """cr_find5 and cr_find7 over one line. `block` is the window's rows over n columns,
    first_col being the column of the first. It returns how many it has added."""
    real = np.float32
    half = window // 2
    npix = window * window
    nborder = 4 * window - 4
    nobig = npix - 2
    rows, cols = _order(window)
    # z[j1], the centre, by the inner part's order: row half, column half
    j1 = nborder + (half - 1) * (window - 2) + (half - 1)
    # The four neighbours, by the same order
    up = nborder + (half - 2) * (window - 2) + (half - 1)
    left = nborder + (half - 1) * (window - 2) + (half - 2)
    right = nborder + (half - 1) * (window - 2) + half
    down = nborder + half * (window - 2) + (half - 1)
    z = np.zeros(npix, real)
    added = 0
    i = half
    while i <= n - 1 - half:
        p = block[half, i]
        maximum = True
        for a in range(window):
            for b in range(i - half, i + half + 1):
                if p < block[a, b]:
                    maximum = False
                    break
            if not maximum:
                break
        if not maximum:
            i += 1
            continue
        for k in range(npix):
            z[k] = block[rows[k], i - half + cols[k]]
        j2 = 0
        for j in range(1, npix):
            if j == j1:
                continue
            if z[j] > z[j2]:
                j2 = j
        total = real(0.0)
        for k in range(npix):
            total = real(total + z[k])
        flux = real(real(real(total - z[j1]) - z[j2]) / real(nobig))
        if p < real(flux + threshold):
            i += 1
            continue
        if j2 < nborder:
            w[j2] = real(0.0)
            _fit(x, y, z, w, nborder, matrix2, factor2, coeff)
            w[j2] = real(1.0)
        else:
            _refit(x, y, z, w, nborder, sf1, coeff)
        for k in range(npix):
            value = real(real(coeff[0] + real(x[k] * coeff[1])) + real(y[k] * coeff[2]))
            z[k] = real(z[k] - value)
        p = z[j1]
        total = real(0.0)
        for k in range(npix):
            total = real(total + z[k])
        flux = real(real(real(total - z[j1]) - z[j2]) / real(nobig))
        replace = real(0.0)
        count = 0
        if j2 != up:
            replace = real(replace + block[half - 1, i])
            count += 1
        if j2 != left:
            replace = real(replace + block[half, i - 1])
            count += 1
        if j2 != right:
            replace = real(replace + block[half, i + 1])
            count += 1
        if j2 != down:
            replace = real(replace + block[half + 1, i])
            count += 1
        replace = real(replace / real(count))
        flux = real(real(100.0) * flux)
        at = start + added
        if at >= out_col.size:
            return -added - 1
        out_col[at] = first_col + i
        out_line[at] = line
        out_flux[at] = flux
        out_ratio[at] = real(flux / p)
        out_replace[at] = replace
        added += 1
        i += half + 1
    return added


@dataclass
class Candidates:
    """cr_open's list."""

    col: list = field(default_factory=list)
    line: list = field(default_factory=list)
    flux: list = field(default_factory=list)
    ratio: list = field(default_factory=list)
    wt: list = field(default_factory=list)
    replace: list = field(default_factory=list)
    flag: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.col)


class Finder:
    """The plane's fixed parts, and cr_find with the list it adds to."""

    def __init__(self, threshold: float, window: int) -> None:
        self.threshold = REAL(threshold)
        self.window = window
        self.x, self.y = _grid(window)
        self.w = np.zeros(window * window, REAL)
        nborder = 4 * window - 4
        self.w[:nborder] = REAL(1.0)
        self.matrix1 = np.zeros((3, 3), REAL)
        self.factor1 = np.zeros((3, 3), REAL)
        self.coeff = np.zeros(3, REAL)
        self.matrix2 = np.zeros((3, 3), REAL)
        self.factor2 = np.zeros((3, 3), REAL)
        _fit(self.x, self.y, np.zeros(window * window, REAL), self.w, nborder, self.matrix1,
             self.factor1, self.coeff)

    def find(self, block: np.ndarray, first_col: int, line: int, found: Candidates) -> None:
        room = 256
        while True:
            col = np.zeros(room, np.int64)
            lines = np.zeros(room, np.int64)
            flux, ratio = np.zeros(room, REAL), np.zeros(room, REAL)
            replace = np.zeros(room, REAL)
            added = _find_row(block, block.shape[1], first_col, line, self.threshold,
                              self.window, self.x, self.y, self.factor1, self.coeff,
                              self.matrix2, self.factor2, self.w, col, lines, flux, ratio,
                              replace, 0)
            if added >= 0:
                break
            room *= 4
        found.col += col[:added].tolist()
        found.line += lines[:added].tolist()
        found.flux += flux[:added].tolist()
        found.ratio += ratio[:added].tolist()
        found.replace += replace[:added].tolist()
        found.wt += [0.0] * added
        found.flag += [NO] * added


def _padded(data: np.ndarray, half: int) -> np.ndarray:
    """The image with IM_TYBNDRY BT_NEAREST around it."""
    return np.pad(np.asarray(data, REAL), half, mode="edge")


def flags(found: Candidates, fluxratio) -> None:
    """cr_flags."""
    limit = REAL(fluxratio)
    for i in range(len(found)):
        if found.flag[i] in (ALWAYSYES, ALWAYSNO):
            continue
        found.flag[i] = NO if REAL(found.ratio[i]) > limit else YES


def first_pass(data: np.ndarray, threshold: float, window: int) -> tuple[Candidates, Finder]:
    finder = Finder(threshold, window)
    half = window // 2
    padded = _padded(data, half)
    found = Candidates()
    for number in range(data.shape[0]):
        finder.find(padded[number:number + window], 1 - half, number + 1, found)
    return found, finder


def replace_from(found: Candidates, offset: int, image: np.ndarray,
                 disk: np.ndarray | None = None, pixtype=REAL) -> int:
    """cr_replace: how many. `disk`, if given, is what a read would see: a write reaches it
    with the next write."""
    count = 0
    last = None
    for i in range(offset, len(found)):
        if found.flag[i] in (NO, ALWAYSNO):
            continue
        at = found.line[i] - 1, found.col[i] - 1
        if disk is not None:
            if last is None:
                disk[:] = image
            else:
                disk[last] = image[last]
        image[at] = np.asarray(found.replace[i], REAL).astype(pixtype)
        last = at
        count += 1
    return count


def later_passes(image: np.ndarray, found: Candidates, finder: Finder, fluxratio, npasses: int,
                 pixtype=REAL) -> list[int]:
    """The passes after the first, around what the last one replaced: how many each replaced.

    A pass reads what IMIO has written, and the last pixel a pass replaces waits in the output
    buffer for the next write: the next pass still sees it as it was, and may find it again.
    """
    window = finder.window
    half = window // 2
    ny, nx = image.shape
    each = []
    ncrlast = 0
    disk = image.copy()
    for _ in range(2, npasses + 1):
        ncr = len(found)
        padded = _padded(disk, half)
        for j in range(ncrlast, ncr):
            if found.flag[j] in (NO, ALWAYSNO):
                continue
            c, l0 = found.col[j], found.line[j]
            c1 = max(1 - half, c - (window - 1))
            c2 = min(nx + half, c + (window - 1))
            top = max(1 - half, l0 - (window - 1))
            bottom = min(ny + half, l0 + (window - 1))
            section = padded[top - 1 + half:bottom + half, c1 - 1 + half:c2 + half]
            first = max(1, l0 - half)
            last = min(ny, l0 + half)
            for t, line in enumerate(range(first, last + 1)):
                finder.find(section[t:t + window], c1, line, found)
        flags(found, fluxratio)
        each.append(replace_from(found, ncr, image, disk, pixtype))
        ncrlast = ncr
    return each


def weight(fluxratio, found: Candidates):
    """cr_weight."""
    out = REAL(0.0)
    for r, w in zip(found.ratio, found.wt, strict=True):
        if REAL(r) > fluxratio:
            if w < 0:
                out = REAL(out - REAL(w))
        elif w > 0:
            out = REAL(out + REAL(w))
    return out


def find_threshold(found: Candidates):
    """cr_findthresh."""
    fluxratio = REAL(found.ratio[0])
    least = weight(fluxratio, found)
    for i in range(1, len(found)):
        if found.wt[i] == 0.0:
            continue
        r = REAL(found.ratio[i])
        w = weight(r, found)
        if w <= least:
            if w == least:
                fluxratio = min(fluxratio, r)
            else:
                least = w
                fluxratio = r
    return fluxratio


def _axis(place: int, logs: bool) -> float:
    """The world of a whole place on glabax's axis, its log or elog, always rounded out."""
    if logs or abs(place) > 1:
        return math.copysign(10.0 ** abs(place), place) if not logs else 10.0 ** place
    return 10.0 * place


@dataclass
class Session:
    """cr_examine's keys and cr_train's, with no window: the graph is flux against flux ratio,
    with a threshold. A read is in the coordinates of its own: flux and ratio for the graph,
    column and line for the training."""

    found: Candidates
    fluxratio: float
    title: str = ""
    show: int = 0                #: 0 all, 1 the training objects
    limits: list = field(default_factory=lambda: [10.0, None, 0.0, None])   #: gtools'
    flips: list = field(default_factory=lambda: [False, False])
    window: tuple = (10.0, 100.0, 0.0, 1.0)
    logs: bool = True
    printed: list = field(default_factory=list)
    save: list = field(default_factory=list)
    done: bool = False
    newgraph: bool = False
    state: str = ""
    first: tuple = (0.0, 0.0)
    trained: int = 0
    graphing: bool = False

    def __post_init__(self) -> None:
        # glb_set_viewport on stdvdm, cw .0125 and ch .0294: a title of the system's line, the
        # image's if it has one, and the graph's
        lines = 3 if self.title.strip() else 2
        self._width = 1.0 - 2 * 11 * 0.0125
        self._height = 1.0 - 5 * 0.0294 - (lines + 1) * 0.0294
        self.graph()

    # ------------------------------------------------------------- the graph

    def shown(self) -> list[int]:
        """cr_show."""
        if self.show == 0:
            return list(range(len(self.found)))
        return [i for i in range(len(self.found)) if self.found.wt[i] != 0.0]

    def graph(self) -> None:
        """gt_ascale, gt_swind and glabax on the log axis of flux. A list with nothing to show
        leaves the window as it was."""
        shown = self.shown()
        if not shown:
            return
        xs = np.array([self.found.flux[i] for i in shown], float)
        ys = np.array([self.found.ratio[i] for i in shown], float)
        xmin, xmax, ymin, ymax = (
            ((MAX_REAL if at % 2 else -MAX_REAL) if value is None else value)
            for at, value in enumerate(self.limits))
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        inside = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
        # gclear has left the window at unity, and with nothing in it gt_ascale leaves it there
        x1, x2, y1, y2 = 0.0, 1.0, 0.0, 1.0
        if inside.any():
            x1, x2 = xs[inside].min(), xs[inside].max()
            y1, y2 = ys[inside].min(), ys[inside].max()
        dx, dy = x2 - x1, y2 - y1
        low = x1 - BUFFER * dx if self.limits[0] is None else self.limits[0]
        high = x2 + BUFFER * dx if self.limits[1] is None else self.limits[1]
        bottom = y1 - BUFFER * dy if self.limits[2] is None else self.limits[2]
        top = y2 + BUFFER * dy if self.limits[3] is None else self.limits[3]
        if self.flips[0]:
            low, high = high, low
        if self.flips[1]:
            bottom, top = top, bottom
        self.logs = low > 0 and high > 0
        out = (math.floor, math.ceil) if low <= high else (math.ceil, math.floor)
        self.window = (_axis(out[0](self._scale(low)), self.logs),
                       _axis(out[1](self._scale(high)), self.logs),
                       *gio.widened(bottom, top))

    def _scale(self, x: float) -> float:
        return math.log10(x) if self.logs else float(gio.elog(x))

    def _ndc(self, x: float, y: float) -> tuple[float, float] | None:
        """gctran, up to where the viewport starts. On a log axis what is at or under zero has
        no place, and is too far for anything to be nearer; glabax takes a window that reaches
        it to its elog axis, where it has one."""
        if self.logs and x <= 0:
            return None
        x0, x1, y0, y1 = self.window
        start, end = self._scale(x0), self._scale(x1)
        return ((self._scale(x) - start) / (end - start) * self._width,
                (y - y0) / (y1 - y0) * self._height)

    def nearest(self, wx: float, wy: float, skip=()) -> int | None:
        """cr_nearest, cr_delete, cr_undelete: the nearest of those not to skip."""
        cursor = self._ndc(wx, wy)
        if cursor is None:
            return None
        best, nearest = MAX_REAL, None
        for i in self.shown():
            if self.found.flag[i] in skip:
                continue
            point = self._ndc(self.found.flux[i], self.found.ratio[i])
            if point is None:
                continue
            r2 = float(REAL(REAL(point[0] - cursor[0]) ** 2 + REAL(point[1] - cursor[1]) ** 2))
            if r2 < best:
                best, nearest = r2, i
        return nearest

    def _region(self, a, b, deleting: bool) -> None:
        """cr_delete_reg and cr_undelete_reg."""
        one, two = self._ndc(*a), self._ndc(*b)
        if one is None or two is None:
            return
        low_x, high_x = min(one[0], two[0]), max(one[0], two[0])
        low_y, high_y = min(one[1], two[1]), max(one[1], two[1])
        skip = (YES, ALWAYSYES) if deleting else (NO, ALWAYSNO)
        for i in self.shown():
            if self.found.flag[i] in skip:
                continue
            point = self._ndc(self.found.flux[i], self.found.ratio[i])
            if point and low_x < point[0] < high_x and low_y < point[1] < high_y:
                self._mark(i, deleting)

    def _mark(self, i: int, deleting: bool) -> None:
        self.found.flag[i] = ALWAYSYES if deleting else ALWAYSNO
        self.found.wt[i] = -1.0 if deleting else 1.0

    def update(self, wy, show: int | None = None) -> None:
        """cr_update: the threshold, and what it decides."""
        self.fluxratio = float(REAL(wy))
        if (self.show if show is None else show) == 1:
            return
        for i in range(len(self.found)):
            flag = self.found.flag[i]
            if flag in (ALWAYSYES, ALWAYSNO):
                continue
            if flag == NO:
                if REAL(self.found.ratio[i]) < REAL(self.fluxratio):
                    self.found.flag[i] = YES
            elif REAL(self.found.ratio[i]) >= REAL(self.fluxratio):
                self.found.flag[i] = NO

    def _window(self, key: str, wx: float, wy: float) -> None:
        """gt_window1, for what the mouse does in the panel."""
        x1, x2, y1, y2 = self.window
        dx, dy = x2 - x1, y2 - y1
        set_to = {
            "a": (None, None, None, None), "b": (..., ..., wy, ...), "j": (wx, ..., ..., ...),
            "k": (..., wx, ..., ...), "t": (..., ..., ..., wy),
            "c": (wx - dx / 2, wx + dx / 2, wy - dy / 2, wy + dy / 2),
            "d": (..., ..., y1 - 0.75 * dy, y2 - 0.75 * dy),
            "l": (x1 - 0.75 * dx, x2 - 0.75 * dx, ..., ...), "m": (None, None, ..., ...),
            "n": (..., ..., None, None), "p": (wx - dx, wx + dx, wy - dy, wy + dy),
            "r": (x1 + 0.75 * dx, x2 + 0.75 * dx, ..., ...),
            "u": (..., ..., y1 + 0.75 * dy, y2 + 0.75 * dy),
            "x": (wx - dx / 4, wx + dx / 4, ..., ...),
            "y": (..., ..., wy - dy / 4, wy + dy / 4),
            "z": (wx - dx / 4, wx + dx / 4, wy - dy / 4, wy + dy / 4),
        }
        if key in ("f", "g"):
            at = 0 if key == "f" else 1
            self.flips[at] = not self.flips[at]
        elif key in set_to:
            for at, value in enumerate(set_to[key]):
                if value is not ...:
                    self.limits[at] = value
        else:
            return
        self.newgraph = True

    # -------------------------------------------------------------- the keys

    def key(self, key: str, wx: float = 0.0, wy: float = 0.0) -> None:
        """A read of cr_examine's cursor loop."""
        state, self.state = self.state, ""
        if state in ("e", "v"):
            self._region(self.first, (wx, wy), state == "e")
        elif state == "any":
            self.newgraph = True
        elif state == "window":
            if key == "e":
                self.first, self.state = (wx, wy), "window2"
            else:
                self._window(key, wx, wy)
        elif state == "window2":
            x1, x2, y1, y2 = self.window
            if abs(wx - self.first[0]) > 0.001 * abs(x2 - x1):
                self.limits[:2] = self.first[0], wx
            if abs(wy - self.first[1]) > 0.001 * abs(y2 - y1):
                self.limits[2:] = self.first[1], wy
            self.newgraph = True
        elif key == "a":
            self.show = 0 if self.show else 1
            self.newgraph = True
        elif key == "d":
            nearest = self.nearest(wx, wy, (YES, ALWAYSYES))
            if nearest is not None:
                self._mark(nearest, True)
        elif key == "u":
            nearest = self.nearest(wx, wy, (NO, ALWAYSNO))
            if nearest is not None:
                self._mark(nearest, False)
        elif key in ("e", "v"):
            self.first, self.state = (wx, wy), key
        elif key == "q":
            self.done = True
        elif key == "r":
            self.newgraph = True
        elif key == "s":
            self.state = "any"
        elif key == "t":
            self.update(wy)
        elif key == "w":
            self.state = "window"
        elif key == " ":
            nearest = self.nearest(wx, wy)
            if nearest is not None:
                self.printed.append(f"{self.found.col[nearest]} {self.found.line[nearest]}")
        if self.newgraph:
            self.graph()
            self.newgraph = False

    def train(self, key: str, x: float, y: float, wcs: int = 1, examine=None) -> None:
        """A read of cr_train's loop: s and c at the image, the candidate nearest it within
        CR_RMAX, and g the graph, which is examine(session)."""
        found = self.found
        if key == "q":
            self.done = True
        elif key == "g":
            if not self.graphing:
                if self.trained > 1:
                    self.fluxratio = float(find_threshold(found))
                flags(found, self.fluxratio)
                self.graphing = True
            if examine is not None:
                examine(self)
                self.done = False
        elif key in ("s", "c") and len(found):
            self._train_on(key, REAL(x), REAL(y), wcs)

    def _train_on(self, key: str, x, y, wcs: int) -> None:
        found = self.found
        count = len(found)

        def r2(i: int):
            dc, dl = REAL(REAL(found.col[i - 1]) - x), REAL(REAL(found.line[i - 1]) - y)
            return REAL(REAL(dc * dc) + REAL(dl * dl))

        low, high = REAL(y - REAL(RMAX)), REAL(y + REAL(RMAX))
        # 1-based, as it goes by tens to where the lines start
        i = 10
        while i < count and low > REAL(found.line[i - 1]):
            i += 10
        j = i - 9
        rmin = r2(j)
        i = j + 1
        while i < count and high > REAL(found.line[i - 1]):
            r = r2(i)
            if r < rmin:
                rmin, j = r, i
            i += 1
        if float(np.sqrt(np.float64(rmin))) > RMAX:
            return
        found.wt[j - 1] = 1.0 if key == "s" else -1.0
        found.flag[j - 1] = ALWAYSNO if key == "s" else ALWAYSYES
        self.trained += 1
        if self.graphing and self.trained > 1:
            self.update(find_threshold(found), 0)
        self.save.append(f"{iraffmt.g(float(x))} {iraffmt.g(float(y))} {wcs} {key}\n")


def reads(text: str) -> Iterator[tuple[float, float, int, str]]:
    """The lines of a cursor file, x y wcs key: a blank is written \\040."""
    for line in text.splitlines():
        words = line.split(None, 3)
        if len(words) < 4:
            continue
        key = " " if words[3].strip() == "\\040" else words[3].strip()
        yield float(words[0]), float(words[1]), int(float(words[2])), key


def examine_reads(session: Session, cursor: Iterator) -> None:
    """cr_examine, from a file of reads. The next call goes on where this one has stopped."""
    session.done = False
    for x, y, _, key in cursor:
        session.key(key, x, y)
        if session.done:
            return


@dataclass
class Result:
    image: np.ndarray
    mask: np.ndarray
    replaced: int
    per_pass: list[int]
    found: Candidates
    fluxratio: float
    saved: list[str]
    printed: list[str]


def run(data: np.ndarray, threshold: float = 25.0, fluxratio: float = 2.0, npasses: int = 5,
        window: int = 5, objects: Iterable | None = None, examine=None,
        pixtype=REAL, title: str = "") -> Result:
    """t_cosmicrays on one image. `objects` are the reads of the training, and `examine` what
    looks at the graph, given the session: None, for no one. A `pixtype` of integers holds
    what is replaced as integers, and later passes see them so."""
    data = np.asarray(data, REAL)
    ny, nx = data.shape
    if nx < window or ny < window:
        raise ValueError("Image size is too small")
    found, finder = first_pass(data, threshold, window)
    session = Session(found, float(REAL(fluxratio)), title)
    if objects is not None:
        for x, y, wcs, key in objects:
            session.train(key, x, y, wcs, examine)
            if session.done:
                break
    flags(found, session.fluxratio)
    if examine is not None:
        session.graph()
        examine(session)
    image = data.copy()
    each = [replace_from(found, 0, image, pixtype=pixtype)]
    if npasses > 1:
        each += later_passes(image, found, finder, session.fluxratio, npasses, pixtype)
    mask = np.zeros(data.shape, bool)
    for i in range(len(found)):
        if found.flag[i] not in (NO, ALWAYSNO):
            mask[found.line[i] - 1, found.col[i] - 1] = True
    return Result(image, mask, sum(each), each, found, session.fluxratio, session.save,
                  session.printed)
