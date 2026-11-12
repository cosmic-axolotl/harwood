"""wcslab -- a world coordinate grid, labelled, over an image's pixels.

From pkg/images/tv/wcslab (iraf-community 2.18, Jonathan Eisenhamer's, STScI):
t_wcslab, wcslab (wl_create, wl_gr_inparams, wl_get_system_type), wl_wcslab and
wl_graphics, wl_setup (the pole, the graph type, the ranges from twenty points of
the edges, wl_ra_range, wl_round_ra/dec and wl_generic_round, the rotation), wl_grid
(the lines of constant axis 1 and axis 2, broken into segments, where they cross
the edges: wl_point_to_label, wl_axis_on_line, wl_mark_tick, wl_vector_angle),
wl_label (the sides, wl_lab_edges, wl_full_label_position, wl_write_label and its
offsets, wl_hms and wl_dms, wl_polar_label, wl_title) and wl_map_viewport,
wl_imd_viewport. What it asks of GIO -- three coordinate systems (the caller's, an
NDC one over the viewport, the plot's from the pixels to the viewport), gctran
between them, clipping to the viewport (and off for ticks drawn outwards), gmark's
plus at the pole and gtext's attributes -- is emulated here, and the drawing comes
out in NDC: polylines already clipped, and texts with their up vector, size and
justification.

What IRAF does and harwood keeps:

* The system: two axes whose MWCS function is tan, arc, sin or tnx are RA and Dec
  (hms and dms labels, the pole, polar and near-polar graphs); two linear ones a
  linear system; anything else (zea, car, no WCS ...) the same rounding as linear
  in world units. RA goes on the axis whose CTYPE says RA, wherever it is.
* The ranges come from twenty points on the edges only, so a field whose extreme
  declination is mid-edge is rounded from the corners'; begin, end and interval
  are all recomputed when any one of the three is not given.
* A label is written once per crossing of an edge; the nearest to the corner of
  the two labelled sides is written in full (h m s), unless one of the others
  already is (a round value); the units ("h m s", the degree's o) are a second text
  half a character above the number.
* With rotate (the default) the labels of a side take the mean angle of their lines
  on the page, and within 10 degrees of horizontal or vertical are written
  horizontally; on a polar graph with labels inside, each its own, through
  wl_string_angle twice.
* The view the labels need is added to the viewport's sides, and the titles go
  outside that: two character heights beyond.

IRAF's MWCS has a slip in mw_v2tran (sys/mwcs/mwv2tran.gx): for a linear transform
with a rotation (its LRO branch) it reads p2[1], the output it has not written,
where it means p1[2], the input's y. wcslab meets it twice. In wl_rotate (real), the
offsets of a rotated label's units and of a label inside lose their y: that is kept
(see _rotate) -- the units then share the number's line, 13h28m00s, which IRAF's
users see and read. In wl_w2ld and wl_l2wd (double), a linear WCS with a rotation
loses the axis it varies along and every line of constant axis 1 is a point, with
nothing labelled: harwood draws that grid as the WCS says.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research
in Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import iraffmt

f32 = np.float32
LEFT, RIGHT, BOTTOM, TOP = 1, 2, 3, 4
SIDES = {"left": LEFT, "right": RIGHT, "bottom": BOTTOM, "top": TOP}
SIDE_NAMES = {v: k for k, v in SIDES.items()}
NORMAL, POLAR, NEAR_POLAR = 1, 2, 3
GRAPH_TYPES = {"normal": NORMAL, "polar": POLAR, "near_polar": NEAR_POLAR}
RA_DEC, LINEAR, OTHER = 1, 2, 3
SYSTEM_NAMES = {RA_DEC: "ra_dec", LINEAR: "linear", OTHER: "other"}
HOUR = DEGREE = 1
MINUTE, SECOND, SUBSEC_LOW, SUBSEC_HIGH = 2, 3, 4, 5
LINE_TYPES = {"solid": 1, "dashed": 2, "dotted": 3, "dotdash": 4}
N_EDGES = 20
MAX_LABEL_POINTS = 100
MAX_INT = 2147483647
DISTANCE_TO_POLE = 0.1
POLE_MARK_SIZE = 3.0
MIN_ANGLE = 10.0
HOW_CLOSE = 3.0
EPSILON = 1.192e-7
RA_DEC_TYPES = ("tan", "arc", "sin", "tnx")


class WcslabError(RuntimeError):
    """What stops IRAF's wcslab: its message."""


@dataclass(frozen=True)
class Device:
    """What wcslab asks of graphcap: character height and width (NDC), the aspect
    ratio and the resolution. stdvdm's, which harwood's own window takes too."""

    ch: float = 0.0294
    cw: float = 0.0125
    ar: float = 0.77
    xr: float = 1024.0
    yr: float = 1024.0


VDM = Device()


@dataclass
class Text:
    x: float                #: NDC
    y: float
    text: str
    up: float               #: the up vector, whole degrees (90: upright)
    size: float             #: in character heights
    h: str = "c"            #: justification: c, l, r
    v: str = "c"            #: c, t, b


@dataclass
class Stroke:
    x: np.ndarray           #: NDC, clipped
    y: np.ndarray
    ltype: int = 1


@dataclass
class Label:
    x: float                #: the plot's (pixel) coordinates
    y: float
    value: float
    axis: int
    angle: float = 0.0


@dataclass
class Drawing:
    """wcslab's drawing: in NDC, and the plot's system that maps the image's
    pixels onto the viewport."""

    strokes: list[Stroke]
    texts: list[Text]
    viewport: tuple[float, float, float, float]
    window: tuple[float, float, float, float]
    new_view: tuple[float, float, float, float]
    system: str
    graph_type: str
    begin: tuple[float, float]
    end: tuple[float, float]
    interval: tuple[float, float]
    labels: list[Label] = field(default_factory=list)
    title: str = ""

    def char_height(self, size: float, device: Device = VDM) -> float:
        """A character of this size, in the plot's y units."""
        vl, vr, vb, vt = self.viewport
        c1, c2, l1, l2 = self.window
        return device.ch * size * abs(l2 - l1) / abs(vt - vb)

    def to_plot(self, x, y):
        """NDC to the plot's coordinates (the image's pixels)."""
        vl, vr, vb, vt = self.viewport
        c1, c2, l1, l2 = self.window
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return (c1 + (x - vl) * (c2 - c1) / (vr - vl),
                l1 + (y - vb) * (l2 - l1) / (vt - vb))


# ------------------------------------------------------------------ the sky


class Sky:
    """The image's world system as wcslab reads it: which axes, of what type, and
    the transforms between logical pixels and (axis 1, axis 2) of the world."""

    def __init__(self, header, naxis=(1, 1)):
        self.header = header
        types = [str(header.get(f"CTYPE{i}", "")).strip() for i in (1, 2)]
        has_wcs = any(k in header for i in (1, 2)
                      for k in (f"CRVAL{i}", f"CRPIX{i}", f"CDELT{i}", f"CTYPE{i}",
                                f"CD{i}_1", f"CD{i}_2"))
        wtypes, axtypes = [], []
        for ctype in types:
            low = ctype.lower()
            m = re.match(r"^([a-z]+)-+([a-z0-9]{3})$", low)
            if m:
                axtypes.append(m.group(1))
                wtypes.append(m.group(2))
            else:
                axtypes.append("")
                wtypes.append("linear" if has_wcs else "")
        self.wtypes, self.axtypes = wtypes, axtypes
        system = OTHER
        if all(w in RA_DEC_TYPES for w in wtypes):
            system = RA_DEC
        elif all(w == "linear" for w in wtypes):
            system = LINEAR
        self.system = system
        index1, index2 = 0, 1
        if system == RA_DEC:
            ra = [i for i, a in enumerate(axtypes) if a.startswith("ra")]
            dec = [i for i, a in enumerate(axtypes) if a.startswith("dec")]
            if ra and dec:
                index1, index2 = ra[-1], dec[-1]
        self.index = (index1, index2)
        self.flip = index1 > index2
        self._wcs = None
        if has_wcs:
            from astropy.wcs import WCS, FITSFixedWarning

            h = header.copy()
            for i in (1, 2):
                if wtypes[i - 1] == "tnx":
                    h[f"CTYPE{i}"] = types[i - 1][:-3] + "TAN"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                warnings.simplefilter("ignore", UserWarning)
                self._wcs = WCS(h, naxis=2, relax=True)
        crval = [_number(header, f"CRVAL{i}", 0.0) for i in (1, 2)]
        crpix = [_number(header, f"CRPIX{i}", 0.0) for i in (1, 2)]
        self.world_center = (crval[index1], crval[index2]) if has_wcs else (0.0, 0.0)
        self.logical_center = (crpix[index1], crpix[index2])

    def l2w(self, x, y):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        if self._wcs is None:
            return x.copy(), y.copy()
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = self._wcs.wcs_pix2world(np.column_stack([x, y]), 1)
        return w[:, self.index[0]], w[:, self.index[1]]

    def w2l(self, a1, a2):
        a1 = np.atleast_1d(np.asarray(a1, float))
        a2 = np.atleast_1d(np.asarray(a2, float))
        if self._wcs is None:
            return a1.copy(), a2.copy()
        world = np.empty((a1.size, 2))
        world[:, self.index[0]] = a1
        world[:, self.index[1]] = a2
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = self._wcs.wcs_world2pix(world, 1)
        return p[:, 0], p[:, 1]


def user_header(p: dict):
    """wl_wcs_params: a WCS from wcspars (ctype, crpix, crval, cd, all reals)."""
    from astropy.io import fits

    h = fits.Header()
    h["NAXIS"] = 2
    for i in (1, 2):
        ctype = str(p.get(f"ctype{i}", "linear")).strip()
        low = ctype.lower().replace("_", "-")
        if low.startswith("ra--"):
            ctype = "RA---" + low.rsplit("-", 1)[-1].upper()
        elif low.startswith("dec-"):
            ctype = "DEC--" + low.rsplit("-", 1)[-1].upper()
        else:
            ctype = "LINEAR"
        h[f"CTYPE{i}"] = ctype
        h[f"CRPIX{i}"] = float(f32(p.get(f"crpix{i}", 0.0)))
        h[f"CRVAL{i}"] = float(f32(p.get(f"crval{i}", 0.0)))
    for i in (1, 2):
        for j in (1, 2):
            h[f"CD{i}_{j}"] = float(f32(p.get(f"cd{i}_{j}", float(i == j))))
    return h


def _number(header, key, default):
    try:
        return float(header[key]) if key in header else default
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ GIO


class _Gio:
    """The little of GIO wcslab uses: WCS (window, viewport, clipping), gctran,
    polylines (clipped as gadraw does, to the viewport or to the device), gmark's
    plus, gtext's attributes."""

    def __init__(self, device: Device):
        self.device = device
        self.wcs = {i: [f32(0), f32(1), f32(0), f32(1), f32(0), f32(1), f32(0), f32(1), True]
                    for i in range(17)}
        self.cur = 0
        self.pltype = 1
        self.txsize = 1.0
        self.txup = 90.0
        self.strokes: list[Stroke] = []
        self.texts: list[Text] = []
        self._pts: list[tuple[float, float]] = []
        self._box = (0.0, 1.0, 0.0, 1.0)
        self._ltype = 1

    # the systems
    def set_view(self, vl, vr, vb, vt):
        self.wcs[self.cur][4:8] = [f32(vl), f32(vr), f32(vb), f32(vt)]

    def set_window(self, x1, x2, y1, y2):
        self.wcs[self.cur][0:4] = [f32(x1), f32(x2), f32(y1), f32(y2)]

    def view(self):
        return tuple(float(v) for v in self.wcs[self.cur][4:8])

    def set_clip(self, on: bool):
        self.wcs[self.cur][8] = on

    def _ndc(self, x, y, w):
        if w == 0:
            return f32(x), f32(y)
        wx1, wx2, wy1, wy2, sx1, sx2, sy1, sy2, _ = self.wcs[w]
        xs = f32(f32(sx2 - sx1) / f32(wx2 - wx1))
        ys = f32(f32(sy2 - sy1) / f32(wy2 - wy1))
        return (f32(f32(f32(f32(x) - wx1) * xs) + sx1),
                f32(f32(f32(f32(y) - wy1) * ys) + sy1))

    def ctran(self, x, y, a, b):
        nx, ny = self._ndc(x, y, a)
        if b == 0:
            return nx, ny
        wx1, wx2, wy1, wy2, sx1, sx2, sy1, sy2, _ = self.wcs[b]
        xs = f32(f32(sx2 - sx1) / f32(wx2 - wx1))
        ys = f32(f32(sy2 - sy1) / f32(wy2 - wy1))
        return (f32(f32(f32(nx - sx1) / xs) + wx1), f32(f32(f32(ny - sy1) / ys) + wy1))

    def scale(self):
        """ggscale: world units per NDC in x and y."""
        wx1, wx2, wy1, wy2, sx1, sx2, sy1, sy2, _ = self.wcs[self.cur]
        return f32(f32(wx2 - wx1) / f32(sx2 - sx1)), f32(f32(wy2 - wy1) / f32(sy2 - sy1))

    # polylines
    def move(self, x, y):
        self.flush()
        w = self.wcs[self.cur]
        self._box = (float(w[4]), float(w[5]), float(w[6]), float(w[7])) if w[8] \
            else (0.0, 1.0, 0.0, 1.0)
        self._ltype = self.pltype
        if not (math.isfinite(x) and math.isfinite(y)):
            self._pts = []
            return
        self._pts = [tuple(float(v) for v in self._ndc(x, y, self.cur))]

    def draw(self, x, y):
        if not (math.isfinite(x) and math.isfinite(y)):
            self.flush()
            return
        point = tuple(float(v) for v in self._ndc(x, y, self.cur))
        if not self._pts:
            self._pts = [point]
            return
        self._pts.append(point)

    def flush(self):
        if len(self._pts) >= 2:
            for piece in _clip(self._pts, self._box):
                if len(piece) >= 2:
                    a = np.array(piece)
                    self.strokes.append(Stroke(a[:, 0], a[:, 1], self._ltype))
        self._pts = []

    def plus(self, x, y, size):
        """gmark (GM_PLUS) of a size in standard marker sizes: greset's ch*i/4, x
        times the aspect ratio."""
        index = min(4, int(size))
        ndc = f32(f32(self.device.ch) * index / f32(4.0))
        dx, dy = self.scale()
        xs = f32(f32(ndc * f32(self.device.ar)) * abs(dx))
        ys = f32(ndc * abs(dy))
        x1, y1 = f32(f32(x) - xs / f32(2)), f32(f32(y) - ys / f32(2))
        ux, uy = (.5, .5, .5, 0, 1), (0, 1, .5, .5, .5)
        saved, self.pltype = self.pltype, 1
        self.move(f32(ux[0] * xs + x1), f32(uy[0] * ys + y1))
        for a, b in zip(ux[1:], uy[1:], strict=True):
            self.draw(f32(a * xs + x1), f32(b * ys + y1))
        self.move(x, y)
        self.pltype = saved

    def text(self, x, y, text, fmt: str = "", up: float | None = None):
        self.flush()
        h = v = "c"
        u = self.txup if up is None else up
        for part in fmt.split(";"):
            key, _, value = part.partition("=")
            if key == "h":
                h = value
            elif key == "v":
                v = value
            elif key == "u":
                u = float(int(float(value)))        # gtxset's ctoi: the integer part
        # where GKI would clamp it to the device's edge, it is kept: a page has margins
        nx, ny = self._ndc(x, y, self.cur)
        self.texts.append(Text(float(nx), float(ny), text, float(u), float(self.txsize), h, v))


def _clip(points, box):
    """A polyline clipped to box (inclusive), as the pieces that are inside."""
    x1, x2, y1, y2 = box
    pieces, cur = [], []
    for (ax, ay), (bx, by) in zip(points[:-1], points[1:], strict=True):
        seg = _clip_segment(ax, ay, bx, by, x1, x2, y1, y2)
        if seg is None:
            if cur:
                pieces.append(cur)
                cur = []
            continue
        (px, py), (qx, qy) = seg
        if cur and (abs(cur[-1][0] - px) > 1e-12 or abs(cur[-1][1] - py) > 1e-12):
            pieces.append(cur)
            cur = []
        if not cur:
            cur = [(px, py)]
        cur.append((qx, qy))
        if (qx, qy) != (bx, by):
            pieces.append(cur)
            cur = []
    if cur:
        pieces.append(cur)
    return pieces


def _clip_segment(ax, ay, bx, by, x1, x2, y1, y2):
    """Liang-Barsky."""
    t0, t1 = 0.0, 1.0
    dx, dy = bx - ax, by - ay
    for p, q in ((-dx, ax - x1), (dx, x2 - ax), (-dy, ay - y1), (dy, y2 - ay)):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    start = (ax, ay) if t0 == 0 else (ax + t0 * dx, ay + t0 * dy)
    end = (bx, by) if t1 == 1 else (ax + t1 * dx, ay + t1 * dy)
    return start, end


# ------------------------------------------------------------------ viewports


def map_viewport(c1, c2, l1, l2, ux1, ux2, uy1, uy2, fill: bool, device: Device = VDM):
    """wl_map_viewport: the viewport (fill, or the image's shape and the device's
    aspect ratio), from what is given of vl, vr, vb, vt (None: not given)."""
    c1, c2, l1, l2 = (f32(v) for v in (c1, c2, l1, l2))
    ncols = _nint(c2 - c1) + 1
    nlines = _nint(l2 - l1) + 1
    ncolsr, nlinesr = f32(ncols), f32(nlines)
    if fill:
        x1, x2, y1, y2 = f32(0), f32(1), f32(0), f32(1)
    else:
        x1, x2, y1, y2 = f32(0.1), f32(0.9), f32(0.12), f32(0.92)
    x1 = f32(ux1) if ux1 is not None else x1
    x2 = f32(ux2) if ux2 is not None else x2
    y1 = f32(uy1) if uy1 is not None else y1
    y2 = f32(uy2) if uy2 is not None else y2
    if not fill:
        ratio = f32(min(ncolsr, nlinesr) / max(ncolsr, nlinesr))
        if ratio >= f32(0.0625):
            if ncols > nlines:
                y2 = f32(f32(f32(y2 - y1) * nlinesr) / ncolsr + y1)
            else:
                x2 = f32(f32(f32(x2 - x1) * ncolsr) / nlinesr + x1)
    xdis, ydis = f32(x2 - x1), f32(y2 - y1)
    xcen, ycen = f32(f32(x2 + x1) / f32(2)), f32(f32(y2 + y1) / f32(2))
    if not fill:
        aspect = f32(device.ar) or f32(1.0)
        if aspect < 1:
            xdis = f32(xdis * aspect)
        elif aspect > 1:
            ydis = f32(ydis / aspect)
    return (float(f32(xcen - xdis / f32(2))), float(f32(xcen + xdis / f32(2))),
            float(f32(ycen - ydis / f32(2))), float(f32(ycen + ydis / f32(2))))


def imd_viewport(ncols, nlines, frame_cols=None, frame_lines=None, vl=None, vr=None,
                 vb=None, vt=None):
    """wl_imd_viewport for an image displayed whole and centred in a frame buffer
    of its own size (harwood's window shows the image 1:1): the edges of the
    image's pixels as fractions of the frame, for the sides not given."""
    fc = frame_cols or ncols
    fl = frame_lines or nlines
    c1, c2, l1, l2 = 1.0, float(ncols), 1.0, float(nlines)
    ox, oy = (fc - ncols) // 2, (fl - nlines) // 2
    fx1, fx2, fy1, fy2 = 1 + ox, ncols + ox, 1 + oy, nlines + oy
    vx1 = max(0.0, min(1.0, (fx1 - 0.5) / fc))
    vx2 = max(0.0, min(1.0, (fx2 + 0.5) / fc))
    vy1 = max(0.0, min(1.0, (fy1 - 0.5) / fl))
    vy2 = max(0.0, min(1.0, (fy2 + 0.5) / fl))
    return ((c1, c2, l1, l2),
            (vx1 if vl is None else vl, vx2 if vr is None else vr,
             vy1 if vb is None else vb, vy2 if vt is None else vt))


def _nint(x) -> int:
    x = float(x)
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


# ------------------------------------------------------------------ parameters

#: wlpars, as IRAF has them.
WLPARS = {
    "major_grid": True, "minor_grid": False, "dolabel": True, "remember": False,
    "axis1_beg": "", "axis1_end": "", "axis1_int": "", "axis2_beg": "", "axis2_end": "",
    "axis2_int": "", "major_line": "solid", "major_tick": 0.03, "axis1_minor": 5,
    "axis2_minor": 5, "minor_line": "dotted", "minor_tick": 0.01, "tick_in": True,
    "axis1_side": "default", "axis2_side": "default", "axis2_dir": "",
    "justify": "default", "labout": True, "rotate": True, "full_label": False,
    "label_size": 1.0, "title": "imtitle", "axis1_title": "", "axis2_title": "",
    "title_side": "top", "axis1_title_side": "default", "axis2_title_side": "default",
    "title_size": 1.0, "axis_title_size": 1.0, "graph_type": "default",
}


def _strdic(value: str, names) -> int | None:
    """strdic: the entry the value names, or abbreviates without ambiguity."""
    value = str(value).strip().lower()
    if not value:
        return None
    if value in names:
        return names[value]
    hits = [v for k, v in names.items() if k.startswith(value)]
    return hits[0] if len(hits) == 1 else None


def _to_internal(text, system, axis):
    """wl_string_to_internal: a number (h:m:s too); RA in hours to degrees."""
    text = str(text).strip()
    if not text:
        return None
    value = iraffmt.ctod(text.split()[0])
    if value is None:
        return None
    if system == RA_DEC and axis == 1:
        value = value * 15
    return value


def _sides(text):
    text = str(text).lower()
    return {s: (name in text) for name, s in SIDES.items()}


def _line_type(text, warn):
    if text in LINE_TYPES:
        return LINE_TYPES[text]
    warn("Pattern unknown, using 'solid'.")
    return 1


# ------------------------------------------------------------------ the labeller


class _WL:
    def __init__(self, sky: Sky, pars: dict, title: str, gio: _Gio, boundary, device):
        p = {**WLPARS, **{k: v for k, v in pars.items() if v is not None}}
        self.p = p
        self.sky = sky
        self.gio = gio
        self.device = device
        self.system = sky.system
        self.messages: list[str] = []
        self.title = title if p["title"] == "imtitle" else str(p["title"])
        self.axis_title = [str(p["axis1_title"]), str(p["axis2_title"])]
        self.always_full = bool(p["full_label"])
        self.axis_title_size = float(f32(p["axis_title_size"]))
        self.label_rotate = bool(p["rotate"])
        self.label_size = float(f32(p["label_size"]))
        self.labon = bool(p["dolabel"])
        self.labout = bool(p["labout"])
        self.maj_gridon = bool(p["major_grid"])
        self.maj_tick = float(f32(p["major_tick"]))
        self.min_gridon = bool(p["minor_grid"])
        self.minor = [int(p["axis1_minor"]), int(p["axis2_minor"])]
        self.min_tick = float(f32(p["minor_tick"]))
        self.tick_in = bool(p["tick_in"])
        self.title_size = float(f32(p["title_size"]))
        self.graph_type = _strdic(p["graph_type"], GRAPH_TYPES)
        self.label_side = [_sides(p["axis1_side"]), _sides(p["axis2_side"])]
        self.polar_direction = _strdic(p["justify"], SIDES)
        self.major = [_to_internal(p["axis1_int"], self.system, 1),
                      _to_internal(p["axis2_int"], self.system, 2)]
        self.begin = [_to_internal(p["axis1_beg"], self.system, 1),
                      _to_internal(p["axis2_beg"], self.system, 2)]
        self.end = [_to_internal(p["axis1_end"], self.system, 1),
                    _to_internal(p["axis2_end"], self.system, 2)]
        self.polar_position = _to_internal(p["axis2_dir"], self.system, 1)
        self.axis_title_side = [_strdic(p["axis1_title_side"], SIDES),
                                _strdic(p["axis2_title_side"], SIDES)]
        self.maj_ltype = _line_type(p["major_line"], self.messages.append)
        self.min_ltype = _line_type(p["minor_line"], self.messages.append)
        ts = str(p["title_side"]).strip().lower()
        self.title_side = _strdic(ts, SIDES) if ts else None
        self.boundary = {LEFT: float(boundary[0]), RIGHT: float(boundary[1]),
                         BOTTOM: float(boundary[2]), TOP: float(boundary[3])}
        self.labels: list[Label] = []
        self.bad_side = None
        self.rota = 0.0

    # transforms
    def w2l(self, a1, a2):
        x, y = self.sky.w2l(a1, a2)
        return x, y

    def w2l1(self, a1, a2):
        x, y = self.sky.w2l([a1], [a2])
        return float(x[0]), float(y[0])

    def l2w1(self, x, y):
        a, b = self.sky.l2w([x], [y])
        return float(a[0]), float(b[0])

    # wl_wcslab and wl_graphics
    def run(self):
        g = self.gio
        old_wcs = g.cur
        self.ndc_wcs, self.plot_wcs = (old_wcs + 1, old_wcs + 2) if old_wcs < 14 \
            else (old_wcs - 1, old_wcs - 2)
        saved = {w: list(g.wcs[w]) for w in (self.ndc_wcs, self.plot_wcs)}
        self.graphics()
        self.setup()
        self.grid()
        if self.labon:
            self.label()
        g.flush()
        for w, v in saved.items():
            g.wcs[w][:8] = v[:8]
        g.cur = old_wcs

    def graphics(self):
        g = self.gio
        vl, vr, vb, vt = g.view()
        g.cur = self.ndc_wcs
        g.set_view(vl, vr, vb, vt)
        g.set_window(vl, vr, vb, vt)
        g.set_clip(True)
        self.new_view = {LEFT: vl, RIGHT: vr, BOTTOM: vb, TOP: vt}
        g.pltype = 1
        g.move(vl, vb)
        g.draw(vr, vb)
        g.draw(vr, vt)
        g.draw(vl, vt)
        g.draw(vl, vb)
        rel = float(f32(max(abs(f32(vr) - f32(vl)), abs(f32(vt) - f32(vb)))))
        self.maj_tick = float(f32(rel * f32(self.maj_tick)))
        self.min_tick = float(f32(rel * f32(self.min_tick)))
        self.title_size = float(f32(f32(self.title_size) * rel))
        self.axis_title_size = float(f32(f32(self.axis_title_size) * rel))
        self.label_size = float(f32(f32(self.label_size) * rel))
        g.cur = self.plot_wcs
        g.set_view(vl, vr, vb, vt)
        b = self.boundary
        g.set_window(b[LEFT], b[RIGHT], b[BOTTOM], b[TOP])
        g.set_clip(True)
        self.segments = int(min(self.device.xr, self.device.yr) / 5)

    # wl_setup
    def setup(self):
        b = self.boundary
        north = self.system == RA_DEC and self.sky.world_center[1] > 0
        pole = (0.0, 0.0)
        if self.system == RA_DEC:
            pole = self.w2l1(self.sky.world_center[0], 90.0 if north else -90.0)
            pole = tuple(float(f32(MAX_INT)) if (not math.isfinite(v) or abs(v) > MAX_INT)
                         else v for v in pole)
        self.pole = pole
        self.north = north
        # wl_determine_graph_type
        if self.system == RA_DEC:
            xcen = (b[LEFT] + b[RIGHT]) / 2.0
            ycen = (b[BOTTOM] + b[TOP]) / 2.0
            max_dist = min((b[LEFT] - xcen) ** 2, (b[TOP] - ycen) ** 2)
            pole_dist = (pole[0] - xcen) ** 2 + (pole[1] - ycen) ** 2
            if self.graph_type is None:
                if pole_dist < max_dist:
                    self.graph_type = POLAR
                elif pole_dist < HOW_CLOSE * max_dist:
                    self.graph_type = NEAR_POLAR
                else:
                    self.graph_type = NORMAL
        else:
            self.graph_type = NORMAL
        ex, ey = _edges(b)
        a1, a2 = self.sky.l2w(ex, ey)
        lo = [float(np.nanmin(a1)), float(np.nanmin(a2))]
        hi = [float(np.nanmax(a1)), float(np.nanmax(a2))]
        rng = [abs(hi[0] - lo[0]), abs(hi[1] - lo[1])]
        if self.system == RA_DEC:
            self.sky_extrema(a1, lo, hi, rng)
        self.rota = self.coord_rotation(self.sky.world_center[0], hi[1],
                                        self.sky.world_center[0], lo[1])
        for axis in (0, 1):
            if self.major[axis] is None or self.begin[axis] is None or self.end[axis] is None:
                self.round_axis(axis, lo[axis], hi[axis], rng[axis])

    def sky_extrema(self, ra, lo, hi, rng):
        b = self.boundary
        pole = self.pole
        is_pole = not (pole[0] < b[LEFT] or pole[0] > b[RIGHT] or pole[1] < b[BOTTOM]
                       or pole[1] > b[TOP])
        if is_pole:
            lo[0], hi[0], rng[0] = 0.0, 359.9, 360.0
            if self.north:
                hi[1] = 90.0 - ((90.0 - lo[1]) * DISTANCE_TO_POLE)
            else:
                lo[1] = -90.0 + ((90.0 + hi[1]) * DISTANCE_TO_POLE)
            rng[1] = abs(hi[1] - lo[1])
            self.gio.plus(f32(pole[0]), f32(pole[1]), POLE_MARK_SIZE)
        else:
            lo[0], hi[0], rng[0] = _ra_range([float(v) for v in ra])
        if self.graph_type == POLAR:
            if self.polar_position is None:
                self.axis2_label_direction()
            else:
                self.bad_side = self.direction_from_axis1()
                if self.bad_side is None:
                    self.bad_side = BOTTOM
            if self.polar_direction is None:
                self.polar_direction = _opposite(self.bad_side)
        elif self.graph_type == NEAR_POLAR:
            xcen = (b[LEFT] + b[RIGHT]) / 2.0
            ycen = (b[BOTTOM] + b[TOP]) / 2.0
            nx, ny = _axis_on_line(xcen, ycen, pole[0], pole[1], b)
            if nx is None:
                self.bad_side, self.polar_direction = BOTTOM, LEFT
            else:
                self.bad_side = _find_side(nx, ny, b)
                if self.bad_side in (LEFT, RIGHT):
                    self.polar_direction = BOTTOM if abs(ny - b[BOTTOM]) < abs(ny - b[TOP]) \
                        else TOP
                else:
                    self.polar_direction = LEFT if abs(nx - b[LEFT]) < abs(nx - b[RIGHT]) \
                        else RIGHT

    def axis2_label_direction(self):
        b, pole = self.boundary, self.pole
        dif = abs(b[TOP] - pole[1])
        self.bad_side = TOP
        if abs(b[BOTTOM] - pole[1]) < dif:
            self.bad_side = BOTTOM
        if self.bad_side == TOP:
            self.polar_position = self.l2w1(pole[0], b[BOTTOM])[0]
        else:
            self.polar_position = self.l2w1(pole[0], b[TOP])[0]

    def direction_from_axis1(self):
        b = self.boundary
        lat = self.begin[1] if self.north else self.end[1]
        if lat is None:
            return None
        nx, ny = self.w2l1(self.polar_position, lat)
        cx, cy = _axis_on_line(self.pole[0], self.pole[1], nx, ny, b)
        if cx is None:
            return None
        return _find_side(cx, cy, b)

    def coord_rotation(self, wx1, wy1, wx2, wy2):
        x1, y1 = self.w2l1(wx1, wy1)
        x2, y2 = self.w2l1(wx2, wy2)
        dx, dy = x2 - x1, y2 - y1
        if not (math.isfinite(dx) and math.isfinite(dy)) or (dx == 0 and dy == 0):
            rota = 0.0
        else:
            rota = math.degrees(math.atan2(dy, dx))
        return rota + 360.0 if rota < 0 else rota

    def round_axis(self, axis, lo, hi, rng):
        if self.system == RA_DEC:
            if axis == 0:
                self.begin[0], self.end[0], self.major[0] = _round_ra(lo, hi, rng, 6)
            else:
                n = 4 if self.graph_type == POLAR else 6
                self.begin[1], self.end[1], self.major[1] = _round_dec(lo, hi, rng, n)
        else:
            self.begin[axis], self.end[axis], self.major[axis] = _generic_round(lo, hi, rng)

    # wl_grid
    def grid(self):
        g = self.gio
        old_type = g.pltype
        minor = self.major[0] / float(self.minor[0])
        if self.graph_type == NEAR_POLAR:
            if abs(self.begin[1]) < abs(self.end[1]):
                tb, te = self.begin[1], 90.0
            else:
                tb, te = -90.0, self.end[1]
        else:
            tb, te = self.begin[1], self.end[1]
        current, k = self.begin[0], 0
        while True:
            if k % self.minor[0] == 0:
                g.pltype = self.maj_ltype
                self.constant_axis(0, current, tb, te, self.maj_gridon, self.labon,
                                   self.maj_tick)
            else:
                g.pltype = self.min_ltype
                self.constant_axis(0, current, tb, te, self.min_gridon, False,
                                   self.min_tick)
            k += 1
            current = self.begin[0] + minor * k
            if f32(current) > f32(self.end[0]):
                break
        minor = self.major[1] / float(self.minor[1])
        if self.end[1] < self.begin[1]:
            current, minor, te = self.end[1], -minor, self.begin[1]
        else:
            current, te = self.begin[1], self.end[1]
        k, tb = 0, current
        while True:
            if k % self.minor[1] == 0:
                g.pltype = self.maj_ltype
                n_old = len(self.labels)
                self.constant_axis(1, current, self.begin[0], self.end[0], self.maj_gridon,
                                   self.labon, self.maj_tick)
                if self.graph_type == POLAR and self.maj_gridon and self.labon:
                    del self.labels[n_old:]
                    x, y = self.w2l1(self.polar_position, current)
                    self.labels.append(Label(x, y, current, 2))
            else:
                g.pltype = self.min_ltype
                self.constant_axis(1, current, self.begin[0], self.end[0], self.min_gridon,
                                   False, self.min_tick)
            k += 1
            current = tb + minor * k
            if f32(current) > f32(te):
                break
        g.pltype = old_type

    def constant_axis(self, axis, value, vmin, vmax, gridon, label, tick):
        """wl_graph_constant_axis1/2: the line of the axis at value, from vmin to vmax
        in segments, drawn (gridon) and its crossings of the edges labelled."""
        g = self.gio
        inc = (vmax - vmin) / self.segments
        seq, v = [], vmin
        while True:
            seq.append(v)
            done = v < vmax if inc < 0 else v > vmax
            v = v + inc
            if done or len(seq) > 100000:
                break
        seq = np.array(seq)
        const = np.full(seq.size, value)
        lx, ly = self.w2l(const, seq) if axis == 0 else self.w2l(seq, const)
        lastx, lasty = float(lx[0]), float(ly[0])
        g.move(f32(lastx), f32(lasty))
        for x, y in zip(lx.tolist(), ly.tolist(), strict=True):
            self.point_to_label(lastx, lasty, x, y, axis + 1, value, gridon, label, tick)
            if gridon:
                g.draw(f32(x), f32(y))
            lastx, lasty = x, y

    def _out(self, x, y):
        b = self.boundary
        return x <= b[LEFT] or x >= b[RIGHT] or y <= b[BOTTOM] or y >= b[TOP]

    def _in(self, x, y):
        b = self.boundary
        return b[LEFT] < x < b[RIGHT] and b[BOTTOM] < y < b[TOP]

    def point_to_label(self, x1, y1, x2, y2, axis, value, gridon, label, tick):
        for outer, inner in (((x1, y1), (x2, y2)), ((x2, y2), (x1, y1))):
            if not (self._out(*outer) and self._in(*inner)):
                continue
            nx, ny = _axis_on_line(*outer, *inner, self.boundary)
            if nx is None:
                continue
            if not gridon:
                tx, ty = self.mark_tick(tick, *outer, *inner, nx, ny)
                if self.tick_in != self.labout:
                    nx, ny = tx, ty
            if label and len(self.labels) < MAX_LABEL_POINTS:
                self.labels.append(Label(nx, ny, value, axis,
                                         self.vector_angle(x1, y1, x2, y2)))

    def mark_tick(self, size, x0, y0, x1, y1, sx, sy):
        g = self.gio
        old_wcs, old_line = g.cur, g.pltype
        g.cur, g.pltype = self.ndc_wcs, 1
        nsx, nsy = g.ctran(f32(sx), f32(sy), old_wcs, self.ndc_wcs)
        n0x, n0y = g.ctran(f32(x0), f32(y0), old_wcs, self.ndc_wcs)
        n1x, n1y = g.ctran(f32(x1), f32(y1), old_wcs, self.ndc_wcs)
        dx, dy = f32(n1x - n0x), f32(n1y - n0y)
        dist = f32(math.sqrt(float(f32(f32(n0x - n1x) ** 2 + f32(n0y - n1y) ** 2))))
        t = f32(f32(size) / dist) if dist else f32(0)
        if not self.tick_in:
            t = -t
            g.set_clip(False)
        ex, ey = f32(f32(t * dx) + nsx), f32(f32(t * dy) + nsy)
        g.move(nsx, nsy)
        g.draw(ex, ey)
        g.flush()
        if not self.tick_in:
            g.set_clip(True)
        g.cur, g.pltype = old_wcs, old_line
        tx, ty = g.ctran(ex, ey, self.ndc_wcs, old_wcs)
        return float(tx), float(ty)

    def vector_angle(self, x1, y1, x2, y2):
        g = self.gio
        ax, ay = g.ctran(f32(x1), f32(y1), g.cur, 0)
        bx, by = g.ctran(f32(x2), f32(y2), g.cur, 0)
        dy, dx = f32(by - ay), f32(bx - ax)
        if _equalr(dx, 0) and _equalr(dy, 0):
            return 0.0
        return float(f32(math.degrees(math.atan2(float(dy), float(dx)))))

    # wl_label
    def label(self):
        g = self.gio
        offset = {s: 0.0 for s in SIDES.values()}
        no1 = not any(self.label_side[0].values())
        no2 = not any(self.label_side[1].values())
        if self.graph_type == POLAR:
            self.polar_label()
            if no1:
                for s in SIDES.values():
                    self.label_side[0][s] = True
                if self.axis_title_side[0] is None:
                    self.axis_title_side[0] = BOTTOM
        elif self.graph_type == NEAR_POLAR:
            if no1:
                self.label_side[0][self.bad_side] = True
                if self.axis_title_side[0] is None:
                    self.axis_title_side[0] = self.bad_side
            if no2:
                self.label_side[1][self.polar_direction] = True
                if self.axis_title_side[1] is None:
                    self.axis_title_side[1] = self.polar_direction
        else:
            if _invert(self.rota):
                s1, s2 = LEFT, BOTTOM
            else:
                s1, s2 = BOTTOM, LEFT
            if no1:
                self.label_side[0][s1] = True
            if no2:
                self.label_side[1][s2] = True
        axes = (0,) if self.graph_type == POLAR else (0, 1)
        for axis in axes:
            for s in (LEFT, RIGHT, BOTTOM, TOP):
                if self.label_side[axis][s]:
                    off = self.lab_edges(axis + 1, s)
                    if self.axis_title_side[axis] is None:
                        self.axis_title_side[axis] = s
                else:
                    off = 0.0
                if abs(off) > abs(offset[s]):
                    offset[s] = off
        for s in SIDES.values():
            self.new_view[s] = float(f32(f32(self.new_view[s]) + f32(offset[s])))
        self.write_title(self.axis_title[0], self.axis_title_side[0], self.axis_title_size)
        if self.graph_type != POLAR:
            self.write_title(self.axis_title[1], self.axis_title_side[1],
                             self.axis_title_size)
        if self.title_side is not None:
            self.write_title(self.title, self.title_side, self.title_size)
        g.flush()

    def polar_label(self):
        g = self.gio
        ch = f32(f32(self.device.ch) * f32(g.txsize) / f32(2))
        cw = f32(f32(self.device.cw) * f32(g.txsize) / f32(2))
        old = g.txsize
        g.txsize = float(f32(f32(old) * f32(0.8)))
        prec = self.precision(2)
        for lab in self.labels:
            if lab.axis != 2:
                continue
            text, units = wl_dms(lab.value, prec, True)
            nx, ny = g.ctran(f32(lab.x), f32(lab.y), self.plot_wcs, self.ndc_wcs)
            d = self.polar_direction
            if d == BOTTOM:
                lf, uf, ny = "h=c;v=t", "h=c;v=c", f32(ny - ch)
            elif d == TOP:
                lf, uf, ny = "h=c;v=c", "h=c;v=b", f32(ny + ch)
            elif d == LEFT:
                lf, uf, nx = "h=r;v=c", "h=r;v=b", f32(nx - cw)
            else:
                lf, uf, nx = "h=l;v=c", "h=l;v=b", f32(nx + cw)
            tx, ty = g.ctran(nx, ny, self.ndc_wcs, self.plot_wcs)
            if self._in(float(tx), float(ty)):
                g.text(tx, ty, text, lf)
                g.text(tx, ty, units, uf)
        g.txsize = old

    def lab_edges(self, axis, side):
        g = self.gio
        old_wcs = g.cur
        g.cur = self.ndc_wcs
        old_size = g.txsize
        g.txsize = self.label_size
        prec = self.precision(axis)
        offset = 0.0
        chosen = [i for i, lab in enumerate(self.labels)
                  if lab.axis == axis and _find_side(lab.x, lab.y, self.boundary) == side]
        if chosen:
            full = self.full_label_position(chosen, axis, side, prec)
            if not self.labout and self.graph_type != NORMAL and self.label_rotate:
                angle = None
            elif self.graph_type == NORMAL and (self.label_rotate
                                                or (not self.labout and self.maj_gridon)):
                total = 0.0
                for i in chosen:
                    total += self.labels[i].angle
                angle = float(f32(total / len(chosen)))
            else:
                angle = 0.0
            for i in chosen:
                lab = self.labels[i]
                nx, ny = g.ctran(f32(lab.x), f32(lab.y), old_wcs, self.ndc_wcs)
                tangle = (wl_string_angle(lab.angle, self.labout) if angle is None
                          else angle)
                offset = self.write_label(lab.value, side, nx, ny, tangle, axis, prec,
                                          i == full or self.always_full, offset)
        g.txsize = old_size
        g.cur = old_wcs
        return offset

    def precision(self, axis):
        if self.system != RA_DEC:
            return None
        v = self.major[axis - 1]
        if axis == 1:
            steps = (3600.0 / 240, 60.0 / 240, 1.0 / 240, 0.01 / 240)
        else:
            steps = (3600.0 / 3600, 60.0 / 3600, 1.0 / 3600, 0.01 / 3600)
        for prec, step in zip((HOUR, MINUTE, SECOND, SUBSEC_LOW), steps, strict=True):
            if v >= step:
                return prec
        return SUBSEC_HIGH

    def full_label_position(self, chosen, axis, side, prec):
        other = self.label_side[1 if axis == 1 else 0]
        if side in (TOP, BOTTOM):
            yside, xside = side, None
            if other[RIGHT]:
                xside = RIGHT
            if other[LEFT]:
                xside = LEFT
            xside = LEFT if xside is None else xside
        else:
            xside, yside = side, None
            if other[TOP]:
                yside = TOP
            if other[BOTTOM]:
                yside = BOTTOM
            yside = BOTTOM if yside is None else yside
        b = self.boundary
        cx, cy = b[xside], b[yside]
        cur = chosen[0]
        cur_dist = math.hypot(cx - self.labels[cur].x, cy - self.labels[cur].y)
        for i in chosen[1:]:
            lab = self.labels[i]
            full = False
            if self.system == RA_DEC:
                if lab.axis == 1:
                    full = wl_hms(lab.value, prec, False, report=True)
                else:
                    full = wl_dms(lab.value, prec, False, report=True)
            if full:
                return None
            dist = math.hypot(cx - lab.x, cy - lab.y)
            if dist < cur_dist:
                cur_dist, cur = dist, i
        return cur

    def write_label(self, value, side, x, y, angle, axis, prec, do_full, offset):
        g = self.gio
        ch = f32(f32(self.device.ch) * f32(g.txsize))
        cw = f32(f32(self.device.cw) * f32(g.txsize))
        ta = f32(wl_string_angle(angle, self.labout))
        uox, uoy = _rotate(0.0, float(ch / f32(2)), float(ta) - 90.0)
        if not self.labout and self.maj_gridon:
            iox, ioy = _rotate(0.0, float(f32(0.75) * ch), float(ta) - 90.0)
        else:
            iox = ioy = f32(0)
        if self.system == RA_DEC:
            if axis == 1:
                text, units = wl_hms(value, prec, do_full)
            else:
                text, units = wl_dms(value, prec, do_full)
        else:
            text, units = iraffmt.printf(value, "g", 2, double=True), ""
        fmt = f"h=c;v=c;u={float(ta):f}"
        length = f32(f32(f32(0.5) * cw) * f32(2 + len(text)))
        rx, ry = _rotate(float(length), 0.0, float(ta) - 90.0)
        rx, ry = abs(rx), abs(ry)
        tside = side if self.labout else _opposite(side)
        x, y = f32(x), f32(y)
        if tside == TOP:
            ly = f32(f32(f32(y + ry) + ioy) + uoy)
            if _equalr(ta, 90.0):
                lx = x
                ly = f32(ly + uoy)
            elif ta < 90:
                lx = f32(x - rx)
            else:
                lx = f32(x + rx)
            lx = f32(lx + iox)
            new_offset = f32(ry + ry)
        elif tside == BOTTOM:
            ly = f32(f32(f32(y - ry) - ioy) - uoy)
            if _equalr(ta, 90.0):
                lx = x
                ly = f32(ly - uoy)
            elif ta < 90:
                lx = f32(x + rx)
            else:
                lx = f32(x - rx)
            lx = f32(lx - iox)
            new_offset = f32(ry + ry)
        elif tside == LEFT:
            lx = f32(f32(x - rx) - abs(uox))
            if ta < 90:
                ly = f32(f32(y + ry) - ioy)
                lx = f32(lx - iox)
            else:
                ly = f32(f32(y - ry) + ioy)
                lx = f32(lx + iox)
            new_offset = f32(f32(rx + rx) + abs(uox))
        else:
            lx = f32(f32(x + rx) + abs(uox))
            if ta < 90:
                ly = f32(f32(y - ry) + ioy)
                lx = f32(lx + iox)
            else:
                ly = f32(f32(y + ry) - ioy)
                lx = f32(lx - iox)
            new_offset = f32(f32(rx + rx) + abs(uox))
        lx = f32(lx - f32(uox / f32(2)))
        ly = f32(ly - f32(uoy / f32(2)))
        ux, uy = f32(lx + uox), f32(ly + uoy)
        g.text(lx, ly, text, fmt)
        if self.system == RA_DEC:
            g.text(ux, uy, units, fmt)
        if self.labout and abs(offset) < new_offset:
            offset = -float(new_offset) if side in (LEFT, BOTTOM) else float(new_offset)
        return offset

    def write_title(self, title, side, size):
        """wl_title: in NDC, two character heights beyond the labels' view."""
        g = self.gio
        if not title or side is None:
            return
        left, right, bottom, top = g.view()
        old_size, old_up = g.txsize, g.txup
        g.txsize = size
        ch = f32(f32(self.device.ch) * f32(size))
        cw = f32(f32(self.device.cw) * f32(size))
        old_wcs = g.cur
        g.cur = 0
        v = self.new_view
        if side == TOP:
            g.txup = 90.0
            x = f32(f32(f32(right) + f32(left)) / f32(2))
            y = f32(f32(v[TOP]) + f32(2) * ch)
            v[TOP] = float(f32(y + ch / f32(2)))
        elif side == BOTTOM:
            g.txup = 90.0
            x = f32(f32(f32(right) + f32(left)) / f32(2))
            y = f32(f32(v[BOTTOM]) - f32(2) * ch)
            v[BOTTOM] = float(f32(y - ch / f32(2)))
        elif side == RIGHT:
            g.txup = 180.0
            x = f32(f32(v[RIGHT]) + f32(2) * cw)
            y = f32(f32(f32(top) + f32(bottom)) / f32(2))
            v[RIGHT] = float(f32(x + cw / f32(2)))
        else:
            g.txup = 180.0
            x = f32(f32(v[LEFT]) - f32(2) * cw)
            y = f32(f32(f32(top) + f32(bottom)) / f32(2))
            v[LEFT] = float(f32(x - cw / f32(2)))
        g.text(x, y, title, "h=c;v=c")
        g.cur = old_wcs
        g.txsize, g.txup = old_size, old_up


# ------------------------------------------------------------------ helpers


def _edges(b):
    """wl_construct_edge_vectors: five points on each side."""
    n = N_EDGES // 4
    x, y = np.zeros(N_EDGES), np.zeros(N_EDGES)
    interval = (b[RIGHT] - b[LEFT]) / n
    current = b[LEFT]
    for i in range(n):
        x[i], y[i] = current + interval, b[BOTTOM]
        x[i + n], y[i + n] = current, b[TOP]
        current = current + interval
    interval = (b[TOP] - b[BOTTOM]) / n
    current = b[BOTTOM]
    for i in range(n):
        x[i + 2 * n], y[i + 2 * n] = b[LEFT], current
        x[i + 3 * n], y[i + 3 * n] = b[RIGHT], current + interval
        current = current + interval
    return x, y


def _ra_range(ra):
    """wl_ra_range: the pair of edge RAs furthest apart, across 0h when nearer."""
    best = None
    for i in range(len(ra)):
        for j in range(i + 1, len(ra)):
            v1, v2 = ra[i], ra[j]
            if abs(v1 - v2) > 180.0:
                lo, hi = (v2, v1 + 360.0) if v1 < v2 else (v1, v2 + 360.0)
            else:
                lo, hi = (v1, v2) if v1 < v2 else (v2, v1)
            if best is None or hi - lo > best[2]:
                best = (lo, hi, hi - lo)
    return best if best is not None else (ra[0], ra[0], 0.0)


RAGAP = (1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
         1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1.2e3, 1.8e3, 3.6e3,
         7.2e3)
DECGAP = RAGAP + (1.8e4, 3.6e4)


def _round_up(x, y):
    """wl_round_upd."""
    z = 0.0 if x < 0 else y
    return y * float(int((x + z) / y))


def _check_array(ex, arr):
    """wl_check_arrayd: the gap nearest ex."""
    n = len(arr)
    j = 1
    while j < n and ex - arr[j - 1] > 0:
        j += 1
    if 1 < j < n and abs(ex - arr[j - 2]) < abs(ex - arr[j - 1]):
        j -= 1
    return arr[j - 1]


def _round_ra(lo, hi, rng, n):
    major = _check_array(240 * rng / n, RAGAP)
    minimum = (_round_up(240 * lo, major) - major) / 240
    maximum = _round_up(240 * hi, major) / 240
    return minimum, maximum, major / 240


def _round_dec(lo, hi, rng, n):
    major = _check_array(3600 * rng / n, DECGAP)
    minimum = (_round_up(3600 * lo, major) - major) / 3600
    maximum = _round_up(3600 * hi, major) / 3600
    major = major / 3600
    return max(minimum, -90.0 + major), min(maximum, 90.0 - major), major


def _generic_round(lo, hi, rng):
    if rng == 0 or not math.isfinite(rng):
        rng = 1.0
    diff = math.log10(abs(rng) / 4.0)
    iexp = int(diff)
    if diff < 0:
        iexp -= 1
    amant = diff - iexp
    num = 1 if amant < 0.15 else 2 if amant < 0.5 else 5 if amant < 0.85 else 10
    interval = num * 10.0 ** iexp
    return _round_up(lo, interval) - interval, _round_up(hi, interval), interval


def _axis_on_line(x0, y0, x1, y1, b):
    """wl_axis_on_line: where the line through two points crosses an edge within
    their box (the last side, in the order left right bottom top, that does)."""
    xv = {LEFT: b[LEFT], RIGHT: b[RIGHT]}
    yv = {TOP: b[TOP], BOTTOM: b[BOTTOM]}
    fx = None if min(x0, x1) >= max(x0, x1) else (x0, y0, x1, y1)
    fy = None if min(y0, y1) >= max(y0, y1) else (y0, x0, y1, x1)

    def line(f, v):
        a0, b0, a1, b1 = f
        return b0 + (v - a0) * (b1 - b0) / (a1 - a0)

    yv[LEFT] = b[LEFT] if fx is None else line(fx, b[LEFT])
    yv[RIGHT] = b[RIGHT] if fx is None else line(fx, b[RIGHT])
    xv[TOP] = b[TOP] if fy is None else line(fy, b[TOP])
    xv[BOTTOM] = b[BOTTOM] if fy is None else line(fy, b[BOTTOM])
    tx0, tx1 = sorted((x0, x1))
    ty0, ty1 = sorted((y0, y1))
    nx = ny = None
    for side in (LEFT, RIGHT, BOTTOM, TOP):
        x, y = xv[side], yv[side]
        if (f32(tx0) <= f32(x) <= f32(tx1) and f32(ty0) <= f32(y) <= f32(ty1)
                and b[LEFT] <= x <= b[RIGHT] and b[BOTTOM] <= y <= b[TOP]):
            nx, ny = x, y
    return nx, ny


def _find_side(x, y, b):
    side, dif = LEFT, abs(x - b[LEFT])
    for s, d in ((RIGHT, abs(x - b[RIGHT])), (BOTTOM, abs(y - b[BOTTOM])),
                 (TOP, abs(y - b[TOP]))):
        if d < dif:
            side, dif = s, d
    return side


def _opposite(side):
    return {LEFT: RIGHT, RIGHT: LEFT, TOP: BOTTOM, BOTTOM: TOP}.get(side, side)


def _invert(rota):
    return rota < 45 or rota > 315 or 135 < rota < 225


def _equalr(x, y) -> bool:
    from .plotgraph import fp_equalr

    return fp_equalr(f32(x), f32(y))


def _rotate(x, y, angle):
    """wl_rotate: (x, y) turned by angle degrees -- through mw_rotate and
    mw_v2tranr, whose branch for a rotated linear transform (LRO) reads p2[1], the
    output it has not written, where it means p1[2], the input's y: in IRAF's
    single precision that is a zero, so y counts only when the angle is 0 (the
    matrix diagonal, the LNR branch). wcslab's rotated labels have their units on the
    number's own line, then, and no offset inwards: "13 28 00 " and "  h  m  s"
    overprint into 13h28m00s."""
    if f32(angle) == 0:
        return f32(x), f32(y)
    a = float(f32(math.radians(float(f32(angle)))))
    return f32(x * math.cos(a)), f32(x * math.sin(a))


def wl_string_angle(angle, right_to_up: bool) -> float:
    out = angle
    if out > 90.0:
        out -= 180.0
    if out < -90.0:
        out += 180.0
    if right_to_up and (math.fmod(abs(out), 90.0) < MIN_ANGLE
                        or 90.0 - math.fmod(abs(out), 90.0) < MIN_ANGLE):
        out = 0.0
    return out + 90.0


def wl_hms(value, precision, full, report=False):
    """wl_hms: an RA (degrees) as its label and the units under it; `full` writes
    h m s. With report, whether the label came out in full."""
    accuracy = 0.0001 if precision == SUBSEC_HIGH else 0.01
    fraction = abs(240 * value)
    if precision in (SUBSEC_LOW, SUBSEC_HIGH):
        sec = int(fraction)
        fraction -= sec
    else:
        sec = int(fraction + 0.5)
        fraction = 0.0
    if sec >= 86400:
        sec %= 86400
    s, m, h = sec % 60, (sec // 60) % 60, sec // 3600
    text = units = ""
    if precision in (SUBSEC_LOW, SUBSEC_HIGH):
        fraction = s + fraction
        if precision == SUBSEC_LOW:
            text, units = f"{fraction:05.2f}"[:5], "  s  "
        else:
            text, units = f"{fraction:07.4f}"[:7], "  s    "
        if not full:
            full = fraction < accuracy
    elif precision == SECOND:
        text, units = f"{s:02d} "[:3], "  s"
        if not full:
            full = s == 0
    if precision == MINUTE or (precision > MINUTE and full):
        tt, tu = text, units
        text, units = f"{m:02d} "[:3], "  m"
        if full:
            text, units = text + tt, units + tu
        else:
            full = m == 0
    if precision == HOUR or full:
        tt, tu = text, units
        text, units = f"{h:2d} "[:3], "  h"
        if full:
            text, units = text + tt, units + tu
    return full if report else (text, units)


def wl_dms(value, precision, full, report=False):
    """wl_dms: a declination (degrees) as its label and its units' line."""
    accuracy = 0.0001 if precision == SUBSEC_HIGH else 0.01
    fraction = abs(3600 * value)
    if precision in (SUBSEC_LOW, SUBSEC_HIGH):
        sec = int(fraction)
        fraction -= sec
    else:
        sec = _nint(fraction)
        fraction = 0.0
    s, m, h = abs(sec) % 60, (abs(sec) // 60) % 60, abs(sec) // 3600
    text = units = ""
    if precision in (SUBSEC_LOW, SUBSEC_HIGH):
        fraction = s + fraction
        if precision == SUBSEC_LOW:
            text, units = f'{fraction:05.2f}"'[:6] + text, "      " + units
        else:
            text, units = f'{fraction:07.4f}"'[:8] + text, "        " + units
        if not full:
            full = fraction < accuracy
    elif precision == SECOND:
        text, units = f'{s:02d}"'[:3] + text, "   " + units
        if not full:
            full = s == 0
    if precision == MINUTE or (precision > MINUTE and full):
        text, units = f"{m:02d}'"[:3] + text, "   " + units
        if not full:
            full = m == 0
    if precision == DEGREE or full:
        if sec + fraction < accuracy:
            d = " 0 "
        elif value < 0:
            d = f"-{h:d} "[:4]
        else:
            d = f"+{h:d} "[:4]
        text, units = d + text, (" " * (len(d) - 1) + "o")[:4] + units
    return full if report else (text, units)


# ------------------------------------------------------------------ the task


def wcslab(header, c1, c2, l1, l2, viewport, title: str = "", pars: dict | None = None,
           device: Device = VDM) -> tuple[Drawing, list[str]]:
    """wcslab on a viewport already mapped (vl, vr, vb, vt), the window c1..c2,
    l1..l2 the image's pixels; header is the WCS (an image's, or user_header's)."""
    sky = Sky(header)
    gio = _Gio(device)
    gio.cur = 1
    gio.set_view(*viewport)
    gio.set_window(c1, c2, l1, l2)
    wl = _WL(sky, pars or {}, title, gio, (c1, c2, l1, l2), device)
    wl.run()
    d = Drawing(gio.strokes, gio.texts, tuple(float(v) for v in viewport),
                (float(c1), float(c2), float(l1), float(l2)),
                tuple(wl.new_view[s] for s in (LEFT, RIGHT, BOTTOM, TOP)),
                SYSTEM_NAMES[sky.system],
                {v: k for k, v in GRAPH_TYPES.items()}[wl.graph_type],
                (wl.begin[0], wl.begin[1]), (wl.end[0], wl.end[1]),
                (wl.major[0], wl.major[1]), wl.labels, wl.title)
    return d, wl.messages


def labelled(image: str, params: dict | None = None, viewport=None,
             device: Device = VDM) -> tuple[Drawing, list[str]]:
    """t_wcslab: the image (or none: wcspars' log_x1..log_y2), its WCS (or with
    usewcs wcspars'), the title "image: its title", the viewport from vl..vt and fill
    (or the one given, as append and imd decide it), and wcslab."""
    from . import images

    p = dict(params or {})
    if image:
        data, header = images.read(image)
        if np.ndim(data) != 2:
            raise WcslabError(f"Image: {image} is not 2-dimensional")
        ny, nx = np.shape(data)
        c1, c2, l1, l2 = 1.0, float(nx), 1.0, float(ny)
        title = f"{image}: {str(header.get('OBJECT', '')).strip()}"
        if p.get("usewcs"):
            header = user_header(p)
    else:
        header = user_header(p)
        c1, c2 = float(f32(p.get("log_x1", 0.0))), float(f32(p.get("log_x2", 1.0)))
        l1, l2 = float(f32(p.get("log_y1", 0.0))), float(f32(p.get("log_y2", 1.0)))
        title = ""
    if viewport is None:
        viewport = map_viewport(c1, c2, l1, l2, p.get("vl"), p.get("vr"), p.get("vb"),
                                p.get("vt"), bool(p.get("fill", True)), device)
    return wcslab(header, c1, c2, l1, l2, viewport, title,
                  {k: v for k, v in p.items() if k in WLPARS}, device)
