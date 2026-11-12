"""surface -- a perspective view of an image as a surface, as IRAF's plot.surface
frames it.

IRAF's, ported: the grid (plt_getdata: block averaged or subsampled to xres by
yres), surf_limits (the floor and ceiling, their messages and the clipping), and
EZSRFC's and SRFACE's geometry -- the grid spaced so that it is twice as wide as
the data's range is high, the eye ten such ranges away at the angles angh
(horizontal) and angv (vertical), looking at the origin, TRN32S's perspective, and
the picture made square and put in NCAR's 10..1014 frame. So each point of the
grid lands where IRAF puts it.

The hidden lines are taken out by painting the grid's cells from the farthest to
the nearest, each over the ones behind it, where NCAR's SRFACE keeps a floating
horizon of 1024 columns; the lines that show are the same, their ends where they
disappear behind a ridge are not IRAF's to the point. NCAR's own (SRFACE's
horizon) is in harwood's porting queue with the rest of NCAR.

Translated from IRAF (pkg/plot/t_surface.x) and its NCAR utilities
(sys/gio/ncarutil/srface.f), Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; see LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .contour import getdata

f32 = np.float32
FACT1, FACT2 = 2.0, 5.0
#: TRN32S's frame for a single picture (LL = 1): u from 10 to 1014, v likewise.
NLU, NRU, NBV, NTV = 10, 1014, 10, 1014


class SurfaceError(RuntimeError):
    """What stops IRAF's surface: its message (printed, the plot not drawn)."""


def surf_limits(z: np.ndarray, floor, ceiling):
    """surf_limits: the data clipped to the floor and ceiling, and what IRAF says."""
    dmin, dmax = f32(z.min()), f32(z.max())
    from .iraffmt import _equal

    if _equal(float(dmin), float(dmax)):
        raise SurfaceError("Constant valued array; no plot drawn")
    tf = dmin if floor is None else f32(floor)
    tc = dmax if ceiling is None else f32(ceiling)
    delta = f32(tc - tf)
    message = ""
    apply = True
    if delta < 0:
        fl, ce = tc, tf
    elif _equal(float(delta), 0.0):
        apply = False
        fl, ce = dmin, dmax
        message = "Floor and ceiling are degenerate values and will be ignored\n"
    else:
        fl, ce = tf, tc
    if apply:
        if dmax <= fl:
            raise SurfaceError("Entire image is at or below specified floor")
        if dmin >= ce:
            raise SurfaceError("Entire image is at or above specified ceiling")
        z = np.minimum(ce, np.maximum(fl, z)).astype(f32)
    return z, message


@dataclass
class Surface:
    """The projected grid (u, v in NCAR's 1..1024 frame), its cells in the order they
    are painted, the title and the messages."""
    title: str
    u: np.ndarray
    v: np.ndarray
    order: list[tuple[int, int]]
    message: str = ""
    corners: dict = field(default_factory=dict)


def project(z: np.ndarray, angh: float, angv: float):
    """EZSRFC and SRFGK: the grid's x and y, the eye, and TRN32S's u, v of every point
    (z[j, i] is Fortran's Z(I,J)), in the 10..1014 frame."""
    ny, mx = z.shape
    ang1 = f32(f32(angh) * f32(3.14159265358979) / f32(180.0))
    ang2 = f32(f32(angv) * f32(3.14159265358979) / f32(180.0))
    flo, hi = f32(z.min()), f32(z.max())
    delta = f32(f32(f32(hi - flo) / f32(max(mx, ny) - 1.0)) * f32(FACT1))
    xmin = f32(-(f32(f32(mx // 2) * delta) + f32(f32((mx + 1) % 2) * delta)))
    ymin = f32(-(f32(f32(ny // 2) * delta) + f32(f32((ny + 1) % 2) * delta)))
    x = np.array([f32(xmin + f32(f32(i) * delta)) for i in range(mx)], f32)
    y = np.array([f32(ymin + f32(f32(j) * delta)) for j in range(ny)], f32)
    facte = f32(f32(f32(hi - flo) * f32(FACT1)) * f32(FACT2))
    cang2 = f32(math.cos(ang2))
    eye = (f32(f32(facte * cang2) * f32(math.cos(ang1))),
           f32(f32(facte * cang2) * f32(math.sin(ang1))),
           f32(f32(facte * f32(math.sin(ang2))) + f32(f32(flo + hi) * f32(0.5))))
    look = (f32(0), f32(0), f32(0))
    trn = _Trn(look, eye)
    uu = np.empty((ny, mx))
    vv = np.empty((ny, mx))
    for j in range(ny):
        for i in range(mx):
            uu[j, i], vv[j, i] = trn(x[i], y[j], z[j, i])
    umin, umax, vmin, vmax = uu.min(), uu.max(), vv.min(), vv.max()
    width, height = f32(umax - umin), f32(vmax - vmin)
    dif = f32(f32(0.5) * f32(width - height))
    if dif < 0:
        umin, umax = f32(umin + dif), f32(umax - dif)
    elif dif > 0:
        vmin, vmax = f32(vmin - dif), f32(vmax + dif)
    u3 = f32(f32(NRU - NLU) / f32(umax - umin))
    v3 = f32(f32(NTV - NBV) / f32(vmax - vmin))
    # xx = u1 + u3 * (fact * xx - u0) in real, and M(1,I,J) = UT truncates it
    m1 = np.trunc((f32(NLU) + (u3 * (uu.astype(f32) - f32(umin))).astype(f32))
                  .astype(f32)).astype(float)
    m2 = np.trunc((f32(NBV) + (v3 * (vv.astype(f32) - f32(vmin))).astype(f32))
                  .astype(f32)).astype(float)
    def to_frame(px, py, pz):
        """TRN32S after SRFGK: a point of the space in the 10..1014 frame."""
        uu1, vv1 = trn(px, py, pz)
        return (float(f32(f32(NLU) + f32(u3 * f32(f32(uu1) - f32(umin))))),
                float(f32(f32(NBV) + f32(v3 * f32(f32(vv1) - f32(vmin))))))

    return x, y, eye, m1, m2, to_frame


class _Trn:
    """TRN32S: the perspective onto the plane through the point looked at."""

    def __init__(self, look, eye):
        ax, ay, az = (f32(c) for c in look)
        ex, ey, ez = (f32(c) for c in eye)
        dx, dy, dz = f32(ax - ex), f32(ay - ey), f32(az - ez)
        d = f32(math.sqrt(float(f32(f32(f32(dx * dx) + f32(dy * dy)) + f32(dz * dz)))))
        self.cosal, self.cosbe, self.cosga = f32(dx / d), f32(dy / d), f32(dz / d)
        singa = f32(math.sqrt(float(f32(1.0) - f32(self.cosga * self.cosga))))
        self.top = singa < 0.0001
        if self.top:
            sinbe = f32(math.sqrt(float(f32(1.0) - f32(self.cosbe * self.cosbe))))
            self.r = f32(f32(1.0) / sinbe)
        else:
            self.r = f32(f32(1.0) / singa)
        self.a, self.e, self.d = (ax, ay, az), (ex, ey, ez), d

    def __call__(self, xx, yy, zz):
        ax, ay, az = self.a
        ex, ey, ez = self.e
        xx, yy, zz = f32(xx), f32(yy), f32(zz)
        q = f32(self.d / f32(f32(f32(f32(xx - ex) * self.cosal)
                                 + f32(f32(yy - ey) * self.cosbe))
                             + f32(f32(zz - ez) * self.cosga)))
        px = f32(f32(ex + f32(q * f32(xx - ex))) - ax)
        py = f32(f32(ey + f32(q * f32(yy - ey))) - ay)
        pz = f32(f32(ez + f32(q * f32(zz - ez))) - az)
        if not self.top:
            u = f32(f32(f32(px * self.cosbe) - f32(py * self.cosal)) * self.r)
            v = f32(pz * self.r)
        else:
            u = f32(f32(f32(pz * self.cosal) - f32(px * self.cosga)) * self.r)
            v = f32(py * self.r)
        return float(u), float(v)


def surface(data: np.ndarray, image: str, title_param: str = "imtitle",
            imtitle: str = "", floor=None, ceiling=None, angh: float = -33.0,
            angv: float = 25.0, xres: int = 64, yres: int = 64, subsample: bool = False,
            preserve: bool = True) -> Surface:
    """t_surface up to EZSRFC: the grid, the limits, the projection and the order the
    cells are painted in (the farthest from the eye first)."""
    data = np.asarray(data, f32)
    if data.ndim == 1:
        data = data[None, :]
    title = f"{image}: {imtitle}" if title_param == "imtitle" else title_param
    z, message = getdata(data, subsample, preserve, xres, yres)
    z, more = surf_limits(z, floor, ceiling)
    x, y, eye, u, v, to_frame = project(z, angh, angv)
    ny, nx = z.shape
    cells = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            cx = 0.5 * (float(x[i]) + float(x[i + 1]))
            cy = 0.5 * (float(y[j]) + float(y[j + 1]))
            cz = 0.25 * float(z[j, i] + z[j, i + 1] + z[j + 1, i] + z[j + 1, i + 1])
            dist = (cx - eye[0]) ** 2 + (cy - eye[1]) ** 2 + (cz - eye[2]) ** 2
            cells.append((dist, i, j))
    cells.sort(reverse=True)
    out = Surface(title, u, v, [(i, j) for _, i, j in cells], message + more)
    out.to_frame = to_frame
    out.z = z
    out.ncols, out.nlines = data.shape[1], data.shape[0]
    out.angh, out.angv = float(angh), float(angv)
    return out


def perimeter(s: Surface):
    """srf_perimeter (label yes): the two axes along the base of the surface on the
    sides toward the eye, their ticks, and the labels X-AXIS, Y-AXIS, 1 and the last
    column and line -- as (lines, texts) in the frame."""
    z = s.z
    nlines, ncols = z.shape
    flo, hi = f32(z.min()), f32(z.max())
    delta = f32(f32(f32(hi - flo) / f32(max(ncols, nlines) - 1.0)) * f32(FACT1))
    xmin = f32(-(f32(f32(ncols // 2) * delta) + f32(f32((ncols + 1) % 2) * delta)))
    ymin = f32(-(f32(f32(nlines // 2) * delta) + f32(f32((nlines + 1) % 2) * delta)))
    dl = f32(2.0 * delta)
    x1p, y1p = f32(xmin - delta), f32(ymin - delta)
    x2p = f32(xmin + f32(f32(ncols) * delta))
    y2p = f32(ymin + f32(f32(nlines) * delta))
    xv = [float(f32(x1p + f32(f32(i) * delta))) for i in range(ncols + 2)]
    yv = [float(f32(y1p + f32(f32(i) * delta))) for i in range(nlines + 2)]
    angh, angv = s.angh, s.angv
    angh = angh - 360 if angh > 180 else angh + 360 if angh < -180 else angh
    angv = angv - 360 if angv > 180 else angv + 360 if angv < -180 else angv
    xcen, ycen = 0.5 * (x1p + x2p), 0.5 * (y1p + y2p)
    fr = s.to_frame
    lines, texts = [], []
    wc2, wl2 = str(int(s.ncols)), str(int(s.nlines))

    def axis(xs, ys):
        pts = [fr(a, b, flo) for a, b in zip(xs, ys, strict=True)]
        lines.append(np.asarray(pts, float))

    def label(px, py, text):
        texts.append((*fr(px, py, flo), text))

    if angh >= 0:
        x_side, y_side = (y2p, x2p) if angv >= 0 else (y1p, x1p)
    else:
        x_side, y_side = (y1p, x2p) if angv > 0 else (y2p, x1p)
    sx = 1 if x_side == y2p else -1
    sy = 1 if y_side == x2p else -1
    front_x = (angh >= 0 and angv >= 0) or (angh < 0 and angv > 0)
    front_y = (angh >= 0 and angv >= 0) or (angh < 0 and angv <= 0)
    xs = xv[1:ncols + 2] if front_x else xv[:ncols + 1]
    ys = yv[1:nlines + 2] if front_y else yv[:nlines + 1]
    axis(xs, [x_side] * len(xs))
    for i in range(1, ncols + 1):
        lines.append(np.asarray([fr(xv[i], x_side, flo), fr(xv[i], x_side + sx * delta, flo)]))
    label(xcen, x_side + sx * dl, "X-AXIS")
    label(xmin, x_side + sx * dl, "1")
    label(xv[ncols], x_side + sx * dl, wc2)
    axis([y_side] * len(ys), ys)
    for j in range(1, nlines + 1):
        lines.append(np.asarray([fr(y_side, yv[j], flo), fr(y_side + sy * delta, yv[j], flo)]))
    label(y_side + sy * dl, ycen, "Y-AXIS")
    label(y_side + sy * dl, ymin, "1")
    label(y_side + sy * dl, yv[nlines], wl2)
    return lines, texts


def polygons(s: Surface):
    """The cells as quadrilaterals in the 1..1024 frame, in painting order."""
    for i, j in s.order:
        xs = [s.u[j, i], s.u[j, i + 1], s.u[j + 1, i + 1], s.u[j + 1, i], s.u[j, i]]
        ys = [s.v[j, i], s.v[j, i + 1], s.v[j + 1, i + 1], s.v[j + 1, i], s.v[j, i]]
        yield np.asarray(xs, float), np.asarray(ys, float)
