"""surface -- a perspective view of an image as a surface, as IRAF's plot.surface
frames it.

IRAF's, ported: the grid (plt_getdata: block averaged or subsampled to xres by
yres), surf_limits (the floor and ceiling, their messages and the clipping), and
EZSRFC's and SRFACE's geometry -- the grid spaced so that it is twice as wide as
the data's range is high, the eye ten such ranges away at the angles angh
(horizontal) and angv (vertical), looking at the origin, TRN32S's perspective, and
the picture made square and put in NCAR's 10..1014 frame -- and the drawing
(draw): SRFGK's lines in the order that lets SRFACE's floating horizon (DRAWS,
core.ncar.Horizon) hide what a ridge is in front of, and with label
srf_perimeter's axes, ticks and PWRZS's stroked labels. Every polyline IRAF puts
into its metacode, to the GKI unit, down to what DRAWS reads past the end of its
horizons for a label outside the frame.

Translated from IRAF (pkg/plot/t_surface.x) and its NCAR utilities
(sys/gio/ncarutil/srface.f, pwrzs.f), Copyright (c) 1986 Association of
Universities for Research in Astronomy Inc.; see LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import ncar
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
    """The grid and its projection: u, v the grid's points in NCAR's 1..1024 frame as
    SRFGK keeps them (M(1,I,J), M(2,I,J), truncated: u[j-1, i-1]), x and y the grid's
    coordinates in space, the eye, the box of the data (x, y and z's extremes); the
    title and the messages. The drawing is draw()'s."""
    title: str
    u: np.ndarray
    v: np.ndarray
    x: np.ndarray
    y: np.ndarray
    eye: tuple
    z: np.ndarray
    box: tuple
    to_frame: object
    ncols: int
    nlines: int
    angh: float
    angv: float
    message: str = ""

    @property
    def m1(self) -> np.ndarray:
        return self.u.astype(int)

    @property
    def m2(self) -> np.ndarray:
        return self.v.astype(int)


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
    # S(1) = FACTE*CANG2*COS(ANG1): the intrinsic's double carries the product
    eye = (f32(float(f32(facte * cang2)) * math.cos(float(ang1))),
           f32(float(f32(facte * cang2)) * math.sin(float(ang1))),
           f32(float(facte) * math.sin(float(ang2)) + float(f32(f32(flo + hi) * f32(0.5)))))
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

    to_frame.limits = (float(umin), float(umax), float(vmin), float(vmax))
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
    """t_surface up to EZSRFC: the grid, the limits and the projection."""
    data = np.asarray(data, f32)
    if data.ndim == 1:
        data = data[None, :]
    title = f"{image}: {imtitle}" if title_param == "imtitle" else title_param
    z, message = getdata(data, subsample, preserve, xres, yres)
    z, more = surf_limits(z, floor, ceiling)
    x, y, eye, u, v, to_frame = project(z, angh, angv)
    box = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]),
           float(z.min()), float(z.max()))
    return Surface(title, u, v, x, y, eye, z, box, to_frame, data.shape[1],
                   data.shape[0], float(angh), float(angv), message + more)


CSIZE = 24                  # t_surface's character size for PWRZS
DELCRT = NRU - NLU          # TRN32S's frame width, for a single picture
WINDOW = (1.0, 1024.0, 1.0, 1024.0)


def viewport(label: bool) -> tuple[float, float, float, float]:
    """noaovp: the whole device, or 0.1..0.9 of it to leave room for the axes."""
    return (0.1, 0.9, 0.1, 0.9) if label else (0.0, 1.0, 0.0, 1.0)


def draw(s: Surface, label: bool = False) -> ncar.Gio:
    """What t_surface puts into the metacode after its title: SRFACE's mesh, with its
    hidden lines taken out by the floating horizon, and with label srf_perimeter's
    axes, ticks and PWRZS's labels -- polylines in GKI units."""
    vp = viewport(label)
    gio, horizon = ncar.srfgk(s.m1, s.m2, s.x, s.y, s.eye, vp,
                              frame=s.to_frame.limits)
    if label:
        perimeter(s, gio, horizon, vp)
    return gio


def perimeter(s: Surface, gio: ncar.Gio, horizon: ncar.Horizon, vp) -> None:
    """srf_perimeter: the two axes along the base of the surface on the sides toward
    the eye, their ticks, and the labels X-AXIS, Y-AXIS, 1 and the last column and
    line, in IRAF's order, clipping off (t_surface's G_CLIP no)."""
    z = s.z
    nlines, ncols = z.shape
    flo, hi = f32(z.min()), f32(z.max())
    delta = f32(f32(f32(hi - flo) / f32(max(ncols, nlines) - 1.0)) * f32(FACT1))
    xmin = f32(-(f32(f32(ncols // 2) * delta) + f32(f32((ncols + 1) % 2) * delta)))
    ymin = f32(-(f32(f32(nlines // 2) * delta) + f32(f32((nlines + 1) % 2) * delta)))
    dl = f32(f32(2.0) * delta)
    x1p, y1p = f32(xmin - delta), f32(ymin - delta)
    x2p = f32(xmin + f32(f32(ncols) * delta))
    y2p = f32(ymin + f32(f32(nlines) * delta))
    xv = [f32(x1p + f32(f32(i) * delta)) for i in range(ncols + 2)]
    yv = [f32(y1p + f32(f32(i) * delta)) for i in range(nlines + 2)]
    angh, angv = f32(s.angh), f32(s.angv)
    if angh > 180:
        angh = f32(angh - f32(360))
    elif angh < -180:
        angh = f32(angh + f32(360))
    if angv > 180:
        angv = f32(angv - f32(360))
    elif angv < -180:
        angv = f32(angv + f32(360))
    xcen = f32(f32(0.5) * f32(x1p + x2p))
    ycen = f32(f32(0.5) * f32(y1p + y2p))
    gio.clip = False
    gio.set_window(WINDOW)
    spps = ncar.Spps(vp, WINDOW, gio=gio)
    fr = s.to_frame
    wc2, wl2 = str(int(s.ncols)), str(int(s.nlines))

    def axis(xs, ys):
        pts = [fr(a, b, flo) for a, b in zip(xs, ys, strict=True)]
        gio.polyline([f32(u) for u, _ in pts], [f32(v) for _, v in pts])

    def ticks(pairs):
        for (xa, ya), (xb, yb) in pairs:
            ua, va = fr(xa, ya, flo)
            ub, vb = fr(xb, yb, flo)
            gio.polyline([f32(ua), f32(ub)], [f32(va), f32(vb)])

    def lab(x, y, text, path, up):
        pwrzs(s, spps, horizon, x, y, flo, text, CSIZE, path, up, 0)

    if angh >= 0 and angv >= 0:
        side, xs, sx, xpath, xup = y2p, xv[1:], f32(y2p + dl), -1, -2
        yside, ys, sy, yup = x2p, yv[1:], f32(x2p + dl), -1
        tx, ty = f32(y2p + delta), f32(x2p + delta)
    elif angh >= 0:
        side, xs, sx, xpath, xup = y1p, xv, f32(y1p - dl), -1, 2
        yside, ys, sy, yup = x1p, yv, f32(x1p - dl), 1
        tx, ty = f32(y1p - delta), f32(x1p - delta)
    elif angv > 0:
        side, xs, sx, xpath, xup = y1p, xv[1:], f32(y1p - dl), 1, 2
        yside, ys, sy, yup = x2p, yv, f32(x2p + dl), -1
        tx, ty = f32(y1p - delta), f32(x2p + delta)
    else:
        side, xs, sx, xpath, xup = y2p, xv, f32(y2p + dl), 1, -2
        yside, ys, sy, yup = x1p, yv[1:], f32(x1p - dl), 1
        tx, ty = f32(y2p + delta), f32(x1p - delta)
    xs, ys = xs[:ncols + 1], ys[:nlines + 1]
    axis(xs, [side] * len(xs))
    lab(xcen, sx, "X-AXIS", xpath, xup)
    ticks(((xv[i], side), (xv[i], tx)) for i in range(1, ncols + 1))
    lab(xmin, sx, "1", xpath, xup)
    lab(xv[ncols], sx, wc2, xpath, xup)
    axis([yside] * len(ys), ys)
    lab(sy, ycen, "Y-AXIS", 2, yup)
    ticks(((yside, yv[j]), (ty, yv[j])) for j in range(1, nlines + 1))
    lab(sy, ymin, "1", 2, yup)
    lab(sy, yv[nlines], wl2, 2, yup)
    spps.plotit(0, 0, 2)
    gio.flush()


def _font():
    import json
    from pathlib import Path as _P

    path = _P(__file__).resolve().parent.parent / "data" / "ncar_pwrz_font.json"
    return json.loads(path.read_text(encoding="utf-8"))


_FONT = None


def pwrzs(s: Surface, spps: ncar.Spps, horizon: ncar.Horizon, x, y, z, text: str,
          isize: int, lin3: int, itop: int, icnt: int) -> None:
    """NCAR's PWRZS: characters of its own stroked font written in the space of the
    surface -- along the axis `lin3` (1 x, 2 y, 3 z, negative backwards), their tops
    toward `itop` -- each stroke through TRN32S. A label whose corners are both
    outside the surface's box on the eye's side is drawn as it is (PLOTIT); one that
    is not goes through the floating horizon (DRAWS), hidden where the surface is in
    front of it."""
    global _FONT
    if _FONT is None:
        _FONT = _font()
    kx, ky, index = _FONT["kx"], _FONT["ky"], _FONT["index"]
    spps.plotif(0, 0, 2)
    nn = len(text)
    if nn <= 0:
        return
    fnnm1 = f32(nn - 1)
    q = f32(f32(isize) / f32(6.0)) if isize > 3 else f32({0: 1.3334, 1: 2.0, 2: 2.6667,
                                                          3: 4.0}[isize])
    ct = f32(float(q) * math.cos(0.0))
    st = f32(float(q) * math.sin(0.0))
    x, y, z = f32(x), f32(y), f32(z)
    xxmin, xxmax, yymin, yymax, zzmin, zzmax = (f32(v) for v in s.box)
    ex, ey, ez = (f32(v) for v in s.eye)
    delmax = max(f32(xxmax - xxmin), f32(yymax - yymin), f32(zzmax - zzmin))
    factor = f32(delmax / f32(DELCRT))
    factx = f32(math.copysign(float(factor), float(lin3)))
    facty = f32(math.copysign(float(factor), float(itop)))
    lin, ito = max(1, min(3, abs(lin3))), max(1, min(3, abs(itop)))
    if lin == ito:
        return                      # SETER: LINE OR ITOP IMPROPER
    st8 = {"ivis": -1, "iviss": 0, "ix1": 0, "iy1": 0}

    def point(ix, iy):
        a = f32(factx * f32(ix))
        b = f32(facty * f32(iy))
        xx, yy, zz = x, y, z
        if (lin, ito) == (2, 1):
            xx, yy = f32(x + b), f32(y + a)
        elif (lin, ito) == (3, 1):
            xx, zz = f32(x + b), f32(z + a)
        elif (lin, ito) == (1, 2):
            xx, yy = f32(x + a), f32(y + b)
        elif (lin, ito) == (3, 2):
            yy, zz = f32(y + b), f32(z + a)
        elif (lin, ito) == (1, 3):
            xx, zz = f32(x + a), f32(z + b)
        else:
            yy, zz = f32(y + a), f32(z + b)
        return xx, yy, zz

    def corner(ix, iy, iz):                                  # INITZS entry 2
        xx, yy, zz = point(ix, iy)
        n = 0
        n += ex > xxmax and xx > xxmax
        n += ey > yymax and yy > yymax
        n += ez > zzmax and zz > zzmax
        n += ex < xxmin and xx < xxmin
        n += ey < yymin and yy < yymin
        n += ez < zzmin and zz < zzmin
        if iz == 1:
            st8["iviss"] = n
        else:
            st8["ivis"] = min(st8["iviss"], n)

    vx1, vx2, vy1, vy2 = spps.vp
    scale = f32(f32(32768.0) / f32(1023.0))

    def stroke(ix, iy, first: bool):                         # INITZS entries 3, 4
        xt, yt = s.to_frame(*point(ix, iy))
        xt, yt = f32(xt), f32(yt)
        if st8["ivis"] > 0:
            mx = f32(f32(f32(scale * f32(vx2 - vx1)) * f32(xt - f32(1)))
                     + f32(vx1 * f32(32768.0)))
            my = f32(f32(f32(scale * f32(vy2 - vy1)) * f32(yt - f32(1)))
                     + f32(vy1 * f32(32768.0)))
            spps.plotit(ncar._ifix(mx), ncar._ifix(my), 0 if first else 1)
        elif first:
            st8["ix1"], st8["iy1"] = ncar._ifix(xt), ncar._ifix(yt)
        else:
            ix2, iy2 = ncar._ifix(xt), ncar._ifix(yt)
            horizon.draws(st8["ix1"], st8["iy1"], ix2, iy2, 1, 0)
            st8["ix1"], st8["iy1"] = ix2, iy2

    xc = f32(f32(f32(0) - f32(f32(2.0) * ct)) + f32(f32(3.5) * st))
    yc = f32(f32(f32(0) - f32(f32(2.0) * st)) - f32(f32(3.5) * ct))
    jcnt = max(-1, min(1, icnt)) + 2
    if jcnt == 2:
        xc = f32(xc - f32(f32(ct * fnnm1) * f32(3)))
        yc = f32(yc - f32(f32(st * fnnm1) * f32(3)))
    elif jcnt == 1:
        xc = f32(xc + f32(ct * f32(2)))
        yc = f32(yc + f32(st * f32(2)))
    else:
        xc = f32(xc - f32(ct * f32(2)))
        yc = f32(yc - f32(st * f32(2)))
        xc = f32(xc - f32(f32(ct * fnnm1) * f32(6)))
        yc = f32(yc - f32(f32(st * fnnm1) * f32(6)))
    corner(ncar._ifix(xc), ncar._ifix(yc), 1)
    corner(ncar._ifix(f32(xc + f32(f32(ct * f32(6)) * fnnm1))),
           ncar._ifix(f32(yc + f32(f32(st * f32(6)) * fnnm1))), 2)
    stroke(ncar._ifix(xc), ncar._ifix(yc), True)
    for k in range(nn):
        xb, yb = xc, yc
        ip = 1
        start = index.get(text[k], index[" "])
        for ell in range(20):
            nx, ny = kx[start - 1 + ell], ky[start - 1 + ell]
            if nx == 7:
                ip = 0
                if ny == 7:
                    break
                continue
            xc = f32(f32(xb + f32(f32(nx) * ct)) - f32(f32(ny) * st))
            yc = f32(f32(yb + f32(f32(nx) * st)) + f32(f32(ny) * ct))
            stroke(ncar._ifix(f32(xc + f32(0.5))), ncar._ifix(f32(yc + f32(0.5))), ip == 0)
            ip = 1
    spps.plotit(0, 0, 0)
    spps.plotif(0, 0, 2)
