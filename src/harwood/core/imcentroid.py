"""IRAF's imcentroid (images.immatch), ported.

From iraf-community 2.18, pkg/images/immatch/src/listmatch/t_imctroid.x.

Each star is centred by IRAF's "MPC" algorithm (ia_center): in a box of
boxsize pixels around the nearest pixel, the background is the mean of the
box (or `background`); only pixels above it, less it, enter the x and y
marginals; the centre is the first moment of each; repeat from the new
nearest pixel until it stops moving by more than `tolerance` whole pixels,
or give up after `niterate` passes.

Without a shifts file, the first star in the list is centred in each image
with the big box, and the offset it gives is the initial shift for every
other star. A star is rejected when it falls off the image, fails to centre,
or lands more than `maxshift` from where it was looked for.

The shift of an image is the mean over the stars centred in both it and the
reference of (reference - image): the shift that, given to imshift, moves
the image onto the reference. Its error is the centring errors propagated;
the "internal" error is the scatter of the individual shifts about their
mean. Then the section every shifted image covers: #Trim_Section.

IRAF computes in single precision; so does this, where it decides a pixel.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nlfit import square

f32 = np.float32
MAX_REAL = 3.4028235e38


class CentreError(Exception):
    """A star IRAF would refuse, with IRAF's reason."""


@dataclass
class Settings:
    boxsize: int = 7
    bigbox: int = 11
    negative: bool = False
    background: float | None = None
    lower: float | None = None
    upper: float | None = None
    niterate: int = 3
    tolerance: int = 0
    maxshift: float | None = None

    def __post_init__(self):
        if self.lower is not None and self.upper is not None:
            self.lower, self.upper = min(self.lower, self.upper), max(self.lower, self.upper)

    @property
    def small(self) -> int:
        """Half-width of the fine box (ia_init rounds an even size up)."""
        size = self.boxsize + 1 if self.boxsize % 2 == 0 else self.boxsize
        return (size - 1) // 2

    @property
    def big(self) -> int:
        size = self.bigbox + 1 if self.bigbox % 2 == 0 else self.bigbox
        return (size - 1) // 2

    @property
    def tries(self) -> int:
        return max(self.niterate, 2)

    @property
    def limit(self) -> float:
        return MAX_REAL if self.maxshift is None else abs(self.maxshift)


def nint(value: float | np.floating) -> int:
    """SPP's nint: half away from zero."""
    return int(np.floor(value + 0.5)) if value >= 0 else -int(np.floor(-value + 0.5))


def centre(data: np.ndarray, x: float, y: float, half: int,
           s: Settings) -> tuple[float, float, float, float]:
    """ia_center: (xcenter, ycenter, xsigma, ysigma), 1-based. Raises
    CentreError when IRAF would return ERR."""
    ny, nx = data.shape
    xold, yold = f32(x), f32(y)
    for _ in range(s.tries):
        x1, x2 = max(nint(xold) - half, 1), min(nint(xold) + half, nx)
        y1, y2 = max(nint(yold) - half, 1), min(nint(yold) + half, ny)
        raster = data[y1 - 1:y2, x1 - 1:x2]
        back, lo, hi = _threshold(raster, s)
        use = (raster >= lo) & (raster <= hi)
        signal = np.where(use, raster - back, f32(0)).astype(f32)
        # The marginals, summed in single precision in IRAF's order.
        row = np.zeros(raster.shape[1], dtype=f32)
        for line in signal:
            row = (row + line).astype(f32)
        col = np.zeros(raster.shape[0], dtype=f32)
        for column in signal.T:
            col = (col + column).astype(f32)
        sign = f32(-1) if s.negative else f32(1)
        row = (row / (sign * f32(raster.shape[0]))).astype(f32)
        col = (col / (sign * f32(raster.shape[1]))).astype(f32)
        if row.min() < 0:
            raise CentreError("Negative value in marginal row")
        if col.min() < 0:
            raise CentreError("Negative value in marginal column")
        cx, ex = _moment(row)
        cy, ey = _moment(col)
        xnew, ynew = f32(x1 + cx), f32(y1 + cy)
        if (abs(nint(xnew) - nint(xold)) <= s.tolerance
                and abs(nint(ynew) - nint(yold)) <= s.tolerance):
            return float(xnew), float(ynew), float(ex), float(ey)
        xold, yold = xnew, ynew
    raise CentreError(f"failed to converge near ({nint(x)},{nint(y)})")


def _threshold(raster, s: Settings):
    """ia_threshold: the local background and the data range used."""
    lo = f32(raster.min()) if s.lower is None else f32(s.lower)
    hi = f32(raster.max()) if s.upper is None else f32(s.upper)
    if s.background is None:
        values = raster.astype(np.float64).ravel()
        if hi != lo:
            values = values[(values >= lo) & (values <= hi)]
        if values.size == 0:
            raise CentreError("no pixels between thresholds")
        back = f32(values.sum() / values.size)
    else:
        back = f32(s.background)
    if s.negative:
        return back, lo, min(hi, back)
    return back, max(lo, back), hi


def _moment(a):
    """ia_ctr1d: the first moment from 0 and its error."""
    sumi = sumix = sumix2 = f32(0)
    for i, pix in enumerate(a):
        sumi = f32(sumi + pix)
        sumix = f32(sumix + pix * f32(i))
        sumix2 = f32(sumix2 + pix * f32(i * i))
    if sumi == 0:
        raise CentreError("zero marginal vector")
    centroid = f32(sumix / sumi)
    err = f32(sumix2 / sumi - centroid * centroid)
    err = f32(np.sqrt(err / sumi)) if err > 0 else f32(0)
    return centroid, err


@dataclass
class Star:
    x: float = np.nan
    y: float = np.nan
    xerr: float = np.nan
    yerr: float = np.nan
    rejected: bool = True
    reason: str = ""


@dataclass
class Image:
    name: str
    nx: int
    ny: int
    stars: list[Star]
    xshift: float | None = None      # initial, then measured
    yshift: float | None = None
    xerr: float | None = None        # propagated centring errors
    yerr: float | None = None
    xinternal: float | None = None   # scatter of the stars about the mean
    yinternal: float | None = None
    nstars: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class Result:
    images: list[Image]
    reference: Image | None
    trim: tuple[int, int, int, int] | None = None
    vignette: tuple[int, int, int, int] | None = None


def measure(frames: list[tuple[str, np.ndarray]], reference: tuple[str, np.ndarray] | None,
            coords: list[tuple[float, float]], s: Settings,
            shifts: list[tuple[float, float]] | None = None) -> Result:
    """t_imcentroid over images already read."""
    if not coords:
        raise ValueError("No coordinates found")
    if shifts is not None and len(shifts) != len(frames):
        raise ValueError("Number of shifts doesn't match images")
    images = []
    ncentred = 0
    for k, (name, data) in enumerate(frames):
        data = np.asarray(data, dtype=f32)
        ny, nx = data.shape
        image = Image(name, nx, ny, [])
        if shifts is None:
            try:
                x, y, _, _ = centre(data, coords[0][0], coords[0][1], s.big, s)
            except CentreError as error:
                raise RuntimeError(f"{name}: Problem with coarse centering ({error})") \
                    from error
            xs, ys = f32(f32(coords[0][0]) - f32(x)), f32(f32(coords[0][1]) - f32(y))
        else:
            xs, ys = f32(shifts[k][0]), f32(shifts[k][1])
        image.xshift, image.yshift = float(xs), float(ys)
        image.stars = [_star(data, f32(f32(cx) - xs), f32(f32(cy) - ys), s)
                       for cx, cy in coords]
        if any(not star.rejected for star in image.stars):
            ncentred += 1
        else:
            image.warnings.append(f"Warning: no sources centered in {name}")
        images.append(image)

    if reference is None:
        return Result(images, None)
    rname, rdata = reference
    rdata = np.asarray(rdata, dtype=f32)
    ref = Image(rname, rdata.shape[1], rdata.shape[0],
                [_star(rdata, f32(cx), f32(cy), s) for cx, cy in coords])
    result = Result(images, ref)
    if all(star.rejected for star in ref.stars):
        ref.warnings.append("Warning: no sources centered in reference")
        return result
    _stats(images, ref)
    if ncentred > 1:
        result.trim, result.vignette = _trim(images)
    return result


def _star(data, x, y, s: Settings) -> Star:
    ny, nx = data.shape
    if x < 1 or x > nx or y < 1 or y > ny:
        return Star(reason="off the image")
    try:
        cx, cy, ex, ey = centre(data, float(x), float(y), s.small, s)
    except CentreError as error:
        return Star(reason=str(error))
    if abs(f32(cx) - x) > s.limit or abs(f32(cy) - y) > s.limit:
        return Star(cx, cy, ex, ey, True, "moved more than maxshift")
    return Star(cx, cy, ex, ey, False)


def _stats(images: list[Image], ref: Image) -> None:
    """ia_stats, in single precision."""
    for image in images:
        xsum = ysum = xsum2 = ysum2 = xsig2 = ysig2 = f32(0)
        n = 0
        for star, rstar in zip(image.stars, ref.stars, strict=True):
            if star.rejected or rstar.rejected:
                continue
            xs = f32(f32(rstar.x) - f32(star.x))
            ys = f32(f32(rstar.y) - f32(star.y))
            xsum, ysum = f32(xsum + xs), f32(ysum + ys)
            xsum2, ysum2 = f32(xsum2 + xs * xs), f32(ysum2 + ys * ys)
            xsig2 = f32(xsig2 + square(rstar.xerr) + square(star.xerr))
            ysig2 = f32(ysig2 + square(rstar.yerr) + square(star.yerr))
            n += 1
        image.nstars = n
        if n == 0:
            image.xshift = image.yshift = None
            continue
        image.xshift, image.yshift = float(f32(xsum / f32(n))), float(f32(ysum / f32(n)))
        if n > 1:
            fn = f32(n)
            xvar = f32((fn * xsum2 - xsum * xsum) / (fn * f32(n - 1)))
            yvar = f32((fn * ysum2 - ysum * ysum) / (fn * f32(n - 1)))
            image.xinternal = float(f32(np.sqrt(max(xvar / fn, f32(0)))))
            image.yinternal = float(f32(np.sqrt(max(yvar / fn, f32(0)))))
        image.xerr = float(f32(np.sqrt(max(xsig2, f32(0))) / f32(n)))
        image.yerr = float(f32(np.sqrt(max(ysig2, f32(0))) / f32(n)))


def _trim(images: list[Image]):
    """ia_trim: the section every shifted image covers, and the vignetted
    one IRAF warns about when the images are smaller than it."""
    first = True
    ixlo = ixhi = iylo = iyhi = 0
    xmin = ymin = 0.0
    for image in images:
        if image.xshift is None or image.yshift is None:
            continue
        xlo, ylo = f32(1 + f32(image.xshift)), f32(1 + f32(image.yshift))
        xhi = f32(f32(image.nx) + f32(image.xshift))
        yhi = f32(f32(image.ny) + f32(image.yshift))
        lo_x = int(xlo) + (1 if xlo > int(xlo) else 0)
        hi_x = int(xhi) - (1 if xhi < int(xhi) else 0)
        lo_y = int(ylo) + (1 if ylo > int(ylo) else 0)
        hi_y = int(yhi) - (1 if yhi < int(yhi) else 0)
        if first:
            ixlo, ixhi, iylo, iyhi = lo_x, hi_x, lo_y, hi_y
            xmin, ymin = image.nx, image.ny
            first = False
        else:
            ixlo, ixhi = max(ixlo, lo_x), min(ixhi, hi_x)
            iylo, iyhi = max(iylo, lo_y), min(iyhi, hi_y)
            xmin, ymin = min(image.nx, xmin), min(image.ny, ymin)
    if first:
        return None, None
    v = (max(1, min(ixlo, int(xmin))), max(1, min(ixhi, int(xmin))),
         max(1, min(iylo, int(ymin))), max(1, min(iyhi, int(ymin))))
    trim = (ixlo, ixhi, iylo, iyhi)
    return trim, (v if v != trim else None)


def report(result: Result, verbose: bool = True) -> list[str]:
    """What IRAF's imcentroid prints, line for line."""
    lines = []
    if verbose:
        lines.append("#Coords         Image     X-center   Err      Y-center   Err     Num")
        for image in result.images:
            for j, star in enumerate(image.stars, 1):
                if not star.rejected:
                    lines.append(f"{image.name:>20s}   {star.x:9.3f} ({star.xerr:.3f})   "
                                 f"{star.y:9.3f} ({star.yerr:.3f})  {j:4d}")
            lines.append("")
    ref = result.reference
    if ref is None:
        return lines
    if verbose:
        header = False
        for j, star in enumerate(ref.stars, 1):
            if star.rejected:
                continue
            if not header:
                lines.append("#Refcoords           Reference     X-center   Err      "
                             "Y-center   Err     Num")
                header = True
            lines.append(f"{ref.name:>20s}   {star.x:9.3f} ({star.xerr:.3f})   "
                         f"{star.y:9.3f} ({star.yerr:.3f})  {j:4d}")
        if header:
            lines.append("")
    header = False
    for image in result.images:
        if image.xshift is None or image.nstars == 0:
            continue
        if not header:
            lines.append("#Shifts         Image    X-shift   Err      Y-shift   Err      "
                         "N      Internal")
            header = True
        internal = (f"({image.xinternal:.3f},{image.yinternal:.3f})"
                    if image.xinternal is not None else "(INDEF,INDEF)")
        lines.append(f"{image.name:>20s}   {image.xshift:8.3f} ({image.xerr:.3f})   "
                     f"{image.yshift:8.3f} ({image.yerr:.3f}) {image.nstars:4d}   {internal}")
    if result.trim is not None:
        lines.append("")
        if result.vignette is not None:
            lines.append("#Vignette_Section = [{}:{},{}:{}]".format(*result.vignette))
        lines.append("#Trim_Section = [{}:{},{}:{}]".format(*result.trim))
    return lines
