"""onedspec's splot, its measurements and its cursor loop without a window:
noao/onedspec/splot (splot.x, getimage.x, avgsnr.x, eqwidth.x, sumflux.x,
gfit.x, spdeblend.x, eqwidthcp.x, smooth.x, splotfun.x, fudgept.x, fudgex.x,
conflam.x, confnu.x, usercoord.x, splotcolon.x, anshdr.x), as IRAF 2.18 runs
them.

splot plots a spectrum and measures it under the cursor: the coordinates and the
data at a point (space), the mean, rms and signal to noise between two (m), an
equivalent width and a flux against a straight continuum (e), a profile fitted to
a line (k), several fitted together (d), and the spectrum smoothed, fixed, put
through arithmetic, converted to flux per frequency or per wavelength. The
spectrum is opened as onedspec opens it (core/shdr.py); the profiles are those
of fitprofs, splot's deblend.x (core/fitprofs.py).

What follows from the code, kept:

* everything is in real; the cursor's x is taken to the nearest pixel of the
  spectrum, and between two reads the ends are swapped into order (fixx);
* the equivalent width sums whole pixels inside the range and fractions of the
  two at its ends, and the centre is the mean of x weighted by the absolute
  difference from the continuum to the power 1.5 (sumflux);
* a profile fit (k, d) scales the data by their largest absolute value first;
  k guesses the width from the two pixels beside the peak;
* the Monte-Carlo noise of k and d is gasdev on urand from seed 1 at each fit,
  and gasdev keeps its pair of deviates from one fit to the next;
* the log (save_file) gets the date and the image when a measurement is the
  first of a spectrum, and the column header when the key changes -- m only
  gets no header.
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import fitprofs, iraffmt, shdr
from . import units as unitsmod

f32 = np.float32
MAX_REAL = float(np.finfo(np.float32).max)
INDEF = None
VLIGHT = f32(2.997925e18)
OPTIONS = ("auto", "zero", "xydraw", "histogram", "nosysid", "wreset", "flip",
           "overplot")

#: splot.key, the keys a panel lists.
HELP = [("space", "cursor position and the spectrum there"), ("a a", "expand between"),
        ("z", "zoom x by 2"), (",  .", "slide left, right"), ("c", "the whole spectrum"),
        ("b", "base at zero, or not"), ("w", "window (gtools)"), ("r", "redraw"),
        ("e e", "equivalent width, flux, centre"), ("k k", "fit a line (g l v)"),
        ("d d", "deblend: g l v at the lines, q"), ("-  -", "subtract the fit"),
        ("h", "width at half: a b c, or at y: l r k"), ("m m", "mean, rms, s/n"),
        ("s", "smooth"), ("f", "functions: l n e d s a i p m + - * /"),
        ("j", "fix a pixel to y"), ("x x", "connect two points"),
        ("l  n", "to f-lambda, f-nu"), ("$", "pixels / world"), ("v", "velocity"),
        ("o", "overplot the next"), ("g", "another image"), ("( )  #", "line, aperture"),
        ("q", "quit, next image"), (":", "colon commands: units, log, # note, show")]

#: What each query parameter asks, as splot.par words it.
PROMPTS = {"line": "Image line/aperture to plot", "band": "Image band to plot",
           "next_image": "Next image to plot", "new_image": "Image to create",
           "overwrite": "Overwrite image?", "spec2": "Spectrum",
           "constant": "Constant to be applied", "wavelength": "Dispersion coordinate:",
           "linelist": "File", "wstart": "Starting wavelength",
           "wend": "Ending wavelength", "dw": "Wavelength per pixel",
           "boxsize": "Smoothing box size (odd number)"}


def printf(value, fmt: str, decpl: int | None = None, width: int | None = None) -> str:
    """IRAF's %w.dg / %w.df of a real, right justified."""
    text = "INDEF" if value is None else iraffmt.printf(float(value), fmt, decpl, width)
    return text.rjust(width) if width else text


def g(value, width: int, decpl: int) -> str:
    return printf(value, "g", decpl, width)


def fx(value, width: int, decpl: int) -> str:
    return printf(value, "f", decpl, width)


def _pow15(value) -> f32:
    """abs(delta) ** 1.5 of a real: f2c's pow_dd, rounded back."""
    return f32(abs(float(value)) ** 1.5)


def fp_equalr(a, b) -> bool:
    if a == b:
        return True
    if a == 0 or b == 0:
        return False
    na, ea = math.frexp(float(a))
    nb, eb = math.frexp(float(b))
    return ea == eb and abs(na - nb) <= 32 * float(np.finfo(np.float32).eps)


def nint(value) -> int:
    v = float(value)
    return math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)


# ------------------------------------------------------------- the spectrum

@dataclass
class Image:
    """An open image: its header, its pixels in real, its SMW."""
    name: str
    header: object
    data: np.ndarray
    smw: shdr.Smw


def open_image(name: str, dispaxis: int, nsum: tuple[int, int]) -> Image:
    """immap and smw_openim: the file's first image (or harwood's spectra as
    multispec)."""

    from . import imsum, spectrum

    path = Path(name)
    if not path.exists() and Path(f"{name}.fits").exists():
        path = Path(f"{name}.fits")
    if spectrum.count(path) > 0:
        smw, data = shdr.from_harwood(path)
        return Image(name, smw.header, data, smw)
    from . import images

    hdul, hdu = images.open_hdu(images.ImageSpec(path, None))
    try:
        if hdu is None or hdu.data is None:
            raise shdr.ShdrError(f"Can't get image {name}")
        # harwood's files keep DISPAXIS and the like in the primary header, which
        # the image extension inherits
        header = images.inherited(hdul, hdu)
        data = imsum.convert(imsum.ieee_in(np.asarray(hdu.data)), np.float32)
    finally:
        hdul.close()
    shape = tuple(reversed(data.shape))
    smw = shdr.open_smw(header, shape, dispaxis, nsum)
    return Image(name, header, data, smw)


def section_of(name: str) -> tuple[str, str]:
    """imgimage and imgsection."""
    m = re.match(r"^(.*?)(\[[^\]]*\])?$", name)
    return m.group(1), m.group(2) or ""


# ---------------------------------------------------------- the measurements

def fixx(sh: shdr.Shdr, x1, x2, y1, y2):
    """fixx: the two cursor reads in pixel order, x on the pixels' coordinates."""
    z1 = max(0.5, min(sh.sn + .499, sh.wl(float(f32(x1)))))
    z2 = max(0.5, min(sh.sn + .499, sh.wl(float(f32(x2)))))
    if z1 > z2:
        y1, y2 = y2, y1
        z1, z2 = z2, z1
    return f32(sh.lw(z1)), f32(sh.lw(z2)), f32(y1), f32(y2), nint(z1), nint(z2)


def avgsnr(sh, x1, y1, x2, y2):
    """avgsnr: the mean, rms and signal to noise between two reads; (text, line)."""
    x1, x2, y1, y2, i1, i2 = fixx(sh, x1, x2, y1, y2)
    if i1 == i2:
        return "Cannot determine SNR - move cursor", None
    y = sh.y
    n = i2 - i1 + 1
    avg = rms = snr = f32(0)
    for i in range(i1, i2 + 1):
        avg = f32(avg + y[i - 1])
    avg = f32(avg / f32(n))
    if n > 1:
        seg = y[i1 - 1:i2]
        span = f32(f32(seg.max()) - f32(seg.min()))
        if span > 0:
            for i in range(i1, i2 + 1):
                t = f32(f32(y[i - 1] - avg) / span)
                rms = f32(rms + f32(t * t))
            rms = f32(span * f32(math.sqrt(float(f32(rms / f32(n - 1))))))
            snr = f32(avg / rms)
    line = f"avg: {g(avg, 10, 4)}  rms: {g(rms, 10, 4)}   snr: {fx(snr, 8, 2)}"
    return line + "\n", line


def sumflux(sh, x, y, s, eqx1, eqx2, eqy1, eqy2):
    """sumflux: (sum, rsum, esum, ctr) each [value, error], esum None for INDEF, and
    the reads as fixx leaves them."""
    eqx1, eqx2, eqy1, eqy2, i1, i2 = fixx(sh, eqx1, eqx2, eqy1, eqy2)
    n = sh.sn
    X = lambda i: x[i - 1]          # noqa: E731
    Y = lambda i: y[i - 1]          # noqa: E731
    slope = f32(f32(eqy2 - eqy1) / f32(eqx2 - eqx1))
    sum1 = rsum1 = csum1 = sum21 = f32(0)
    esum1 = f32(0)
    scale = f32(0)
    for i in range(i1 + 1, i2):
        scale = max(scale, Y(i))
    if scale <= 0:
        scale = f32(1)
    for i in range(i1 + 1, i2):
        ramp = f32(eqy1 + f32(slope * f32(X(i) - eqx1)))
        sum1 = f32(sum1 + Y(i))
        rsum1 = f32(rsum1 + ramp)
        if esum1 is not None:
            if fp_equalr(0.0, f32(ramp / scale)):
                esum1 = None
            else:
                esum1 = f32(esum1 + f32(f32(1) - f32(Y(i) / ramp)))
    for i in range(i1 + 1, i2):
        ramp = f32(eqy1 + f32(slope * f32(X(i) - eqx1)))
        delta = f32(f32(Y(i) - ramp) / scale)
        p = _pow15(delta)
        csum1 = f32(csum1 + f32(p * X(i)))
        sum21 = f32(sum21 + p)

    def frac(xa, ia):
        if xa < X(ia):
            return f32(f32(X(ia) - xa) / f32(X(ia) - X(ia - 1))) if ia > 1 else \
                f32(f32(X(ia) - xa) / f32(X(ia + 1) - X(ia)))
        return f32(f32(X(ia) - xa) / f32(X(ia + 1) - X(ia))) if ia < n else \
            f32(f32(X(ia) - xa) / f32(X(ia) - X(ia - 1)))

    w1 = frac(eqx1, i1)
    w2 = f32(f32(1) - frac(eqx2, i2))
    sum1 = f32(f32(sum1 + f32(w1 * Y(i1))) + f32(w2 * Y(i2)))
    rsum1 = f32(f32(rsum1 + f32(w1 * eqy1)) + f32(w2 * eqy2))
    if esum1 is not None:
        if fp_equalr(0.0, f32(eqy1 / scale)) or fp_equalr(0.0, f32(eqy2 / scale)):
            esum1 = None
        else:
            esum1 = f32(f32(esum1 + f32(w1 * f32(f32(1) - f32(Y(i1) / eqy1))))
                        + f32(w2 * f32(f32(1) - f32(Y(i2) / eqy2))))
    delta = f32(f32(Y(i1) - eqy1) / scale)
    csum1 = f32(csum1 + f32(f32(w1 * _pow15(delta)) * eqx1))
    sum21 = f32(sum21 + f32(w1 * _pow15(delta)))
    delta = f32(f32(Y(i2) - eqy2) / scale)
    csum1 = f32(csum1 + f32(f32(w2 * _pow15(delta)) * eqx2))
    sum21 = f32(sum21 + f32(w2 * _pow15(delta)))
    ctr1 = f32(csum1 / sum21) if sum21 != 0 else f32(0)
    if i1 != i2:
        wpc = abs(f32(f32(X(i2) - X(i1)) / f32(i2 - i1)))
    elif i1 < n:
        wpc = abs(f32(X(i1 + 1) - X(i1)))
    else:
        wpc = abs(f32(X(i1 - 1) - X(i1)))
    sum1 = f32(sum1 * wpc)
    if esum1 is not None:
        esum1 = f32(esum1 * wpc)
    rsum1 = f32(rsum1 * wpc)
    sum2 = esum2 = ctr2 = None
    if s is not None:
        S = lambda i: s[i - 1]      # noqa: E731
        s2 = e2 = c2 = f32(0)
        for i in range(i1 + 1, i2):
            ramp = f32(eqy1 + f32(slope * f32(X(i) - eqx1)))
            s2 = f32(s2 + f32(S(i) * S(i)))
            if esum1 is not None:
                if fp_equalr(0.0, f32(ramp / scale)):
                    e2 = None
                elif e2 is not None:
                    q = f32(S(i) / ramp)
                    e2 = f32(e2 + f32(q * q))
        for i in range(i1 + 1, i2):
            ramp = f32(eqy1 + f32(slope * f32(X(i) - eqx1)))
            delta = f32(f32(Y(i) - ramp) / scale)
            q = f32(f32(X(i) - ctr1) * S(i))
            c2 = f32(c2 + f32(abs(delta) * f32(q * q)))
        a, b = f32(w1 * S(i1)), f32(w2 * S(i2))
        s2 = f32(f32(s2 + f32(a * a)) + f32(b * b))
        if esum1 is not None and e2 is not None:
            if fp_equalr(0.0, f32(eqy1 / scale)) or fp_equalr(0.0, f32(eqy2 / scale)):
                e2 = None
            else:
                a, b = f32(f32(w1 * S(i1)) / eqy1), f32(f32(w2 * S(i2)) / eqy2)
                e2 = f32(f32(e2 + f32(a * a)) + f32(b * b))
        delta = f32(f32(Y(i1) - eqy1) / scale)
        q = f32(f32(w1 * f32(eqx1 - ctr1)) * S(i1))
        c2 = f32(c2 + f32(abs(delta) * f32(q * q)))
        delta = f32(f32(Y(i2) - eqy2) / scale)
        q = f32(f32(w2 * f32(eqx2 - ctr1)) * S(i2))
        c2 = f32(c2 + f32(abs(delta) * f32(q * q)))
        ctr2 = f32(f32(f32(f32(1.5) / scale) * f32(math.sqrt(float(c2)))) / sum21) \
            if sum21 != 0 else f32(0)
        sum2 = f32(f32(math.sqrt(float(s2))) * wpc)
        if e2 is not None:
            esum2 = f32(math.sqrt(float(e2)))
            if esum1 is not None:
                esum2 = f32(esum2 * wpc)
    return ([sum1, sum2], [rsum1, None], [esum1, esum2], [ctr1, ctr2],
            (eqx1, eqx2, eqy1, eqy2))


def noise(y, sigma0, invgain):
    """eqwidth's sigmas: None without both noise terms."""
    if sigma0 is None or invgain is None or sigma0 < 0 or invgain < 0:
        return None, None, None
    s0, ig = f32(sigma0), f32(invgain)
    out = np.empty(y.size, f32)
    for i, v in enumerate(y):
        out[i] = f32(math.sqrt(float(f32(f32(s0 * s0) + f32(ig * v))))) if v > 0 else \
            f32(math.sqrt(float(f32(s0 * s0))))
    return out, s0, ig


def eqwidth(sh, x1, y1, x2, y2, sigma0, invgain):
    """eqwidth: the status line, the log lines and what is drawn."""
    if f32(x1) == f32(x2):
        return "Cannot get EQW - move cursor", [], []
    s, s0, _ = noise(sh.y, sigma0, invgain)
    sums, rsum, esum, ctr, (a, b, ya, yb) = sumflux(sh, sh.x, sh.y, s, x1, x2, y1, y2)
    flux_diff = f32(sums[0] - rsum[0])
    cont = f32(f32(0.5) * f32(ya + yb))
    text = (f"center = {g(ctr[0], 9, 7)}, eqw = {fx(esum[0], 9, 4)}, "
            f"continuum = {g(cont, 9, 7)} flux = {g(flux_diff, 9, 6)}\n")
    log = [f" {g(ctr[0], 9, 7)} {g(cont, 9, 7)} {g(flux_diff, 9, 6)} {g(esum[0], 9, 4)}"]
    if s0 is not None:
        log.append(f"  ({g(ctr[1], 7, 5)}) {'':9s} ({g(sums[1], 7, 4)}) "
                   f"({g(esum[1], 7, 2)})")
    i = max(1, min(sh.sn, nint(sh.wl(float(ctr[0])))))
    drawn = [("line", (a, ya, b, yb)), ("line", (ctr[0], cont, ctr[0], sh.y[i - 1]))]
    return text, log, drawn


def smooth(y: np.ndarray, boxsize: int) -> tuple[np.ndarray, str]:
    """smooth: a box average, the box made odd, shrinking at the ends."""
    said = ""
    if boxsize % 2 == 0:
        boxsize += 1
        said = f"WARNING: Using a box size of {boxsize}"
    n = y.size
    half = boxsize // 2
    out = np.empty(n, f32)
    for i in range(1, n + 1):
        if half < i < n - half:
            d = half
        elif i <= half:
            d = i // 2
        else:
            d = (n - i + 1) // 2
        total = f32(0)
        count = 0
        for j in range(i - d, i + d + 1):
            count += 1
            total = f32(total + y[j - 1])
        out[i - 1] = f32(total / f32(count))
    return out, said


def fun_do(key: str, y: np.ndarray, constant=None) -> np.ndarray:
    """fun_do's functions of one spectrum."""
    y = np.asarray(y, f32).copy()
    if key == "a":
        return np.abs(y).astype(f32)
    if key == "d":
        const = f32(math.log10(MAX_REAL))
        for i, v in enumerate(y):
            y[i] = f32(10.0 ** float(v)) if abs(v) < const else \
                (f32(MAX_REAL) if v >= const else f32(0))
        return y
    if key == "e":
        const = f32(math.log(MAX_REAL))
        for i, v in enumerate(y):
            y[i] = f32(math.exp(float(v))) if abs(v) < const else \
                (f32(MAX_REAL) if v >= const else f32(0))
        return y
    if key == "i":
        with np.errstate(divide="ignore"):
            return np.where(y != 0, f32(1) / y, f32(0)).astype(f32)
    if key in ("l", "n"):
        fn = math.log10 if key == "l" else math.log
        return np.array([f32(fn(float(v))) if v > 0 else f32(-0.5) for v in y], f32)
    if key == "m":
        return (y * f32(constant)).astype(f32)
    if key == "p":
        return (y + f32(constant)).astype(f32)
    if key == "s":
        return np.array([f32(math.sqrt(float(v))) if v >= 0 else f32(0) for v in y], f32)
    raise ValueError("Unknown function")


def conflam(sh) -> None:
    """conflam: the data times c / lambda^2, lambda in Angstroms."""
    ang = unitsmod.decode("angstroms")
    lam = unitsmod.convert(sh.x, sh.un, ang, real=True).astype(f32)
    sh.y = (sh.y * VLIGHT / (lam * lam).astype(f32)).astype(f32)


def confnu(sh) -> None:
    ang = unitsmod.decode("angstroms")
    lam = unitsmod.convert(sh.x, sh.un, ang, real=True).astype(f32)
    sh.y = (sh.y * (lam * lam).astype(f32) / VLIGHT).astype(f32)


def fudgept(sh, wx, wy):
    """fudgept: the nearest pixel set to the cursor's y; what is redrawn."""
    n = sh.sn
    i1 = max(1, min(n, nint(sh.wl(float(f32(wx))))))
    sh.y[i1 - 1] = f32(wy)
    if 1 < i1 < n:
        start, count = i1 - 1, 3
    elif i1 == 1:
        start, count = i1, 2
    else:
        start, count = n - 1, 2
    return sh.x[start - 1:start - 1 + count], sh.y[start - 1:start - 1 + count]


def fudgex(sh, x1, y1, x2, y2, xydraw: bool):
    """fudgex: the pixels between two reads on the straight line joining them (or
    the pixels' own ends, without xydraw)."""
    x1, x2, y1, y2, i1, i2 = fixx(sh, x1, x2, y1, y2)
    if not xydraw:
        y1, y2 = sh.y[i1 - 1], sh.y[i2 - 1]
    slope = f32(0) if fp_equalr(x1, x2) else f32(f32(y2 - y1) / f32(x2 - x1))
    for i in range(i1, i2 + 1):
        sh.y[i - 1] = f32(y1 + f32(f32(sh.x[i - 1] - x1) * slope))
    return sh.x[i1 - 1:i2], sh.y[i1 - 1:i2]


# ------------------------------------------------------------- profile fits

FITS = {"g": fitprofs.GAUSS, "l": fitprofs.LORENTZ, "v": fitprofs.VOIGT}


@dataclass
class Fitted:
    """The last k or d fit: the profiles and the continuum, what - subtracts."""
    xg: np.ndarray
    yg: np.ndarray
    sg: np.ndarray
    lg: np.ndarray
    pg: np.ndarray


def _scaled(sh, i1, npts, sigma0, invgain, deblend: bool):
    x = sh.x[i1 - 1:i1 - 1 + npts].astype(f32).copy()
    y = sh.y[i1 - 1:i1 - 1 + npts].astype(f32).copy()
    warn = ""
    if sigma0 is None or invgain is None or sigma0 < 0 or invgain < 0 or \
            (sigma0 == 0 and invgain == 0):
        sigma0 = invgain = None
    scale = f32(0)
    for v in y:
        # gfit asks for a gain that is not INDEF; sp_deblend only that it is not 0,
        # and INDEF is not: it warns with no noise given at all
        if v <= 0 and (invgain is not None and invgain != 0 or deblend):
            sigma0 = invgain = None
            warn += ("WARNING: Cannot compute errors with non-zero gain and negative "
                     "pixel values.\n")
        scale = max(scale, abs(v))
    if sigma0 is None:
        s = np.ones(npts, f32)
        sscale = f32(1)
    else:
        s0, ig = f32(sigma0), f32(invgain)
        s = np.array([f32(math.sqrt(float(f32(f32(s0 * s0) + f32(ig * v))))) for v in y],
                     f32)
        total = f32(0)
        for v in s:
            total = f32(total + v)
        sscale = f32(total / f32(npts))
    y = (y / scale).astype(f32)
    s = (s / sscale).astype(f32)
    return x, y, s, scale, sscale, sigma0, invgain, warn


def _results(i, kind, w, flux, eqw, gfwhm, lfwhm) -> str:
    if kind == fitprofs.GAUSS:
        return (f"\n{i}: center = {g(w, 8, 6)}, flux = {g(flux, 8, 4)}, "
                f"eqw = {g(eqw, 6, 4)}, gfwhm = {g(gfwhm, 6, 4)}")
    if kind == fitprofs.LORENTZ:
        return (f"\n{i}: center = {g(w, 8, 6)}, flux = {g(flux, 8, 4)}, "
                f"eqw = {g(eqw, 6, 4)}, lfwhm = {g(lfwhm, 6, 4)}")
    return (f"\n{i}: center = {g(w, 8, 6)}, eqw = {g(eqw, 6, 4)}, "
            f"gfwhm = {g(gfwhm, 6, 4)}, lfwhm = {g(lfwhm, 6, 4)}")


def _log_line(w, cont, flux, eqw, peak, gfwhm, lfwhm) -> str:
    return (f" {g(w, 9, 7)} {g(cont, 9, 7)} {g(flux, 9, 6)} {g(eqw, 9, 4)} "
            f"{g(peak, 9, 6)} {g(gfwhm, 9, 4)} {g(lfwhm, 9, 4)}")


def _eqw(flux, cont):
    return f32(-flux / cont) if cont > 0 else None


def _montecarlo(fit, x, y, s, npts, dw, xg, yg, sg, lg, pg, wyc, slope, scale, sscale,
                mc_n, gasdev, say):
    """gfit's and sp_deblend's errors: mc_n refits of the model with noise; the 68th
    percentile of each parameter's deviations, by profile."""
    ng = len(pg)
    mc_p = nint(mc_n * 10 / 100.0)
    mc_sig = nint(mc_n * 68 / 100.0)
    # Fortran's model + wyc + slope * w is (model + wyc) + slope * w
    ym = np.array([f32(f32(fitprofs.model(x[i], dw, fitprofs.NSUB, xg, yg, sg, lg, pg)
                           + wyc) + f32(slope * x[i])) for i in range(npts)], f32)
    state = [1]
    errs = {k: np.zeros((ng, mc_n), f32) for k in
            ("cont", "x", "y", "s", "l", "flux", "eqw")}
    for i in range(mc_n):
        if i > 0 and i % mc_p == 0:
            say(f"{100 * i // mc_n:2d} ")
        yy = np.array([f32(ym[j] + f32(f32(f32(sscale / scale) * s[j]) * gasdev(state)))
                       for j in range(npts)], f32)
        xg1, yg1, sg1, lg1 = xg.copy(), yg.copy(), sg.copy(), lg.copy()
        wyc1, slope1 = fitprofs.dorefit(fit, x, yy, s, dw, fitprofs.NSUB, wyc, slope,
                                        xg1, yg1, sg1, lg1, pg)
        for j in range(ng):
            cont = f32(wyc + f32(slope * xg[j]))
            cont1 = f32(wyc1 + f32(slope1 * xg[j]))
            flux = fitprofs.flux_of(pg[j], yg[j], sg[j], lg[j])
            flux1 = fitprofs.flux_of(pg[j], yg1[j], sg1[j], lg1[j])
            if cont > 0 and cont1 > 0:
                eqw, eqw1 = f32(-flux / cont), f32(-flux1 / cont1)
            else:
                eqw = eqw1 = f32(0)
            errs["cont"][j, i] = abs(f32(cont1 - cont))
            errs["x"][j, i] = abs(f32(xg1[j] - xg[j]))
            errs["y"][j, i] = abs(f32(yg1[j] - yg[j]))
            errs["s"][j, i] = abs(f32(sg1[j] - sg[j]))
            errs["l"][j, i] = abs(f32(lg1[j] - lg[j]))
            errs["flux"][j, i] = abs(f32(flux1 - flux))
            errs["eqw"][j, i] = abs(f32(eqw1 - eqw))
    for k in errs:
        errs[k].sort(axis=1)
    for k in ("cont", "y", "flux"):
        errs[k] = (errs[k] * scale).astype(f32)
    return {k: v[:, mc_sig] for k, v in errs.items()}


@dataclass
class Session:
    """splot's cursor loop on one spectrum at a time. `reads` gives the cursor
    reads (x, y, wcs, key, command); `answers` the query parameters' values, in
    the order IRAF asks them. What it prints accumulates in `out`, the log lines
    in `log` (the save_file) and `measured` (what :show shows)."""

    params: dict
    answers: Iterator[str] = field(default_factory=lambda: iter(()))
    out: str = ""
    err: str = ""
    log: list = field(default_factory=list)
    measured: list = field(default_factory=list)
    logging: bool = True
    key1: str = ""
    fitted: Fitted | None = None
    drawn: list = field(default_factory=list)
    asking: str = ""                #: the query parameter being asked, for a window

    def __post_init__(self) -> None:
        self.gasdev = fitprofs.Gasdev()
        self.options = {o: o in self.params.get("options", "auto wreset").split()
                        for o in OPTIONS}

    def say(self, text: str) -> None:
        self.out += text

    def answer(self, name: str) -> str:
        self.asking = name
        try:
            return next(self.answers)
        except StopIteration:
            raise EOFError(name) from None

    def number(self, name: str):
        """A query parameter's number. One given with the task (line=21) is not asked
        again for the rest of it, as the CL does not ask a parameter set on the
        command line."""
        if self.params.get(name) is not None:
            return float(self.params[name])
        text = self.answer(name).strip()
        if text.upper().startswith("INDEF") or text == "":
            return None
        return float(text)

    # ------------------------------------------------------------- the log

    def ans_hdr(self, sh, newimage: bool, key: str) -> None:
        if newimage:
            stamp = iraffmt.cnvdate()
            line = f"{stamp} [{sh.image}{sh.imsec}]: {sh.title}"
            self._write(["", line])
        if key != self.key1:
            if key != "m":
                self._write(["".join(f"{w:>10s}" for w in
                                     ("center", "cont", "flux", "eqw", "core",
                                      "gfwhm", "lfwhm"))])
            self.key1 = key

    def _write(self, lines: list[str]) -> None:
        if self.logging and self.params.get("save_file", "splot.log"):
            self.log.extend(lines)
        self.measured.extend(lines)

    # -------------------------------------------------------------- k and d

    def gfit(self, sh, x1, y1, read2, key2: str) -> None:
        """gfit: one profile between two continuum points."""
        x2, y2 = read2
        x1, x2, y1, y2, i1, j = fixx(sh, x1, x2, y1, y2)
        npts = j - i1 + 1
        if npts < 3:
            self.err += "At least 3 points are required\n"
            return
        mc_n = int(self.params.get("nerrsample", 0))
        x, y, s, scale, sscale, sigma0, _, warn = _scaled(
            sh, i1, npts, self.params.get("sigma0"), self.params.get("invgain"), False)
        self.err += warn
        slope = f32(f32(f32(y2 - y1) / f32(x2 - x1)) / scale)
        wyc = f32(f32(y1 / scale) - f32(slope * x1))
        wx = f32(0)
        jbest = 0
        xg0 = yg0 = f32(0)
        for i in range(npts):
            w = x[i]
            wy = f32(f32(y[i] - wyc) - f32(slope * w))
            if abs(wy) > wx:
                wx = abs(wy)
                jbest = i
                xg0, yg0 = w, wy
        if 0 < jbest < npts - 1:
            def width(k):
                w = x[k]
                wy = min(f32(0.99), max(f32(0.01), f32(abs(f32(f32(y[k] - wyc)
                                                              - f32(slope * w))) / wx)))
                d = f32(w - xg0)
                return f32(f32(2.355) * f32(math.sqrt(float(
                    f32(f32(f32(-0.5) * f32(d * d)) / f32(math.log(float(wy))))))))
            gfwhm = width(jbest - 1)
            gfwhm = f32(f32(gfwhm + width(jbest + 1)) / f32(2))
        else:
            gfwhm = f32(f32(0.3) * abs(f32(x[npts - 1] - x[0])))
        kind = FITS.get(key2, fitprofs.GAUSS)
        sg = {fitprofs.GAUSS: gfwhm, fitprofs.LORENTZ: f32(0),
              fitprofs.VOIGT: f32(f32(0.5) * gfwhm)}[kind]
        lg = {fitprofs.GAUSS: f32(0), fitprofs.LORENTZ: gfwhm,
              fitprofs.VOIGT: f32(f32(0.5) * gfwhm)}[kind]
        xg = np.array([xg0], f32)
        yg = np.array([yg0], f32)
        sgv = np.array([sg], f32)
        lgv = np.array([lg], f32)
        pg = np.array([kind], np.int64)
        n = sh.sn
        dw = f32(f32(sh.x[n - 1] - sh.x[0]) / f32(n - 1))
        fit0 = [1, 2, 2, 2, 2]          # background held, the rest single
        try:
            wyc, slope = fitprofs.dofit(fit0, x, y, s, dw, fitprofs.NSUB, wyc, slope,
                                        xg, yg, sgv, lgv, pg)
        except (ValueError, ArithmeticError):
            self.fitted = None
            return
        errs = None
        if mc_n > 9 and sigma0 is not None:
            errs = _montecarlo(fit0, x, y, s, npts, dw, xg, yg, sgv, lgv, pg, wyc, slope,
                               scale, sscale, mc_n, self.gasdev, self.say)
        yg = (yg * scale).astype(f32)
        wyc = f32(f32(wyc + f32(slope * x1)) * scale)
        slope = f32(slope * scale)
        z = np.array([f32(f32(fitprofs.model(sh.x[i1 - 1 + i], dw, fitprofs.NSUB, xg,
                                             yg, sgv, lgv, pg) + wyc)
                          + f32(slope * f32(sh.x[i1 - 1 + i] - x1)))
                      for i in range(npts)], f32)
        self.drawn.append(("curve", (sh.x[i1 - 1:i1 - 1 + npts].copy(), z), 2))
        self.drawn.append(("line", (x1, wyc, x2, f32(wyc + f32(slope * f32(x2 - x1)))),
                           3))
        self.fitted = Fitted(xg, yg, sgv, lgv, pg)
        self._report(xg, yg, sgv, lgv, pg, wyc, slope, x1, errs, gfit=True)

    def _report(self, xg, yg, sg, lg, pg, wyc, slope, x1, errs, gfit: bool) -> None:
        lines = []
        for i in range(len(pg)):
            w = xg[i]
            cont = f32(wyc + f32(slope * f32(w - x1)))
            peak, gfwhm, lfwhm = yg[i], sg[i], lg[i]
            flux = fitprofs.flux_of(pg[i], peak, gfwhm, lfwhm)
            eqw = _eqw(flux, cont)
            if gfit:
                self.say(_results(i + 1, pg[i], w, flux, eqw, gfwhm, lfwhm))
            lines.append(_log_line(w, cont, flux, eqw, peak, gfwhm, lfwhm))
            if errs is not None:
                e = {k: v[i] for k, v in errs.items()}
                if gfit:            # gfit's (%7w) is seven blanks, and no value
                    lines.append(f"  ({g(e['x'], 7, 5)}) ({'':7s}) ({g(e['flux'], 7, 4)}) "
                                 f"({g(e['eqw'], 7, 4)}) ({g(e['y'], 7, 4)}) "
                                 f"({g(e['s'], 7, 4)}) ({g(e['l'], 7, 4)})")
                else:
                    lines.append(f"  ({g(e['x'], 7, 5)}) ({g(e['cont'], 7, 5)}) "
                                 f"({g(e['flux'], 7, 4)}) ({g(e['eqw'], 7, 4)}) "
                                 f"({g(e['y'], 7, 4)}) ({g(e['s'], 7, 4)}) "
                                 f"({g(e['l'], 7, 4)})")
        self._write(lines)

    # ------------------------------------------------------------- reads

    def read(self):
        """clgcur: the next cursor read, None at the end."""
        return next(self.reads, None)

    def deblend(self, sh, x1, y1) -> None:
        """sp_deblend."""
        self.say("d again:")
        r = self.read()
        if r is None:
            return
        x2, y2 = r[0], r[1]
        x1, x2, y1, y2, i1, j = fixx(sh, x1, x2, y1, y2)
        npts = j - i1 + 1
        if npts < 3:
            self.err += "At least 3 points are required\n"
            return
        mc_n = int(self.params.get("nerrsample", 0))
        x, y, s, scale, sscale, sigma0, _, warn = _scaled(
            sh, i1, npts, self.params.get("sigma0"), self.params.get("invgain"), True)
        self.err += warn
        prompt = "Lines ('f'ile, 'g'aussian, 'l'orentzian, 'v'oigt, 't'ype, 'q'uit:"
        waves, peaks, gws, lws, types = [], [], [], [], []
        fitg = fitl = False
        self.say(prompt)

        def add(wx, wy, peak, gw, lw, kind):
            nonlocal fitg, fitl
            if any(f32(wx) == w for w in waves):
                return
            waves.append(f32(wx))
            peaks.append(peak)
            gws.append(gw)
            lws.append(lw)
            types.append(kind)
            fitg = fitg or kind in (fitprofs.GAUSS, fitprofs.VOIGT)
            fitl = fitl or kind in (fitprofs.LORENTZ, fitprofs.VOIGT)
            self.drawn.append(("vline", (f32(wx), f32(wy))))

        while True:
            r = self.read()
            if r is None:
                break
            wx, wy, _, key, cmd = r
            if key == "f":
                name = self.answer("linelist")
                self.say(prompt)
                for kind, values in _linelist(name, sh.x):
                    add(values[0], wy, *values[1:], kind)
                continue
            if key in ("g", "l", "v"):
                add(wx, wy, None, None, None, FITS[key])
            elif key == "t":
                wx = self.number("wavelength")
                self.say(prompt)
                add(wx, wy, None, None, None, fitprofs.GAUSS)
            elif key == "q":
                self.say("\n")
                break
            else:
                self.say(prompt + "\007")
        fitit = False
        result = None
        if waves:
            nlines = len(waves)
            fit = [1, 1, 3, 1, 1]
            ifit = 0
            while True:
                answer = self._choose("Fit positions (fixed, single, all, quit) ",
                                      "fsa")
                if answer in (None, "q"):
                    break
                fit[1] = "fsa".index(answer) + 1
                if fitg:
                    answer = self._choose(
                        "Fit Gaussian widths (fixed, single, all, quit) ", "fsa")
                    if answer in (None, "q"):
                        break
                    fit[3] = "fsa".index(answer) + 1
                if fitl:
                    answer = self._choose(
                        "Fit Lorentzian widths (fixed, single, all, quit) ", "fsa")
                    if answer in (None, "q"):
                        break
                    fit[4] = "fsa".index(answer) + 1
                answer = self._choose("Fit background (no, yes, quit) ", "ny")
                if answer in (None, "q"):
                    break
                fit[0] = "ny".index(answer) + 1
                self.say("Fitting...")
                if ifit == 0:
                    slope = f32(f32(f32(y2 - y1) / f32(x2 - x1)) / scale)
                    wyc = f32(f32(y1 / scale) - f32(slope * x1))
                    eqw = f32(abs(f32(x[npts - 1] - x[0])) / f32(nlines))
                    xg = np.array(waves, f32)
                    yg = np.zeros(nlines, f32)
                    sg = np.zeros(nlines, f32)
                    lg = np.zeros(nlines, f32)
                    pg = np.array(types, np.int64)
                    for i in range(nlines):
                        w = waves[i]
                        jj = max(1, min(sh.sn, nint(sh.wl(float(w)))))
                        yg[i] = f32(f32(f32(sh.y[jj - 1] / scale) - wyc)
                                    - f32(slope * w)) if peaks[i] is None else \
                            f32(f32(peaks[i]) / scale)
                        if types[i] == fitprofs.GAUSS:
                            sg[i] = f32(f32(0.3) * eqw) if gws[i] is None else gws[i]
                        elif types[i] == fitprofs.LORENTZ:
                            lg[i] = f32(f32(0.3) * eqw) if lws[i] is None else lws[i]
                        else:
                            sg[i] = f32(f32(0.1) * eqw) if gws[i] is None else gws[i]
                            lg[i] = f32(f32(0.1) * eqw) if lws[i] is None else lws[i]
                else:
                    yg = (yg / scale).astype(f32)
                    slope = f32(slope / scale)
                    wyc = f32(f32(wyc / scale) - f32(slope * x1))
                n = sh.sn
                dw = f32(f32(sh.x[n - 1] - sh.x[0]) / f32(n - 1))
                try:
                    wyc, slope = fitprofs.dofit(fit, x, y, s, dw, fitprofs.NSUB, wyc,
                                                slope, xg, yg, sg, lg, pg)
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                    self.err += f"Warning: {error}\n"
                    continue
                ifit += 1
                errs = None
                if mc_n > 9 and sigma0 is not None:
                    errs = _montecarlo(fit, x, y, s, npts, dw, xg, yg, sg, lg, pg, wyc,
                                       slope, scale, sscale, mc_n, self.gasdev, self.say)
                chisq = fitprofs.LAST_CHISQ     # the last Monte-Carlo refit's, if any
                yg = (yg * scale).astype(f32)
                wyc = f32(f32(wyc + f32(slope * x1)) * scale)
                slope = f32(slope * scale)
                fitit = True
                result = (xg, yg, sg, lg, pg, wyc, slope, errs)
                answer = self._choose("Overplot (total, components, both, none) ", None)
                if answer is None:
                    break
                rms = f32(0)
                z = np.zeros(npts, f32)
                for i in range(npts):
                    w = x[i]
                    z[i] = f32(f32(fitprofs.model(w, dw, fitprofs.NSUB, xg, yg, sg, lg, pg)
                                   + wyc) + f32(slope * f32(w - x1)))
                    d = f32(f32(z[i] / scale) - y[i])
                    rms = f32(rms + f32(d * d))
                if answer in ("t", "b"):
                    self.drawn.append(("curve", (x.copy(), z.copy()), 2))
                if answer in ("c", "b"):
                    for jj in range(len(pg)):
                        zc = np.array([f32(f32(fitprofs.model(x[i], dw, fitprofs.NSUB,
                                                              xg[jj:jj + 1],
                                                              yg[jj:jj + 1],
                                                              sg[jj:jj + 1],
                                                              lg[jj:jj + 1],
                                                              pg[jj:jj + 1]) + wyc)
                                           + f32(slope * f32(x[i] - x1)))
                                       for i in range(npts)], f32)
                        self.drawn.append(("curve", (x.copy(), zc), 3))
                if answer != "n":
                    self.drawn.append(("line", (x1, wyc, x2,
                                                f32(wyc + f32(slope * f32(x2 - x1)))), 4))
                i = 1
                key = ""
                while True:
                    if key == "-":
                        i = nlines if i - 1 < 1 else i - 1
                    elif key == "+":
                        i = 1 if i + 1 > nlines else i + 1
                    elif key == "q":
                        self.say("\n")
                        break
                    if key == "r":
                        value = f32(scale * f32(math.sqrt(float(f32(chisq /
                                                                    f32(npts))))))
                        self.say(f"\nrms = {g(value, 8, 4)}")
                    else:
                        w = xg[i - 1]
                        cont = f32(wyc + f32(slope * f32(w - x1)))
                        flux = fitprofs.flux_of(pg[i - 1], yg[i - 1], sg[i - 1], lg[i - 1])
                        self.say(_results(i, pg[i - 1], w, flux, _eqw(flux, cont),
                                          sg[i - 1], lg[i - 1]))
                    self.say("  (+,-,r,q):")
                    r = self.read()
                    if r is None:
                        break
                    key = r[3]
                del rms
        self.say("Deblending done\n")
        if fitit:
            xg, yg, sg, lg, pg, wyc, slope, errs = result
            self.fitted = Fitted(xg, yg, sg, lg, pg)
            self._report(xg, yg, sg, lg, pg, wyc, slope, x1, errs, gfit=False)
        else:
            self.fitted = None

    def _choose(self, prompt: str, keys: str | None):
        """A prompt repeated until one of keys (or q) is typed; None at EOF."""
        while True:
            self.say(prompt)
            r = self.read()
            if r is None:
                return None
            if keys is None or r[3] == "q" or r[3] in keys:
                return r[3]


def _linelist(name: str, x: np.ndarray):
    """sp_deblend's 'f': position [peak [type [gfwhm [lfwhm]]]] a line, those within
    the spectrum."""
    out = []
    lo, hi = min(x[0], x[-1]), max(x[0], x[-1])
    for line in Path(name).read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        values = []
        for w in words[:2]:
            try:
                values.append(float(f32(float(w))))
            except ValueError:
                break
        if not values:
            continue
        wx = f32(values[0])
        if wx < lo or wx > hi:
            continue
        peak = f32(values[1]) if len(values) > 1 else None
        kind_word = words[2] if len(words) > 2 and len(values) > 1 else ""
        kinds = [i for i, n in enumerate(fitprofs.PTYPES, 1) if n.startswith(kind_word)] \
            if kind_word else []
        kind = kinds[0] if len(kinds) == 1 else fitprofs.GAUSS
        rest = []
        for w in words[3:5]:
            try:
                rest.append(f32(float(w)))
            except ValueError:
                break
        nscan = len(values) + (1 if kind_word else 0) + len(rest)
        gw = lw = None
        if nscan <= 2:
            kind = fitprofs.GAUSS
        elif nscan == 4:
            if kind == fitprofs.LORENTZ:
                lw = rest[0]
            else:
                gw = rest[0]
        elif nscan >= 5:
            gw, lw = rest[0], rest[1]
        out.append((kind, (wx, peak, gw, lw)))
    return out


# --------------------------------------------------------------- the loop

def eqwidth_cp(sh, center, cont, ylevel, key: str):
    """eqwidth_cp: a gaussian from the line's core and its width at a level --
    half way to the core (a, b, c) or the level given (l, r, k); one or both
    sides. (status line, error, log line, drawn, the profile) in double, as IRAF
    computes it."""
    if cont <= 0:
        return "", "Continuum cannot be less than zero.\n", None, [], None
    edge = 1 if key in "al" else (2 if key in "br" else 3)
    y = sh.y
    n = sh.sn
    icore = max(1, min(n, nint(sh.wl(float(f32(center))))))
    i1, i2 = max(1, icore - 3), min(n, icore + 3)
    if edge == 1:
        i2 = max(i2 - 2, icore + 1)
    if edge == 2:
        i1 = min(i1 + 2, icore - 1)
    c0 = icore
    for i in range(i1, i2 + 1):
        if abs(f32(y[i - 1] - f32(cont))) > abs(f32(y[c0 - 1] - f32(cont))):
            c0 = i
    icore = c0
    xp = [float(icore - 1), float(icore), float(icore + 1)]
    yp = [float(y[icore - 2]), float(y[icore - 1]), float(y[icore])]
    x12, x13, x23 = xp[0] - xp[1], xp[0] - xp[2], xp[1] - xp[2]
    x213, x223 = x13 * x13, x23 * x23
    y13, y23 = yp[0] - yp[2], yp[1] - yp[2]
    c3 = (y13 - y23 * x13 / x23) / (x213 - x223 * x13 / x23)
    c2 = (y23 - c3 * x223) / x23
    c1 = yp[2]
    c1 = c1 - xp[2] * (c2 - c3 * xp[2])
    c2 = c2 - 2 * c3 * xp[2]
    del x12
    xcore = -c2 / 2.0 / c3
    ycore = c1 + c2 * xcore + c3 * xcore ** 2
    cont = float(f32(cont))
    yl = (cont + ycore) / 2.0 if ylevel is None else float(f32(ylevel))
    rcore, rinter = abs(ycore - cont), abs(yl - cont)
    if rcore <= rinter:
        return "", "Y cursor must be between the continuum and the line core\n", None, \
            [], None
    xleft = xright = 0.0
    if edge in (1, 3):
        i = icore
        while i >= 1 and not abs(f32(y[i - 1] - f32(cont))) < rinter:
            i -= 1
        if i < 1:
            return "", "Can't find left edge of line\n", None, [], None
        xleft = float(i) + (yl - float(y[i - 1])) / float(f32(y[i] - y[i - 1]))
        if edge == 1:
            xright = xcore + (xcore - xleft)
    if edge in (2, 3):
        i = icore
        while i <= n and not abs(f32(y[i - 1] - f32(cont))) < rinter:
            i += 1
        if i > n:
            return "", "Can't find right edge of line\n", None, [], None
        xright = float(i) - (yl - float(y[i - 1])) / float(f32(y[i - 2] - y[i - 1]))
        if edge == 2:
            xleft = xcore - (xright - xcore)
    w, w1, w2 = sh.lw(xcore), sh.lw(xleft), sh.lw(xright)
    # SPP's 1.665109 and 1.064467 are reals, even in a double expression
    k1, k2 = float(f32(1.665109)), float(f32(1.064467))
    gfwhm = k1 * abs(w2 - w1) / 2.0 / math.sqrt(math.log(rcore / rinter))
    rcore = ycore - cont
    flux = k2 * rcore * gfwhm
    eqw = -flux / cont
    text = f"center = {_gd(w, 9, 7)}, eqw = {_gd(eqw, 9, 4)}, gfwhm = {_gd(gfwhm, 9, 4)}\n"
    log = (f" {_gd(w, 9, 7)} {g(cont, 9, 7)} {_gd(flux, 9, 6)} {_gd(eqw, 9, 4)} "
           f"{_gd(ycore - cont, 9, 6)} {_gd(gfwhm, 9, 4)} {_gd(0.0, 9, 4)}")
    drawn = [("line", (f32(w), f32(cont), f32(w), f32(ycore))),
             ("line", (f32(w1), f32(yl), f32(w2), f32(yl)))]
    a = w - 2 * gfwhm
    pts = [(f32(a), f32(cont + rcore * math.exp(-(k1 * (a - w) / gfwhm) ** 2)))]
    while a <= w + 2 * gfwhm:
        pts.append((f32(a), f32(cont + rcore * math.exp(-(k1 * (a - w) / gfwhm) ** 2))))
        a = a + float(f32(0.05)) * gfwhm
    drawn.append(("polyline", pts, 2))
    profile = Fitted(np.array([w], f32), np.array([rcore], f32), np.array([gfwhm], f32),
                     np.array([0.0], f32), np.array([1], np.int64))
    return text, "", log, drawn, profile


def _gd(value, width, decpl):
    """%w.dg of a double."""
    return iraffmt.printf(float(value), "g", decpl, width, double=True).rjust(width)


def subblend(sh, x1, y1, x2, y2, fitted: Fitted):
    """subblend: the fitted profiles taken out of the pixels between two reads."""
    x1, x2, y1, y2, i1, j = fixx(sh, x1, x2, y1, y2)
    n = sh.sn
    dw = f32(f32(sh.x[n - 1] - sh.x[0]) / f32(n - 1))
    for i in range(i1, j + 1):
        sh.y[i - 1] = f32(sh.y[i - 1] - fitprofs.model(sh.x[i - 1], dw, fitprofs.NSUB,
                                                       fitted.xg, fitted.yg, fitted.sg,
                                                       fitted.lg, fitted.pg))
    return sh.x[i1 - 1:j], sh.y[i1 - 1:j]


class Splot(Session):
    """splot.x's loop over images and cursor reads."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sh = None
        self.image = None
        self.window = (0.0, 1.0, 0.0, 1.0)
        self.graphs: list = []
        from .gtools import GTools

        self.gt = GTools()
        self.gt.type = 3 if self.options["histogram"] else 2
        self.gt.xflip = self.options["flip"]
        self.gt.sysid = not self.options["nosysid"]
        self._reset_limits()

    def _reset_limits(self) -> None:
        p = self.params
        for name in ("xmin", "xmax", "ymin", "ymax"):
            self.gt.setr(name, p.get(name))
        if self.options["zero"]:
            self.gt.setr("ymin", 0.0)

    # ---------------------------------------------------------- getimage

    def getimage(self, name: str, nline: int, nband: int, nap, wave_scl: bool,
                 units: str):
        """getimage: the spectrum asked for (a section names it), the line or the
        band asked for when there are several."""
        root, section = section_of(name)
        if self.image is None or self.image.name != root:
            self.image = open_image(root, int(self.params.get("dispaxis", 1)),
                                    tuple(self.params.get("nsum", (1, 1))))
            self.da = 0
            if section:
                nline, nband, self.da = self._section(section, nline, nband)
        smw = self.image.smw
        if smw.format == shdr.ND:
            nap = None
            if smw.llen[1] > 1:
                if nline == 0:
                    nline = max(1, min(smw.llen[1], int(self.number("line"))))
            else:
                nline = 0
            if smw.llen[2] > 1:
                if nband == 0:
                    nband = max(1, min(smw.llen[2], int(self.number("band"))))
            else:
                nband = 0
        else:
            if smw.nspec > 1:
                if nline == 0:
                    nline = int(self.number("line"))
                    nap = nline
            else:
                nline, nap = 0, None
            if smw.nbands > 1:
                if nband == 0:
                    nband = max(1, min(smw.nbands, int(self.number("band"))))
            else:
                nband = 0
        sh = shdr.shdr_open(smw, self.image.data, Path(root).name if False else root,
                            nline, nband, nap)
        nap = sh.ap
        nline = sh.lindex[0]
        if not wave_scl:
            sh.set_system("physical")
        else:
            with contextlib.suppress(shdr.ShdrError):
                sh.set_units(units)
        if self.da > 0:
            a, b = self.gt.getr("xmin"), self.gt.getr("xmax")
            if a is None and b is None:
                s1, s2 = self.sec[self.da - 1][:2]
                if not wave_scl:
                    self.gt.setr("xmin", s1)
                    self.gt.setr("xmax", s2)
                else:
                    self.gt.setr("xmin", f32(sh.lw(float(s1))))
                    self.gt.setr("xmax", f32(sh.lw(float(s2))))
        self.mktitle(sh)
        return sh, nline, nband, nap

    def _section(self, section: str, nline: int, nband: int):
        """getimage's reading of an image section: which line or band it names."""
        smw = self.image.smw
        n = smw.ldim
        shape = smw.shape
        sec = [[1, shape[i], 1] for i in range(n)]
        parts = section.strip("[]").split(",")
        for i, part in enumerate(parts[:n]):
            part = part.strip()
            if part in ("*", ""):
                continue
            if ":" in part:
                a, b = part.split(":")[:2]
                sec[i][0], sec[i][1] = int(a), int(b)
            else:
                sec[i][0] = sec[i][1] = int(part)
        self.sec = sec
        da = 0
        if smw.format == shdr.ND:
            if n == 1:
                da = 1
            elif n == 2:
                if sec[0][1] == sec[0][0]:
                    nline, da = sec[0][0], 2
                elif sec[1][1] == sec[1][0]:
                    nline, da = sec[1][0], 1
            if da > 0:
                header = self.image.header.copy()
                header["DISPAXIS"] = da
                self.image.smw = shdr.open_smw(header, shape[:n],
                                               int(self.params.get("dispaxis", 1)),
                                               tuple(self.params.get("nsum", (1, 1))))
        else:
            da = 1
            if n > 1 and sec[1][1] == sec[1][0]:
                nline = sec[1][0]
            if n > 2 and sec[2][1] == sec[2][0]:
                nband = sec[2][0]
        return nline, nband, da

    def mktitle(self, sh) -> None:
        """mktitle."""
        it = "INDEF" if sh.it is None else iraffmt.printf(float(sh.it), "s")
        del it
        self.gt.title = (f"[{sh.image}{sh.imsec}]: {sh.title} "
                         f"{_two(sh.it)} ap:{sh.ap} beam:{sh.beam}")
        if sh.un.label:
            self.gt.xlabel, self.gt.xunits = sh.un.label, sh.un.units
        else:
            self.gt.xlabel, self.gt.xunits = sh.label, sh.units

    # ---------------------------------------------------------- the graph

    def replot(self, clear: bool) -> None:
        """replot: gt_ascale over what the limits let in, gt_swind; the graph kept
        as (window, curve) for the panel and the tests."""
        sh = self.sh
        x, y = sh.x, sh.y
        if clear:
            self.drawn = []             # gclear takes what was drawn over the graph
            gt = self.gt
            limits = [gt.xmin, gt.xmax, gt.ymin, gt.ymax]
            lo_x = min(v for v in limits[:2] if v is not None) if None not in limits[:2] \
                else None
            del lo_x
            xmin, xmax = gt.xmin, gt.xmax
            ymin, ymax = gt.ymin, gt.ymax
            a = -MAX_REAL if xmin is None else min(xmin, xmax if xmax is not None else xmin)
            b = MAX_REAL if xmax is None else max(xmax, xmin if xmin is not None else xmax)
            inside = (x >= a) & (x <= b)
            ys = y[inside] if inside.any() else y
            c = -MAX_REAL if ymin is None else ymin
            d = MAX_REAL if ymax is None else ymax
            ok = (ys >= min(c, d)) & (ys <= max(c, d))
            ys = ys[ok] if ok.any() else ys
            xs = x[inside] if inside.any() else x
            x_span = (float(xs.min()), float(xs.max()))
            y_span = (float(ys.min()), float(ys.max()))
            self.window = gt.swind(x_span, y_span)
            label = gt.xlabel + (f" ({gt.xunits})" if gt.xunits else "")
            self.graphs.append({"window": self.window, "title": gt.title,
                                "xlabel": label, "curves": [],
                                "histogram": gt.type == 3})
        if self.graphs:
            self.graphs[-1]["curves"].append((x.copy(), y.copy()))

    def auto_exp(self, key: str, wx1: float) -> None:
        """auto_exp: a, z, ',' and '.' on the window as drawn."""
        x1, x2, y1, y2 = self.window
        dx = f32(f32(x2) - f32(x1))
        if key == "a":
            self.say("again:\n")
            r = self.read()
            wx2 = r[0] if r is not None else 0.0
            x1, x2 = f32(wx1), f32(wx2)
        elif key == ",":
            x1, x2 = f32(x1 - f32(0.85) * dx), f32(x2 - f32(0.85) * dx)
        elif key == ".":
            x1, x2 = f32(x1 + f32(0.85) * dx), f32(x2 + f32(0.85) * dx)
        else:
            x1, x2 = f32(x1 + f32(0.25) * dx), f32(x2 - f32(0.25) * dx)
        if x1 == x2:
            x1 = x2 = ymin = ymax = None
        else:
            lo, hi = min(x1, x2), max(x1, x2)
            sel = (self.sh.x >= lo) & (self.sh.x <= hi)
            if not sel.any():
                ymin, ymax = y1, y2
            else:
                ymin, ymax = float(self.sh.y[sel].min()), float(self.sh.y[sel].max())
                if y1 > y2:
                    ymin, ymax = ymax, ymin
        self.gt.setr("xmin", x1)
        self.gt.setr("xmax", x2)
        self.gt.setr("ymin", ymin)
        self.gt.setr("ymax", ymax)
        self.replot(True)

    # ------------------------------------------------------------- run

    def run(self, images: list[str], reads) -> None:
        """splot: each image in turn, the cursor loop until q or the reads end."""
        self.reads = iter(reads)
        units = self.params.get("units", "") or "display"
        units = units.replace(" ", "") and units
        self.units, self.units1, self.units2 = units, units, units
        self.nline = self.nband = 0
        self.nap = 0
        for name in images:
            wave_scl = True
            try:
                self.sh, self.nline, self.nband, self.nap = self.getimage(
                    name, self.nline, self.nband, self.nap, wave_scl, self.units)
            except (OSError, shdr.ShdrError) as error:
                self.err += f"Can't get image {name}\n"
                del error
                continue
            self.name = name
            self.wave_scl = wave_scl
            newimage = True
            overplot = self.options["overplot"]
            key, wx, wy, cmd = "r", 0.0, 0.0, ""
            ended = False
            while True:
                newgraph, overplot, newimage, quit_ = self.key(key, wx, wy, cmd, newimage,
                                                              overplot)
                if quit_:
                    break
                if newgraph:
                    if overplot:
                        self.say(f"Overplotting: {self.name}")
                        if self.nline > 0:
                            self.say(f"({self.nline},{self.nband})" if self.nband > 0
                                     else f"({self.nline})")
                        self.replot(False)
                    else:
                        self.replot(True)
                    overplot = self.options["overplot"]
                r = self.read()
                if r is None:
                    ended = True
                    break
                wx, wy, _, key, cmd = r
            if ended:
                break

    def key(self, key, wx, wy, cmd, newimage, overplot):
        """One read of splot's loop: (newgraph, overplot, newimage, quit)."""
        sh = self.sh
        auto = self.options["auto"]
        newgraph = False
        if key == ":":
            if cmd.startswith("/"):
                said, redraw = self.gt.colon(cmd, self.window)
                self.say(said)
                newgraph = bool(redraw)
            else:
                newgraph = self.colon(cmd, wx, wy) or False
                overplot = self.options["overplot"]
        elif key in "az,.":
            self.auto_exp(key, wx)
        elif key == "b":
            if not self.options["zero"]:
                self.gt.setr("ymin", 0.0)
                self.options["zero"] = True
            else:
                self.gt.setr("ymin", None)
                self.options["zero"] = False
            newgraph, overplot = auto, False
        elif key == "c":
            self.gt.reset_window()
            if self.options["zero"]:
                self.gt.setr("ymin", 0.0)
            newgraph, overplot = True, False
        elif key == "d":
            self.ans_hdr(sh, newimage, key)
            self.deblend(sh, wx, wy)
            newimage = False
        elif key == "k":
            self.ans_hdr(sh, newimage, key)
            self.say("k again:")
            r = self.read()
            if r is not None:
                self.gfit(sh, wx, wy, (r[0], r[1]), r[3])
            newimage = False
        elif key == "e":
            self.ans_hdr(sh, newimage, key)
            self.say("e again:")
            r = self.read()
            x2, y2 = (r[0], r[1]) if r is not None else (wx, wy)
            text, log, drawn = eqwidth(sh, wx, wy, x2, y2, self.params.get("sigma0"),
                                       self.params.get("invgain"))
            self.say(text)
            self._write(log)
            self.drawn.extend(drawn)
            newimage = False
        elif key == "h":
            self.ans_hdr(sh, newimage, key)
            k, x, y = key, wx, wy
            while True:
                if k in "abc":
                    self._cp(sh, x, y, None, k)
                    break
                if k in "lrk":
                    self._cp(sh, x, f32(1), y, k)
                    break
                self.say("Set cursor and type a, b, c, l, r, or k:")
                r = self.read()
                if r is None:
                    break
                x, y, _, k, _ = r
            newimage = False
        elif key == "m":
            self.ans_hdr(sh, newimage, key)
            self.say("m again:")
            r = self.read()
            x2, y2 = (r[0], r[1]) if r is not None else (wx, wy)
            text, line = avgsnr(sh, wx, wy, x2, y2)
            self.say(text)
            if line is not None:
                self._write([line])
            newimage = False
        elif key == "o":
            overplot = True
        elif key in "g#%()":
            i, j = self.nline, self.nband
            smw = self.image.smw
            name = self.name
            if key == "(":
                if smw.shape[1] > 1:
                    self.nline, self.nap = max(1, min(smw.shape[1], self.nline - 1)), None
                elif smw.shape[2] > 1:
                    self.nband = max(1, min(smw.shape[2], self.nband - 1))
            elif key == ")":
                if smw.shape[1] > 1:
                    self.nline, self.nap = max(1, min(smw.shape[1], self.nline + 1)), None
                elif smw.shape[2] > 1:
                    self.nband = max(1, min(smw.shape[2], self.nband + 1))
            elif key == "#":
                self.nline = 0
            elif key == "%":
                self.nband = 0
            else:
                other = self.answer("next_image").strip()
                if other != self.name:
                    if not (Path(other).exists() or Path(f"{other}.fits").exists() or
                            Path(section_of(other)[0]).exists() or
                            Path(f"{section_of(other)[0]}.fits").exists()):
                        self.err += f"Can't get {other}\n"
                        return newgraph, overplot, newimage, False
                    self.image = None
                    newimage = True
                name = other
                self.name = other
                self.nline = self.nband = 0
            self.sh, self.nline, self.nband, self.nap = self.getimage(
                name, self.nline, self.nband, self.nap, self.wave_scl, self.units)
            if self.options["wreset"] and not overplot:
                self._reset_limits()
            if self.nline != i or self.nband != j:
                newimage = True
            newgraph = True
        elif key == "w":
            if self._gt_window():
                newgraph, overplot = auto, False
        elif key == "l":
            if sh.fc == shdr.FCNO:
                self.err += "Warning: (>flam) spectrum not calibrated\n"
            conflam(sh)
            self._flux_reset()
            newgraph, overplot = auto, False
        elif key == "n":
            if sh.fc == shdr.FCNO:
                self.err += "Warning: (>fnu) spectrum not calibrated\n"
            confnu(sh)
            self._flux_reset()
            newgraph, overplot = auto, False
        elif key == "f":
            self._functions()
            overplot = False
        elif key == "q":
            if self.options["wreset"]:
                self._reset_limits()
            return True, overplot, newimage, True
        elif key == "r":
            newgraph, overplot = True, False
        elif key == "s":
            box = int(self.number("boxsize"))
            sh.y, said = smooth(sh.y, box)
            self.err += said
            newgraph = auto
        elif key == "j":
            xs, ys = fudgept(sh, wx, wy)
            self.drawn.append(("polyline", list(zip(xs, ys, strict=True)), 1))
        elif key == "x":
            self.say("x again:")
            r = self.read()
            x2, y2 = (r[0], r[1]) if r is not None else (wx, wy)
            xs, ys = fudgex(sh, wx, wy, x2, y2, self.options["xydraw"])
            self.drawn.append(("polyline", list(zip(xs, ys, strict=True)), 1))
        elif key == "-":
            if self.fitted is not None:
                self.say("- again:")
                r = self.read()
                if r is not None:
                    xs, ys = subblend(sh, wx, wy, r[0], r[1], self.fitted)
                    self.drawn.append(("polyline", list(zip(xs, ys, strict=True)), 1))
                    self.fitted = None
        elif key == "$":
            if self.wave_scl:
                sh.set_system("physical")
                self.wave_scl = False
                self.gt.xlabel, self.gt.xunits = "Pixel", ""
            else:
                sh.set_system("world")
                self.wave_scl = True
                self.gt.xlabel, self.gt.xunits = sh.un.label, sh.un.units
            self.gt.reset_window()
            if self.options["zero"]:
                self.gt.setr("ymin", 0.0)
            newgraph, overplot = auto, False
        elif key == "v":
            self._velocity(wx)
            newgraph, overplot = True, False
        elif key in "pu":
            newgraph, overplot = self._usercoord(key, wx, wy, auto)
        elif key in ("/", "?"):
            pass
        elif key == "I":
            raise KeyboardInterrupt("Interrupt")
        else:
            i = max(1, min(sh.sn, nint(sh.wl(float(f32(wx))))))
            self.say(f"x,y,z(x): {fx(f32(wx), 10, 3)} {g(f32(wy), 10, 4)} "
                     f"{g(sh.y[i - 1], 10, 4)}\n")
        return newgraph, overplot, newimage, False

    def _gt_window(self) -> bool:
        """gt_window: 'window:', a key of gtools' window (e takes a second corner,
        after 'again:'); whether to redraw."""
        self.say("window:")
        r = self.read()
        if r is None:
            return False
        wx, wy, _, k, _ = r
        if k == "e":
            self.say("again:")
            r2 = self.read()
            if r2 is None:
                return False
            return self.gt.window2((wx, wy), (r2[0], r2[1]), self.window)
        said, redraw = self.gt.window1(k, wx, wy, self.window)
        self.say(said)
        return redraw

    def _flux_reset(self) -> None:
        self.gt.setr("ymin", None)
        self.gt.setr("ymax", None)
        if self.options["zero"]:
            self.gt.setr("ymin", 0.0)

    def _cp(self, sh, center, cont, ylevel, key) -> None:
        text, error, log, drawn, profile = eqwidth_cp(sh, center, cont, ylevel, key)
        self.say(text)
        self.err += error
        if log is not None:
            self._write([log])
            self.drawn.extend(drawn)
            self.fitted = profile

    def _functions(self) -> None:
        """f: fun_help and fun_do until q."""
        help_ = ("q=quit l,n=log10,e d,e=d,exp s=sqrt a=abs i=1/s p=+k m=*k  "
                 "+,-,*,/=2spec ops\n")
        self.say(help_)
        while True:
            r = self.read()
            if r is None:
                break
            k = r[3]
            if k == "?":
                self.say(help_)
            elif k == "q":
                break
            elif k == "I":
                raise KeyboardInterrupt("Interrupt")
            else:
                try:
                    if k in "+-*/":
                        self.say("Second spectrum ")
                        other = self.answer("spec2").strip()
                        if other:
                            self._binary(k, other)
                    elif k == "r":
                        pass
                    else:
                        const = self.number("constant") if k in "mp" else None
                        self.sh.y = fun_do(k, self.sh.y, const)
                    self._flux_reset()
                    if self.options["auto"]:
                        self.replot(True)
                    self.say(help_)
                except ValueError as error:
                    self.err += f"Warning: {error}\n"
                    self.say(help_)
        self.say("\n")

    def _binary(self, op: str, other: str) -> None:
        img = open_image(section_of(other)[0], int(self.params.get("dispaxis", 1)),
                         tuple(self.params.get("nsum", (1, 1))))
        smw = img.smw
        line = 1
        sh2 = shdr.shdr_open(smw, img.data, section_of(other)[0], line, 1, None)
        with contextlib.suppress(shdr.ShdrError):
            sh2.set_units("angstroms")
        other_y = rebin(sh2, self.sh)
        y = self.sh.y
        if op == "+":
            self.sh.y = (y + other_y).astype(f32)
        elif op == "-":
            self.sh.y = (y - other_y).astype(f32)
        elif op == "*":
            self.sh.y = (y * other_y).astype(f32)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                self.sh.y = np.where(other_y == 0, f32(0), y / other_y).astype(f32)

    def _velocity(self, wx) -> None:
        """v: the coordinates as a velocity about the cursor, or back."""
        sh = self.sh
        if sh.un.kind == unitsmod.VEL:
            self.units = self.units1
            target = self.units2
        else:
            self.units1 = self.units
            self.units2 = sh.units
            ref = float(unitsmod.convert([float(f32(wx))], sh.un,
                                         unitsmod.decode("angstroms"), real=True)[0])
            self.units = f"km/s {iraffmt.g(float(f32(ref)))} angstroms"
            target = self.units
        new = unitsmod.decode(target)
        for name in ("xmin", "xmax"):
            v = self.gt.getr(name)
            if v is not None:
                self.gt.setr(name, float(unitsmod.convert([v], sh.un, new, real=True)[0]))
        sh.x = unitsmod.convert(sh.x, sh.un, new, real=True).astype(f32)
        sh.un = new
        self.gt.xlabel, self.gt.xunits = new.label, new.units

    def _usercoord(self, key, wx, wy, auto):
        """p and u: new coordinates from a start and an end, or from marked features."""
        self.err += ("p and u (new dispersion coordinates) are not ported yet; "
                     "use dispcor, dopcor or specshift\n")
        return False, False

    def colon(self, cmd: str, wx, wy) -> bool:
        """splot_colon."""
        words = cmd.split(None, 1)
        word = words[0] if words else ""
        rest = words[1] if len(words) > 1 else ""
        sh = self.sh
        if word == "show":
            if self.measured:
                self.say("\n".join(self.measured) + "\n")
            else:
                self.say("No measurements\n")
        elif word == "nolog":
            self.say(f"Logging to {self.params.get('save_file', '')} disabled")
            self.logging = False
        elif word == "log":
            self.logging = True
            self.say(f"Logging to {self.params.get('save_file', '')} enabled")
        elif word == "#":
            self.ans_hdr(sh, False, "m")
            self._write([cmd])
        elif word in ("dispaxis", "nsum"):
            if self.image.smw.format != shdr.ND:
                self.say("Image is not two dimensional\n" if word == "dispaxis"
                         else "Invalid image format\n")
            else:
                self.err += f":{word} is not ported yet\n"
        elif word == "units":
            try:
                new = unitsmod.decode(rest.strip())
                for name in ("xmin", "xmax"):
                    v = self.gt.getr(name)
                    if v is not None:
                        self.gt.setr(name, float(unitsmod.convert([v], sh.un, new,
                                                                  real=True)[0]))
                sh.x = unitsmod.convert(sh.x, sh.un, new, real=True).astype(f32)
                sh.un = new
                self.units = rest.strip()
                self.gt.xlabel, self.gt.xunits = new.label, new.units
                return True
            except ValueError as error:
                self.err += f"Warning: {error}\n"
        elif word in OPTIONS or word == "hist":
            name = "histogram" if word == "hist" else word
            value = rest.strip().lower()
            if not value:            # %b of an int: YES or NO
                self.say(f"{word} {'YES' if self.options[name] else 'NO'}\n")
                return False
            flag = value in ("yes", "y", "true", "+")
            self.options[name] = flag
            if name == "zero" and flag:
                self.gt.setr("ymin", 0.0)
            if name == "histogram":
                self.gt.type = 3 if flag else 2
            if name == "nosysid":
                self.gt.sysid = not flag
            if name == "flip":
                self.gt.xflip = flag
            return self.options["auto"] if name in ("zero", "histogram", "nosysid") \
                else False
        elif word in ("label", "mabove", "mbelow"):
            self.drawn.append((word, (wx, wy, rest)))
        else:
            self.say("Unrecognized or ambiguous command\007")
        return False


def _two(value) -> str:
    """mktitle's %.2s of a real: IRAF's %s of a number is its %g, the precision not
    applied -- 900. (the metacode's title says so)."""
    return "INDEF" if value is None else iraffmt.g(float(value))


def rebin(sh2, sh1) -> np.ndarray:
    """shdr_rebin, the simple case: the second spectrum on the first's pixels by
    linear interpolation of its coordinates (onedinterp is not consulted)."""
    return np.interp(sh1.x.astype(float), sh2.x.astype(float),
                     sh2.y.astype(float)).astype(f32)


def cursor_reads(text: str):
    """IRAF's cursor file: 'x y wcs key [command]' a line, ':command' one, a key
    as \\ooo in octal (\\040 a space)."""
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith(":"):
            yield (0.0, 0.0, 1, ":", line[1:])
            continue
        words = line.split(None, 3)
        if len(words) < 4:
            continue
        key = words[3]
        if key.startswith("\\") and len(key) >= 4 and key[1:4].isdigit():
            key = chr(int(key[1:4], 8)) + key[4:]
        yield (float(words[0]), float(words[1]), int(words[2]), key[0], key[1:].strip())
