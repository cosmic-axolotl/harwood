"""zfind -- a redshift by scanning chi-square, as today's surveys measure them.

IRAF and RVSAO decide a redshift from one line: emsao takes the tallest peak near
Halpha (or [O III], Hbeta, [O II]) as that line, and when another is taller --
[N II] 6583 over Halpha in a LINER or a starburst nucleus such as NGC 660's -- the
redshift is wrong by the distance between them. A modern redshift fitter (SDSS's
spectro1d, DESI's Redrock: Bolton et al. 2012, Bailey et al. 2023) asks instead how
well the whole spectrum is explained at each redshift, and takes the best one over
the whole range:

* At each trial redshift z (steps uniform in ln(1+z), `dv` km/s apart) the model is
  a Legendre polynomial continuum plus a Gaussian at every line of the list that
  falls in the spectrum, each with its own non-negative amplitude -- a linear
  least-squares fit, so every line counts and none has to be found first. Several
  line widths are tried; a template spectrum (`templates`), redshifted, can stand
  in for or join the lines, for absorption-line spectra.
* chi2(z) is minimised over the grid, and the minimum refined by a parabola through
  a finer scan around it; the error of z is where chi2 has risen by 1 (from its
  curvature), with the noise scaled so that the best fit has a reduced chi2 of 1
  when the spectrum has no errors of its own.
* deltachi2 -- the chi2 of the best other minimum (more than `separation` km/s
  away) minus the best's -- says how sure the answer is: under 9, Redrock's
  threshold, the redshift is flagged as doubtful, and the next candidates are
  listed with their own chi2.

Held to synthetic spectra whose redshift is known (the error's pull distribution
too) and to the course's NGC 660 nucleus, where emsao's search fails and its
guided fit agrees with this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

C = 299792.458              # km/s

#: The lines fit by default: RVSAO's emission list (air wavelengths).
DEFAULT_WIDTHS = (60.0, 120.0, 250.0, 500.0)       # Gaussian sigma, km/s


@dataclass
class Candidate:
    z: float
    chi2: float
    width: float


@dataclass
class ZResult:
    z: float
    z_err: float
    chi2: float
    dof: int
    deltachi2: float
    width: float
    doubtful: bool
    candidates: list[Candidate] = field(default_factory=list)
    lines: list[dict] = field(default_factory=list)
    grid: np.ndarray | None = None
    chi2_grid: np.ndarray | None = None
    noise_scale: float = 1.0

    @property
    def cz(self) -> float:
        return C * self.z

    @property
    def cz_err(self) -> float:
        return C * self.z_err


def _legendre(x: np.ndarray, n: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    if n > 1:
        cols.append(x)
    for k in range(2, n):
        cols.append(((2 * k - 1) * x * cols[-1] - (k - 1) * cols[-2]) / k)
    return np.stack(cols[:n], axis=1)


class Fitter:
    """The linear model at each redshift: polynomial + lines (+ templates)."""

    def __init__(self, wave, flux, ivar, rest_lines, npoly: int = 4, templates=None):
        self.wave = np.asarray(wave, float)
        self.flux = np.asarray(flux, float)
        self.ivar = np.asarray(ivar, float)
        good = np.isfinite(self.flux) & np.isfinite(self.ivar) & (self.ivar > 0)
        self.wave, self.flux, self.ivar = self.wave[good], self.flux[good], self.ivar[good]
        self.sw = np.sqrt(self.ivar)
        lo, hi = self.wave.min(), self.wave.max()
        x = 2 * (self.wave - lo) / (hi - lo) - 1
        self.poly = _legendre(x, npoly)
        self.npoly = npoly
        self.rest = [(float(w), str(n)) for w, n in rest_lines]
        self.templates = templates or []

    def design(self, z: float, width: float):
        cols = [self.poly]
        names: list[str] = []
        for rest, name in self.rest:
            centre = rest * (1 + z)
            sigma = centre * width / C
            if centre - 3 * sigma < self.wave[0] or centre + 3 * sigma > self.wave[-1]:
                continue
            cols.append(np.exp(-0.5 * ((self.wave - centre) / sigma) ** 2)[:, None])
            names.append((name if name else f"{rest:.2f}", rest))
        for k, (tw, tf) in enumerate(self.templates):
            shifted = np.interp(self.wave, tw * (1 + z), tf, left=np.nan, right=np.nan)
            if np.isnan(shifted).mean() > 0.2:
                continue
            cols.append(np.nan_to_num(shifted)[:, None])
            names.append((f"template{k + 1}", None))
        return np.hstack(cols), names

    def fit(self, z: float, width: float):
        """chi2 and the coefficients: the polynomial free, the lines >= 0."""
        a, names = self.design(z, width)
        aw = a * self.sw[:, None]
        bw = self.flux * self.sw
        coef = self._nonneg(aw, bw, self.npoly)
        resid = bw - aw @ coef
        return float(resid @ resid), coef, names, a

    @staticmethod
    def _nonneg(aw, bw, nfree: int):
        """Least squares with the first nfree coefficients free and the rest >= 0:
        an active set on the bounded ones (the problem is small)."""
        coef, *_ = np.linalg.lstsq(aw, bw, rcond=None)
        active = np.ones(aw.shape[1], bool)
        for _ in range(aw.shape[1]):
            bad = np.where(active & (np.arange(coef.size) >= nfree) & (coef < 0))[0]
            if bad.size == 0:
                break
            active[bad] = False
            coef = np.zeros(aw.shape[1])
            sol, *_ = np.linalg.lstsq(aw[:, active], bw, rcond=None)
            coef[active] = sol
        return coef


def zscan(wave, flux, error=None, rest_lines=None, zmin: float = -0.005,
          zmax: float = 1.0, dv: float = 30.0, widths=DEFAULT_WIDTHS, npoly: int = 4,
          templates=None, separation: float = 1000.0, ncandidates: int = 5) -> ZResult:
    """The redshift that best explains the spectrum, its error and how sure it is."""
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    if rest_lines is None:
        rest_lines = default_lines()
    if error is None:
        ivar = np.ones_like(flux)
        scaled = True
    else:
        error = np.asarray(error, float)
        ivar = np.where(error > 0, 1.0 / np.where(error > 0, error, 1.0) ** 2, 0.0)
        scaled = False
    fitter = Fitter(wave, flux, ivar, rest_lines, npoly, templates)
    lmin, lmax = math.log1p(zmin), math.log1p(zmax)
    step = math.log1p(dv / C)
    grid = np.expm1(np.arange(lmin, lmax + step / 2, step))
    best_chi2 = np.full(grid.size, np.inf)
    best_width = np.zeros(grid.size)
    for i, z in enumerate(grid):
        for w in widths:
            chi2 = fitter.fit(float(z), float(w))[0]
            if chi2 < best_chi2[i]:
                best_chi2[i], best_width[i] = chi2, w
    # the noise: without errors, scaled so the best fit is reduced-chi2 1
    i0 = int(np.argmin(best_chi2))
    ndata = fitter.flux.size
    npar = fitter.design(float(grid[i0]), float(best_width[i0]))[0].shape[1]
    dof = max(1, ndata - npar)
    scale = best_chi2[i0] / dof if scaled else 1.0
    chi2 = best_chi2 / scale
    minima = _minima(grid, chi2, best_width, separation)
    z0, w0 = minima[0].z, minima[0].width
    w0 = _refine_width(fitter, z0, w0, scale)
    zf, cf = _refine(fitter, z0, w0, dv, scale)
    w0 = _refine_width(fitter, zf, w0, scale)
    zf, cf = _refine(fitter, zf, w0, dv, scale)
    zerr = _error(fitter, zf, w0, dv, scale)
    others = [m for m in minima[1:] if abs(m.z - zf) * C / (1 + zf) > separation]
    delta = (others[0].chi2 - cf) if others else math.inf
    _, coef, names, a = fitter.fit(zf, w0)
    lines = []
    for k, (name, rest) in enumerate(names):
        amp = coef[fitter.npoly + k]
        if rest is None:
            lines.append({"name": name, "amplitude": float(amp)})
            continue
        sigma = rest * (1 + zf) * w0 / C
        lines.append({"name": name, "rest": rest, "observed": rest * (1 + zf),
                      "amplitude": float(amp),
                      "flux": float(amp * sigma * math.sqrt(2 * math.pi))})
    return ZResult(zf, zerr, cf, dof, delta, w0, delta < 9.0,
                   [Candidate(m.z, m.chi2, m.width) for m in minima[:ncandidates]],
                   lines, grid, chi2, math.sqrt(scale))


def _minima(grid, chi2, widths, separation) -> list[Candidate]:
    """The local minima of chi2(z), best first, none within `separation` km/s of a
    better one."""
    idx = [i for i in range(grid.size)
           if (i == 0 or chi2[i] <= chi2[i - 1])
           and (i == grid.size - 1 or chi2[i] <= chi2[i + 1])]
    idx.sort(key=lambda i: chi2[i])
    kept: list[int] = []
    for i in idx:
        if all(abs(grid[i] - grid[j]) * C / (1 + grid[j]) > separation for j in kept):
            kept.append(i)
    return [Candidate(float(grid[i]), float(chi2[i]), float(widths[i])) for i in kept]


def _refine_width(fitter, z, width, scale, rounds: int = 3) -> float:
    """The line width that fits best at z: a geometric scan about the grid's, narrowed
    each round (the error of z depends on it: a width off by 20% inflates it 10%)."""
    best = width
    span = 2.0
    for _ in range(rounds):
        trial = best * np.geomspace(1 / span, span, 13)
        chi2 = [_chi2(fitter, z, float(w), scale) for w in trial]
        best = float(trial[int(np.argmin(chi2))])
        span = span ** 0.25
    return best


def _chi2(fitter, z, width, scale) -> float:
    return fitter.fit(z, width)[0] / scale


def _refine(fitter, z0, width, dv, scale):
    """A finer scan about the minimum, and the parabola through its lowest three."""
    dz = (1 + z0) * dv / C
    zs = z0 + dz * np.linspace(-1.5, 1.5, 31)
    cs = np.array([_chi2(fitter, float(z), width, scale) for z in zs])
    i = int(np.clip(np.argmin(cs), 1, zs.size - 2))
    x, y = zs[i - 1:i + 2], cs[i - 1:i + 2]
    denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
    a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
    b = (x[2] ** 2 * (y[0] - y[1]) + x[1] ** 2 * (y[2] - y[0])
         + x[0] ** 2 * (y[1] - y[2])) / denom
    if a <= 0:
        return float(zs[i]), float(cs[i])
    zf = -b / (2 * a)
    return float(zf), _chi2(fitter, float(zf), width, scale)


def _error(fitter, z, width, dv, scale) -> float:
    """sigma_z where chi2 rises by 1: sqrt(2 / chi2''), from a symmetric difference."""
    h = (1 + z) * dv / C / 5
    c0 = _chi2(fitter, z, width, scale)
    cp = _chi2(fitter, z + h, width, scale)
    cm = _chi2(fitter, z - h, width, scale)
    curvature = (cp + cm - 2 * c0) / (h * h)
    return math.sqrt(2.0 / curvature) if curvature > 0 else math.inf


def default_lines() -> list[tuple[float, str]]:
    """RVSAO's emission lines (emlines.dat), air wavelengths, as (rest, name)."""
    from .emsao import DEFAULTS, read_lines

    lines = read_lines({**DEFAULTS, "ablines": "", "emcombine": "", "emsearch": ""})
    return list(zip(lines.wlref, lines.nmref, strict=True))
