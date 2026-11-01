"""Where an image is on the sky: stars of the frame against a catalogue, TAN and, if asked, SIP
distortion (Shupe et al. 2005). IRAF's ccmap fits a plate from stars already paired by hand;
here the pairs are found. Given about where it is and about its pixel scale, the rotation and
the side of the sky are not needed: every rotation, mirror and scale in the range is tried,
with a vote among all pairs of stars for an offset between the lists.

The sky is projected about a centre that the fit moves. What is fitted is

    xi = a0 + a1 u + a2 v     eta = b0 + b1 u + b2 v

with (u, v) the pixel from CRPIX, the middle of the frame; then, for SIP, the polynomials f and
g in u and v, of order 2 and above, in xi, eta = CD (u + f, v + g).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import advice
from .jit import jit

DEG = math.pi / 180.0


def project(ra, dec, ra0: float, dec0: float):
    """Gnomonic, in degrees: the tangent plane at (ra0, dec0)."""
    a, d = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    a0, d0 = math.radians(ra0), math.radians(dec0)
    cosine = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosine
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosine
    return np.degrees(xi), np.degrees(eta)


def unproject(xi, eta, ra0: float, dec0: float):
    x, y = np.radians(np.asarray(xi, float)), np.radians(np.asarray(eta, float))
    a0, d0 = math.radians(ra0), math.radians(dec0)
    bottom = np.cos(d0) - y * np.sin(d0)
    ra = a0 + np.arctan2(x, bottom)
    dec = np.arctan2((np.sin(d0) + y * np.cos(d0)) * np.cos(ra - a0), bottom)
    return np.mod(np.degrees(ra), 360.0), np.degrees(dec)


@jit
def _vote(xs, ys, us, vs, cos_t, sin_t, mirror, scale, cell, half, counts):
    """The offset that most pairs share, for one rotation, mirror and scale: `us` and `vs` are
    the catalogue in degrees, `xs` and `ys` the frame in pixels from the middle."""
    counts[:, :] = 0
    size = 2 * half
    best = 0
    where_x = 0
    where_y = 0
    # East is to the left on the sky as it is seen, which is no mirror
    sign = 1.0 if mirror else -1.0
    for j in range(us.size):
        px = (cos_t * us[j] * sign - sin_t * vs[j]) * scale
        py = (sin_t * us[j] * sign + cos_t * vs[j]) * scale
        for i in range(xs.size):
            bx = int(math.floor((xs[i] - px) / cell)) + half
            by = int(math.floor((ys[i] - py) / cell)) + half
            if 0 <= bx < size and 0 <= by < size:
                counts[by, bx] += 1
                if counts[by, bx] > best:
                    best = counts[by, bx]
                    where_x = bx
                    where_y = by
    return best, (where_x - half + 0.5) * cell, (where_y - half + 0.5) * cell


@dataclass
class Guess:
    votes: int
    runner_up: int
    rotation: float      #: degrees, of the catalogue on the frame
    mirror: bool
    pixel_scale: float   #: arcsec a pixel
    dx: float
    dy: float


def search(xs, ys, us, vs, scale: float, scale_error: float, cell: float = 4.0,
           rotations=None, mirrors=(False, True)) -> Guess:
    """Every rotation, mirror and scale in the range, by the vote. `xs`, `ys` from the middle
    of the frame in pixels, `us`, `vs` the catalogue in degrees."""
    extent = float(max(np.ptp(xs), np.ptp(ys))) or 1.0
    step_angle = math.degrees(cell / extent)
    if rotations is None:
        rotations = np.arange(0.0, 360.0, step_angle)
    step = cell / extent
    count = max(int(math.ceil(scale_error / step)), 0)
    scales = scale * (1.0 + step * np.arange(-count, count + 1))
    half = int(math.ceil(1.5 * extent / cell)) + 1
    counts = np.zeros((2 * half, 2 * half), np.int32)
    results = []
    for mirror in mirrors:
        for pixel_scale in scales:
            for angle in rotations:
                cos_t, sin_t = math.cos(math.radians(angle)), math.sin(math.radians(angle))
                votes, dx, dy = _vote(xs, ys, us, vs, cos_t, sin_t, bool(mirror),
                                      3600.0 / pixel_scale, cell, half, counts)
                results.append((int(votes), float(angle), bool(mirror), float(pixel_scale),
                                float(dx), float(dy)))
    results.sort(key=lambda row: -row[0])
    best = results[0]
    # The runner-up is the best that is not next to it in rotation and scale
    other = 0
    for row in results[1:]:
        away = abs((row[1] - best[1] + 180.0) % 360.0 - 180.0)
        if row[2] != best[2] or away > 3 * step_angle:
            other = row[0]
            break
    return Guess(best[0], other, best[1], best[2], best[3], best[4], best[5])


@dataclass
class Solution:
    crval: tuple[float, float]
    crpix: tuple[float, float]
    cd: np.ndarray
    sip: int = 0
    a: dict = field(default_factory=dict)
    b: dict = field(default_factory=dict)
    ap: dict = field(default_factory=dict)
    bp: dict = field(default_factory=dict)
    pairs: np.ndarray | None = None       #: (index in the frame, index in the catalogue)
    rms: float = 0.0                      #: arcsec
    rejected: int = 0

    @property
    def pixel_scale(self) -> float:
        return float(math.sqrt(abs(np.linalg.det(self.cd))) * 3600.0)

    @property
    def rotation(self) -> float:
        """Of north from the frame's +y, degrees, towards +x."""
        return float(math.degrees(math.atan2(self.cd[1, 0], self.cd[1, 1])))

    @property
    def flipped(self) -> bool:
        """East is to the right of north, against the sky as it is seen."""
        return bool(np.linalg.det(self.cd) > 0)

    def pixel(self, u, v):
        """u, v plus the distortion."""
        return (u + poly(self.a, u, v), v + poly(self.b, u, v)) if self.sip else (u, v)

    def sky(self, x, y):
        """RA and Dec of 1-based pixels."""
        u, v = self.pixel(np.asarray(x, float) - self.crpix[0],
                          np.asarray(y, float) - self.crpix[1])
        xi = self.cd[0, 0] * u + self.cd[0, 1] * v
        eta = self.cd[1, 0] * u + self.cd[1, 1] * v
        return unproject(xi, eta, *self.crval)

    def cards(self) -> list[tuple[str, object, str]]:
        """What a header needs, for any FITS tool."""
        suffix = "-SIP" if self.sip else ""
        out: list[tuple[str, object, str]] = [
            ("CTYPE1", f"RA---TAN{suffix}", "tangent plane" + (", SIP" if self.sip else "")),
            ("CTYPE2", f"DEC--TAN{suffix}", ""), ("CRVAL1", float(self.crval[0]), "deg"),
            ("CRVAL2", float(self.crval[1]), "deg"), ("CRPIX1", float(self.crpix[0]), ""),
            ("CRPIX2", float(self.crpix[1]), ""), ("CD1_1", float(self.cd[0, 0]), ""),
            ("CD1_2", float(self.cd[0, 1]), ""), ("CD2_1", float(self.cd[1, 0]), ""),
            ("CD2_2", float(self.cd[1, 1]), "")]
        if self.sip:
            for name, values in (("A", self.a), ("B", self.b), ("AP", self.ap),
                                 ("BP", self.bp)):
                out.append((f"{name}_ORDER", self.sip, "SIP order"))
                for (p, q), value in values.items():
                    out.append((f"{name}_{p}_{q}", float(value), ""))
        return out


def terms(order: int, first: int = 2) -> list[tuple[int, int]]:
    return [(p, q) for total in range(first, order + 1) for p in range(total, -1, -1)
            for q in [total - p]]


def poly(coefficients: dict, u, v):
    total = np.zeros_like(np.asarray(u, float))
    for (p, q), value in coefficients.items():
        total = total + value * u ** p * v ** q
    return total


def _design(u, v, wanted):
    return np.column_stack([u ** p * v ** q for p, q in wanted])


def _pairs(x, y, px, py, radius: float):
    """Each frame star's nearest catalogue star, within the radius, one for one."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([px, py]))
    distance, nearest = tree.query(np.column_stack([x, y]), distance_upper_bound=radius)
    order = np.argsort(distance)
    used, frame, catalogue = set(), [], []
    for at in order:
        if not np.isfinite(distance[at]) or nearest[at] in used:
            continue
        used.add(int(nearest[at]))
        frame.append(int(at))
        catalogue.append(int(nearest[at]))
    return np.array(frame, int), np.array(catalogue, int)


def _linear_fit(u, v, xi, eta, keep):
    design = np.column_stack([np.ones(u.size), u, v])[keep]
    first = np.linalg.lstsq(design, xi[keep], rcond=None)[0]
    second = np.linalg.lstsq(design, eta[keep], rcond=None)[0]
    return first, second


def fit(x, y, ra, dec, guess: Guess, crpix, centre, sip: int = 0, tolerance: float = 3.0,
        clip: float = 3.0) -> Solution:
    """From the vote, in steps: pairs within a radius that closes, the plate fitted to them."""
    u, v = np.asarray(x, float) - crpix[0], np.asarray(y, float) - crpix[1]
    angle = math.radians(guess.rotation)
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    sign = 1.0 if guess.mirror else -1.0
    pixels = 3600.0 / guess.pixel_scale
    inverse = pixels * np.array([[cos_t * sign, -sin_t], [sin_t * sign, cos_t]])
    shift = np.array([guess.dx, guess.dy])
    crval = centre
    cd = np.linalg.inv(inverse)
    # The middle of the frame is where the vote's offset says: move the centre there
    middle = cd @ (-shift)
    crval = unproject(middle[0], middle[1], *crval)
    crval = (float(crval[0]), float(crval[1]))
    radius = max(tolerance, 8.0)
    frame, catalogue, keep = np.array([], int), np.array([], int), np.array([], bool)
    a, b = {}, {}
    wanted = terms(sip) if sip else []
    for round_ in range(12):
        xi, eta = project(ra, dec, *crval)
        back = np.linalg.inv(cd)
        px = back[0, 0] * xi + back[0, 1] * eta
        py = back[1, 0] * xi + back[1, 1] * eta
        tu, tv = u.copy(), v.copy()
        if a:
            tu, tv = u + poly(a, u, v), v + poly(b, u, v)
        frame, catalogue = _pairs(tu, tv, px, py, radius)
        if frame.size < 4:
            advice.fail(f"only {frame.size} stars of the frame are near one of the catalogue",
                        "the vote found no real match: check the centre and the scale, and "
                        "that the catalogue covers the frame.")
        fu, fv, fxi, feta = tu[frame], tv[frame], xi[catalogue], eta[catalogue]
        keep = np.ones(frame.size, bool)
        for _ in range(4):
            first, second = _linear_fit(fu, fv, fxi, feta, keep)
            left = np.hypot(np.column_stack([np.ones(fu.size), fu, fv]) @ first - fxi,
                            np.column_stack([np.ones(fu.size), fu, fv]) @ second - feta)
            sigma = 1.4826 * np.median(left[keep])
            new = left < max(clip * sigma, 1e-9)
            if new.sum() < 3 or (new == keep).all():
                break
            keep = new
        cd = np.array([[first[1], first[2]], [second[1], second[2]]])
        moved = unproject(first[0], second[0], *crval)
        crval = (float(moved[0]), float(moved[1]))
        if sip and round_ >= 3 and keep.sum() > 2 * len(wanted):
            xi, eta = project(ra, dec, *crval)
            back = np.linalg.inv(cd)
            target_u = back[0, 0] * xi[catalogue] + back[0, 1] * eta[catalogue]
            target_v = back[1, 0] * xi[catalogue] + back[1, 1] * eta[catalogue]
            raw_u, raw_v = u[frame][keep], v[frame][keep]
            design = _design(raw_u, raw_v, wanted)
            solved_u = np.linalg.lstsq(design, target_u[keep] - raw_u, rcond=None)[0]
            solved_v = np.linalg.lstsq(design, target_v[keep] - raw_v, rcond=None)[0]
            a = dict(zip(wanted, solved_u, strict=True))
            b = dict(zip(wanted, solved_v, strict=True))
        radius = max(tolerance, radius * 0.7)
    done = Solution(crval, tuple(crpix), cd, sip if a else 0, a, b)
    if a:
        done.ap, done.bp = _inverse_sip(a, b, sip, u, v)
    ras, decs = done.sky(np.asarray(x, float)[frame[keep]], np.asarray(y, float)[frame[keep]])
    done.pairs = np.column_stack([frame[keep], catalogue[keep]])
    done.rejected = int((~keep).sum())
    dx = (ras - np.asarray(ra)[catalogue[keep]]) * np.cos(np.radians(decs)) * 3600.0
    dy = (decs - np.asarray(dec)[catalogue[keep]]) * 3600.0
    done.rms = float(np.sqrt(np.mean(dx ** 2 + dy ** 2)))
    return done


def _inverse_sip(a, b, order: int, u, v) -> tuple[dict, dict]:
    """AP and BP, fitted over the frame, one order higher being of no use at these sizes: what
    takes u + f, v + g back to u, v, for those who do not iterate."""
    grid_u, grid_v = np.meshgrid(np.linspace(u.min(), u.max(), 40),
                                 np.linspace(v.min(), v.max(), 40))
    grid_u, grid_v = grid_u.ravel(), grid_v.ravel()
    forward_u = grid_u + poly(a, grid_u, grid_v)
    forward_v = grid_v + poly(b, grid_u, grid_v)
    wanted = terms(order, 1)
    design = _design(forward_u, forward_v, wanted)
    ap = np.linalg.lstsq(design, grid_u - forward_u, rcond=None)[0]
    bp = np.linalg.lstsq(design, grid_v - forward_v, rcond=None)[0]
    return dict(zip(wanted, ap, strict=True)), dict(zip(wanted, bp, strict=True))


def proper_motion(ra, dec, pmra, pmdec, reference: float, epoch: float):
    """Straight on, which is enough over a century at these rates. pmra has cos(dec) in it."""
    years = epoch - reference
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    step = np.nan_to_num(np.asarray(pmra, float)) * years / 3.6e6
    moved = ra + step / np.cos(np.radians(dec))
    return moved, dec + np.nan_to_num(np.asarray(pmdec, float)) * years / 3.6e6


GAIA_SERVICES = ("https://gea.esac.esa.int/tap-server/tap/sync",
                 "https://gaia.ari.uni-heidelberg.de/tap/sync")


def gaia_query(ra: float, dec: float, radius: float, top: int = 3000) -> str:
    return (f"SELECT TOP {top} ra,dec,pmra,pmdec,phot_g_mean_mag FROM gaiadr3.gaia_source "
            f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra:.6f},{dec:.6f},"
            f"{radius:.5f})) AND phot_g_mean_mag IS NOT NULL ORDER BY phot_g_mean_mag")


def gaia(ra: float, dec: float, radius: float, top: int = 3000):
    """Gaia DR3 around a place, from ESA's service, or Heidelberg's when that does not answer.
    Only the place and the radius leave the machine."""
    from urllib.error import URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    from astropy.table import Table

    body = urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                      "QUERY": gaia_query(ra, dec, radius, top)}).encode()
    problems = []
    for url in GAIA_SERVICES:
        try:
            with urlopen(Request(url, data=body), timeout=120) as answer:
                text = answer.read().decode()
            if not text.lstrip().startswith("ra,"):
                problems.append(f"{url}: not a table")
                continue
            return Table.read(text, format="ascii.csv"), url
        except (URLError, OSError) as error:
            problems.append(f"{url}: {error}")
    advice.fail("no service gave the catalogue: " + "; ".join(problems),
                "try again later, or give a table of ra and dec with --catalog.")


def chance_matches(frame_stars: int, catalogue_stars: int, area: float,
                   radius: float) -> float:
    """How many of a frame's stars have a catalogue star within `radius` pixels by chance, with
    that many catalogue stars over `area` square pixels."""
    return frame_stars * min(catalogue_stars / max(area, 1.0) * math.pi * radius * radius, 1.0)


def judged(solution: Solution, frame_stars: int, xi, eta, shape, tolerance: float):
    """(pairs, what chance gives, what they are worth in sigma of chance): a plate fits
    anything a little, so it has to beat what a list of stars at random would have."""
    ny, nx = shape
    back = np.linalg.inv(solution.cd)
    moved_xi, moved_eta = project(*unproject(xi, eta, *solution.crval), *solution.crval)
    u = back[0, 0] * moved_xi + back[0, 1] * moved_eta
    v = back[1, 0] * moved_xi + back[1, 1] * moved_eta
    inside = int(((np.abs(u) < nx / 2) & (np.abs(v) < ny / 2)).sum())
    chance = chance_matches(frame_stars, inside, nx * ny, tolerance)
    found = solution.pairs.shape[0]
    return found, chance, (found - chance) / math.sqrt(max(chance, 1.0))
