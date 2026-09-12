"""Matching coordinate lists between frames.

Two frames of the same field are never on the same pixels. The telescope
drifts, the rotator moves, the guider nudges. On the 1987 M92 frames the shift
between two exposures taken minutes apart was 0.75 by 1.75 pixels, and between
the B and V frames it was nearly five -- so anything that compares magnitudes
across frames has to solve this first.

Two algorithms, as IRAF's `xyxymatch` offers them:

``tolerance``
    Nearest neighbour within a radius, after an optional initial shift. Fast,
    and right when you already know roughly where the frames sit.

``triangles``
    Pattern matching with no prior at all (Groth 1986; Valdes et al. 1995).
    Triangles are described by quantities that survive translation, rotation
    and scale, so a triangle of three stars looks the same in both frames
    however the frame moved. Matching triangles vote for point pairs, and the
    pairs with the most votes are the ones to trust.

Everything here takes arrays and returns arrays: no FITS, no tasks, no Qt.
Coordinates are 1-based (x, y) because that is what the rest of harwood is,
but nothing in the arithmetic depends on the origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Match", "by_tolerance", "by_triangles", "estimate_shift", "fit_transform"]


@dataclass
class Match:
    """Which point of one list corresponds to which point of the other."""

    reference: np.ndarray        #: indices into the reference list
    input: np.ndarray            #: indices into the input list
    shift: tuple[float, float] = (0.0, 0.0)
    rms: float = float("nan")
    method: str = "tolerance"
    message: str = ""

    def __len__(self) -> int:
        return int(self.reference.size)

    def __bool__(self) -> bool:
        return bool(len(self) > 0 and np.isfinite(self.rms))

    def summary(self) -> str:
        if not self:
            return self.message or "no match"
        return (f"{len(self)} pairs by {self.method}, "
                f"shift {self.shift[0]:+.2f} {self.shift[1]:+.2f}, "
                f"rms {self.rms:.3f}")


def estimate_shift(rx, ry, ix, iy, search: float = 50.0, bins: int = 200
                   ) -> tuple[float, float]:
    """Find the translation between two point sets, without pairing anything.

    Every pair of points contributes one candidate offset; the true offset is
    the one many pairs agree on, so it stands out as a peak in the histogram of
    all offsets. Robust to the lists having quite different lengths, which they
    always do -- one frame goes deeper than the other.
    """
    rx, ry, ix, iy = (np.asarray(a, dtype=float) for a in (rx, ry, ix, iy))
    dx = (ix[None, :] - rx[:, None]).ravel()
    dy = (iy[None, :] - ry[:, None]).ravel()
    near = (np.abs(dx) <= search) & (np.abs(dy) <= search)
    if not near.any():
        return 0.0, 0.0

    span = (-search, search)
    counts, edges = np.histogram(dx[near], bins=bins, range=span)
    peak_x = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
    counts, edges = np.histogram(dy[near], bins=bins, range=span)
    peak_y = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])

    # Refine: the histogram bin is coarse, so take the mean of the offsets
    # that fall inside it.
    width = 2 * search / bins
    close = near & (np.abs(dx - peak_x) < width) & (np.abs(dy - peak_y) < width)
    if close.sum() >= 3:
        return float(dx[close].mean()), float(dy[close].mean())
    return float(peak_x), float(peak_y)


def apply_transform(x, y, coefficients) -> tuple[np.ndarray, np.ndarray]:
    """Map coordinates through the six coefficients of a linear transform."""
    a, b, c, d, e, f = coefficients
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return a + b * x + c * y, d + e * x + f * y


def by_tolerance(rx, ry, ix, iy, tolerance: float = 2.0,
                 xin: float | None = None, yin: float | None = None,
                 transform=None) -> Match:
    """Pair points that land within `tolerance` of one another.

    With no initial shift given, one is estimated first -- otherwise a
    two-pixel tolerance finds nothing on frames that moved five. A full
    `transform` may be given instead, which is what matters once rotation is
    in play: half a degree is three pixels at the corner of a 512 chip, so a
    translation-only match loses the edges of the field and keeps the middle,
    which looks like success.
    """
    rx, ry, ix, iy = (np.asarray(a, dtype=float) for a in (rx, ry, ix, iy))
    if rx.size == 0 or ix.size == 0:
        return Match(np.array([], int), np.array([], int), (0.0, 0.0), float("nan"))

    if transform is not None:
        mapped_x, mapped_y = apply_transform(rx, ry, transform)
        shift = (float(transform[0]), float(transform[3]))
    else:
        if xin is None or yin is None:
            shift = estimate_shift(rx, ry, ix, iy)
        else:
            shift = (float(xin), float(yin))
        mapped_x, mapped_y = rx + shift[0], ry + shift[1]

    distance = np.hypot(mapped_x[:, None] - ix[None, :],
                        mapped_y[:, None] - iy[None, :])
    nearest = distance.argmin(axis=1)
    keep = distance.min(axis=1) <= tolerance

    # One input point must not be claimed by two reference points; when it is,
    # the closer claim wins. A double match is a real ambiguity in a crowded
    # field, not a detail.
    reference = np.flatnonzero(keep)
    candidate = nearest[keep]
    order = np.argsort(distance[reference, candidate])
    seen: set[int] = set()
    pairs = []
    for index in order:
        target = int(candidate[index])
        if target in seen:
            continue
        seen.add(target)
        pairs.append((int(reference[index]), target))
    pairs.sort()
    if not pairs:
        return Match(np.array([], int), np.array([], int), shift, float("nan"))

    ref_index = np.array([p[0] for p in pairs])
    in_index = np.array([p[1] for p in pairs])
    residual = np.hypot(ix[in_index] - mapped_x[ref_index],
                        iy[in_index] - mapped_y[ref_index])
    return Match(ref_index, in_index, shift, float(np.sqrt(np.mean(residual ** 2))),
                 "tolerance")


def _brightest(flux, size: int, limit: int) -> np.ndarray:
    """Indices of the points that take part in the pattern search."""
    if flux is None:
        return np.arange(min(size, limit))
    order = np.argsort(np.asarray(flux, dtype=float))[::-1]
    return order[:min(size, limit)]


def _triangles(x: np.ndarray, y: np.ndarray, limit: int):
    """Every triangle from the first `limit` points, with its shape.

    A triangle is described by the ratio of its longest side to its shortest
    and by the cosine of the angle between them. Both survive translation,
    rotation and any change of scale, so the same three stars give the same
    two numbers in any frame.
    """
    from itertools import combinations

    count = min(len(x), limit)
    indices = list(combinations(range(count), 3))
    if not indices:
        return np.empty((0, 3), int), np.empty((0, 2))

    shapes = np.empty((len(indices), 2))
    ordered = np.empty((len(indices), 3), int)
    for row, (a, b, c) in enumerate(indices):
        points = np.array([[x[a], y[a]], [x[b], y[b]], [x[c], y[c]]])
        sides = np.array([
            np.hypot(*(points[1] - points[2])),   # opposite vertex 0
            np.hypot(*(points[0] - points[2])),   # opposite vertex 1
            np.hypot(*(points[0] - points[1])),   # opposite vertex 2
        ])
        if sides.min() <= 0:
            shapes[row] = (np.inf, np.inf)
            ordered[row] = (a, b, c)
            continue
        # Sort the vertices by the length of the side opposite them, so the
        # correspondence between two matching triangles is unambiguous.
        order = np.argsort(sides)[::-1]
        vertices = np.array([a, b, c])[order]
        longest, _, shortest = sides[order]
        ratio = longest / shortest
        cosine = (sides[order][1] ** 2 + sides[order][2] ** 2 - longest ** 2) / \
                 (2 * sides[order][1] * sides[order][2])
        shapes[row] = (ratio, cosine)
        ordered[row] = vertices
    return ordered, shapes


def by_triangles(rx, ry, ix, iy, nmatch: int = 30, tolerance: float = 2.0,
                 ratio_tolerance: float = 0.02, votes: int = 2,
                 rflux=None, iflux=None) -> Match:
    """Match two lists with no prior knowledge of the transformation.

    `nmatch` is how many points from each list take part in the pattern
    search. **Which** points matters more than how many: the two subsets have
    to overlap, or the triangles have nothing in common and the vote matches
    noise. Pass `rflux` and `iflux` and the brightest are used, which is the
    one ordering that agrees between two frames of the same field -- it is why
    IRAF's xyxymatch asks for sorted lists and why `daofind` writes its
    catalogue brightest first.

    Without fluxes, the first `nmatch` of each list are used as given; if the
    lists are not already in a comparable order, raise `nmatch` to cover them
    both.
    """
    rx, ry, ix, iy = (np.asarray(a, dtype=float) for a in (rx, ry, ix, iy))
    empty = np.array([], int)
    if min(rx.size, ix.size) < 3:
        return Match(empty, empty, (0.0, 0.0), float("nan"), "triangles",
                     "fewer than three points in one of the lists")

    ref_pick = _brightest(rflux, rx.size, nmatch)
    in_pick = _brightest(iflux, ix.size, nmatch)
    ref_tri, ref_shape = _triangles(rx[ref_pick], ry[ref_pick], len(ref_pick))
    in_tri, in_shape = _triangles(ix[in_pick], iy[in_pick], len(in_pick))
    if len(ref_tri) == 0 or len(in_tri) == 0:
        return Match(empty, empty, (0.0, 0.0), float("nan"), "triangles",
                     "no triangles could be formed")

    ballot: dict[tuple[int, int], int] = {}
    for row in range(len(ref_tri)):
        ratio, cosine = ref_shape[row]
        if not np.isfinite(ratio):
            continue
        close = (np.abs(in_shape[:, 0] - ratio) < ratio_tolerance * ratio) & \
                (np.abs(in_shape[:, 1] - cosine) < ratio_tolerance)
        for other in np.flatnonzero(close):
            for a, b in zip(ref_tri[row], in_tri[other]):
                key = (int(ref_pick[a]), int(in_pick[b]))
                ballot[key] = ballot.get(key, 0) + 1

    elected = [(a, b) for (a, b), count in ballot.items() if count >= votes]
    if len(elected) < 3:
        return Match(empty, empty, (0.0, 0.0), float("nan"), "triangles",
                     f"only {len(elected)} point pairs won {votes} votes or more; "
                     "the two lists may not overlap, or nmatch is too small to "
                     "cover the same stars in both")

    # A point may win votes for more than one partner; keep its best.
    best: dict[int, tuple[int, int]] = {}
    for a, b in elected:
        count = ballot[(a, b)]
        if a not in best or count > best[a][1]:
            best[a] = (b, count)
    ref_index = np.array(sorted(best))
    in_index = np.array([best[a][0] for a in ref_index])

    # The vote gives a handful of pairs, some of them wrong. Fit the transform
    # with rejection, use it to match everything, refit, match again. Two
    # passes recover the corners of the field, which a translation-only match
    # always loses while looking successful.
    # The vote alone is never the answer: a set of pairs with no transform
    # behind it is a guess, and returning it would be a silent wrong result.
    # Only a match that produced a transform within tolerance is returned.
    best = Match(empty, empty, (0.0, 0.0), float("nan"), "triangles",
                 "the voted pairs did not fit a consistent transform")
    for _ in range(3):
        if ref_index.size < 3:
            break
        model = fit_transform(rx[ref_index], ry[ref_index], ix[in_index], iy[in_index],
                              niterate=3, reject=2.5)
        if model["rms"] > max(tolerance, 1.0):
            break
        candidate = by_tolerance(rx, ry, ix, iy, tolerance=tolerance,
                                 transform=model["coefficients"])
        candidate.method = "triangles"
        candidate.shift = (model["xshift"], model["yshift"])
        candidate.rms = model["rms"]
        if len(candidate) <= len(best) and best:
            break
        best = candidate
        ref_index, in_index = candidate.reference, candidate.input
    return best


def fit_transform(rx, ry, ix, iy, niterate: int = 0, reject: float = 3.0) -> dict:
    """Least-squares linear transform taking reference coordinates to input.

        x' = a + b*x + c*y
        y' = d + e*x + f*y

    Returned with the shift, scale and rotation read out of it, because those
    are what a person wants to see -- "rotated by 0.03 degrees" says more than
    six coefficients.

    With `niterate` above zero, pairs further than `reject` sigma from the fit
    are dropped and the fit repeated. Least squares has no defence against a
    single wrong pair, and a mismatched star a hundred pixels away will bend
    the whole solution towards itself while still returning a number.
    """
    rx, ry, ix, iy = (np.asarray(a, dtype=float) for a in (rx, ry, ix, iy))
    if rx.size < 3:
        raise ValueError(f"a linear transform needs at least 3 pairs; got {rx.size}")

    keep = np.ones(rx.shape, dtype=bool)
    for _ in range(max(0, int(niterate)) + 1):
        design = np.column_stack([np.ones_like(rx), rx, ry])[keep]
        coefficients_x, *_ = np.linalg.lstsq(design, ix[keep], rcond=None)
        coefficients_y, *_ = np.linalg.lstsq(design, iy[keep], rcond=None)
        full = np.column_stack([np.ones_like(rx), rx, ry])
        residual = np.hypot(ix - full @ coefficients_x, iy - full @ coefficients_y)
        if niterate <= 0:
            break

        # A robust sigma, from the median absolute deviation rather than the
        # standard deviation. With the plain standard deviation each pass
        # shrinks sigma, which rejects more, which shrinks it again -- on the
        # real M92 frames that cascade threw away 40% of good pairs and
        # returned an rms of 0.05 px, far better than centroids can actually
        # be, which is how it announced itself.
        centre = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - centre))
        sigma = 1.4826 * mad
        if sigma <= 1e-9:
            break
        drop = keep & (residual > centre + reject * sigma)
        if not drop.any() or (keep & ~drop).sum() < 3:
            break
        keep &= ~drop

    a, b, c = coefficients_x
    d, e, f = coefficients_y
    return {
        "used": keep,
        "xshift": float(a), "yshift": float(d),
        "xscale": float(np.hypot(b, e)), "yscale": float(np.hypot(c, f)),
        "rotation": float(np.degrees(np.arctan2(e, b))),
        "coefficients": (float(a), float(b), float(c), float(d), float(e), float(f)),
        "rms": float(np.sqrt(np.mean(residual[keep] ** 2))),
        "npairs": int(keep.sum()),
        "nrejected": int((~keep).sum()),
    }
