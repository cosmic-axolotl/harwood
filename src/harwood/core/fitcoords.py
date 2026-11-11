"""twodspec.longslit's fitcoords without the task: noao/twodspec/longslit/lstrans
(fcgetcoords.x, fcfitcoords.x, fcdbio.x, fcdlist.x) and the igsfit solutions,
as IRAF 2.18 runs them.

identify and reidentify leave, for every line or column they measured of a long
slit arc, the pixel where each lamp line fell and its wavelength. fitcoords
gathers them all and fits one surface, the wavelength as a function of the
column and line, which transform then inverts to resample the frame onto
straight lines of constant wavelength. The surface is gsurfit's (core/gsurfit.py),
in double precision; the measurements are kept in real, as IRAF keeps them.

What follows from the code, kept:

* a feature is used when its user coordinate is not INDEF and its weight -- the
  sixth column, or with five the last there is -- is above zero; the positions
  IRAF wrote are physical pixels, turned to the frame's logical ones with its
  LTV and LTM (a harwood database already holds logical ones);
* the features are sorted by their user coordinate with IRAF's quicksort and
  numbered, equal coordinates the same feature;
* the surface is fitted in two passes: one of orders min(2, xorder) and
  min(2, yorder) without cross terms, then all the orders with them on what the
  first leaves, the two added; with a single feature, z = y + f(x) (or x + f(y));
* a point on the deletion list closer than 10 epsilon to a feature is fitted
  with weight zero;
* the log's second corner is evaluated at (xmax, xmin), not (xmax, ymin).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import gsurfit, iraffmt
from . import units as unitsmod
from .sensfunc import qsort

f32 = np.float32
EPSILONR = float(np.finfo(np.float32).eps)
INDEFR = float(f32(1.6e38))
#: The columns of the points, igsfit.h's order.
FEATURE, X, Y, Z, W, S, R = range(7)
FUNCTIONS = ("chebyshev", "legendre")


class FitcoordsError(ValueError):
    """What fitcoords stops on, in IRAF's words."""


@dataclass
class Record:
    """One identify record: its image (with the section), units, and features as
    (pixel, user, weight), each None where IRAF reads INDEF."""
    task: str
    image: str
    units: str | None
    features: list[tuple]
    physical: bool = True           #: IRAF's positions; harwood's are logical


@dataclass
class Coords:
    """fc_getcoords' result."""
    axis: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    points: np.ndarray              #: (7, n) real: feature, x, y, z, w, s, r
    units: str | None
    warnings: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.points.shape[1]


# -------------------------------------------------------------- the database

def _real(text: str):
    """gargr: a real, or None for INDEF; ValueError when it is not a number."""
    if text.upper().startswith("INDEF"):
        return None
    return float(f32(float(text)))


def read_iraf(path: Path) -> list[Record]:
    """dtmap's records of an IRAF text database: each `begin` block's task, image,
    units and the rows after `features`, read as fc_getcoords reads them."""
    records = []
    current = None
    reading = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if reading > 0:
            reading -= 1
            current["rows"].append(stripped.split())
            continue
        if not stripped or stripped.startswith("#"):
            continue
        words = stripped.split(None, 1)
        key = words[0]
        value = words[1].strip() if len(words) > 1 else ""
        if key == "begin":
            current = {"rows": []}
            records.append(current)
            continue
        if current is None:
            continue
        current.setdefault(key, value)
        if key == "features":
            reading = int(value.split()[0])
    out = []
    for rec in records:
        if "task" not in rec:
            continue
        features = []
        for row in rec["rows"]:
            values = []
            for text in row[:6]:
                try:
                    values.append(_real(text))
                except ValueError:
                    break
            if not values:
                continue
            pixel = values[0]
            user = values[2] if len(values) > 2 else None
            # gargr leaves its variable as it was when the row has run out: the
            # weight is the last of the width, type and weight columns there is.
            weight = values[min(len(values), 6) - 1] if len(values) > 3 else user
            features.append((pixel, user, weight))
        out.append(Record(rec.get("task", ""), rec.get("image", ""), rec.get("units"),
                          features, physical=True))
    return out


def read_records(path: Path) -> list[Record]:
    """IRAF's database file, or harwood's: a harwood record's feature is used or
    not, and its positions are logical already."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"^begin\s", text, re.M):
        return read_iraf(path)
    from . import database

    out = []
    for sol in database._read_harwood_all(path, text):
        used = np.ones(len(sol), bool) if sol.used is None else np.asarray(sol.used, bool)
        rows = []
        for p, w, u in zip(sol.pixel, sol.wavelength, used, strict=True):
            user = None if (w is None or not np.isfinite(w)) else float(f32(w))
            rows.append((float(f32(p)), user, 1.0 if u else 0.0))
        out.append(Record("identify", f"{database._root(sol.image)}{sol.section}",
                          sol.units or None, rows, physical=False))
    return out


def get_root(image: str) -> str:
    """get_root: the image's name without its section."""
    return image.split("[", 1)[0]


def strip_extension(name: str) -> str:
    """fc_getim takes off .imh and .hhh; harwood also .fits and .fit, so that
    arc.fits and arc are one image, as harwood names them everywhere."""
    for suffix in (".imh", ".hhh", ".fits", ".fit"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _section(image: str, shape: tuple[int, int]) -> tuple[int, int]:
    """IM_VMAP and IM_VOFF of a 1-D section of the frame: (axis, index) -- [N,*]
    is column N along axis 2, [*,N] line N along axis 1."""
    m = re.search(r"\[\s*([^,\]]+)\s*,\s*([^\]]+?)\s*\]\s*$", image)
    if m is None:
        raise FitcoordsError(f"{image}: not a line or a column of a 2-D image")
    a, b = m.group(1), m.group(2)
    if a == "*" and re.fullmatch(r"\d+", b):
        return 1, int(b)
    if b == "*" and re.fullmatch(r"\d+", a):
        return 2, int(a)
    raise FitcoordsError(f"{image}: only [N,*] and [*,N] sections are read")


def _ltm(header) -> tuple[np.ndarray, np.ndarray]:
    """mw_gltermr with fc_getcoords' corrections: a transposed frame's matrix
    turned back to diagonal, any other rotation dropped."""
    ltm = np.array([[float(header.get("LTM1_1", 1.0)), float(header.get("LTM1_2", 0.0))],
                    [float(header.get("LTM2_1", 0.0)), float(header.get("LTM2_2", 1.0))]])
    ltv = np.array([float(header.get("LTV1", 0.0)), float(header.get("LTV2", 0.0))])
    if ltm[0, 0] == 0.0 and ltm[1, 1] == 0.0:
        ltm = np.array([[ltm[1, 0], 0.0], [0.0, ltm[0, 1]]])
    elif ltm[0, 1] != 0.0 or ltm[1, 0] != 0.0:
        ltm = np.eye(2)
        ltv = np.zeros(2)
    return ltm, ltv


def get_coords(images: list[str], database_dir: Path, open_header) -> Coords:
    """fc_getcoords: the features of every image's records, sorted and numbered.
    open_header(root) gives the frame's header and its shape (ny, nx)."""
    xs, ys, users = [], [], []
    axis = 0
    xmin = xmax = ymin = ymax = None
    units = None
    ntotal = 0
    warnings = []
    for image1 in images:
        path = database_dir / f"id{image1}"
        if not path.is_file():
            raise FitcoordsError(f"Cannot open database ({path})")
        for rec in read_records(path):
            if rec.task != "identify":
                continue
            if get_root(rec.image) != image1:
                continue
            header, shape = open_header(get_root(rec.image))
            j, index = _section(rec.image, shape)
            xmin, xmax = 1.0, float(shape[1])
            ymin, ymax = 1.0, float(shape[0])
            if axis == 0:
                axis = j
            if j != axis:
                warnings.append("Warning: Fit axes don't agree for combine option.  "
                                f"Ignoring {image1}.")
                break
            ltm, ltv = _ltm(header) if rec.physical else (np.eye(2), np.zeros(2))
            if rec.units is not None:
                units = unitsmod.decode(rec.units).units
            for pixel, user, weight in rec.features:
                # an INDEF position goes through the matrix as the number it is
                p = INDEFR if pixel is None else pixel
                position = float(f32(ltm[j - 1, j - 1] * p + ltv[j - 1]))
                if user is not None and weight is not None and weight > 0.0:
                    if axis == 1:
                        xs.append(position)
                        ys.append(float(index))
                    else:
                        xs.append(float(index))
                        ys.append(position)
                    users.append(user)
                ntotal += 1
    if not users:
        if ntotal == 0:
            raise FitcoordsError("No coordinates found in database")
        raise FitcoordsError("Only INDEF coordinates found in database")
    order = qsort([f32(u) for u in users])
    n = len(users)
    pts = np.zeros((7, n), dtype=f32)
    pts[X] = np.array(xs, f32)[order]
    pts[Y] = np.array(ys, f32)[order]
    pts[Z] = np.array(users, f32)[order]
    pts[W] = f32(1.0)
    pts[FEATURE, 0] = f32(1.0)
    for i in range(1, n):
        pts[FEATURE, i] = pts[FEATURE, i - 1] + (f32(1.0) if pts[Z, i] != pts[Z, i - 1]
                                                 else f32(0.0))
    return Coords(axis, xmin, xmax, ymin, ymax, pts, units, warnings)


def read_deletions(path: Path | None, coords: Coords) -> None:
    """fc_dlread: the points of the list closer than 10 epsilon to a feature's
    (user coordinate, position across) get weight zero."""
    if path is None or not Path(path).is_file():
        return
    pts = coords.points
    other = pts[Y] if coords.axis == 1 else pts[X]
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        if len(words) < 2:
            continue
        try:
            xdel, ydel = f32(float(words[0])), f32(float(words[1]))
        except ValueError:
            continue
        dx = (pts[Z] - xdel).astype(f32)
        dy = (other - ydel).astype(f32)
        r = np.sqrt((dx * dx + dy * dy).astype(f32)).astype(f32)
        pts[W][r < f32(10 * EPSILONR)] = f32(0.0)


def write_deletions(path: Path, coords: Coords) -> None:
    """fc_dlwrite: the points of weight zero, user coordinate and position across."""
    pts = coords.points
    other = pts[Y] if coords.axis == 1 else pts[X]
    lines = [f"{iraffmt.g(float(pts[Z, i]))} {iraffmt.g(float(other[i]))}"
             for i in range(coords.n) if pts[W, i] == 0.0]
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")


# ------------------------------------------------------------------- the fit

def _xgsfit(sf, x, y, z, w):
    status = gsurfit.fit(sf, np.asarray(x, np.float64), np.asarray(y, np.float64),
                         np.asarray(z, np.float64), np.asarray(w, np.float64))
    if status == gsurfit.NO_DEG_FREEDOM:
        raise FitcoordsError("No degrees of freedom")
    return ["Singular solution"] if status == gsurfit.SINGULAR else []


def _xgsvector(sf, x, y) -> np.ndarray:
    return gsurfit.vector(sf, np.asarray(x, np.float64),
                          np.asarray(y, np.float64)).astype(f32)


def solve1(function, xorder, yorder, limits, x, y, z, w):
    """igs_solve1: z = f(x, y), a low-order surface first and the full one fitted
    to what it leaves."""
    nfunc = gsurfit.NAMES[function]
    said = []
    sf1 = gsurfit.init(nfunc, min(2, xorder), min(2, yorder), gsurfit.XNONE, *limits)
    said += _xgsfit(sf1, x, y, z, w)
    resid = (np.asarray(z, f32) - _xgsvector(sf1, x, y)).astype(f32)
    sf2 = gsurfit.init(nfunc, xorder, yorder, gsurfit.XFULL, *limits)
    said += _xgsfit(sf2, x, y, resid, w)
    return gsurfit.add(sf1, sf2), said


def solve2(function, xorder, yorder, limits, x, y, z, w):
    """igs_solve2: z = x + f(y)."""
    nfunc = gsurfit.NAMES[function]
    xmin, xmax, ymin, ymax = limits
    sf1 = gsurfit.init(nfunc, 1, yorder, gsurfit.XNONE, *limits)
    said = _xgsfit(sf1, x, y, (np.asarray(z, f32) - np.asarray(x, f32)).astype(f32), w)
    sf = gsurfit.init(nfunc, 2, yorder, gsurfit.XNONE, *limits)
    a = f32(gsurfit.gcoeff(sf1, 1, 1))
    a = f32(a + f32(f32(f32(xmin) + f32(xmax)) / f32(2)))
    gsurfit.scoeff(sf, 1, 1, float(a))
    gsurfit.scoeff(sf, 2, 1, float(f32(f32(f32(xmax) - f32(xmin)) / f32(2))))
    for i in range(2, yorder + 1):
        gsurfit.scoeff(sf, 1, i, float(f32(gsurfit.gcoeff(sf1, 1, i))))
    return sf, said


def solve3(function, xorder, yorder, limits, x, y, z, w):
    """igs_solve3: z = y + f(x)."""
    nfunc = gsurfit.NAMES[function]
    xmin, xmax, ymin, ymax = limits
    sf1 = gsurfit.init(nfunc, xorder, 1, gsurfit.XNONE, *limits)
    said = _xgsfit(sf1, x, y, (np.asarray(z, f32) - np.asarray(y, f32)).astype(f32), w)
    sf = gsurfit.init(nfunc, xorder, 2, gsurfit.XNONE, *limits)
    a = f32(gsurfit.gcoeff(sf1, 1, 1))
    a = f32(a + f32(f32(f32(ymin) + f32(ymax)) / f32(2)))
    gsurfit.scoeff(sf, 1, 1, float(a))
    gsurfit.scoeff(sf, 1, 2, float(f32(f32(f32(ymax) - f32(ymin)) / f32(2))))
    for i in range(2, xorder + 1):
        gsurfit.scoeff(sf, i, 1, float(f32(gsurfit.gcoeff(sf1, i, 1))))
    return sf, said


def solver(coords: Coords):
    """fc_fitcoords' choice: igs_fit1 unless every point is one feature, then the
    surface that is the position across plus a function of the other."""
    single = coords.points[FEATURE, -1] == 1
    if coords.axis == 1:
        return solve2 if single else solve1
    return solve3 if single else solve1


def fit(coords: Coords, function: str, xorder: int, yorder: int):
    """igs_fit's solution and its surface and residual columns."""
    pts = coords.points
    limits = (float(f32(coords.xmin)), float(f32(coords.xmax)),
              float(f32(coords.ymin)), float(f32(coords.ymax)))
    sf, said = solver(coords)(function, xorder, yorder, limits, pts[X], pts[Y], pts[Z],
                              pts[W])
    pts[S] = _xgsvector(sf, pts[X], pts[Y])
    pts[R] = (pts[Z] - pts[S]).astype(f32)
    return sf, said


def xgseval(sf, x, y) -> float:
    """xgseval: the point and the value in real."""
    return float(f32(gsurfit.evaluate(sf, float(f32(x)), float(f32(y)))))


# ---------------------------------------------------------------- the logs

def header_lines(sysid: str, fitname: str, database: str, images: list[str]) -> list[str]:
    return ["", sysid, f"  Longslit coordinate fit name is {fitname}.",
            f"  Longslit database is {database}.", "  Features from images:",
            *[f"    {name}" for name in images]]


def _d(value: float) -> str:
    """%d of a real: the integer IRAF prints for it."""
    return str(int(round(value)))


def result_lines(coords: Coords, sf, function: str, xorder: int, yorder: int) -> list[str]:
    xmin, xmax, ymin, ymax = coords.xmin, coords.xmax, coords.ymin, coords.ymax
    g = iraffmt.g
    return [
        f"  Map User coordinates for axis {coords.axis} using image features:",
        f"  Number of feature coordnates = {coords.n}",
        f"  Mapping function = {function}",
        f"    X order = {xorder}", f"    Y order = {yorder}",
        "  Fitted coordinates at the corners of the images:",
        f"    ({_d(xmin)}, {_d(ymin)}) = {g(xgseval(sf, xmin, ymin))}  "
        f"({_d(xmax)}, {_d(ymin)}) = {g(xgseval(sf, xmax, xmin))}",
        f"    ({_d(xmin)}, {_d(ymax)}) = {g(xgseval(sf, xmin, ymax))}  "
        f"({_d(xmax)}, {_d(ymax)}) = {g(xgseval(sf, xmax, ymax))}",
    ]


# ------------------------------------------------------- the fc database

def fc_path(database_dir: Path, fitname: str) -> Path:
    return Path(database_dir) / f"fc{fitname.split('[', 1)[0]}"


def write_fc(database_dir: Path, fitname: str, axis: int, units: str | None, sf,
             when: float | None = None) -> Path:
    """fc_dbwrite: appended to database/fc<fitname>, the surface saved as gssave
    lays it out, each number as IRAF's %g of a double."""
    path = fc_path(database_dir, fitname)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {iraffmt.cnvtime(when)}", f"begin\t{fitname}", "\ttask\tfitcoords",
             f"\taxis\t{axis}"]
    if units is not None:
        lines.append(f"\tunits\t{units}")
    saved = gsurfit.save(sf)
    lines.append(f"\tsurface\t{len(saved)}")
    lines += [f"\t\t{iraffmt.g(v, double=True)}" for v in saved]
    with open(path, "a", encoding="utf-8") as out:
        out.write("".join(line + "\n" for line in lines))
    return path


@dataclass
class Fit:
    """lm_dbread: the axis, the units and the surface of one fc record."""
    name: str
    axis: int
    units: str | None
    surface: gsurfit.Surface


def read_fc(database_dir: Path, fitname: str) -> Fit:
    """lm_dbread: database/fc<fitname>'s record of that name -- the last one, as
    dtlocate finds it -- read with ctod."""
    path = fc_path(database_dir, fitname)
    if not path.is_file():
        raise FitcoordsError(f"Cannot open database ({path})")
    blocks: list[list[str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("begin"):
            blocks.append([stripped])
        elif blocks and stripped and not stripped.startswith("#"):
            blocks[-1].append(stripped)
    found = [b for b in blocks if b[0].split(None, 1)[1:] == [fitname]]
    if not found:
        raise FitcoordsError(f"Database record not found ({fitname})")
    block = found[-1]
    fields: dict[str, str] = {}
    values: list[float] = []
    reading = 0
    for stripped in block[1:]:
        if reading > 0:
            values.append(float(iraffmt.ctod(stripped.split()[0])))
            reading -= 1
            continue
        words = stripped.split(None, 1)
        fields.setdefault(words[0], words[1] if len(words) > 1 else "")
        if words[0] == "surface":
            reading = int(words[1].split()[0])
    if "axis" not in fields:
        raise FitcoordsError("Database field not found (axis)")
    sf = gsurfit.restore(values)
    return Fit(fitname, int(fields["axis"]), fields.get("units"), sf)


def is_finite(value: float) -> bool:
    return value is not None and math.isfinite(value)
