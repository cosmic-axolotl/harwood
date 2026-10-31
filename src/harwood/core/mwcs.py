"""An image's logical, physical and world coordinates, as mw_ctrand gives them.

Logical is the pixel of the image or section as read; physical undoes the
section (LTV, LTM: logical = LTM physical + LTV); world is the header's WCS.
A FITS image's CRPIX and CD are logical in IRAF, so world goes straight from
logical, and core.pixels.section_header has already moved them with a section.
An axis the header says nothing about has no world: IRAF gives its physical.

From iraf-community 2.18: sys/mwcs. See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import re

import numpy as np

from . import advice

SYSTEMS = ("logical", "physical", "world")
RADIAN = float(np.float32(57.295779513082320877))


def transform(header, system: str, logical: np.ndarray) -> np.ndarray:
    """logical[axis, n], 1-based, in `system`, as a new array."""
    logical = np.asarray(logical, dtype=np.float64)
    if system == "logical":
        return logical.copy()
    if system == "physical":
        return _physical(header, logical)
    if system == "world":
        return _world(header, logical)
    raise ValueError(f"no coordinate system '{system}'")


def _physical(header, logical: np.ndarray) -> np.ndarray:
    n = logical.shape[0]
    ltm = np.eye(n)
    for i in range(n):
        for j in range(n):
            ltm[i, j] = float(header.get(f"LTM{i + 1}_{j + 1}", ltm[i, j]))
    ltv = np.array([float(header.get(f"LTV{i + 1}", 0.0)) for i in range(n)])
    offset = logical - ltv[:, None]
    if np.array_equal(ltm, np.diag(np.diag(ltm))):
        return offset / np.diag(ltm)[:, None]
    return np.linalg.solve(ltm, offset)


def _has_world(header, axis: int, n: int) -> bool:
    return (any(f"{key}{axis}" in header for key in ("CRVAL", "CRPIX", "CDELT", "CTYPE"))
            or any(f"CD{axis}_{j}" in header for j in range(1, n + 1)))


def _world(header, logical: np.ndarray) -> np.ndarray:
    if "multispec" in str(header.get("WAT0_001", "")):
        advice.fail("the world system of a multispec image is not read here",
                    "use --wcs physical, or take the wavelengths from `wspectext`")
    n = logical.shape[0]
    types = [str(header.get(f"CTYPE{i + 1}", "")).upper() for i in range(n)]
    out = _physical(header, logical)
    projected = [i for i, ctype in enumerate(types) if ctype.endswith("-TAN")]
    if any(re.search(r"-[A-Z0-9]{3}$", ctype) for ctype in types) and len(projected) != 2:
        return _celestial(header, logical)
    cd = _matrix(header, n)
    crpix = np.array([float(header.get(f"CRPIX{i + 1}", 0.0)) for i in range(n)])
    crval = np.array([float(header.get(f"CRVAL{i + 1}", 0.0)) for i in range(n)])
    # mw_sctran's order: IRAF keeps these in physical pixels, and its compiled
    # transform is M times the pixel, plus LTV; the last digits depend on it.
    ltm = np.array([[float(header.get(f"LTM{i + 1}_{j + 1}", float(i == j))) for j in range(n)]
                    for i in range(n)])
    ltv = np.array([float(header.get(f"LTV{i + 1}", 0.0)) for i in range(n)])
    inverse = np.linalg.inv(ltm)
    matrix = cd @ ltm
    moved = (matrix @ inverse) @ logical
    shift = -(matrix @ (inverse @ (crpix - ltv))) - (matrix @ (inverse @ ltv))
    for axis in range(n):
        if axis not in projected and _has_world(header, axis + 1, n):
            out[axis] = moved[axis] + (crval[axis] + shift[axis])
    if projected:
        ira, idec = projected if not types[projected[0]].startswith("DE") else projected[::-1]
        out[ira], out[idec] = _tan(moved[ira] + shift[ira], moved[idec] + shift[idec],
                                   crval[ira], crval[idec])
    return out


def _matrix(header, n: int) -> np.ndarray:
    """The CD matrix, or with none in the header the diagonal of CDELT."""
    if any(f"CD{i + 1}_{j + 1}" in header for i in range(n) for j in range(n)):
        return np.array([[float(header.get(f"CD{i + 1}_{j + 1}", 0.0)) for j in range(n)]
                         for i in range(n)])
    return np.diag([float(header.get(f"CDELT{i + 1}", 1.0)) for i in range(n)])


def _tan(xi, eta, ra0: float, dec0: float):
    """wf_tan_fwd, sys/mwcs/wftan.x; xi and eta in degrees. Its RADTODEG and
    DEGTORAD divide by math.h's RADIAN, a real: 1.2e-8 off, which shows at the
    tenth decimal of an RA."""
    xi, eta = xi / RADIAN, eta / RADIAN
    cosdec, sindec = np.cos(dec0 / RADIAN), np.sin(dec0 / RADIAN)
    x = cosdec - eta * sindec
    z = sindec + eta * cosdec
    ra = np.arctan2(xi, x) * RADIAN + ra0
    dec = np.arctan2(z, np.sqrt(x * x + xi * xi)) * RADIAN
    return np.where(ra < 0, ra + 360.0, np.where(ra > 360.0, ra - 360.0, ra)), dec


def _celestial(header, logical: np.ndarray) -> np.ndarray:
    from astropy.wcs import WCS

    wcs = WCS(header, naxis=logical.shape[0], relax=True)
    return wcs.wcs_pix2world(logical.T, 1).T
