"""An image's logical, physical and world coordinates, as mw_ctrand gives them.

Logical is the pixel of the image or section as read; physical undoes the
section (LTV, LTM: logical = LTM physical + LTV); world is the header's WCS.
A FITS image's CRPIX and CD are logical in IRAF, so world goes straight from
logical, and core.pixels.section_header has already moved them with a section.
An axis the header says nothing about has no world: IRAF gives its physical.

From iraf-community 2.18: sys/mwcs. See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
import re

import numpy as np

from . import advice, iraffmt

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


def logical_of_physical(header, physical: np.ndarray) -> np.ndarray:
    """physical[axis] in logical: LTM times it, plus LTV."""
    n = physical.shape[0]
    ltm = np.array([[_number(header, f"LTM{i + 1}_{j + 1}", float(i == j)) for j in range(n)]
                    for i in range(n)])
    ltv = np.array([_number(header, f"LTV{i + 1}", 0.0) for i in range(n)])
    return _vmul(ltm, physical) + ltv


def logical_of(header, world: np.ndarray) -> np.ndarray:
    """world[axis] in logical, the other way from transform; for whole-pixel offsets,
    so it is not held to IRAF's last digits. With no WCS, world is physical."""
    world = np.asarray(world, dtype=np.float64)
    n = world.shape[0]
    if not any(_has_world(header, axis + 1, n) for axis in range(n)):
        return logical_of_physical(header, world)
    types = [str(header.get(f"CTYPE{i + 1}", "")).upper() for i in range(n)]
    projected = [i for i, ctype in enumerate(types) if ctype.endswith("-TAN")]
    if any(re.search(r"-[A-Z0-9]{3}$", ctype) for ctype in types) and len(projected) != 2:
        from astropy.wcs import WCS

        return WCS(header, naxis=n, relax=True).wcs_world2pix(world[None, :], 1)[0]
    crval = np.array([_number(header, f"CRVAL{i + 1}", 0.0) for i in range(n)])
    crpix = np.array([_number(header, f"CRPIX{i + 1}", 0.0) for i in range(n)])
    plane = world - crval
    if projected:
        ira, idec = projected if not types[projected[0]].startswith("DE") else projected[::-1]
        ra, dec = math.radians(world[ira] - crval[ira]), math.radians(world[idec])
        dec0 = math.radians(crval[idec])
        z = math.sin(dec) * math.sin(dec0) + math.cos(dec) * math.cos(dec0) * math.cos(ra)
        plane[ira] = math.degrees(math.cos(dec) * math.sin(ra) / z)
        plane[idec] = math.degrees((math.sin(dec) * math.cos(dec0)
                                    - math.cos(dec) * math.sin(dec0) * math.cos(ra)) / z)
    return np.linalg.solve(_matrix(header, n), plane) + crpix


def _number(header, key: str, default: float) -> float:
    """A number of the header as IRAF has it: ctod of what the card says, whose
    digits are worked out in steps and can end a unit from the nearest double. A
    card a section has changed is not in the file, and has its value as it is."""
    if key not in header:
        return default
    card = header.cards[key]
    if getattr(card, "_modified", True):
        return float(header[key])
    image = card.image
    value = iraffmt.ctod(image[10:].split("/", 1)[0].strip()) if len(image) > 10 else None
    return float(header[key]) if value is None else value


def _physical(header, logical: np.ndarray) -> np.ndarray:
    n = logical.shape[0]
    ltm = np.eye(n)
    for i in range(n):
        for j in range(n):
            ltm[i, j] = _number(header, f"LTM{i + 1}_{j + 1}", ltm[i, j])
    ltv = np.array([_number(header, f"LTV{i + 1}", 0.0) for i in range(n)])
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
    crpix = np.array([_number(header, f"CRPIX{i + 1}", 0.0) for i in range(n)])
    crval = np.array([_number(header, f"CRVAL{i + 1}", 0.0) for i in range(n)])
    # mw_sctran's order: IRAF keeps these in physical pixels, and its compiled
    # transform is M times the pixel, plus LTV; the last digits depend on it.
    ltm = np.array([[_number(header, f"LTM{i + 1}_{j + 1}", float(i == j)) for j in range(n)]
                    for i in range(n)])
    ltv = np.array([_number(header, f"LTV{i + 1}", 0.0) for i in range(n)])
    inverse = np.linalg.inv(ltm)
    matrix = _mmul(cd, ltm)
    moved = _vmul(_mmul(matrix, inverse), logical)
    reference = _vmul(inverse, crpix - ltv)
    for i in range(n):
        card = header.cards[f"CRPIX{i + 1}"] if f"CRPIX{i + 1}" in header else None
        if card is not None and getattr(card, "_modified", False):
            # A section moved it; IRAF still has the file's, through ctod.
            reference[i] = iraffmt.ctod(repr(float(reference[i])))
    # mw_sctran: LTV = W - CD*R, and an axis with a function undoes the +W again, which
    # rounds; then the input side, CD*(-inv(LTM)*LTV).
    shift = crval + _vmul(matrix, -reference)
    for axis in projected:
        shift[axis] = shift[axis] - crval[axis]
    shift = shift + _vmul(matrix, -_vmul(inverse, ltv))
    for axis in range(n):
        if axis not in projected and _has_world(header, axis + 1, n):
            out[axis] = moved[axis] + shift[axis]
    if projected:
        ira, idec = projected if not types[projected[0]].startswith("DE") else projected[::-1]
        # mw_ltran, for the general case: from the translation, then term by term
        full = _mmul(matrix, inverse)
        plane = shift[:, None] + full[:, 0, None] * logical[0]
        for j in range(1, n):
            plane = plane + full[:, j, None] * logical[j]
        out[ira], out[idec] = _tan(plane[ira], plane[idec], crval[ira], crval[idec])
    return out


def _vmul(matrix, vector) -> np.ndarray:
    """mw_vmuld: the sums term by term, in order. `@` may go through BLAS, whose fused
    multiply and add round differently, by a unit in the last digit."""
    out = matrix[:, 0] * vector[0] if vector.ndim == 1 else None
    if out is None:
        out = matrix[:, 0, None] * vector[0]
    out = out.copy()
    for j in range(1, matrix.shape[1]):
        if vector.ndim == 1:
            out = out + matrix[:, j] * vector[j]
        else:
            out = out + matrix[:, j, None] * vector[j]
    return out


def _mmul(a, b) -> np.ndarray:
    """mw_mmuld, as mw_vmuld."""
    return np.stack([_vmul(a, b[:, j]) for j in range(b.shape[1])], axis=1)


def _matrix(header, n: int) -> np.ndarray:
    """The CD matrix, or with none in the header the diagonal of CDELT."""
    if any(f"CD{i + 1}_{j + 1}" in header for i in range(n) for j in range(n)):
        return np.array([[_number(header, f"CD{i + 1}_{j + 1}", 0.0) for j in range(n)]
                         for i in range(n)])
    return np.diag([_number(header, f"CDELT{i + 1}", 1.0) for i in range(n)])


def _tan(xi, eta, ra0: float, dec0: float):
    """wf_tan_fwd, sys/mwcs/wftan.x; xi and eta in degrees. Its RADTODEG and
    DEGTORAD divide by math.h's RADIAN, a real: 1.2e-8 off, which shows at the
    tenth decimal of an RA."""
    xi, eta = xi / RADIAN, eta / RADIAN
    # libm's, as IRAF's: numpy's own are within a unit in the last place of them
    cosdec, sindec = math.cos(dec0 / RADIAN), math.sin(dec0 / RADIAN)
    x = cosdec - eta * sindec
    z = sindec + eta * cosdec
    atan2 = np.frompyfunc(math.atan2, 2, 1)
    ra = atan2(xi, x).astype(float) * RADIAN + ra0
    dec = atan2(z, np.sqrt(x * x + xi * xi)).astype(float) * RADIAN
    return np.where(ra < 0, ra + 360.0, np.where(ra > 360.0, ra - 360.0, ra)), dec


def _celestial(header, logical: np.ndarray) -> np.ndarray:
    from astropy.wcs import WCS

    wcs = WCS(header, naxis=logical.shape[0], relax=True)
    return wcs.wcs_pix2world(logical.T, 1).T
