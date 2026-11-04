"""Extracted spectra written as IRAF writes them: multispec, echelle, onedspec, strip.

harwood keeps spectra as a table per aperture (core/spectrum.py). These are for
taking the work to IRAF, PyRAF's splot or anything else that reads IRAF's
spectra: apextract.x's ap_output and apmw.x, ported. A multispec or echelle
image holds one line per aperture (or part of one) and, with the extras, a
plane per band behind the spectrum -- raw, background, sigma -- named by
BANDIDn; APNUMn gives each line's aperture, beam and limits across the slit;
the world coordinates are IRAF's equispec system, pixels along the dispersion
until dispcor gives them wavelengths. onedspec writes each aperture to an image
of its own, root.0001; strip, the aperture's pixels shifted along the trace so
that its lower edge is the first line, one 2-D image per aperture.

Every card of the frame's header comes along, as IRAF's NEW_COPY brings it,
less DISPAXIS. Its world coordinates are rewritten as apmw.x writes them: the
dispersion axis's (pixels of a trimmed frame counted as the untrimmed frame
counts them, or its own linear WCS) becomes the spectrum's first axis, cards
the frame already has keep their place, the rest come after the APNUMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10
from pathlib import Path

import numpy as np

FORMATS = ("multispec", "echelle", "onedspec", "strip")

#: World coordinate cards: the frame's are kept where they are when the spectrum
#: has them too, and dropped when it does not.
_WCS = ("WCSDIM", "CTYPE", "CRVAL", "CRPIX", "CDELT", "CD1_", "CD2_", "CD3_", "LTV",
        "LTM", "WAT", "CROTA")
#: Cards IRAF deletes outright.
_DROPPED = ("DISPAXIS", "SLFIB")
_STRUCTURE = ("SIMPLE", "BITPIX", "NAXIS", "EXTEND", "BSCALE", "BZERO", "PCOUNT",
              "GCOUNT", "XTENSION", "EXTNAME", "EXTVER", "INHERIT", "CHECKSUM", "DATASUM")


@dataclass
class Part:
    """One extracted spectrum and what IRAF says of it."""

    number: int                 # aperture number, + 1000 per part after the first
    beam: int
    low: float                  # limits across the slit, image pixels
    high: float
    spectrum: np.ndarray
    raw: np.ndarray | None = None
    sky: np.ndarray | None = None
    sigma: np.ndarray | None = None
    strip: np.ndarray | None = None
    aperture: int = 0           # the aperture it is part of


def names(fmt: str, root: Path, parts: list[Part], nsubaps: int, default: bool) -> list[Path]:
    """ap_check's output names. `default`: no output was given, so the name is the
    frame's, with IRAF's extension."""
    if fmt == "multispec":
        return [root.with_name(root.name + (".ms" if default else "") + ".fits")]
    if fmt == "echelle":
        if nsubaps == 1:
            return [root.with_name(root.name + (".ec" if default else "") + ".fits")]
        digits = int(log10(nsubaps)) + 1
        return [root.with_name(f"{root.name}{part:0{digits}d}" + (".ec" if default else "")
                               + ".fits") for part in range(1, nsubaps + 1)]
    digits = int(log10(nsubaps)) + 4
    return [root.with_name(f"{root.name}.{p.number:0{digits}d}.fits") for p in parts]


def dispersion(header, dispaxis: int) -> tuple[float, float]:
    """apmw_open's W1 and DW: the world coordinate of the first pixel along the
    dispersion and its step -- the frame's linear WCS on that axis, else its
    physical pixels (a trimmed frame's LTV), else 1 and 1."""
    n = dispaxis
    has_wcs = any(f"{key}{n}" in header for key in ("CTYPE", "CRVAL", "CDELT")) \
        or f"CD{n}_{n}" in header
    if has_wcs:
        step = float(header.get(f"CD{n}_{n}", header.get(f"CDELT{n}", 1.0)) or 1.0)
        value = float(header.get(f"CRVAL{n}", 0.0))
        return value + step * (1.0 - float(header.get(f"CRPIX{n}", 0.0))), step
    scale = float(header.get(f"LTM{n}_{n}", 1.0)) or 1.0
    return (1.0 - float(header.get(f"LTV{n}", 0.0))) / scale, 1.0 / scale


def write(fmt: str, root: Path, parts: list[Part], header, *, nsubaps: int = 1,
          extras: bool = True, background: str = "none", weights: str = "none",
          clean: bool = False, default: bool = False, dispaxis: int = 2) -> list[Path]:
    """Write `parts` (aperture by aperture, each one's parts in order) as IRAF's
    `fmt`, from the frame's `header`. Returns the files written."""
    if fmt not in FORMATS:
        raise ValueError(f"format '{fmt}' is not one of {', '.join(FORMATS)}")
    paths = names(fmt, root, parts, nsubaps, default)
    yesno = "yes" if clean else "no"
    world = dispersion(header, dispaxis)
    if fmt == "multispec":
        _image(paths[0], parts, header, extras, background, weights, yesno, " -", None,
               world)
    elif fmt == "echelle":
        for index, path in enumerate(paths):
            mine = parts[index::nsubaps]
            _image(path, mine, header, extras, background, weights, yesno, " -",
                   index + 1 if nsubaps > 1 else None, world)
    elif fmt == "onedspec":
        for path, part in zip(paths, parts, strict=True):
            _image(path, [part], header, extras, background, weights, yesno, ":", None,
                   world)
    else:
        for path, part in zip(paths, parts, strict=True):
            _strip(path, part, header, world)
    return paths


def _copy(header):
    from astropy.io import fits

    out = fits.Header()
    for card in header.cards:
        if (card.keyword in _STRUCTURE or card.keyword.startswith(_DROPPED)
                or card.keyword.startswith("NAXIS")):
            continue
        # A card of its own: the strip's title is changed per aperture.
        out.append((card.keyword, card.value, card.comment), end=True)
    return out


def _bands(parts, extras, background, weights, yesno, colon):
    """The planes and their BANDIDs, in ap_output's order."""
    first = parts[0]
    bands = [("spectrum", f"spectrum{colon} background {background}, weights {weights}, "
                          f"clean {yesno}")]
    if extras:
        if first.raw is not None:
            word = "spectrum:" if colon == ":" else "raw -"  # onedspec says spectrum
            bands.append(("raw", f"{word} background {background}, weights none, clean no"))
        if first.sky is not None:
            bands.append(("sky", f"background{colon} background {background}"))
        if first.sigma is not None:
            bands.append(("sigma", f"sigma - background {background}, weights {weights}, "
                                   f"clean {yesno}"))
    return bands


def _image(path, parts, header, extras, background, weights, yesno, colon, subap, world):
    bands = _bands(parts, extras, background, weights, yesno, colon)
    ny = parts[0].spectrum.size
    cube = np.zeros((len(bands), len(parts), ny), dtype=np.float32)
    for b, (name, _) in enumerate(bands):
        for line, part in enumerate(parts):
            cube[b, line] = getattr(part, "spectrum" if name == "spectrum" else name)
    data = cube
    if len(bands) == 1:
        data = cube[0]
        if len(parts) == 1:
            data = cube[0, 0]
    out = _copy(header)
    for index, (_, text) in enumerate(bands, start=1):
        out[f"BANDID{index}"] = text
    if subap is not None:
        out["SUBAP"] = subap
    for index, part in enumerate(parts, start=1):
        # An echelle image of one part numbers its lines by aperture alone.
        number = part.aperture if subap is not None else part.number
        out[f"APNUM{index}"] = f"{number} {part.beam} {part.low:.2f} {part.high:.2f}"
    _wcs(out, data.ndim, "equispec", world)
    _save(path, data, out)


def _strip(path, part, header, world):
    out = _copy(header)
    if "OBJECT" in out:
        out["OBJECT"] = f"{out['OBJECT']} - Aperture {part.aperture}"
    out["APNUM1"] = f"{part.number} {part.beam} {part.low:.2f} {part.high:.2f}"
    out["DISPAXIS"] = 1
    _wcs(out, 2, "linear", world)
    _save(path, part.strip.astype(np.float32), out)


def _wcs(out, ndim, system, world):
    """apmw_saveim's world coordinates: the dispersion along the first axis, the
    lines and bands numbered. A card the frame has is changed where it stands, as
    IRAF's header kernel does; one it has that the spectrum has not is dropped.
    IRAF writes CDELT beside CD up to two axes, not with three."""
    w1, dw = world
    wanted = {"WCSDIM": ndim, "CTYPE1": "PIXEL"}
    wanted.update({f"CTYPE{axis}": "LINEAR" for axis in range(2, ndim + 1)})
    wanted.update({"CRVAL1": w1, "CRPIX1": 1.0})
    if ndim < 3:
        wanted.update({f"CDELT{axis}": dw if axis == 1 else 1.0
                       for axis in range(1, ndim + 1)})
    wanted.update({f"CD{axis}_{axis}": dw if axis == 1 else 1.0
                   for axis in range(1, ndim + 1)})
    wanted.update({f"LTM{axis}_{axis}": 1.0 for axis in range(1, ndim + 1)})
    wanted["WAT0_001"] = f"system={system}"
    wanted["WAT1_001"] = "wtype=linear label=Pixel"
    wanted.update({f"WAT{axis}_001": "wtype=linear" for axis in range(2, ndim + 1)})
    for key in [k for k in out if k.startswith(_WCS) and k not in wanted]:
        del out[key]
    for key, value in wanted.items():
        out[key] = value


def _save(path, data, header):
    from astropy.io import fits

    from . import provenance

    provenance.stamp_header(header)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.header.set("EXTEND", False, after=f"NAXIS{data.ndim}")
    hdu.writeto(path, overwrite=True)
