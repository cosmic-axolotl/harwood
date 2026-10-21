"""An image as harwood keeps it: the data, its variance and its quality.

Design contract, section 12. A file holds a primary HDU with no pixels and
the header common to everything, then a trio per detector plane::

    SCI   the data
    VAR   its variance          (or ERR, its standard deviation, if asked)
    DQ    why a pixel is not to be trusted, in DRAGONS' bits (core/dq.py)

A detector read through several amplifiers, or a mosaic, repeats the trio
with EXTVER 1, 2, ... Each extension carries INHERIT = T, so a reader that
opens SCI alone still sees OBJECT, EXPTIME and the rest of the primary
header -- core/images.read does that for every task.

Inside harwood the uncertainty is always a variance: variances add when
images are added or combined, and the standard deviation is worked out only
to show it. On disk it is the user's choice (`uncertainty`): VAR, ERR, or
UNCERT, the layout astropy's CCDData reads. Reading recognises all three,
and an image of one HDU -- IRAF's, a camera's -- is read as it is: no
variance (nobody can make one up without the gain), and the DQ from its BPM
mask when it names one.

`--format iraf` writes what IRAF reads: the data alone in one HDU, and the
DQ as a mask image named by BPM, which IRAF's ccdproc and imcombine take.
What that leaves behind -- the variance -- is returned so the task can say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import advice, dq

UNCERTAINTIES = ("VAR", "ERR", "UNCERT")
FORMATS = ("harwood", "iraf")

#: Structural cards astropy writes itself; never copied between headers.
_STRUCTURAL = {"SIMPLE", "XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
               "EXTEND", "PCOUNT", "GCOUNT", "EXTNAME", "EXTVER", "INHERIT", "BSCALE",
               "BZERO", "CHECKSUM", "DATASUM"}

#: Cards that belong to a plane rather than to the exposure: its coordinates
#: and where it sits on the detector.
_PLANE = ("CTYPE", "CRPIX", "CRVAL", "CDELT", "CUNIT", "CD1_", "CD2_", "CD3_",
          "PC1_", "PC2_", "LTV", "LTM", "WAT", "WCSDIM", "WCSNAME", "RADESYS",
          "EQUINOX", "LONPOLE", "LATPOLE", "DATASEC", "TRIMSEC", "BIASSEC", "CCDSEC",
          "DETSEC", "AMPSEC", "BUNIT", "BPM")


@dataclass
class Plane:
    """One detector plane: the data, its variance (None when unknown) and
    its DQ, with the header cards that belong to it."""

    sci: np.ndarray
    var: np.ndarray | None = None
    dq: np.ndarray | None = None
    header: Any = None

    def __post_init__(self):
        from astropy.io import fits

        self.sci = np.asarray(self.sci)
        if self.dq is None:
            self.dq = dq.empty(self.sci.shape)
        if self.header is None:
            self.header = fits.Header()


@dataclass
class Image:
    """An image: the primary header and its planes. `layout` says what the
    file was when read -- harwood, ccddata or single (one HDU)."""

    primary: Any
    planes: list[Plane] = field(default_factory=list)
    layout: str = "harwood"

    @property
    def plane(self) -> Plane:
        """The only plane, for the common case of one detector."""
        if len(self.planes) != 1:
            advice.fail(f"this image has {len(self.planes)} planes (EXTVER 1 to "
                        f"{len(self.planes)}), not one.",
                        "name the one you mean, e.g. image.fits[SCI,2].")
        return self.planes[0]


def is_plane_card(key: str) -> bool:
    return key.startswith(_PLANE)


def split(header) -> tuple[Any, Any]:
    """A one-HDU header as (primary, plane): the coordinates and detector
    sections go with the plane, everything else with the exposure."""
    from astropy.io import fits

    primary, plane = fits.Header(), fits.Header()
    for card in header.cards:
        if card.keyword in _STRUCTURAL:
            continue
        target = plane if is_plane_card(card.keyword) else primary
        target.append(card, end=True)
    return primary, plane


def merged(primary, plane):
    """The header a one-HDU file gets: the exposure's, then the plane's --
    the plane's value winning where both have one, HISTORY and COMMENT
    cards kept from both."""
    from astropy.io import fits

    out = fits.Header()
    for header in (primary, plane):
        for card in header.cards:
            if card.keyword in _STRUCTURAL:
                continue
            if card.keyword in ("HISTORY", "COMMENT", "") or card.keyword not in out:
                out.append(card, end=True)
            else:
                out[card.keyword] = (card.value, card.comment)
    return out


# --------------------------------------------------------------------- reading

def read(path: Path | str, workdir: Path | str = ".") -> Image:
    """Any of the layouts section 12 accepts, as an Image whose variance is
    a variance whatever the file held."""
    from astropy.io import fits

    from . import images

    spec = images.parse(path)
    with fits.open(spec.path, memmap=False) as hdul:
        names = [str(h.header.get("EXTNAME", "")).upper() for h in hdul]
        if "SCI" in names:
            return _read_sci(hdul)
        if hdul[0].data is not None and ("UNCERT" in names or "MASK" in names):
            return _read_ccddata(hdul, names)
        index = next((i for i, h in enumerate(hdul) if h.data is not None), None)
        if index is None:
            advice.fail(f"'{spec.path}' holds no image.",
                        "check the file: every HDU in it is empty.")
        header = hdul[index].header
        sci = np.asarray(hdul[index].data)
    primary, plane_header = split(header)
    mask = None
    if "BPM" in header:
        from . import pixmask

        mask = pixmask.read("BPM", header, sci.shape, workdir)
    quality = dq.empty(sci.shape)
    if mask is not None:
        quality[mask != 0] |= dq.BAD_PIXEL
    return Image(primary, [Plane(sci, None, quality, plane_header)], layout="single")


def _read_sci(hdul) -> Image:
    from astropy.io import fits

    primary = fits.Header([c for c in hdul[0].header.cards
                           if c.keyword not in _STRUCTURAL])
    by_name: dict[tuple[str, int], Any] = {}
    for hdu in hdul[1:]:
        name = str(hdu.header.get("EXTNAME", "")).upper()
        by_name[(name, int(hdu.header.get("EXTVER", 1)))] = hdu
    planes = []
    for version in sorted(v for (n, v) in by_name if n == "SCI"):
        sci_hdu = by_name[("SCI", version)]
        var = None
        if ("VAR", version) in by_name:
            var = np.asarray(by_name[("VAR", version)].data, dtype=np.float64)
        elif ("ERR", version) in by_name:
            var = np.asarray(by_name[("ERR", version)].data, dtype=np.float64) ** 2
        quality = (np.asarray(by_name[("DQ", version)].data).astype(dq.DTYPE)
                   if ("DQ", version) in by_name else None)
        header = fits.Header([c for c in sci_hdu.header.cards
                              if c.keyword not in _STRUCTURAL])
        planes.append(Plane(np.asarray(sci_hdu.data), var, quality, header))
    return Image(primary, planes, layout="harwood")


def _read_ccddata(hdul, names) -> Image:
    """astropy's CCDData: the data in the primary HDU, UNCERT and MASK beside
    it, the uncertainty's kind in UTYPE."""
    primary, plane_header = split(hdul[0].header)
    sci = np.asarray(hdul[0].data)
    var = None
    if "UNCERT" in names:
        hdu = hdul[names.index("UNCERT")]
        values = np.asarray(hdu.data, dtype=np.float64)
        kind = str(hdu.header.get("UTYPE", "StdDevUncertainty"))
        if kind == "VarianceUncertainty":
            var = values
        elif kind == "InverseVariance":
            with np.errstate(divide="ignore"):
                var = np.where(values > 0, 1.0 / values, np.inf)
        else:
            var = values ** 2
    if "DQ" in names:
        quality = np.asarray(hdul[names.index("DQ")].data).astype(dq.DTYPE)
    else:
        quality = dq.empty(sci.shape)
        if "MASK" in names:
            quality[np.asarray(hdul[names.index("MASK")].data) != 0] |= dq.BAD_PIXEL
    return Image(primary, [Plane(sci, var, quality, plane_header)], layout="ccddata")


# --------------------------------------------------------------------- writing

def write(path: Path | str, image: Image, uncertainty: str = "VAR",
          format: str = "harwood", overwrite: bool = True) -> tuple[list[Path], list[str]]:
    """Write `image`. Returns the files written and, for --format iraf, what
    IRAF's single HDU could not hold."""
    if uncertainty not in UNCERTAINTIES:
        advice.fail(f"--uncertainty '{uncertainty}' is not one harwood writes.",
                    "give VAR (the variance), ERR (its square root) or UNCERT "
                    "(astropy's CCDData layout).")
    if format not in FORMATS:
        advice.fail(f"--format '{format}' is not one harwood writes.",
                    "give harwood (SCI, VAR and DQ) or iraf (one HDU, as IRAF reads).")
    path = Path(path)
    if format == "iraf":
        return _write_iraf(path, image, overwrite)
    if uncertainty == "UNCERT":
        return [_write_ccddata(path, image, overwrite)], []
    return [_write_sci(path, image, uncertainty, overwrite)], []


def _extension(data, name, version, header=None):
    from astropy.io import fits

    hdu = fits.ImageHDU(data, header=header, name=name)
    hdu.header["EXTVER"] = version
    hdu.header["INHERIT"] = (True, "the primary header applies to this extension")
    return hdu


def _write_sci(path: Path, image: Image, uncertainty: str, overwrite: bool) -> Path:
    from astropy.io import fits

    hdus = [fits.PrimaryHDU(header=image.primary.copy())]
    for version, plane in enumerate(image.planes, start=1):
        header = plane.header.copy()
        # IRAF takes an image's title from OBJECT in its own HDU, not from
        # the inherited primary: without it, hselect x.fits[SCI] OBJECT is blank.
        if "OBJECT" in image.primary and "OBJECT" not in header:
            header["OBJECT"] = image.primary["OBJECT"]
        hdus.append(_extension(np.asarray(plane.sci, dtype=np.float32), "SCI", version,
                               header))
        if plane.var is not None:
            if uncertainty == "ERR":
                hdus.append(_extension(np.sqrt(plane.var).astype(np.float32), "ERR",
                                       version))
            else:
                hdus.append(_extension(np.asarray(plane.var, dtype=np.float32), "VAR",
                                       version))
        quality = _extension(np.asarray(plane.dq, dtype=dq.DTYPE), "DQ", version)
        dq.header_cards(quality.header)
        hdus.append(quality)
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path


def _write_ccddata(path: Path, image: Image, overwrite: bool) -> Path:
    """CCDData's layout, which astropy reads back with CCDData.read: data in
    the primary HDU, UNCERT as a standard deviation, MASK as good/bad -- and
    the DQ beside them, so harwood reads back every reason, not only 'bad'."""
    from astropy.io import fits

    plane = image.plane
    hdus = [fits.PrimaryHDU(np.asarray(plane.sci, dtype=np.float32),
                            header=merged(image.primary, plane.header))]
    if plane.var is not None:
        uncert = fits.ImageHDU(np.sqrt(plane.var).astype(np.float32), name="UNCERT")
        uncert.header["UTYPE"] = "StdDevUncertainty"
        hdus.append(uncert)
    hdus.append(fits.ImageHDU((np.asarray(plane.dq) != 0).astype(np.uint8), name="MASK"))
    quality = fits.ImageHDU(np.asarray(plane.dq, dtype=dq.DTYPE), name="DQ")
    dq.header_cards(quality.header)
    hdus.append(quality)
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path


def mask_name(path: Path) -> Path:
    """Where --format iraf puts the DQ: beside the image, as IRAF names masks."""
    stem = path.name[:-5] if path.name.endswith(".fits") else path.stem
    return path.with_name(f"{stem}.bpm.fits")


def _write_iraf(path: Path, image: Image, overwrite: bool) -> tuple[list[Path], list[str]]:
    from astropy.io import fits

    if len(image.planes) != 1:
        advice.fail(f"IRAF keeps one image per file, and this has {len(image.planes)} "
                    "planes.",
                    "write each plane on its own -- image.fits[SCI,1], [SCI,2] -- or "
                    "keep harwood's format, which holds them all.")
    plane = image.plane
    header = merged(image.primary, plane.header)
    written, lost = [path], []
    if np.any(plane.dq):
        mask = mask_name(path)
        mask_header = fits.Header()
        mask_header["TITLE"] = (f"DQ of {path.name}", "nonzero is a bad pixel; bits as DQ")
        dq.header_cards(mask_header)
        fits.PrimaryHDU(np.asarray(plane.dq, dtype=np.int16), header=mask_header) \
            .writeto(mask, overwrite=overwrite)
        header["BPM"] = (mask.name, "bad pixel mask: harwood's DQ, nonzero is bad")
        written.append(mask)
    elif "BPM" in header:
        del header["BPM"]
    if plane.var is not None:
        lost.append("the variance")
    fits.PrimaryHDU(np.asarray(plane.sci, dtype=np.float32), header=header) \
        .writeto(path, overwrite=overwrite)
    return written, lost
