"""Extracted spectra: one FITS file, one table per aperture.

IRAF writes extracted spectra as *multispec*: an image whose rows are
apertures, whose bands are things like the raw sum and the sky, and whose
wavelength scale lives in a set of header keywords that encode a polynomial as
a string. It works, it is compact, and it is unreadable without IRAF -- the
wavelength of a pixel is not in the file, it is in an equation in a keyword.

harwood writes a FITS file with one binary table per aperture instead:

    obj.fits            primary header: where it came from, and how
    obj.fits[SPEC,1]    aperture 1, a table
    obj.fits[SPEC,2]    aperture 2

with columns `pixel`, `wavelength`, `flux`, `error`, `sky`, `raw`. Every number
is in the file; nothing has to be recomputed to read it. The aperture's own
geometry -- where it sat, how wide it was, what the trace did -- is in the
extension header beside it.

Two things fall out of that choice. The `[SPEC,2]` addressing is the syntax
harwood already understands for image extensions, so an aperture is nameable
the same way anything else is. And the file can be turned into text without
harwood knowing anything the file does not say, which is what `wspectext`
does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import advice

EXTNAME = "SPEC"

#: Columns a spectrum may carry, in the order they are written.
COLUMNS = ("pixel", "wavelength", "flux", "error", "sky", "raw", "dq")

#: What each column is, for the header and for anyone reading the text dump.
MEANING = {
    "pixel": "position along the dispersion axis, 1-based",
    "wavelength": "wavelength, when a dispersion solution has been applied",
    "flux": "extracted flux, background subtracted",
    "error": "uncertainty on the flux: from the frame's variance when it has one, "
             "else IRAF's sigma band (variance weights only)",
    "sky": "background summed over the aperture, as subtracted (IRAF's background band)",
    "raw": "unweighted, uncleaned aperture sum, background subtracted (IRAF's raw band)",
    "dq": "DQ bits of the frame's pixels the aperture took on this line (core/dq.py)",
}


@dataclass
class Aperture:
    """One extracted spectrum, and where on the frame it came from."""

    number: int
    centre: float                       #: across the slit, 1-based
    low: float                          #: lower edge, relative to the centre
    high: float                         #: upper edge, relative to the centre
    pixel: Any = None
    flux: Any = None
    error: Any = None
    sky: Any = None
    raw: Any = None
    wavelength: Any = None
    dq: Any = None
    trace: Any = field(default=None, repr=False)   #: centre against dispersion
    title: str = ""
    meta: dict = field(default_factory=dict)
    #: The z dopcor has corrected the wavelengths by: they are the dispersion's
    #: over 1+z, as IRAF's world coordinates are (APZ).
    redshift: float = 0.0
    #: IRAF's beam number (APNUM's second), when it is not the aperture's (APBEAM).
    beam: int | None = None
    #: The units of the wavelength column, IRAF's (APUNITS): empty is Angstroms, as
    #: every task but disptrans writes and reads them.
    units: str = ""

    @property
    def width(self) -> float:
        return self.high - self.low

    def describe(self) -> str:
        return (f"aperture {self.number} at {self.centre:.2f}, "
                f"{self.low:+.1f} to {self.high:+.1f} ({self.width:.1f} px wide)")

    def columns(self) -> dict:
        """The columns this aperture actually has, in a fixed order."""
        return {name: getattr(self, name) for name in COLUMNS
                if getattr(self, name) is not None}


def write(path: Path | str, apertures: list[Aperture], header: Any = None,
          source: str = "") -> Path:
    """Write the apertures as one FITS file, a table each."""
    from astropy.io import fits
    from astropy.table import Table

    from . import provenance

    primary = fits.PrimaryHDU()
    if header is not None:
        for card in header.cards:
            if card.keyword in ("SIMPLE", "BITPIX", "EXTEND") or \
                    card.keyword.startswith("NAXIS"):
                continue
            try:
                primary.header[card.keyword] = (card.value, card.comment)
            except ValueError:
                continue
    primary.header["HRWDSPEC"] = (len(apertures),
                                  "extracted apertures, one per table")
    extra = []
    if source:
        # A FITS card holds 80 columns: eight for the keyword, three for the
        # separator, and whatever is left is shared between the value and its
        # comment. A long path would silently lose its end, so the keyword gets
        # the name and HISTORY -- which wraps -- gets the whole thing.
        text = str(source)
        primary.header["HRWDFROM"] = (Path(text).name[:48],
                                      "frame the spectra came from")
        if text != Path(text).name:
            extra.append(f"extracted from {text}")
    provenance.stamp_header(primary.header, extra)

    hdus = [primary]
    for aperture in apertures:
        table = Table(aperture.columns())
        hdu = fits.BinTableHDU(table, name=EXTNAME)
        hdu.header["EXTVER"] = aperture.number
        hdu.header["APNUM"] = (aperture.number, "aperture number")
        hdu.header["APCENTER"] = (round(float(aperture.centre), 4),
                                  "aperture centre across the slit, 1-based")
        hdu.header["APLOW"] = (round(float(aperture.low), 4),
                               "lower edge, relative to the centre")
        hdu.header["APHIGH"] = (round(float(aperture.high), 4),
                                "upper edge, relative to the centre")
        if aperture.title:
            hdu.header["APTITLE"] = aperture.title[:68]
        if aperture.beam is not None and aperture.beam != aperture.number:
            hdu.header["APBEAM"] = (int(aperture.beam), "beam number, IRAF's APNUM")
        if aperture.units:
            hdu.header["APUNITS"] = (aperture.units, "units of the wavelength column")
        if aperture.redshift:
            hdu.header["APZ"] = (float(aperture.redshift),
                                 "dopcor's z: wavelength = dispersion / (1+z)")
        for index, name in enumerate(table.colnames, start=1):
            hdu.header[f"TCOMM{index}"] = MEANING.get(name, "")
        for key, value in aperture.meta.items():
            try:
                hdu.header[key[:8].upper()] = value
            except ValueError:
                continue
        hdus.append(hdu)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def read(path: Path | str, number: int | None = None) -> list[Aperture]:
    """Read a spectrum file. One aperture if `number` is given, else all."""
    from astropy.io import fits
    from astropy.table import Table

    path_text = str(path)
    with fits.open(path_text, memmap=False) as hdul:
        found, present = [], []
        for hdu in hdul:
            if hdu.header.get("EXTNAME", "").upper() != EXTNAME:
                continue
            index = int(hdu.header.get("APNUM", hdu.header.get("EXTVER", 1)))
            present.append(index)
            if number is not None and index != number:
                continue
            table = Table(hdu.data)
            aperture = Aperture(
                number=index,
                centre=float(hdu.header.get("APCENTER", 0.0)),
                low=float(hdu.header.get("APLOW", 0.0)),
                high=float(hdu.header.get("APHIGH", 0.0)),
                title=str(hdu.header.get("APTITLE", "")),
                redshift=float(hdu.header.get("APZ", 0.0)),
                beam=int(hdu.header["APBEAM"]) if "APBEAM" in hdu.header else None,
                units=str(hdu.header.get("APUNITS", "")),
            )
            for name in table.colnames:
                if name in COLUMNS:
                    setattr(aperture, name, table[name].data)
            found.append(aperture)

    if not found:
        # Two different failures, and telling them apart is the whole use of
        # the message: a file with no spectra in it is a wrong file, and a file
        # without *that* aperture is a wrong number.
        if present:
            advice.fail(
                f"'{path}' has no aperture {number}. It holds "
                + ", ".join(str(index) for index in sorted(present)) + ".",
                "use one of those numbers.")
        advice.fail(
            f"'{path}' holds no extracted spectra. harwood writes one table "
            f"per aperture, named {EXTNAME}; this file has none.",
            "run apall on the frame to extract it. If this file came from "
            "IRAF as multispec, it needs converting first.")
    return found


def count(path: Path | str) -> int:
    """How many apertures a file holds, without reading the data."""
    from astropy.io import fits

    with fits.open(str(path), memmap=False) as hdul:
        return sum(1 for hdu in hdul
                   if hdu.header.get("EXTNAME", "").upper() == EXTNAME)


def dispersion_type(wavelength) -> int:
    """IRAF's DC-FLAG of a wavelength column: -1 none, 0 linear, 1 linear in log10,
    2 anything else."""
    import numpy as np

    if wavelength is None:
        return -1
    wave = np.asarray(wavelength, dtype=float)
    if wave.size < 2 or not wave.any():
        return -1
    step = np.diff(wave)
    if np.allclose(step, step[0], rtol=1e-9, atol=0):
        return 0
    if (wave > 0).all():
        logs = np.diff(np.log10(wave))
        if np.allclose(logs, logs[0], rtol=1e-7, atol=0):
            return 1
    return 2


def start_and_step(wavelength, npts: int) -> tuple[float, float]:
    """shdr's W0 and WP, in real: the first wavelength, and the mean step to the last
    (pixels, 1 and 1, for a spectrum without wavelengths)."""
    import numpy as np

    if dispersion_type(wavelength) == -1:
        return 1.0, 1.0
    w0, w1 = np.float32(wavelength[0]), np.float32(wavelength[-1])
    return float(w0), float((w1 - w0) / np.float32(npts - 1))
