"""The images of the implot graph cases, here: python make_frames.py [DIR] [--iraf].

ms.fits is three extracted apertures, 3, 7 and 8, of different linear dispersions: as
harwood writes spectra (a table per aperture), or with --iraf as the multispec image
IRAF reads, its header written here and not by harwood."""
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
where = Path(arguments[0] if arguments else ".")
yy, xx = np.indices((60, 80))
a = (500 + 2 * xx + 3 * yy + 40 * np.sin(xx / 4.0) * np.cos(yy / 5.0)
     + (xx * 7 + yy * 13) % 9).astype(np.float32)
b = (100 + xx * yy / 10.0 + 5 * np.cos(xx / 3.0)).astype(np.float32)
fits.PrimaryHDU(a, fits.Header({"LTV1": -2.0, "LTV2": -1.0, "OBJECT": "synthetic a"})
                ).writeto(where / "sa.fits", overwrite=True)
fits.PrimaryHDU(b, fits.Header({"OBJECT": "synthetic b"})).writeto(where / "sb.fits",
                                                                   overwrite=True)
star = (50 + 1000 * np.exp(-0.5 * (((xx - 39.3) / 2.5) ** 2 + ((yy - 24.6) / 2.5) ** 2))
        + 0.02 * xx).astype(np.float32)
fits.PrimaryHDU(star, fits.Header({"OBJECT": "a star"})).writeto(where / "star.fits",
                                                                 overwrite=True)
fits.PrimaryHDU(np.full((60, 80), 7.0, dtype=np.float32)).writeto(where / "flat.fits",
                                                                  overwrite=True)
wave = (200 + 80 * np.sin(np.arange(300) / 20.0)).astype(np.float32)
fits.PrimaryHDU(wave, fits.Header({
    "OBJECT": "a spectrum", "CRVAL1": 4000.0, "CRPIX1": 1.0, "CD1_1": 2.5, "CDELT1": 2.5,
    "CTYPE1": "LINEAR", "DC-FLAG": 0, "WCSDIM": 1, "LTM1_1": 1.0,
    "WAT0_001": "system=world",
    "WAT1_001": 'wtype=linear label="Rest wavelength" units=Angstroms'})
                ).writeto(where / "spec.fits", overwrite=True)

n = 120
x = np.arange(n)
apertures = [(3, 6000.0, 1.1, 10.0, 20.0, 40.0), (7, 6010.0, 1.2, 30.0, 40.0, 70.0),
             (8, 6020.0, 0.9, 50.0, 60.0, 90.0)]
flux = [(100 + 900 * np.exp(-0.5 * ((x - peak) / 3.0) ** 2) + 0.5 * x).astype(np.float32)
        for *_, peak in apertures]
if "--iraf" in sys.argv:
    header = fits.Header({"OBJECT": "two apertures", "WCSDIM": 2, "CTYPE1": "MULTISPE",
                          "CTYPE2": "MULTISPE", "CDELT1": 1.0, "CDELT2": 1.0,
                          "CD1_1": 1.0, "CD2_2": 1.0, "LTM1_1": 1.0, "LTM2_2": 1.0,
                          "WAT0_001": "system=multispec",
                          "WAT1_001": "wtype=multispec label=Wavelength units=Angstroms"})
    text = "wtype=multispec" + "".join(
        f' spec{i} = "{ap} {ap} 0 {w1:.1f} {dw} {n} 0. {lo} {hi}"'
        for i, (ap, w1, dw, lo, hi, _) in enumerate(apertures, start=1))
    for i in range(0, len(text), 68):
        header[f"WAT2_{i // 68 + 1:03d}"] = text[i:i + 68]
    for i, (ap, _, _, lo, hi, _) in enumerate(apertures, start=1):
        header[f"APNUM{i}"] = f"{ap} {ap} {lo} {hi}"
    fits.PrimaryHDU(np.vstack(flux), header).writeto(where / "ms.fits", overwrite=True)
else:
    hdus = [fits.PrimaryHDU(header=fits.Header({"OBJECT": "two apertures",
                                                "HRWDSPEC": len(apertures)}))]
    for (ap, w1, dw, lo, hi, _), f in zip(apertures, flux, strict=True):
        table = fits.BinTableHDU.from_columns([
            fits.Column("pixel", "D", array=x + 1.0),
            fits.Column("wavelength", "D", array=w1 + dw * x),
            fits.Column("flux", "E", array=f)], name="SPEC")
        table.header.update({"EXTVER": ap, "APNUM": ap, "APCENTER": (lo + hi) / 2,
                             "APLOW": (lo - hi) / 2, "APHIGH": (hi - lo) / 2})
        hdus.append(table)
    fits.HDUList(hdus).writeto(where / "ms.fits", overwrite=True)
