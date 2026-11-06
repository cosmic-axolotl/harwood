"""The spectra the telluric and skytweak cases correct, as IRAF's images (python
twk_inputs.py <folder>, harwood's venv) and as harwood's tables (spectra()):

  s      a star, two apertures, through telluric bands at airmass 1.4
  t      the telluric calibration, the bands alone, at airmass 1.1, shifted by 0.7 px
  t1     the same, one aperture (aperture 1), one-dimensional
  t13    the calibration with apertures 1 and 3, so that aperture 2 has none
  tz     a calibration whose bands go to zero and below, for the threshold
  tn     a calibration with no AIRMASS
  sn     the star with no AIRMASS
  s3     the star in the four bands apall writes
  short  a calibration that covers only part of the star's range
  o      an object with sky lines, two apertures; k the sky, shifted and scaled
"""

import sys
from pathlib import Path

import numpy as np

N = 300
W1, DW = 6800.0, 1.0
BANDS = ("spectrum", "raw", "background", "sigma")
#: telluric bands: (centre, depth, width)
TELLURIC = ((6870.0, 0.55, 4.0), (6884.0, 0.35, 3.0), (6950.0, 0.25, 6.0),
            (7020.0, 0.4, 2.5))
SKY = ((6863.9, 400.0, 1.4), (6923.2, 250.0, 1.4), (6949.0, 600.0, 1.4),
       (7023.5, 300.0, 1.4), (7050.0, 150.0, 1.4))


def _wave(shift=0.0, w1=W1, n=N):
    return w1 + DW * (np.arange(n, dtype=float) - shift)


def transmission(wave, power=1.0):
    t = np.ones(wave.size)
    for centre, depth, width in TELLURIC:
        t *= 1.0 - depth * np.exp(-0.5 * ((wave - centre) / width) ** 2)
    return t ** power


def sky(wave, scale=1.0):
    s = 80.0 + 0.02 * (wave - W1)
    for centre, height, width in SKY:
        s = s + height * np.exp(-0.5 * ((wave - centre) / width) ** 2)
    return scale * s


#: name: (kind, apertures [(ap, beam, w1, title)], npix, cards, nbands)
SPECTRA = {
    "s": ("star", [(1, 1, W1, "star one"), (2, 2, W1 + 0.4, "star two")], N,
          {"AIRMASS": 1.4}, 1),
    "sn": ("star", [(1, 1, W1, "star one"), (2, 2, W1 + 0.4, "star two")], N, {}, 1),
    "s3": ("star", [(1, 1, W1, "star one"), (2, 2, W1 + 0.4, "star two")], N,
           {"AIRMASS": 1.4}, 4),
    "t": ("cal", [(1, 1, W1, "tell one"), (2, 2, W1 + 0.4, "tell two")], N,
          {"AIRMASS": 1.1}, 1),
    "t1": ("cal", [(1, 1, W1, "tell")], N, {"AIRMASS": 1.1}, 1),
    "tn": ("cal", [(1, 1, W1, "tell one"), (2, 2, W1 + 0.4, "tell two")], N, {}, 1),
    "t13": ("cal", [(1, 1, W1, "tell one"), (3, 3, W1 + 0.4, "tell three")], N,
            {"AIRMASS": 1.1}, 1),
    "tz": ("zero", [(1, 1, W1, "zero one"), (2, 2, W1 + 0.4, "zero two")], N,
           {"AIRMASS": 1.1}, 1),
    "short": ("cal", [(1, 1, W1 + 60.0, "short one"), (2, 2, W1 + 60.4, "short two")],
              150, {"AIRMASS": 1.1}, 1),
    "o": ("object", [(1, 1, W1, "object one"), (2, 2, W1 + 0.3, "object two")], N,
          {"AIRMASS": 1.2}, 1),
    "k": ("sky", [(1, 1, W1, "sky one"), (2, 2, W1 + 0.3, "sky two")], N,
          {"AIRMASS": 1.2}, 1),
}


def data(name):
    """[band][line] of pixels, in real."""
    kind, rows, n, _, nbands = SPECTRA[name]
    rng = np.random.default_rng(sum(map(ord, name)) + 3)
    out = np.zeros((nbands, len(rows), n), dtype=np.float32)
    for line, (ap, _, w1, _) in enumerate(rows):
        wave = _wave(0.0, w1, n)
        if kind == "star":
            cont = 2000.0 + 1.5 * (wave - W1) + 100.0 * ap
            flux = cont * transmission(wave, 1.4) + rng.normal(0.0, 8.0, n)
        elif kind == "cal":
            flux = transmission(_wave(0.7, w1, n), 1.1) * (1.0 + 0.0002 * (wave - W1)) \
                + rng.normal(0.0, 0.002, n)
        elif kind == "zero":
            flux = transmission(wave, 1.1) * 1.3 - 0.62
        elif kind == "object":
            flux = 500.0 + 0.3 * (wave - W1) + sky(wave, 1.15) + rng.normal(0.0, 4.0, n)
        else:
            flux = sky(_wave(-0.6, w1, n)) + rng.normal(0.0, 2.0, n)
        out[0, line] = flux
        if nbands > 1:
            out[1, line] = flux * 1.1 + 30.0
            out[2, line] = 50.0 + rng.normal(0.0, 3.0, n)
            out[3, line] = np.sqrt(np.abs(flux)) + 2.0
    return out


def _header(name):
    from astropy.io import fits

    _, rows, n, cards, nbands = SPECTRA[name]
    header = fits.Header()
    header["OBJECT"] = f"spectrum {name}"
    if len(rows) == 1 and nbands == 1:
        ap, beam, w1, title = rows[0]
        header["WCSDIM"] = 1
        header["CTYPE1"] = "LINEAR"
        header["CRVAL1"] = w1
        header["CDELT1"] = DW
        header["CD1_1"] = DW
        header["CRPIX1"] = 1.0
        header["LTM1_1"] = 1.0
        header["WAT0_001"] = "system=equispec"
        header["WAT1_001"] = "wtype=linear label=Wavelength units=angstroms"
        header["APNUM1"] = f"{ap} {beam} {ap * 10 - 3}. {ap * 10 + 3}."
        header["OBJECT"] = title
    else:
        header["WCSDIM"] = 3 if nbands > 1 else 2
        for axis in range(1, header["WCSDIM"] + 1):
            header[f"CTYPE{axis}"] = "MULTISPE" if axis < 3 else "LINEAR"
            header[f"CD{axis}_{axis}"] = 1.0
            header[f"LTM{axis}_{axis}"] = 1.0
        header["WAT0_001"] = "system=multispec"
        header["WAT1_001"] = "wtype=multispec label=Wavelength units=angstroms"
        text = "wtype=multispec " + " ".join(
            f'spec{i} = "{ap} {beam} 0 {w1!r} {DW!r} {n} 0. {ap * 10 - 3}. '
            f'{ap * 10 + 3}."' for i, (ap, beam, w1, _) in enumerate(rows, 1))
        for i in range(0, len(text), 68):
            header[f"WAT2_{i // 68 + 1:03d}"] = text[i:i + 68]
        if nbands > 1:
            header["WAT3_001"] = "wtype=linear"
            for band, what in enumerate(BANDS, 1):
                header[f"BANDID{band}"] = what
        for i, (ap, beam, _, title) in enumerate(rows, 1):
            header[f"APNUM{i}"] = f"{ap} {beam} {ap * 10 - 3}. {ap * 10 + 3}."
            header[f"APID{i}"] = title
    for key, value in cards.items():
        header[key] = value
    return header


def write(folder: Path) -> None:
    from astropy.io import fits

    for name, (_, rows, _, _, nbands) in SPECTRA.items():
        pixels = data(name)
        if nbands == 1:
            pixels = pixels[0]
            if len(rows) == 1:
                pixels = pixels[0]
        fits.writeto(folder / f"{name}.fits", pixels, _header(name), overwrite=True)


def spectra(folder: Path) -> None:
    """The same spectra as harwood's tables."""
    from astropy.io import fits

    from harwood.core import spectrum

    for name, (_, rows, n, cards, nbands) in SPECTRA.items():
        pixels = data(name)
        pixel = np.arange(1.0, n + 1.0)
        header = fits.Header()
        header["OBJECT"] = f"spectrum {name}"
        for key, value in cards.items():
            header[key] = value
        apertures = []
        for line, (ap, beam, w1, title) in enumerate(rows):
            bands = {"flux": pixels[0, line].astype(float)}
            if nbands > 1:
                bands.update(raw=pixels[1, line].astype(float),
                             sky=pixels[2, line].astype(float),
                             error=pixels[3, line].astype(float))
            apertures.append(spectrum.Aperture(
                number=ap, centre=10.0 * ap, low=-3.0, high=3.0, pixel=pixel,
                wavelength=w1 + DW * (pixel - 1), title=title, beam=beam, **bands))
        spectrum.write(folder / f"{name}.fits", apertures, header=header)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
