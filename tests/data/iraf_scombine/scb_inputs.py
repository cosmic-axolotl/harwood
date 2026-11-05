"""The spectra the scombine cases combine, as IRAF's multispec (python sc_inputs.py
<folder>, harwood's venv) and as harwood's tables (spectra()): five of two apertures
on dispersions a little apart, with noise, cosmic rays and their own levels and
scales; `short`, which covers only part of the range; `lg`, two apertures even in
log wavelength; and `scales.txt`, a file of scales."""

import sys
from pathlib import Path

import numpy as np

N = 150
#: name: ([(aperture, beam, dtype, w1, dw, title)], header cards)
SPECTRA = {
    "a": ([(1, 1, 0, 4010.0, 2.0, "a one"), (2, 2, 0, 4020.0, 2.1, "a two")],
          dict(EXPTIME=100.0, SCL=1.0, ZER=10.0, WGT=1.0, RDN=5.0, GN=2.0)),
    "b": ([(1, 1, 0, 4012.5, 1.95, "b one"), (2, 2, 0, 4021.0, 2.05, "b two")],
          dict(EXPTIME=150.0, SCL=1.4, ZER=25.0, WGT=1.5, RDN=5.0, GN=2.0)),
    "c": ([(1, 1, 0, 4007.0, 2.02, "c one"), (2, 2, 0, 4018.0, 2.12, "c two")],
          dict(EXPTIME=120.0, SCL=1.2, ZER=5.0, WGT=1.2, RDN=6.0, GN=2.2)),
    "d": ([(1, 1, 0, 4011.0, 1.98, "d one"), (2, 2, 0, 4019.5, 2.08, "d two")],
          dict(EXPTIME=80.0, SCL=0.8, ZER=-3.0, WGT=0.8, RDN=5.0, GN=2.0)),
    "e": ([(1, 1, 0, 4009.0, 2.01, "e one"), (2, 2, 0, 4022.0, 2.09, "e two")],
          dict(EXPTIME=110.0, SCL=1.1, ZER=15.0, WGT=1.1, RDN=5.0, GN=2.0)),
    "short": ([(1, 1, 0, 4100.0, 2.0, "short one"), (2, 2, 0, 4120.0, 2.1, "short two")],
              dict(EXPTIME=90.0, SCL=0.9, ZER=0.0, WGT=0.9, RDN=5.0, GN=2.0)),
    "lg": ([(1, 1, 1, 3.6031443726201822, 0.0002, "lg one"),
            (2, 2, 1, 3.6042092, 0.00021, "lg two")],
           dict(EXPTIME=100.0, SCL=1.0, ZER=0.0, WGT=1.0, RDN=5.0, GN=2.0)),
}
LEVELS = {"a": (1.0, 0.0), "b": (1.4, 20.0), "c": (1.2, 5.0), "d": (0.8, -3.0),
          "e": (1.1, 15.0), "short": (0.9, 0.0), "lg": (1.0, 0.0)}
SCALES = "1.0\n1.3\n\n1.25\n0.85\n1.05\n"


def npix(name):
    return 60 if name == "short" else N


def data(name):
    """[aperture line] of pixels, in real."""
    rng = np.random.default_rng(sum(map(ord, name)))
    n = npix(name)
    out = []
    for ap, _, dtype, w1, dw, _ in SPECTRA[name][0]:
        x = np.arange(n, dtype=float)
        wave = 10 ** (w1 + dw * x) if dtype == 1 else w1 + dw * x
        scale, zero = LEVELS[name]
        base = 800.0 + 150.0 * np.sin(wave / 25.0) + 30.0 * ap \
            - 300.0 * np.exp(-0.5 * ((wave - 4150.0) / 4.0) ** 2)
        values = scale * base + zero + rng.normal(0.0, np.sqrt(np.abs(base) / 2.0), n)
        for at in rng.choice(n, 4, replace=False):
            values[at] += rng.uniform(800.0, 3000.0)
        out.append(values.astype(np.float32))
    return out


def write(folder: Path) -> None:
    from astropy.io import fits

    for name, (rows, cards) in SPECTRA.items():
        n = npix(name)
        header = fits.Header()
        header["OBJECT"] = f"star {name}"
        header["WCSDIM"] = 2
        for axis in (1, 2):
            header[f"CTYPE{axis}"] = "MULTISPE"
            header[f"CD{axis}_{axis}"] = 1.0
            header[f"LTM{axis}_{axis}"] = 1.0
        header["WAT0_001"] = "system=multispec"
        header["WAT1_001"] = "wtype=multispec label=Wavelength units=angstroms"
        text = "wtype=multispec " + " ".join(
            f'spec{i} = "{ap} {beam} {dtype} {w1!r} {dw!r} {n} 0. {ap * 10 - 3}. '
            f'{ap * 10 + 3}."' for i, (ap, beam, dtype, w1, dw, _) in enumerate(rows, 1))
        for i in range(0, len(text), 68):
            header[f"WAT2_{i // 68 + 1:03d}"] = text[i:i + 68]
        for i, (ap, beam, *_rest, title) in enumerate(rows, 1):
            header[f"APNUM{i}"] = f"{ap} {beam} {ap * 10 - 3}. {ap * 10 + 3}."
            header[f"APID{i}"] = title
        for key, value in cards.items():
            header[key] = value
        fits.writeto(folder / f"{name}.fits", np.array(data(name), np.float32), header,
                     overwrite=True)
    (folder / "scales.txt").write_text(SCALES, encoding="utf-8")


def spectra(folder: Path) -> None:
    """The same spectra as harwood's tables."""
    from astropy.io import fits

    from harwood.core import spectrum

    for name, (rows, cards) in SPECTRA.items():
        n = npix(name)
        pixel = np.arange(1.0, n + 1.0)
        header = fits.Header()
        header["OBJECT"] = f"star {name}"
        for key, value in cards.items():
            header[key] = value
        apertures = []
        for (ap, beam, dtype, w1, dw, title), values in zip(rows, data(name), strict=True):
            world = w1 + dw * (pixel - 1)
            apertures.append(spectrum.Aperture(
                number=ap, centre=10.0 * ap, low=-3.0, high=3.0, pixel=pixel,
                wavelength=10 ** world if dtype == 1 else world, title=title,
                flux=values.astype(float), beam=beam))
        spectrum.write(folder / f"{name}.fits", apertures, header=header)
    (folder / "scales.txt").write_text(SCALES, encoding="utf-8")


if __name__ == "__main__":
    write(Path(sys.argv[1]))
