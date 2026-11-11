"""The images and identify database of IRAF's fitcoords and transform gabarito:
python tr_inputs.py <folder> (harwood's venv, for astropy).

arc: a lamp dispersed along the lines (DISPAXIS=2), 61 columns by 301 lines, its
lines curved along the slit as a real spectrograph's are; gal: a galaxy and its
sky on the same geometry. arcl: the arc trimmed, LTV2 = -20, its database in
physical pixels. arct: the arc transposed, DISPAXIS=1. database/id<image>: the
lines measured every ten columns (rows), as reidentify leaves them, written here in
IRAF's text format -- with one user coordinate INDEF and one feature of weight 0.
one: the arc with a single line identified (fitcoords' z = y + f(x)); gali: the
galaxy in 16-bit integers; bpm: a mask of a few bad pixels; dels.db: two points
for the deletion list."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

NX, NY = 61, 301
WAVES = [6305.2, 6331.8, 6359.0, 6392.4, 6421.7, 6458.3, 6490.1, 6522.6, 6560.0,
         6591.3, 6624.8, 6650.5]


def position(x, w):
    """Where line w falls on column x (1-based pixels): the dispersion, a curvature
    along the slit and one that grows with wavelength."""
    s = (x - 31.0) / 30.0
    return (1.0 + (w - 6300.0) / 1.17 + 2.5e-5 * (w - 6300.0) ** 2 * 0.02 + 1.8 * s
            + 1.2 * s * s + 0.004 * (w - 6450.0) * s)


def frames():
    y, x = np.indices((NY, NX)) + 1.0
    arc = np.full((NY, NX), 50.0)
    for k, w in enumerate(WAVES):
        arc += (2000 + 300 * k) * np.exp(-0.5 * ((y - position(x, w)) / 1.4) ** 2)
    sky = 0.0
    for w in (6363.4, 6498.7, 6577.3):
        sky = sky + 800 * np.exp(-0.5 * ((y - position(x, w)) / 1.5) ** 2)
    gal = (300 + 0.4 * y) * np.exp(-0.5 * ((x - 29.0) / 5.0) ** 2) + 120 + sky
    gal += 900 * np.exp(-0.5 * ((x - 29.0) / 3.0) ** 2) * np.exp(
        -0.5 * ((y - position(x, 6568.0)) / 1.6) ** 2)
    return arc.astype(np.float32), gal.astype(np.float32)


def database(name, columns, transposed=False, offset=0.0, waves=None):
    lines = ["# Mon 10:00:00 10-Nov-2026"]
    for c in columns:
        section = f"[*,{c}]" if transposed else f"[{c},*]"
        lines += [f"begin\tidentify {name}{section}", f"\tid\t{name}{section}",
                  "\ttask\tidentify", f"\timage\t{name}{section}", "\tunits\tAngstroms",
                  f"\tfeatures\t{len(waves or WAVES)}"]
        for k, w in enumerate(waves or WAVES):
            pixel = position(float(c), w) + offset
            user = "INDEF" if (k == 4 and c == 31 and not waves) else f"{w:.4f}"
            weight = 0 if (k == 7 and c == 11 and not waves) else 1
            lines.append(f"\t   {pixel:10.2f} {w:10.8g} {user:>10s}   4.0 1 {weight}")
        lines += ["\tfunction\tchebyshev", "\torder\t3", "\tsample\t*",
                  "\tnaverage\t1", "\tniterate\t0", "\tlow\t3.", "\thigh\t3.",
                  "\tcoefficients\t0", ""]
    return "\n".join(lines) + "\n"


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    (where / "database").mkdir(exist_ok=True)
    arc, gal = frames()
    columns = list(range(1, NX + 1, 10))
    for name, data, cards in (("arc", arc, {"DISPAXIS": 2}), ("gal", gal, {"DISPAXIS": 2}),
                              ("arcl", arc, {"DISPAXIS": 2, "LTV2": -20.0, "LTM2_2": 1.0}),
                              ("arct", arc.T.copy(), {"DISPAXIS": 1}),
                              ("one", arc, {"DISPAXIS": 2}),
                              ("gali", np.round(gal).astype(np.int16), {"DISPAXIS": 2})):
        hdu = fits.PrimaryHDU(data)
        hdu.header.update(cards)
        hdu.header["OBJECT"] = name
        hdu.writeto(where / f"{name}.fits", overwrite=True)
    (where / "database" / "idarc").write_text(database("arc", columns), encoding="utf-8")
    (where / "database" / "idarcl").write_text(database("arcl", columns, offset=20.0),
                                                encoding="utf-8")
    (where / "database" / "idone").write_text(database("one", columns, waves=[6421.7]),
                                               encoding="utf-8")
    bpm = np.zeros((NY, NX), np.int16)
    bpm[100:103, 20:23] = 1
    bpm[250, :] = 3
    fits.PrimaryHDU(bpm).writeto(where / "bpm.fits", overwrite=True)
    (where / "dels.db").write_text(
        f"6359.0 1\n6522.6 41\n{WAVES[2]} 61\n", encoding="utf-8")
    (where / "database" / "idarct").write_text(database("arct", columns, transposed=True),
                                                encoding="utf-8")


if __name__ == "__main__":
    write(Path(sys.argv[1]))
