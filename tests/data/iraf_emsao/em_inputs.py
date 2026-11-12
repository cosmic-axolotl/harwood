"""The spectra of RVSAO's emsao gabarito: python em_inputs.py <folder> (harwood's venv).

Emission-line galaxies at three redshifts -- the NGC 660 nucleus of the course
(Halpha, [N II], [S II], [O I] at cz ~ 780 km/s), a blue spectrum at z = 0.05
([O II], Hgamma, Hbeta, [O III]) and one at z = 0.3 -- and the cases around them:
one line only, none, a log-linear dispersion, a two-aperture multispec, the
header a barycentric correction needs (with a dd/mm/yy date and with an
observatory's name instead of its coordinates), BCV and HCV cards to be taken from
the file, no position, and a spectrum whose continuum is already subtracted."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

NGC = {6548.06: 300, 6562.82: 1500, 6583.57: 800, 6716.44: 350, 6730.82: 260,
       6300.3: 120, 6363.8: 50}
BLUE = {3727.3: 900, 4101.7: 120, 4340.5: 220, 4861.33: 500, 4958.91: 600,
        5006.84: 1700, 3868.75: 150}


def spectrum(w, lines, z, width, cont, rng, noise):
    f = cont + 0.05 * (w - w[0])
    for rest, amp in lines.items():
        f = f + amp * np.exp(-0.5 * ((w - rest * (1 + z)) / width) ** 2)
    return (f + rng.normal(0, noise, w.size)).astype(np.float32)


def linear(name, w0, dw, data, where, **cards):
    h = fits.Header()
    h.update({"OBJECT": name, "CRVAL1": w0, "CRPIX1": 1.0, "CDELT1": dw, "CD1_1": dw,
              "CTYPE1": "LINEAR", "WCSDIM": 1, "DC-FLAG": 0,
              "WAT0_001": "system=equispec",
              "WAT1_001": "wtype=linear label=Wavelength units=angstroms",
              "APNUM1": "1 1 100.00 110.00"})
    h.update(cards)
    fits.writeto(where / f"{name}.fits", data, h, overwrite=True)


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(29)
    w = 6300.0 + 1.1 * np.arange(600)
    ngc = spectrum(w, NGC, 0.0026, 2.2, 400.0, rng, 8.0)
    linear("ngc", 6300.0, 1.1, ngc, where, APID1="nucleus")
    linear("nopos", 6300.0, 1.1, spectrum(w, NGC, 0.0026, 2.0, 300.0, rng, 6.0), where)
    site = {"DATE-OBS": "1985-12-19", "UT": "03:21:00", "EXPTIME": 1800.0,
            "RA": "01:40:21.0", "DEC": "+13:23:41", "EPOCH": 1950.0,
            "SITELONG": 45.5825, "SITELAT": -22.5344, "SITEELEV": 1864.0}
    linear("dated", 6300.0, 1.1, ngc.copy(), where, **site)
    linear("slashdate", 6300.0, 1.1, ngc.copy(), where,
           **{"DATE-OBS": "19/12/85", "UTOPEN": "02:51:00", "EXPOSURE": 3600.0,
              "RA": "01:40:21.0", "DEC": "13:23:41", "EQUINOX": 2000.0,
              "OBS-LONG": 70.815, "OBS-LAT": -30.165, "OBS-ELEV": 2215.0})
    linear("mjd", 6300.0, 1.1, ngc.copy(), where,
           **{"MJD-OBS": 46418.15, "RA": 25.0875, "DEC": 13.3947, "EPOCH": 1950.0,
              "SITELONG": 111.6, "SITELAT": 31.96, "SITEELEV": 2120.0})
    linear("observatory", 6300.0, 1.1, ngc.copy(), where,
           **{"DATE-OBS": "1985-12-19T03:00:00", "UT": "03:21:00", "EXPTIME": 1800.0,
              "RA": "01:40:21.0", "DEC": "+13:23:41", "EPOCH": 1950.0,
              "OBSERVAT": "ctio"})
    linear("filebcv", 6300.0, 1.1, ngc.copy(), where, BCV=12.34, HCV=11.02)
    linear("filehcv", 6300.0, 1.1, ngc.copy(), where, HCV=-8.5)
    single = spectrum(w, {6562.82: 900}, 0.0026, 2.4, 350.0, rng, 7.0)
    linear("single", 6300.0, 1.1, single, where)
    linear("none", 6300.0, 1.1, spectrum(w, {}, 0.0, 2.0, 350.0, rng, 7.0), where)
    sub = spectrum(w, NGC, 0.0026, 2.2, 0.0, rng, 6.0) - (0.05 * (w - w[0])).astype(
        np.float32)
    linear("subtracted", 6300.0, 1.1, sub.astype(np.float32) + 2.0, where, SUBCONT="F")

    wb = 3600.0 + 2.0 * np.arange(1000)
    linear("blue", 3600.0, 2.0, spectrum(wb, BLUE, 0.05, 3.5, 600.0, rng, 10.0), where)
    wf = 4000.0 + 3.0 * np.arange(1300)
    linear("far", 4000.0, 3.0, spectrum(wf, {**BLUE, 6562.82: 700}, 0.3, 5.0, 500.0,
                                        rng, 9.0), where)

    lw0, ldw = np.log10(6300.0), 7.3e-5
    wl = 10 ** (lw0 + ldw * np.arange(600))
    h = fits.Header()
    h.update({"OBJECT": "loglinear", "CRVAL1": lw0, "CRPIX1": 1.0, "CDELT1": ldw,
              "CD1_1": ldw, "CTYPE1": "LINEAR", "WCSDIM": 1, "DC-FLAG": 1,
              "WAT0_001": "system=equispec",
              "WAT1_001": "wtype=linear label=Wavelength units=angstroms",
              "APNUM1": "1 1 100.00 110.00"})
    fits.writeto(where / "loglinear.fits", spectrum(wl, NGC, 0.0026, 2.2, 400.0, rng, 8.0),
                 h, overwrite=True)

    ms = np.array([spectrum(w, NGC, 0.0026, 2.2, 400.0, rng, 8.0),
                   spectrum(w, NGC, 0.0032, 2.3, 380.0, rng, 8.0)], np.float32)
    h = fits.Header()
    h.update({"OBJECT": "two apertures", "WCSDIM": 2, "CTYPE1": "MULTISPE",
              "CTYPE2": "MULTISPE", "CD1_1": 1.0, "CD2_2": 1.0, "LTM1_1": 1.0,
              "LTM2_2": 1.0, "WAT0_001": "system=multispec",
              "WAT1_001": "wtype=multispec label=Wavelength units=angstroms",
              "APID1": "nucleus", "APID2": "disk"})
    wat = ('wtype=multispec spec1 = "1 1 0 6300. 1.1 600 0. 90.00 100.00" '
           'spec2 = "2 2 0 6300. 1.1 600 0. 110.00 120.00"')
    for k in range(0, len(wat), 68):         # IRAF reads no CONTINUE cards
        h[f"WAT2_{k // 68 + 1:03d}"] = wat[k:k + 68]
    fits.writeto(where / "ms.fits", ms, h, overwrite=True)

    (where / "mylines.dat").write_text(
        "6562.82  6400.0 6470.0 15.0 Ha\n"
        "6583.57  6400.0 6470.0 12.0 N2\n"
        "6716.440 6400.0 6470.0 10.0 S1\n"
        "6730.815 6400.0 6470.0 15.0 S2\n", encoding="utf-8")
    (where / "mysearch.dat").write_text("6562.82 6500.0 7700.0 Ha\n", encoding="utf-8")


if __name__ == "__main__":
    write(Path(sys.argv[1]))
