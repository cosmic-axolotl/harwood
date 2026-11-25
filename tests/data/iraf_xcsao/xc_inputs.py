"""The spectra of RVSAO's xcsao gabarito: python xc_inputs.py <folder> (harwood's venv).

Synthetic templates -- a G star (the metal lines, the G band, Mg b, Na D and the
Balmer lines), an A star (broad Balmer lines) and an emission-line galaxy -- with
the header cards xcsao reads from a template (VELOCITY, TSHIFT, SUBCONT, FI-FLAG,
DIVCONT, OVERLAP, ZEROPAD, PEAKFRAC); and objects made from them: a G star at 2500
km/s and one at -150, a galaxy at 6000 km/s broadened by 200 km/s, a noisy one,
one with emission lines over its absorption, one in log-linear dispersion, a
two-aperture multispec, one with an observatory's header (for the barycentric
correction), one with its ends zeroed and one whose counts are all under one."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

C = 299792.5
G_LINES = {3933.66: 0.55, 3968.47: 0.5, 4045.8: 0.2, 4101.74: 0.25, 4226.73: 0.25,
           4300.0: 0.3, 4340.47: 0.3, 4383.55: 0.2, 4861.33: 0.3, 5167.32: 0.2,
           5172.68: 0.25, 5183.60: 0.3, 5269.54: 0.2, 5328.0: 0.12, 5889.95: 0.35,
           5895.92: 0.3, 6122.2: 0.12, 6162.2: 0.12, 6562.80: 0.3}
A_LINES = {3889.05: 0.45, 3970.07: 0.5, 4101.74: 0.55, 4340.47: 0.6, 4861.33: 0.6,
           6562.80: 0.55, 3933.66: 0.15}
EM_LINES = {3727.3: 3.0, 4861.33: 1.5, 4958.91: 1.2, 5006.84: 3.6, 6548.06: 0.8,
            6562.82: 4.5, 6583.57: 2.4, 6716.44: 0.9, 6730.82: 0.7}


def absorb(w, lines, z, width, slope=0.0):
    f = 1000.0 * (1.0 + slope * (w - 5000.0) / 1000.0)
    for rest, depth in lines.items():
        f = f * (1.0 - depth * np.exp(-0.5 * ((w - rest * (1 + z)) / width) ** 2))
    return f


def emit(w, lines, z, width, cont):
    f = np.full(w.size, cont)
    for rest, amp in lines.items():
        f = f + amp * 100.0 * np.exp(-0.5 * ((w - rest * (1 + z)) / width) ** 2)
    return f


def broaden(w, f, sigma_kms):
    """Gaussian broadening in velocity, on the even grid in wavelength."""
    dw = w[1] - w[0]
    sigma = sigma_kms / C * w.mean() / dw
    x = np.arange(-int(5 * sigma), int(5 * sigma) + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return np.convolve(f, k / k.sum(), mode="same")


def linear(name, w0, dw, data, where, **cards):
    h = fits.Header()
    h.update({"OBJECT": name, "CRVAL1": w0, "CRPIX1": 1.0, "CDELT1": dw, "CD1_1": dw,
              "CTYPE1": "LINEAR", "WCSDIM": 1, "DC-FLAG": 0,
              "WAT0_001": "system=equispec",
              "WAT1_001": "wtype=linear label=Wavelength units=angstroms",
              "APNUM1": "1 1 100.00 110.00"})
    h.update(cards)
    fits.writeto(where / f"{name}.fits", np.asarray(data, np.float32), h, overwrite=True)


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(41)
    w = 3700.0 + 1.0 * np.arange(3000)                 # 3700..6699 A
    tg = absorb(w, G_LINES, 0.0, 1.6, 0.15)
    ta = absorb(w, A_LINES, 0.0, 6.0, -0.2)
    tem = emit(w, EM_LINES, 0.0, 2.5, 100.0)
    linear("tg", 3700.0, 1.0, tg, where, VELOCITY=0.0)
    linear("ta", 3700.0, 1.0, ta, where)
    linear("tem", 3700.0, 1.0, tem, where, SUBCONT=True)
    linear("tgvel", 3700.0, 1.0, absorb(w, G_LINES, 25.0 / C, 1.6, 0.15), where,
           VELOCITY=25.0, TSHIFT=3.5, BCV=-12.25)
    linear("tgflat", 3700.0, 1.0, tg - 1000.0 * (1.0 + 0.15 * (w - 5000.0) / 1000.0),
           where, SUBCONT=False)
    linear("tgfilt", 3700.0, 1.0, tg, where, **{"FI-FLAG": 1})
    linear("tgdiv", 3700.0, 1.0, tg, where, DIVCONT=True)
    linear("tgover", 3700.0, 1.0, tg, where, OVERLAP=True)
    linear("tgzpad", 3700.0, 1.0, tg, where, ZEROPAD=True, PEAKFRAC=0.7)
    wr = 6000.0 + 1.0 * np.arange(1500)                # red only: no overlap with blue
    linear("tred", 6000.0, 1.0, absorb(wr, G_LINES, 0.0, 1.6), where)
    wb = 3000.0 + 1.0 * np.arange(500)
    linear("tblue", 3000.0, 1.0, absorb(wb, G_LINES, 0.0, 1.6), where)

    wo = 3800.0 + 1.2 * np.arange(2200)               # 3800..6438.8 A
    def noisy(f, sigma):
        return f + rng.normal(0, sigma, f.size)
    linear("g2500", 3800.0, 1.2, noisy(absorb(wo, G_LINES, 2500.0 / C, 1.6, 0.1), 8.0),
           where)
    linear("gneg", 3800.0, 1.2, noisy(absorb(wo, G_LINES, -150.0 / C, 1.6, 0.1), 8.0),
           where)
    gal = broaden(wo, absorb(wo, G_LINES, 6000.0 / C, 1.6, 0.25), 200.0)
    linear("gal6000", 3800.0, 1.2, noisy(gal, 10.0), where)
    linear("noisy", 3800.0, 1.2, noisy(absorb(wo, G_LINES, 1200.0 / C, 1.6), 120.0),
           where)
    mix = absorb(wo, G_LINES, 4000.0 / C, 1.6) + emit(wo, EM_LINES, 4000.0 / C, 2.0, 0.0)
    emabs = noisy(mix, 8.0)
    linear("emabs", 3800.0, 1.2, emabs, where)
    # emsao's lines in the header, as RVSAO 2.8.5's emsao save_vel wrote them on emabs:
    # what xcsao's graph of the lines (l, dispmode 2) marks as found
    linear("emlines", 3800.0, 1.2, emabs.copy(), where,
           CZEM=4001.10578535799, ZEM=0.0133462504410817, CZEMERR=1.96972652633861,
           CZEMNL=2, CZEMNLF=2,
           EMLINE1="OIII 4958.91 5025.14 130.05 1.9485 4003.87 12.08 0.64 0.120",
           EMLINE2="OIII 5006.84 5073.66 356.97 2.0010 4000.73 4.45 1.79 0.880")
    wlog = 10 ** (np.log10(3800.0) + 1.0e-4 * np.arange(2000))
    flog = noisy(absorb(wlog, G_LINES, 3300.0 / C, 1.6, 0.1), 8.0)
    linear("glog", np.log10(3800.0), 1.0e-4, flog, where, **{"DC-FLAG": 1})
    site = {"DATE-OBS": "1998-10-12", "UT": "04:12:30", "EXPTIME": 1200.0,
            "RA": "02:10:31.0", "DEC": "-05:12:41", "EPOCH": 2000.0,
            "SITELONG": 70.7325, "SITELAT": -30.1653, "SITEELEV": 2215.0}
    linear("gsite", 3800.0, 1.2, noisy(absorb(wo, G_LINES, 2500.0 / C, 1.6, 0.1), 8.0),
           where, **site)
    linear("gbcv", 3800.0, 1.2, noisy(absorb(wo, G_LINES, 2500.0 / C, 1.6, 0.1), 8.0),
           where, BCV=14.75, HCV=14.5)
    ends = noisy(absorb(wo, G_LINES, 1800.0 / C, 1.6, 0.1), 8.0)
    ends[:60] = 0.0
    ends[-90:] = 0.0
    linear("gends", 3800.0, 1.2, ends, where)
    small = absorb(wo, G_LINES, 900.0 / C, 1.6, 0.1) / 2500.0
    linear("gsmall", 3800.0, 1.2, small + rng.normal(0, 0.002, small.size), where)
    # a two-aperture multispec: 2500 and 3500 km/s
    two = np.vstack([noisy(absorb(wo, G_LINES, v / C, 1.6, 0.1), 8.0)
                     for v in (2500.0, 3500.0)]).astype(np.float32)
    h = fits.Header()
    h.update({"OBJECT": "pair", "WCSDIM": 2, "CTYPE1": "MULTISPE", "CTYPE2": "MULTISPE",
              "CD1_1": 1.0, "CD2_2": 1.0, "LTM1_1": 1.0, "LTM2_2": 1.0,
              "WAT0_001": "system=multispec",
              "WAT1_001": "wtype=multispec label=Wavelength units=angstroms",
              "APID1": "first", "APID2": "second"})
    wat = ('wtype=multispec spec1 = "1 1 0 3800. 1.2 2200 0. 10. 20." '
           'spec2 = "2 2 0 3800. 1.2 2200 0. 30. 40."')
    for k in range(0, len(wat), 68):           # IRAF's own way: 68 characters a card
        h[f"WAT2_{k // 68 + 1:03d}"] = wat[k:k + 68]
    fits.writeto(where / "pair.fits", two, h, overwrite=True)
    # save_vel: emsao's results already in the header (vcombine joins them), an
    # earlier xcsao's card (kept as HISTORY), and a spectrum with RA and DEC
    star = noisy(absorb(wo, G_LINES, 2500.0 / C, 1.6, 0.1), 8.0)
    linear("gemnear", 3800.0, 1.2, star, where, CZEM=2512.5, CZEMERR=20.0, CZEMNL=4,
           CZEMNLF=3)
    linear("gemfar", 3800.0, 1.2, star, where, CZEM=2650.0, CZEMERR=12.0, CZEMNL=2,
           CZEMNLF=2)
    linear("gold", 3800.0, 1.2, star, where,
           XCSAO="rvsao.xcsao 2.8.5  01-Jan-2020 10:00 CZXC = 2400.00 R = 9.00",
           VELOCITY=2400.0, BCV=3.25)
    linear("gradec", 3800.0, 1.2, star, where, RA="10:20:30.5", DEC="+12:34:56")
    # what RVSAO 2.8.5's xcsao save_vel wrote on g2500 with tg and ta: correlate=no
    # reads it back instead of correlating
    from astropy.io import fits as _fits
    g2500 = _fits.getdata(where / "g2500.fits")
    saved = {"VELOCITY": 2498.38317783736, "Z": 0.00833370807420919,
             "CZERR": 1.45952986393666, "ZERR": 4.86846690272991E-6, "VELQUAL": "_",
             "BCV": 0.0, "CZXC": 2498.38317783736, "ZXC": 0.00833370807420919,
             "CZXCERR": 1.45952986393666, "CZXCR": 140.894373187141,
             "BESTTEMP": "tg.fits", "CZPARAMS": "2 2048 1 5 20 125 250 0.05",
             "XCTEMP1": "tg.fits 2498.383 1.460 140.894 0.980 276.420 0.000 0.000 no 0",
             "XCTEMP2": "ta.fits 2462.345 49.983 5.25513 0.388 414.609 0.000 0.000 no 0",
             "XCSAO": "rvsao.xcsao 2.8.5  25-Nov-2026 10:22 CZXC = 2498.38 R = 140.89"}
    linear("gsaved", 3800.0, 1.2, g2500.copy(), where, **saved)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
