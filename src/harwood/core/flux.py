"""Turning counts into physical flux.

An extracted, wavelength-calibrated spectrum is counts against Angstroms.
Counts are a property of the detector, the telescope, the grating, the slit
width, the exposure time and how much atmosphere the light came through --
none of which are properties of the star. Flux calibration removes all of it
at once, by observing a star whose true flux somebody else has already
measured and asking what this instrument did to it.

Three steps, and IRAF gives each its own task:

``standard``   measure the calibration star in bands, and pair each band with
               the flux it is supposed to have
``sensfunc``   fit the ratio -- the instrument's response, in magnitudes
``calibrate``  apply it, and the extinction, to the object

The middle step is where the physics is. The ratio of observed counts to true
flux is not a constant: it varies with wavelength because the grating, the
coating and the detector all do, and it varies with *airmass* because the
atmosphere absorbs. Those two have to be separated, because the first belongs
to the instrument and stays put while the second belongs to the night and
changes between exposures. Separating them is what the extinction curve is
for: it says how much the atmosphere takes at each wavelength per unit
airmass, so the observation can be corrected to what it would have been above
the atmosphere before the instrument's own response is fitted.

**The magnitudes in the calibration files** are defined by a zero point rather
than by a filter: a flux density of `FNUZERO` erg/cm^2/s/Hz is magnitude zero.
IRAF's default, 3.68e-20, is the AB system to within 0.015 mag. So

    F_nu   = FNUZERO * 10^(-0.4 m)
    F_lam  = F_nu * c / lambda^2

with c in Angstroms per second, which is the conversion done here and the one
thing in this module that is a convention rather than a measurement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import advice

#: Flux density of a zero-magnitude source, erg/cm^2/s/Hz. IRAF's `fnuzero`.
FNUZERO = 3.68e-20

#: Speed of light in Angstroms per second, for F_nu to F_lambda: IRAF's
#: VLIGHT, which standard and calibrate both use.
LIGHT = 2.997925e18

#: What a calibrated flux is in. Written to the header as BUNIT.
FLUX_UNIT = "erg/cm2/s/angstrom"

#: Where `onedstds$` is looked for: harwood's own copy of IRAF's standards
#: and extinction curves first, then an installed IRAF, for a file someone has
#: added to theirs.
IRAF_STANDARDS = (
    Path(__file__).resolve().parent.parent / "data" / "iraf" / "onedstds",
    # Lower case on purpose: `iraf` is the name IRAF itself sets.
    Path(os.environ.get("iraf", "/usr/lib/iraf")) / "noao" / "lib" / "onedstds",  # noqa: SIM112
    Path("/usr/share/iraf/noao/lib/onedstds"),
)

#: The calibration directories, with what each is for.
CALDIRS = {
    "spec50cal": "50 Angstrom bands, the usual choice for a low-resolution "
                 "spectrograph",
    "spec16cal": "16 Angstrom bands, for higher resolution",
    "spechayescal": "Hayes (1985) spectrophotometric standards",
    "oke1990": "Oke (1990) HST calibration standards",
    "iidscal": "the IIDS standards, 50 Angstrom bands",
    "irscal": "the IRS standards",
    "ctiocal": "CTIO standards",
    "ctionewcal": "the revised CTIO standards",
    "bstdscal": "bright stars, for a small telescope",
    "redcal": "standards extending to the red",
    "blackbody": "blackbody curves rather than measured stars",
}


@dataclass
class Band:
    """One bandpass of a calibration star: what it should be."""

    wavelength: float        #: centre, Angstroms
    width: float             #: full width of the band, Angstroms
    magnitude: float

    @property
    def flux(self) -> float:
        """True flux density, erg/cm^2/s/Angstrom."""
        return magnitude_to_flux(self.magnitude, self.wavelength)


@dataclass
class Measurement:
    """One band of a calibration star: what it should be, and what was seen."""

    wavelength: float
    width: float
    magnitude: float
    counts: float            #: observed, per second per Angstrom
    airmass: float = 1.0
    star: str = ""
    used: bool = True
    wstart: float | None = None  #: the standard spectrum's first wavelength
    wend: float | None = None    #: and its last
    npts: int | None = None      #: and its length
    aperture: int = 1
    image: str = ""              #: the spectrum measured ([obj]-[sky] for a pair)
    flambda: float | None = None  #: the true flux as standard wrote it (its fnuzero)

    @property
    def flux(self) -> float:
        if self.flambda is not None:
            return self.flambda
        return magnitude_to_flux(self.magnitude, self.wavelength)

    def sensitivity(self, extinction: float = 0.0) -> float:
        """Instrument response at this band, in magnitudes.

        The extinction is taken out first, so what is left belongs to the
        instrument rather than to the night:

            S = 2.5 log10( counts_above_atmosphere / true flux )
        """
        corrected = self.counts * 10.0 ** (0.4 * extinction * self.airmass)
        if corrected <= 0 or self.flux <= 0:
            return float("nan")
        return 2.5 * np.log10(corrected / self.flux)


@dataclass
class Standard:
    """Everything the `standard` task measured, and what sensfunc needs."""

    measurements: list[Measurement] = field(default_factory=list)
    extinction_name: str = ""
    caldir: str = ""

    def __len__(self) -> int:
        return len(self.measurements)


def magnitude_to_flux(magnitude: float, wavelength: float,
                      fnuzero: float = FNUZERO) -> float:
    """A calibration magnitude as a flux density per Angstrom."""
    f_nu = fnuzero * 10.0 ** (-0.4 * magnitude)
    return f_nu * LIGHT / (wavelength ** 2)


def flux_to_magnitude(flux: float, wavelength: float,
                      fnuzero: float = FNUZERO) -> float:
    """The inverse, for reporting."""
    f_nu = flux * wavelength ** 2 / LIGHT
    return -2.5 * np.log10(f_nu / fnuzero)


# ------------------------------------------------------ the shipped tables


def resolve(name: str | Path, caldir: str = "") -> Path:
    """Find a standard star or extinction file, IRAF's `onedstds$` accepted.

    IRAF's tables ship with harwood (data/iraf/onedstds), so no IRAF is
    needed; see IRAF_STANDARDS for the order they are looked in.
    """
    text = str(name)
    if "$" in text:
        prefix, _, filename = text.partition("$")
        if prefix != "onedstds":
            advice.fail(
                f"unknown IRAF directory prefix '{prefix}$'.",
                "flux calibration data lives under onedstds$, as in "
                "onedstds$kpnoextinct.dat.")
        for root in IRAF_STANDARDS:
            candidate = root / filename
            if candidate.is_file():
                return candidate
        advice.fail(
            f"there is no '{filename}' under onedstds$.",
            "`harwood standards` lists the calibration directories and their "
            "stars; or give the path of a file of your own.", FileNotFoundError)

    path = Path(text)
    if path.is_file():
        return path
    if caldir:
        found = find_star(text, caldir)
        if found is not None:
            return found
    advice.fail(f"'{path}' does not exist.",
                "give a path to a calibration file, or a star name together "
                "with --caldir. 'harwood standards' lists what is available.",
                FileNotFoundError)


def _roots() -> list[Path]:
    return [root for root in IRAF_STANDARDS if root.is_dir()]


def find_star(name: str, caldir: str = "") -> Path | None:
    """The calibration file for a star, by name or by one of its aliases.

    Standards are known by several names -- Feige 110 is also EG 158 -- and
    IRAF keeps the mapping in a `names.men` beside the data. Both are tried,
    and so are the usual ways of writing a name: `BD+28 4211`, `bd284211`.
    """
    key = _flatten(name)
    # A directory of one's own, as IRAF's caldir="mydir/" allows, first.
    roots = ([None] if caldir and Path(caldir).is_dir() else []) + list(_roots())
    for root in roots:
        if root is None:
            folders = [Path(caldir)]
        else:
            folders = [root / caldir] if caldir else sorted(
                child for child in root.iterdir() if child.is_dir())
        for folder in folders:
            if not folder.is_dir():
                continue
            direct = folder / f"{key}.dat"
            if direct.is_file():
                return direct
            for alias, real in _aliases(folder).items():
                if alias == key:
                    candidate = folder / f"{real}.dat"
                    if candidate.is_file():
                        return candidate
    return None


def where_is(name: str) -> list[tuple[str, Path]]:
    """Every directory that has this star, for when the caldir is in doubt."""
    found = []
    for root in _roots():
        for folder in sorted(child for child in root.iterdir() if child.is_dir()):
            path = find_star(name, folder.name)
            if path is not None:
                found.append((folder.name, path))
    return found


def _flatten(name: str) -> str:
    """`BD+28 4211` and `bd284211` are the same star."""
    return "".join(character for character in str(name).lower()
                   if character.isalnum())


def _aliases(folder: Path) -> dict[str, str]:
    """The `names.men` mapping of alternate names, when there is one."""
    path = folder / "names.men"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            out[_flatten(parts[0])] = parts[1]
    return out


def stars(caldir: str) -> list[str]:
    """Every standard in one calibration directory."""
    names = set()
    for root in _roots():
        folder = root / caldir
        if folder.is_dir():
            names.update(path.stem for path in folder.glob("*.dat"))
    return sorted(names)


def read_bands(path: Path | str) -> list[Band]:
    """A standard star file: wavelength, magnitude, bandwidth."""
    bands = []
    for line in Path(path).read_text(encoding="utf-8",
                                     errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            wavelength, magnitude = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        width = float(parts[2]) if len(parts) > 2 else 50.0
        bands.append(Band(wavelength, width, magnitude))
    if not bands:
        advice.fail(f"'{path}' holds no calibration bands.",
                    "the file should have one line per band: wavelength, "
                    "magnitude, bandwidth. Check it is a standard star table "
                    "and not something else.")
    return sorted(bands, key=lambda band: band.wavelength)


def read_extinction(name: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """An extinction curve: wavelength against magnitudes per airmass."""
    path = resolve(name)
    wavelength, value = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            wavelength.append(float(parts[0]))
            value.append(float(parts[1]))
        except ValueError:
            continue
    if len(wavelength) < 2:
        advice.fail(f"'{path}' holds no extinction curve.",
                    "the file should have one line per point: wavelength, "
                    "magnitudes per airmass. onedstds$kpnoextinct.dat is the "
                    "shape to compare against.")
    order = np.argsort(wavelength)
    return np.array(wavelength)[order], np.array(value)[order]


def extinction_at(wavelength, curve) -> np.ndarray:
    """The extinction coefficient at each wavelength, as IRAF's onedspec
    interpolates it (xtools intrp: a blend of the two parabolas through the
    neighbouring points, which continues past the table's ends)."""
    grid, value = curve
    xs = np.asarray(grid, dtype=f32)
    ys = np.asarray(value, dtype=f32)
    w = np.atleast_1d(np.asarray(wavelength, dtype=float))
    out = np.array([intrp(xs, ys, f32(x)) for x in w], dtype=float)
    return out if np.ndim(wavelength) else out[0]


f32 = np.float32


def intrp(xtab, ytab, x) -> float:
    """xtools intrp.f: interpolation in a table of up to four points around
    x, blending the two parabolas by how far apart their predictions are."""
    ntab = len(xtab)
    if ntab == 1:
        return float(ytab[0])
    idir = 1.0 if xtab[-1] - xtab[0] >= 0 else -1.0
    # srch: the first i (1-based) with xtab(i) at or beyond x; 2 below the
    # table, ntab above it.
    if (x - xtab[0]) * idir < 0:
        index = 2
    elif (x - xtab[-1]) * idir > 0:
        index = ntab
    else:
        index = ntab
        for i in range(2, ntab + 1):
            if not (x - xtab[i - 1]) * idir > 0:
                index = i
                break
    i1, i2 = 2, 3
    iload = max(index - 2, 1)
    if ntab > 2:
        if index == 2:
            i2 = 4
        if index == ntab:
            i1 = 1
    if 2 < index < ntab:
        i1, i2 = 1, 4
    t = [f32(0)] * 5
    u = [f32(0)] * 5
    for i in range(i1, i2 + 1):
        j = iload + (i - i1)
        t[i] = f32(xtab[j - 1])
        u[i] = f32(ytab[j - 1])
    return float(_codim1(f32(x), t, u, i1, i2))


def _codim1(v, t, u, i1, i2):
    one = f32(1)
    a2 = f32(v - t[2])
    al = f32(a2 / (t[3] - t[2]))
    s = f32(al * u[3] + (one - al) * u[2])
    if i1 > 1 and i2 < 4:
        return s
    a3 = f32(v - t[3])
    p1 = p2 = f32(0)
    if i1 <= 1:
        a1 = f32(v - t[1])
        c1 = f32(a2 / (t[1] - t[2]) * a3 / (t[1] - t[3]))
        c2 = f32(a1 / (t[2] - t[1]) * a3 / (t[2] - t[3]))
        c3 = f32(a1 / (t[3] - t[1]) * a2 / (t[3] - t[2]))
        p1 = f32(c1 * u[1] + c2 * u[2] + c3 * u[3])
        if i2 < 4:
            slope1 = abs(f32((u[2] - u[1]) / (t[2] - t[1])))
            slope2 = abs(f32((u[3] - u[2]) / (t[3] - t[2])))
            xe = one
            if slope1 + slope2 != 0:
                xe = f32(one - abs(slope1 - slope2) / (slope1 + slope2))
            p2 = f32(s + xe * (p1 - s))
            return _blend(s, p1, p2, al)
    a4 = f32(v - t[4])
    c4 = f32(a3 / (t[2] - t[3]) * a4 / (t[2] - t[4]))
    c5 = f32(a2 / (t[3] - t[2]) * a4 / (t[3] - t[4]))
    c6 = f32(a2 / (t[4] - t[2]) * a3 / (t[4] - t[3]))
    p2 = f32(c4 * u[2] + c5 * u[3] + c6 * u[4])
    if i1 > 1:
        slope1 = abs(f32((u[4] - u[3]) / (t[4] - t[3])))
        slope2 = abs(f32((u[3] - u[2]) / (t[3] - t[2])))
        xe = one
        if slope1 + slope2 != 0:
            xe = f32(one - abs(slope1 - slope2) / (slope1 + slope2))
        p1 = f32(s + xe * (p2 - s))
    return _blend(s, p1, p2, al)


def _blend(s, p1, p2, al):
    e1 = abs(f32(p1 - s))
    e2 = abs(f32(p2 - s))
    if not e1 + e2 > 0:
        return s
    bt = f32((e1 * al) / (e1 * al + (f32(1) - al) * e2))
    return f32(bt * p2 + (f32(1) - bt) * p1)


# --------------------------------------------------------- the measurement


def measure(aperture, bands: list[Band], exptime: float, airmass: float,
            star: str = "", edge: float = 0.0, curve=None) -> list[Measurement]:
    """IRAF's std_flux over each band: the counts in the pixels the band
    covers, the end pixels by the fraction inside, each corrected for the
    difference between the extinction at its wavelength and at the band's
    centre. A band not wholly inside the spectrum is skipped. Stored per
    second per Angstrom of band.
    """
    wave = np.asarray(aperture.wavelength, dtype=float)
    flux = np.asarray(aperture.flux, dtype=f32)
    if wave.size < 2:
        advice.fail("the spectrum has no wavelength scale.",
                    "run dispcor on it first; a band is a range of wavelengths "
                    "and there is no way to integrate over one without them.")
    n = wave.size
    lw, wl = _pixel_map(wave)
    w3, w4 = lw(0.5), lw(n + 0.5)
    low, high = min(w3, w4) + edge, max(w3, w4) - edge
    am = f32(airmass)

    def factor(i, ec):
        if curve is None:
            return f32(1)
        e = f32(extinction_at(wave[i - 1], curve))
        return f32(f32(10.0) ** f32(f32(0.4) * am * (e - ec)))

    out = []
    for band in bands:
        wavec, dwave = f32(band.wavelength), f32(band.width)
        w1 = float(wavec - dwave / f32(2))
        w2 = float(wavec + dwave / f32(2))
        if w1 < low or w2 > high:
            continue
        a = f32(wl(w1))
        x2 = f32(wl(w2))
        x1, x2 = min(a, x2), max(a, x2)
        i1 = int(np.floor(x1 + f32(0.5)))
        i2 = int(np.floor(x2 - f32(0.00001) + f32(0.5)))
        if x1 == x2 or i1 < 1 or i2 > n:
            continue
        ec = f32(extinction_at(float(wavec), curve)) if curve is not None else f32(0)
        if i1 == i2:
            total = f32((x2 - x1) * flux[i1 - 1] * factor(i1, ec))
        else:
            total = f32((f32(i1) + f32(0.5) - x1) * flux[i1 - 1] * factor(i1, ec))
            for i in range(i1 + 1, i2):
                total = f32(total + flux[i - 1] * factor(i, ec))
            total = f32(total + (x2 - f32(i2) + f32(0.5)) * flux[i2 - 1] * factor(i2, ec))
        counts = float(total) / (band.width * exptime)
        out.append(Measurement(band.wavelength, band.width, band.magnitude,
                               counts, airmass, star,
                               wstart=float(wave[0]), wend=float(wave[-1]), npts=n))
    return out


@dataclass
class SensSpectrum:
    """The sensitivity function sampled as IRAF's sensfunc writes it: a 1-D
    spectrum starting at w1 in steps of dw, in magnitudes."""

    w1: float
    dw: float
    values: np.ndarray

    def write(self, path: Path | str, title: str = "Sensitivity function") -> Path:
        from astropy.io import fits

        header = fits.Header()
        header["OBJECT"] = title
        header["CTYPE1"] = "LINEAR"
        header["CRVAL1"] = self.w1
        header["CRPIX1"] = 1.0
        header["CDELT1"] = self.dw
        header["CD1_1"] = self.dw
        header["DC-FLAG"] = 0
        header["WAT0_001"] = "system=equispec"
        header["WAT1_001"] = "wtype=linear label=Wavelength units=angstroms"
        header["BUNIT"] = "mag"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(np.asarray(self.values, dtype=f32), header).writeto(
            path, overwrite=True)
        return path

    @classmethod
    def read(cls, path: Path | str) -> SensSpectrum:
        """A sensitivity spectrum harwood or IRAF wrote."""
        from astropy.io import fits

        data = np.asarray(fits.getdata(path), dtype=f32).ravel()
        header = fits.getheader(path)
        dw = float(header.get("CDELT1", header.get("CD1_1", 1.0)))
        w1 = float(header["CRVAL1"]) + (1 - float(header.get("CRPIX1", 1.0))) * dw
        return cls(w1, dw, data)


def _pixel_map(wave):
    """lw and wl: wavelength at a pixel position and back, straight past
    the ends -- shdr_lw and shdr_wl for a spectrum given by its wavelengths."""
    wave = np.asarray(wave, dtype=float)
    n = wave.size
    pixel = np.arange(1.0, n + 1.0)
    increasing = wave[-1] > wave[0]

    def lw(p):
        if p < 1:
            return wave[0] + (p - 1) * (wave[1] - wave[0])
        if p > n:
            return wave[-1] + (p - n) * (wave[-1] - wave[-2])
        return float(np.interp(p, pixel, wave))

    def wl(w):
        ws, ps = (wave, pixel) if increasing else (wave[::-1], pixel[::-1])
        if w < ws[0]:
            return ps[0] + (w - ws[0]) / (ws[1] - ws[0]) * (ps[1] - ps[0])
        if w > ws[-1]:
            return ps[-1] + (w - ws[-1]) / (ws[-1] - ws[-2]) * (ps[-1] - ps[-2])
        return float(np.interp(w, ws, ps))

    return lw, wl


#: t_calibrate.x's speed of light, Angstroms per second.
VLIGHT = 2.997925e18
FNU_UNIT = "erg/cm2/s/Hz"


def calibrate(aperture, sens: SensSpectrum | None, curve, airmass: float, exptime: float,
              extinct: bool = True, interp: str = "poly5", fnu: bool = False):
    """IRAF's calibrate, pixel by pixel:

        F = counts x 10^(0.4 X k(lambda)) x 10^(-0.4 S(lambda)) / dlambda / t

    with S interpolated in the sampled sensitivity function (the
    interpolant of 10^(-0.4 S), clamped to its ends) and dlambda the
    width of the pixel in Angstroms. With fnu, each sample of the
    sensitivity is first multiplied by lambda^2 / c at its own wavelength,
    as cal_getflux does, so the result is per hertz. Returns (flux, error,
    pixels outside the sensitivity function's range).
    """
    from .asi import Interpolant

    wave = np.asarray(aperture.wavelength, dtype=float)
    counts = np.asarray(aperture.flux, dtype=f32)
    error = None if aperture.error is None else np.asarray(aperture.error, dtype=f32)
    lw, _ = _pixel_map(wave)
    time = f32(exptime) if exptime and exptime > 0 else f32(1)
    asi = None
    if sens is not None:
        linear = (f32(10.0) ** (f32(-0.4) * np.asarray(sens.values, dtype=f32))).astype(f32)
        if fnu:
            steps = np.arange(linear.size, dtype=f32)
            grid = (f32(sens.w1) + f32(sens.dw) * steps).astype(f32)
            linear = (linear * grid ** 2 / f32(VLIGHT)).astype(f32)
        asi = Interpolant(interp, linear)
        n = linear.size
    am = f32(airmass)
    out = np.zeros(wave.size, dtype=f32)
    err = None if error is None else np.zeros(wave.size, dtype=f32)
    nout = 0
    for k in range(1, wave.size + 1):
        w = wave[k - 1]
        ext = f32(1)
        if extinct and curve is not None:
            ext = f32(f32(10.0) ** f32(f32(0.4) * am * f32(extinction_at(w, curve))))
        fcor = f32(1)
        if asi is not None and sens is not None:
            ical = f32((w - sens.w1) / sens.dw + 1)
            if ical < 1 or ical > n:
                if ical < 0.5 or ical > n + 0.5:
                    nout += 1
                ical = f32(max(f32(1), min(f32(n), ical)))
            dw = f32(abs(lw(k + 0.5) - lw(k - 0.5)))
            fcor = f32(f32(asi(float(ical))) / dw / time)
        out[k - 1] = f32(counts[k - 1] * ext * fcor)
        if err is not None and error is not None:
            err[k - 1] = f32(error[k - 1] * ext * fcor)
    return out, err, nout
