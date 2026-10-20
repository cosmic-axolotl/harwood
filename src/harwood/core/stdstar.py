"""standard's measurement, as IRAF's t_standard.x and getcalib.x make it.

A standard star is measured in bands: the counts inside each band, summed
with the end pixels by the fraction they cover and each pixel corrected for
the extinction difference from the band's centre. The bands are the
calibration table's, or evenly spaced ones (bandwidth, bandsep), and they
belong to the *aperture*: bands added or deleted while looking at one
spectrum stay that way for the next spectrum of the same aperture, as in
IRAF. Each aperture reads its own calibration star unless samestar.

Beam-switched data come as pairs of spectra of one aperture, an object
(OFLAG 1) and a sky (OFLAG 0), in either order; the sky is subtracted from
the object, rebinned onto it when their dispersions differ (shdr_rebin).

A spectrum already corrected for extinction (EX-FLAG 0) has the correction
taken out again first: the extinction belongs in sensfunc.

From iraf-community 2.18: noao/onedspec/t_standard.x, getcalib.x,
getairm.x, smw/shdr.x (shdr_rebin). See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import advice, flux

f32 = np.float32
f64 = np.float64

SKY, OBJECT = 0, 1
ECYES, ECNO = 0, -1

#: clgwrd's dictionary for `answer`, and what each word means.
ANSWERS = ("no", "yes", "N", "Y", "NO", "YES", "NO!", "YES!")
NO1, YES1, N2, Y2, NO2, YES2, NO3, YES3 = range(1, 9)
_EDIT = (YES1, Y2, YES2, YES3)

KEYS = """STANDARD TASK CURSOR KEY OPTIONS

?  Display help page
a  Add a new band by marking the endpoints
d  Delete band nearest the cursor in wavelength
r  Redraw current plot
q  Quit with current bandpass definitions
w  Window plot
I  Interrupt task immediately

:show   Show current bandpass data"""


# --------------------------------------------------------- calibration data


@dataclass
class Calibration:
    """getcalib: a star's bands (centre, width, magnitude), in Angstroms."""

    waves: list
    dwaves: list
    mags: list
    path: Path | None = None
    notes: list[str] = field(default_factory=list)
    star: str = ""

    def magnitude(self, wave: float) -> f32:
        """std_addband's intrp(MAG_LOOKUP) in the table."""
        return f32(flux.intrp(np.asarray(self.waves, f32), np.asarray(self.mags, f32),
                              f32(wave)))


#: units.h: each unit's class and its scale from Angstroms.
_UNITS = {"angstroms": ("wave", 1.0), "nanometers": ("wave", 0.1),
          "millimicrons": ("wave", 0.1), "microns": ("wave", 1e-4),
          "millimeters": ("wave", 1e-7), "centimeters": ("wave", 1e-8),
          "meters": ("wave", 1e-10), "hertz": ("freq", 2.9979e18),
          "kilohertz": ("freq", 2.9979e15), "megahertz": ("freq", 2.9979e12),
          "gigahertz": ("freq", 2.9979e9), "ev": ("energy", 1.2396e4),
          "kev": ("energy", 1.2396e1), "mev": ("energy", 1.2396e-2)}
#: noao$lib/units.dat.
_ABBREVIATIONS = {"nm": "nanometers", "mm": "millimeters", "cm": "centimeters",
                  "m": "meters", "Hz": "hertz", "KHz": "kilohertz", "MHz": "megahertz",
                  "GHz": "gigahertz", "wn": "inverse centimeters"}


def units(text: str, path=None) -> tuple[str, f32, bool, bool]:
    """un_open: (class, scale, log, inverse) for a units string such as
    'microns', 'nm', 'log angstroms' or 'inverse centimeters'."""
    words = _ABBREVIATIONS.get(text.strip(), text.strip()).lower().split()
    log = inverse = False
    for word in words[:3]:
        matches = [name for name in (*_UNITS, "log", "inverse") if name.startswith(word)]
        name = word if word in matches else (matches[0] if len(matches) == 1 else None)
        if name == "log" and not log:
            log = True
        elif name == "inverse" and not inverse:
            inverse = True
        elif name in _UNITS:
            kind, scale = _UNITS[name]
            return kind, f32(scale), log, inverse
        else:
            break
    advice.fail(f"'{path or text}' gives its wavelengths in '{text}', which are not "
                "units IRAF's onedspec knows.",
                "write '# units angstroms' (or nanometers, microns, hertz, ev...).")


def to_angstroms(values, unit: tuple[str, f32, bool, bool]) -> list:
    """un_ctranr from `unit` to Angstroms, in single precision."""
    kind, scale, log, inverse = unit
    out = []
    for v in values:
        v = f32(v)
        if log:
            v = f32(f64(10.0) ** f64(v))
        if inverse:
            v = f32(f32(1) / v)
        v = f32(v / scale) if kind == "wave" else f32(scale / v)
        out.append(v)
    return out


def read_calibration(caldir: str, star: str, mag: float | None = None,
                     magband: str = "", teff: str = "") -> Calibration:
    """getcalib: the star's table in `caldir`, or a blackbody of magnitude
    `mag` in `magband` and temperature (or spectral type) `teff` when the
    table is a '# type blackbody' one."""
    path = flux.find_star(star, caldir)
    if path is None:
        raise LookupError(star)
    kind, unit_name, band, weff = "star", "angstroms", "", None
    waves, mags, dwaves = [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "#":
            what = words[1].lower() if len(words) > 1 else ""
            value = words[2] if len(words) > 2 else ""
            if what == "type":
                kind = value if value in ("star", "blackbody") else "unknown"
            elif what == "units":
                unit_name = value
            elif what == "band":
                band = value
            elif what == "weff":
                weff = _real(value)
            continue
        values = [_real(word) for word in words[:3]]
        if len(values) < 3 or None in values:     # nscan() != 3
            continue
        waves.append(f32(values[0]))
        mags.append(f32(values[1]))
        dwaves.append(f32(values[2]))
    if not waves:
        advice.fail(f"No calibration data found in '{path}'.",
                    "the table needs one line per band: wavelength, magnitude, "
                    "bandwidth.")
    unit = units(unit_name, path)
    for i, (wave, dwave) in enumerate(zip(waves, dwaves, strict=True)):
        w1, w2 = to_angstroms([f32(wave - dwave / f32(2)), f32(wave + dwave / f32(2))], unit)
        waves[i] = f32((w1 + w2) / f32(2))
        dwaves[i] = f32(abs(w1 - w2))
    if weff is not None:
        weff = to_angstroms([weff], unit)[0]
    calibration = Calibration(waves, dwaves, mags, path)
    if kind == "unknown":
        advice.fail(f"Unknown calibration type in '{path}'.",
                    "the '# type' line should say star or blackbody.")
    if kind == "blackbody":
        _blackbody(calibration, band, weff, path.parent, mag, magband, teff)
    return calibration


def _real(text: str):
    """gargr: a number, or None for anything else (INDEF included)."""
    from .iraffmt import INDEF, ctod

    value = ctod(text)
    return None if value is None or abs(value) >= INDEF else float(value)


def colour_column(bands: list[str], band: str) -> int:
    """Which of params.dat's colour columns (counted after Teff) is `band`'s.

    IRAF's getbbcal numbers the columns from the spectral type (V is 3) and
    then counts them after the temperature, so it reads the column two to
    the right: converting a K5V's V magnitude to J it takes K - H (-0.11)
    for J - V (-2.13), and for most bands it finds INDEF and stops. harwood
    reads the band's own column."""
    return bands.index(band)


def _blackbody(cal: Calibration, band: str, weff, caldir: Path, mag, magband, teff) -> None:
    """getbbcal: the table's magnitudes made a blackbody's of magnitude
    `mag` in `magband` and temperature (or spectral type) `teff`."""
    if not band or weff is None:
        advice.fail("Blackbody calibration file has no band or effective wavelength",
                    "the file needs '# band V' and '# weff 0.55' lines.")
    params = caldir / "params.dat"
    lines = params.read_text().splitlines() if params.is_file() else []
    bands: list[str] = []
    for line in lines:
        words = line.split()
        if len(words) > 2 and words[0] == "#" and words[1] == "Type":
            bands = words[3:]
            break
    if band not in bands:
        bands.append(band)
    if mag is None:
        advice.fail("a blackbody calibration needs the star's magnitude.",
                    "give --mag, with --magband for its band and --teff for the "
                    "temperature or the spectral type (A0V).")
    if magband not in bands:
        advice.fail(f"magband '{magband}' is not one of {', '.join(bands)}.",
                    "give --magband as one of those.")
    rows = [line.split() for line in lines if line.split() and line.split()[0] != "#"]
    sptype = "".join(str(teff or "").split())
    temperature = _real(sptype)
    if temperature is None:
        if not sptype:
            advice.fail("a blackbody calibration needs a temperature or a spectral type.",
                        "give --teff 9500, or --teff A0V.")
        default = ""
        for row in rows:
            t = _real(row[1]) if len(row) > 1 else None
            if row[0][:2] != sptype[:2] or t is None:
                continue
            default, temperature = row[0], f32(t)
            if default == sptype:
                break
        if temperature is None:
            advice.fail(f"Failed to determine effective temperature for '{sptype}'",
                        f"use a spectral type {params} lists, or give the temperature.")
        if default != sptype:
            cal.notes.append(f"WARNING: Effective temperature for {sptype} not found "
                             f"- using {default}")
            sptype = default
    else:
        temperature, sptype = f32(temperature), ""
    mag = f32(mag)
    if magband != band:
        if not sptype:
            dt = math.inf
            for row in rows:
                t = _real(row[1]) if len(row) > 1 else None
                if len(row[0]) < 3 or row[0][2] != "V" or t is None:
                    continue
                if abs(f32(t) - temperature) < dt:
                    dt, sptype = abs(f32(t) - temperature), row[0]
            if not sptype:
                advice.fail("Failed to determine spectral type",
                            f"{params} has no dwarf (V) types to convert with.")
            cal.notes.append(f"WARNING: Assuming spectral type of {sptype}")
        col1, col2 = colour_column(bands, band), colour_column(bands, magband)
        dm, default = None, ""
        for row in rows:
            if row[0][:2] != sptype[:2]:
                continue
            values = row[2:]
            m1 = _real(values[col1]) if col1 < len(values) else None
            m2 = _real(values[col2]) if col2 < len(values) else None
            if m1 is not None and m2 is not None:
                default, dm = row[0], f32(f32(m1) - f32(m2))
                if default == sptype:
                    break
        if dm is None:
            advice.fail(f"No information in {params} to convert {magband} mag to "
                        f"{band} mag for {sptype} star",
                        f"give the magnitude in {band} (--magband {band}), or a "
                        "spectral type whose colours the table has.")
        if default != sptype:
            cal.notes.append(f"WARNING: Converting {magband} mag to {band} mag using "
                             f"spectral type {default}")
        mag = f32(mag + dm)
        cal.notes.append(f"Blackbody: {magband} = {float(f32(mag - dm)):.2f}, "
                         f"{band} = {float(mag):.2f}, Teff = {int(temperature)}")
    else:
        cal.notes.append(f"Blackbody: {band} = {float(mag):.2f}, Teff = {int(temperature)}")

    def planck(w):
        # f2c: the cube in single precision, times exp() - 1 in double,
        # stored single before log10 (getcalib.c, r__1).
        w = f32(w)
        x = f32(f64(f32(w * f32(w * w)))
                * (math.exp(f32(f32(1.4387e8) / f32(w * temperature))) - 1))
        return 2.5 * math.log10(x)

    m1 = f32(-planck(weff))
    cal.mags = [f32(f64(f32(f32(m + mag) + m1)) + planck(w))
                for m, w in zip(cal.mags, cal.waves, strict=True)]


# --------------------------------------------------------- the spectrum


@dataclass
class Spectrum:
    """What shdr gives std_calib: one aperture of one image."""

    image: str
    ap: int
    title: str
    wave: np.ndarray            #: wavelength of each pixel (double)
    y: np.ndarray               #: the counts, single precision
    airmass: float | None
    exptime: float | None
    oflag: int = OBJECT
    ec: int = ECNO
    section: str = ""
    object: str = ""            #: the OBJECT keyword, for naming the star

    def __post_init__(self) -> None:
        self.wave = np.asarray(self.wave, f64)
        self.y = np.asarray(self.y, f32).copy()
        self.x = self.wave.astype(f32)
        self.lw, self.wl = flux._pixel_map(self.wave)

    @property
    def n(self) -> int:
        return self.wave.size

    @property
    def w0(self) -> float:
        return float(self.wave[0])

    @property
    def wend(self) -> float:
        return float(self.wave[-1])

    @property
    def wp(self) -> f32:
        """shdr's WP: (W1 - W0) / (n - 1), from the single-precision ends."""
        return f32((f32(self.wend) - f32(self.w0)) / f32(self.n - 1))


def rebin(sky: Spectrum, ref: Spectrum, interp: str = "poly5") -> None:
    """shdr_rebin: `sky` resampled onto `ref`'s pixels, each output pixel
    the average of the interpolant over it. Nothing when they share a
    dispersion."""
    from .asi import Interpolant
    from .iraffmt import _equal

    n = sky.n
    if n == ref.n and _equal(f32(sky.w0), f32(ref.w0)) and \
            _equal(f32((sky.wend - sky.w0) / (n - 1)), f32((ref.wend - ref.w0) / (ref.n - 1))):
        return
    asi = Interpolant(interp, np.concatenate(([sky.y[0]], sky.y, [sky.y[-1]])))
    xmin, xmax = 0.5, n + 0.5

    def nint(v):
        return int(math.floor(v + 0.5))

    out = np.zeros(ref.n, f32)
    b = f32(max(xmin, min(xmax, sky.wl(ref.lw(0.5)))) + 1)
    for i in range(1, ref.n + 1):
        a = b
        b = f32(max(xmin, min(xmax, sky.wl(ref.lw(i + 0.5)))) + 1)
        lo, hi = (a, b) if a <= b else (b, a)
        ia, ib = nint(lo + f32(0.5)), nint(hi - f32(0.5))
        if abs(lo + f32(0.5) - ia) < 0.00001 and abs(hi - f32(0.5) - ib) < 0.00001:
            total = f32(0)
            for j in range(ia, ib + 1):
                total = f32(total + f32(asi(float(j))))
            if ib - ia > 0:
                total = f32(total / f32(ib - ia))
        else:
            total = f32(asi.integral(float(lo), float(hi)))
            if hi - lo > 0:
                total = f32(total / f32(hi - lo))
        out[i - 1] = total
    sky.wave, sky.y, sky.x = ref.wave.copy(), out, ref.x.copy()
    sky.lw, sky.wl = ref.lw, ref.wl


# --------------------------------------------------------- airmass


def airmass(ra, dec, ha, st, latitude) -> float | None:
    """get_airm: from the hour angle (or ST - RA) and the declination,
    John Ball's formula with a scale height of 750; None when the header
    has not the numbers."""
    if ha is None:
        if st is None or ra is None:
            return None
        ha = f32(f32(st) - f32(ra))
    if dec is None:
        return None
    # f2c's arithmetic: quotients and squares in single precision, the
    # trigonometry and the root in double.
    rads, scale = f32(57.29577951), f32(750.0)
    lat, dec, ha = f32(latitude), f32(dec), f32(ha)
    cos_zd = f32(math.sin(lat / rads) * math.sin(dec / rads)
                 + math.cos(lat / rads) * math.cos(dec / rads)
                 * math.cos(f32(ha * f32(15)) / rads))
    x = f32(scale * cos_zd)
    return float(f32(math.sqrt(f32(f32(x * x) + scale * f32(2)) + f32(1)) - f64(x)))


# --------------------------------------------------------- the bands


@dataclass
class Bands:
    """STD_WAVES and the rest: one aperture's bands, kept sorted."""

    cal: Calibration
    waves: list = field(default_factory=list)
    dwaves: list = field(default_factory=list)
    mags: list = field(default_factory=list)
    fluxes: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.waves)

    def add(self, wave, dwave, total=0.0) -> None:
        """std_addband: inserted after every band of lower or equal
        wavelength, its magnitude interpolated in the calibration table."""
        wave, dwave = f32(wave), f32(dwave)
        mag = self.cal.magnitude(wave)
        i = len(self.waves)
        while i > 0 and self.waves[i - 1] > wave:
            i -= 1
        self.waves.insert(i, wave)
        self.dwaves.insert(i, dwave)
        self.mags.insert(i, mag)
        self.fluxes.insert(i, f32(total))

    def delete(self, j: int) -> None:
        for column in (self.waves, self.dwaves, self.mags, self.fluxes):
            del column[j]


def initial_bands(cal: Calibration, spec: Spectrum, bandwidth=None, bandsep=None) -> Bands:
    """The table's bands, or bands of `bandwidth` every `bandsep` from the
    spectrum's first wavelength."""
    bands = Bands(cal)
    if bandwidth is None:
        for wave, dwave in zip(cal.waves, cal.dwaves, strict=True):
            bands.add(wave, dwave)
        return bands
    bw = f32(bandwidth)
    sep = f32(bandsep if bandsep is not None else bandwidth)
    if sep <= 0:
        advice.fail("bandsep has to be positive, or there would be no end of bands.",
                    "give --bandsep, the distance between band centres in Angstroms.")
    wave = f32(f32(spec.w0) + bw / f32(2))
    last = f32(f32(spec.w0) + f32(spec.n - 1) * spec.wp - bw / f32(2))
    while wave <= last:
        bands.add(wave, bw)
        wave = f32(wave + sep)
    return bands


class Extinction:
    """The extinction table as ext_load leaves it, looked up as intrp does."""

    def __init__(self, curve) -> None:
        self.curve = curve

    def __call__(self, wave) -> f32:
        if self.curve is None:
            return f32(0)
        return f32(flux.extinction_at(float(f32(wave)), self.curve))


def band_flux(spec: Spectrum, wave, dwave, ext: Extinction, edge: float = 0.0) -> f32:
    """std_flux: the counts in the band, the end pixels by the fraction
    inside, each corrected by 10^(0.4 X (k - k_centre)); 0 when the band is
    not wholly inside the spectrum (less `edge` Angstroms at each end)."""
    wave, dwave = f32(wave), f32(dwave)
    w1 = f64(f32(wave - dwave / f32(2)))
    w2 = f64(f32(wave + dwave / f32(2)))
    w3, w4 = spec.lw(0.5), spec.lw(spec.n + 0.5)
    if w1 < min(w3, w4) + edge or w2 > max(w3, w4) - edge:
        return f32(0)
    a = f32(spec.wl(w1))
    x2 = f32(spec.wl(w2))
    x1, x2 = min(a, x2), max(a, x2)
    i1 = int(math.floor(x1 + f32(0.5)))
    i2 = int(math.floor(f32(x2 - f32(0.00001)) + f32(0.5)))
    if x1 == x2 or i1 < 1 or i2 > spec.n:
        return f32(0)
    am = f32(spec.airmass)
    ec = ext(wave)
    factor = f32(f32(0.4) * am)

    def corrected(i):
        return f64(spec.y[i - 1]) * 10.0 ** f64(f32(factor * f32(ext(spec.x[i - 1]) - ec)))

    if i1 == i2:
        return f32(f64(f32(x2 - x1)) * corrected(i1))
    total = f32(f64(f32(f32(i1) + f32(0.5) - x1)) * corrected(i1))
    for i in range(i1 + 1, i2):
        total = f32(total + corrected(i))
    return f32(total + f64(f32(f32(x2 - f32(i2)) + f32(0.5))) * corrected(i2))


def prepare(obj: Spectrum, sky: Spectrum | None, ext: Extinction,
            interp: str = "poly5") -> None:
    """std_calib's first steps: the sky taken off, an extinction correction
    already applied taken out."""
    if sky is not None:
        rebin(sky, obj, interp)
        obj.y = (obj.y - sky.y).astype(f32)
    if obj.ec == ECYES:
        am = f32(obj.airmass)
        for i in range(obj.n):
            k = ext(obj.x[i])
            obj.y[i] = f32(f64(obj.y[i]) * 10.0 ** f64(f32(f32(-0.4) * k) * am))


def measure(spec: Spectrum, bands: Bands, ext: Extinction, edge: float = 0.0) -> None:
    for i in range(len(bands)):
        bands.fluxes[i] = band_flux(spec, bands.waves[i], bands.dwaves[i], ext, edge)


# --------------------------------------------------------- the editor


class Editor:
    """std_calib's cursor loop on one spectrum: a marks a band's two ends,
    d deletes the band nearest in wavelength, :show lists the bands, q
    ends. A window and a cursor file (IRAF's cursor) drive it alike."""

    def __init__(self, spec: Spectrum, bands: Bands, ext: Extinction,
                 sky: Spectrum | None = None, fnuzero: float = flux.FNUZERO,
                 edge: float = 0.0) -> None:
        self.spec, self.bands, self.ext, self.sky = spec, bands, ext, sky
        self.fnuzero, self.edge = fnuzero, edge
        self.pending: float | None = None
        self.done = False
        self.messages: list[str] = []

    def key(self, x: float, key: str, command: str = "") -> None:
        if self.pending is not None:          # the second end of an 'a'
            wx1, self.pending = self.pending, None
            self._add(wx1, x)
            return
        if key == ":":
            if command.strip() == "show":
                self.messages.append("\n".join(record_lines(self.spec, self.sky,
                                                            self.bands, self.fnuzero)))
            else:
                self.messages.append("\a")
        elif key == "?":
            self.messages.append(KEYS)
        elif key == "a":
            self.pending = f32(x)
            self.messages.append("a again:")
        elif key == "d":
            self._delete(f32(x))
        elif key == "q":
            self.done = True
        elif key == "I":
            advice.fail("Interrupt", "standard was interrupted from the cursor.")
        elif key in ("r", "w"):
            pass
        else:
            self.messages.append("\a")

    def _add(self, wx1, wx2) -> None:
        wx1, wx2 = f32(wx1), f32(wx2)
        if wx1 == wx2:
            self.messages.append("\aTwo cursor positions required")
            return
        # IRAF keeps the width signed, so a band marked right to left would
        # be written with a negative width that sensfunc then throws out;
        # harwood keeps the band.
        wave = f32((wx1 + wx2) / f32(2))
        dwave = f32(abs(wx2 - wx1))
        self.bands.add(wave, dwave, band_flux(self.spec, wave, dwave, self.ext, self.edge))

    def _delete(self, wx1) -> None:
        if not len(self.bands):
            self.messages.append("\a")
            return
        best, j = None, 0
        for i, wave in enumerate(self.bands.waves):
            if best is None or abs(wx1 - wave) < best:
                best, j = abs(wx1 - wave), i
        self.bands.delete(j)


# --------------------------------------------------------- the output


def record_lines(obj: Spectrum, sky: Spectrum | None, bands: Bands,
                 fnuzero: float = flux.FNUZERO) -> list[str]:
    """std_output: the lines IRAF's standard appends for one spectrum."""
    from .iraffmt import printf

    name = f"[{obj.image}]" + (f"-[{sky.image}]" if sky is not None else "")
    wend = f32(f32(obj.w0) + f32(obj.n - 1) * obj.wp)
    lines = [f"{name} {obj.ap} {obj.n} {f32(obj.exptime):.2f} {f32(obj.airmass):5.3f} "
             f"{f32(obj.w0):9.3f} {wend:9.3f} {obj.title}"]
    zero = f32(fnuzero)
    for wave, dwave, mag, total in zip(bands.waves, bands.dwaves, bands.mags,
                                       bands.fluxes, strict=True):
        if total == 0:
            continue
        flambda = flambda_of(mag, wave, zero)
        lines.append(f"{wave:8.2f} {printf(float(flambda), 'g', 5, 12):>12} {dwave:8.3f} "
                     f"{printf(float(total), 'g', 5, 12):>12}")
    return lines


def flambda_of(mag, wave, fnuzero=flux.FNUZERO) -> f32:
    """fnuzero * 10^(-0.4 mag) * c / wave^2, in single precision."""
    return f32(f64(f32(fnuzero)) * 10.0 ** f64(f32(f32(-0.4) * f32(mag)))
               * f64(f32(flux.LIGHT)) / f64(f32(f32(wave) ** 2)))


# --------------------------------------------------------- t_standard


@dataclass
class Record:
    """One spectrum measured: what std_output writes for it."""

    obj: Spectrum
    sky: Spectrum | None
    waves: list
    dwaves: list
    mags: list
    fluxes: list
    lines: list[str]
    star: str = ""
    table: Path | None = None


@dataclass
class _Std:
    ap: int
    bands: Bands | None = None
    kind: int | None = None          # STD_TYPE: None, SKY or OBJECT
    held: Spectrum | None = None     # STD_SH: the first of a beam-switched pair
    iflag: int = NO3
    star: str = ""


class CursorFile:
    """IRAF's cursor file read as clgcur reads it, across every spectrum:
    'x y wcs key' or 'x y wcs :command'; once at its end, at its end."""

    def __init__(self, path) -> None:
        self.events = []
        for line in Path(path).read_text().splitlines():
            words = line.split(None, 3)
            if len(words) < 4 or line.lstrip().startswith("#"):
                continue
            key = words[3].strip()
            key = {"\\040": " ", "\\40": " "}.get(key, key)
            if key.startswith(":"):
                self.events.append((float(words[0]), ":", key[1:]))
            else:
                self.events.append((float(words[0]), key[:1], ""))
        self.position = 0

    def edit(self, editor: Editor) -> None:
        while not editor.done and self.position < len(self.events):
            x, key, command = self.events[self.position]
            self.position += 1
            editor.key(x, key, command)


def run(spectra, ext: Extinction, calibration, *, samestar: bool = True,
        beam_switch: bool = False, bandwidth=None, bandsep=None, interact: bool = False,
        answer=None, cursor=None, fnuzero: float = flux.FNUZERO, edge: float = 0.0,
        interp: str = "poly5", say=None) -> list[Record]:
    """t_standard's loop over the spectra (already chosen by aperture):
    `calibration(ap)` gives an aperture's Calibration, `answer(spec)` the
    answer to 'Edit bandpasses?', `cursor.edit(editor)` the editing, and
    `say` hears what IRAF prints."""
    say = say or (lambda text: None)
    if beam_switch:
        samestar = True
    interactive = YES1 if interact else NO3
    stds: list[_Std] = []
    records: list[Record] = []
    for spec in spectra:
        if not beam_switch or spec.oflag == OBJECT:
            say(f"{spec.image}{spec.section}({spec.ap}): {spec.title}")
        std = next((s for s in stds if s.ap == spec.ap), None)
        if std is None:
            std = _Std(spec.ap, iflag=interactive)
            cal = calibration(spec.ap) if not samestar or not stds else stds[0].bands.cal
            std.star = cal.star
            for note in cal.notes:
                say(note)
            std.bands = initial_bands(cal, spec, bandwidth, bandsep)
            stds.append(std)
        if beam_switch:
            if std.kind is None:
                std.kind, std.held = spec.oflag, spec
                continue
            if std.kind == SKY:
                obj, sky = spec, std.held
                if spec.oflag == SKY:
                    say(f"{sky.image}[{sky.ap}]: Object spectrum not found")
                    std.held = spec
                    continue
            else:
                obj, sky = std.held, spec
                if spec.oflag == OBJECT:
                    say(f"{obj.image}[{obj.ap}]: Sky spectrum not found")
                    std.held = spec
                    continue
        else:
            obj, sky = spec, None

        prepare(obj, sky, ext, interp)
        measure(obj, std.bands, ext, edge)
        if std.iflag in (NO1, YES1):
            word = answer(obj) if answer is not None else "no"
            if word not in ANSWERS:
                advice.fail(f"'{word}' is not an answer to 'Edit bandpasses?'",
                            "answer one of " + ", ".join(ANSWERS) + ".")
            std.iflag = ANSWERS.index(word) + 1
        if std.iflag in _EDIT and cursor is not None:
            editor = Editor(obj, std.bands, ext, sky, fnuzero, edge)
            cursor.edit(editor)
            for message in editor.messages:
                if message.strip("\a"):
                    say(message.strip("\a"))
        bands = std.bands
        records.append(Record(obj, sky, list(bands.waves), list(bands.dwaves),
                              list(bands.mags), list(bands.fluxes),
                              record_lines(obj, sky, bands, fnuzero), std.star,
                              bands.cal.path))
        if interactive == YES1 and std.iflag in (NO3, YES3):
            interactive = std.iflag
            for other in stds:
                other.iflag = interactive
        if beam_switch:
            std.kind, std.held = None, None
    for std in stds:
        if std.kind == SKY:
            say(f"{std.held.image}[{std.held.ap}]: Object spectrum not found")
        elif std.kind == OBJECT:
            say(f"{std.held.image}[{std.held.ap}]: Sky spectrum not found")
    return records
