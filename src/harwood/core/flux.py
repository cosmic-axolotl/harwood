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

#: Flux density of a zero-magnitude source, erg/cm^2/s/Hz. IRAF's `fnuzero`.
FNUZERO = 3.68e-20

#: Speed of light in Angstroms per second, for F_nu to F_lambda.
LIGHT = 2.99792458e18

#: What a calibrated flux is in. Written to the header as BUNIT.
FLUX_UNIT = "erg/cm2/s/angstrom"

#: Where IRAF keeps its standard stars and extinction curves.
IRAF_STANDARDS = (
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

    @property
    def flux(self) -> float:
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

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Wavelength and sensitivity, for the fit."""
        return (np.array([item.wavelength for item in self.measurements]),
                np.array([item.sensitivity() for item in self.measurements]))


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

    Section 4 wants these shipped with harwood; until their provenance has
    been checked one by one they are read from an installed IRAF, which is the
    common case for somebody migrating.
    """
    text = str(name)
    if "$" in text:
        prefix, _, filename = text.partition("$")
        if prefix != "onedstds":
            raise ValueError(
                f"unknown IRAF directory prefix '{prefix}$'. Flux calibration "
                "data lives under onedstds$.")
        for root in IRAF_STANDARDS:
            candidate = root / filename
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"'{text}' needs IRAF's calibration files. Looked in "
            f"{', '.join(str(root) for root in IRAF_STANDARDS)}. Give a path "
            "instead, or install iraf-noao.")

    path = Path(text)
    if path.is_file():
        return path
    if caldir:
        found = find_star(text, caldir)
        if found is not None:
            return found
    raise FileNotFoundError(f"'{path}' does not exist")


def _roots() -> list[Path]:
    return [root for root in IRAF_STANDARDS if root.is_dir()]


def find_star(name: str, caldir: str = "") -> Path | None:
    """The calibration file for a star, by name or by one of its aliases.

    Standards are known by several names -- Feige 110 is also EG 158 -- and
    IRAF keeps the mapping in a `names.men` beside the data. Both are tried,
    and so are the usual ways of writing a name: `BD+28 4211`, `bd284211`.
    """
    key = _flatten(name)
    for root in _roots():
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
        raise ValueError(f"'{path}' holds no calibration bands")
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
        raise ValueError(f"'{path}' holds no extinction curve")
    order = np.argsort(wavelength)
    return np.array(wavelength)[order], np.array(value)[order]


def extinction_at(wavelength, curve) -> np.ndarray:
    """The extinction coefficient at each wavelength, interpolated.

    Outside the tabulated range the end value is held rather than
    extrapolated: an extinction curve rises steeply into the ultraviolet and a
    straight line continued past its last point goes wrong fast and quietly.
    """
    grid, value = curve
    return np.interp(np.asarray(wavelength, dtype=float), grid, value,
                     left=value[0], right=value[-1])


# --------------------------------------------------------- the measurement


def measure(aperture, bands: list[Band], exptime: float, airmass: float,
            star: str = "", edge: float = 0.0) -> list[Measurement]:
    """Integrate the observed spectrum over each calibration band.

    Counts come back per second per Angstrom, so the exposure time and the
    width of the band are already out and what is left can be compared with a
    flux density directly.

    A band only partly covered by the spectrum is dropped rather than scaled:
    a half-covered band looks like a star half as bright, and there is no way
    to tell the two apart afterwards.
    """
    wave = np.asarray(aperture.wavelength, dtype=float)
    flux = np.asarray(aperture.flux, dtype=float)
    if wave.size < 2:
        raise ValueError("the spectrum has no wavelength scale")

    low, high = float(wave.min()), float(wave.max())
    out = []
    for band in bands:
        first = band.wavelength - band.width / 2.0
        last = band.wavelength + band.width / 2.0
        if first < low + edge or last > high - edge:
            continue
        inside = (wave > first) & (wave < last)
        if inside.sum() < 2:
            continue
        # Integrate over the band and divide by its width: the mean flux
        # density, which is what a magnitude in that band means.
        #
        # The edges have to be interpolated rather than rounded to the nearest
        # sample. Integrating only over the samples that fall inside covers
        # about one pixel less than the band, and dividing that by the *full*
        # width then understates every band by the same fraction -- a pixel in
        # fifty Angstroms is four per cent, which is a flat offset in the
        # fitted response and a flat error in every calibrated flux. It is
        # invisible in the rms, because it is the same for every band.
        positions = np.concatenate(([first], wave[inside], [last]))
        values = np.concatenate((
            [np.interp(first, wave, flux)], flux[inside],
            [np.interp(last, wave, flux)]))
        total = float(np.trapezoid(values, positions))
        counts = total / (band.width * exptime)
        out.append(Measurement(band.wavelength, band.width, band.magnitude,
                               counts, airmass, star))
    return out


def sensitivity(standard: Standard, curve, spec) -> tuple:
    """Fit the instrument's response in magnitudes against wavelength.

    The extinction is removed from each band first, using the airmass of the
    exposure it came from, so what is fitted is a property of the instrument
    rather than of the night. Bands from several stars and several airmasses
    can therefore go into one fit, which is the point of doing it this way.
    """
    from . import icfit

    wavelength = np.array([item.wavelength for item in standard.measurements])
    coefficient = extinction_at(wavelength, curve)
    values = np.array([item.sensitivity(k) for item, k
                       in zip(standard.measurements, coefficient, strict=True)])

    good = np.isfinite(values)
    if good.sum() < 2:
        raise ValueError(
            "not enough usable bands to fit a sensitivity function: "
            f"{int(good.sum())} of {len(values)}. Bands with no counts, or a "
            "standard star file that does not overlap the spectrum, give "
            "nothing to fit.")
    return icfit.fit(wavelength[good], values[good], spec), wavelength[good], values[good]


def calibrate(aperture, response, curve, airmass: float, exptime: float,
              extinct: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Counts to flux, returning the flux and its uncertainty.

    The sensitivity is in magnitudes, so dividing is subtracting in
    magnitudes:

        F(lambda) = counts / (exptime * 10^(0.4 S(lambda)))

    with the counts first corrected to above the atmosphere. The uncertainty
    goes through the same multiplication: the calibration is treated as exact,
    which it is not -- the scatter of the sensitivity fit is a real error and
    it is reported separately rather than folded in, because it is
    systematic across the whole spectrum and adding it in quadrature per
    pixel would misrepresent it as noise.
    """
    wave = np.asarray(aperture.wavelength, dtype=float)
    counts = np.asarray(aperture.flux, dtype=float) / max(exptime, 1e-12)
    error = (np.asarray(aperture.error, dtype=float) / max(exptime, 1e-12)
             if aperture.error is not None else None)

    if extinct:
        coefficient = extinction_at(wave, curve)
        gain = 10.0 ** (0.4 * coefficient * airmass)
        counts = counts * gain
        if error is not None:
            error = error * gain

    scale = 10.0 ** (0.4 * response(wave))
    with np.errstate(divide="ignore", invalid="ignore"):
        calibrated = counts / scale
        uncertainty = None if error is None else error / scale
    return calibrated, uncertainty
