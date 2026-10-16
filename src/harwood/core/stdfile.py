"""The file `standard` writes and `sensfunc` reads.

IRAF's version of this is a positional text file with no header, and reading
it means counting columns. Harwood's says what it is: a comment block naming
the columns and the conventions, then the numbers, as CSV -- so a spreadsheet
opens it, `awk` cuts it, and anyone can see what the flux calibration was
built from without running anything.

It is **appended** to, not replaced. That is the point of the format: running
`standard` on a second star adds its bands to the same file and `sensfunc`
fits them all together, which is how the response gets fitted over more
wavelength than any one standard covers.
"""

from __future__ import annotations

from pathlib import Path

from .flux import Measurement, Standard

COLUMNS = ("wavelength", "width", "magnitude", "flux", "counts", "airmass",
           "star", "used", "wstart", "wend", "npts")

HEADER = [
    "# harwood flux standard measurements.",
    "#",
    "# One line per bandpass of one calibration star. Written by `standard`,",
    "# read by `sensfunc`. Appended to: a second star adds its bands here and",
    "# the response is fitted over all of them at once.",
    "#",
    "#   wavelength  band centre, Angstroms (air)",
    "#   width       full width of the band, Angstroms",
    "#   magnitude   the standard's tabulated magnitude in that band",
    "#   flux        the same as a flux density, erg/cm2/s/Angstrom",
    "#   counts      what was observed, per second per Angstrom",
    "#   airmass     of the exposure this band came from",
    "#   star        which standard",
    "#   used        0 excludes the band from the fit; edit this by hand to",
    "#               throw one out without re-measuring",
    "#   wstart,wend the standard spectrum's first and last wavelengths, and",
    "#   npts        its length: sensfunc fits over that range and samples",
    "#               its output at the finest dispersion among them",
    "#",
    "# The instrument's response at a band is",
    "#   2.5 log10( counts * 10^(0.4 k X) / flux )",
    "# with k the extinction at that wavelength and X the airmass. sensfunc",
    "# fits that against wavelength.",
]


def append(path: Path | str, measurements: list[Measurement],
           extinction: str = "", caldir: str = "") -> int:
    """Add measurements to the file, creating it if it is not there.

    Returns how many bands the file now holds.
    """
    path = Path(path)
    existing = read(path) if path.exists() else Standard()
    combined = existing.measurements + list(measurements)

    lines = list(HEADER)
    lines.append(f"# extinction: {extinction or existing.extinction_name}")
    lines.append(f"# caldir: {caldir or existing.caldir}")
    lines.append(",".join(COLUMNS))
    for item in combined:
        lines.append(",".join((
            f"{item.wavelength:.3f}", f"{item.width:.3f}",
            f"{item.magnitude:.4f}", f"{item.flux:.6e}",
            f"{item.counts:.6e}", f"{item.airmass:.4f}",
            item.star or "-", "1" if item.used else "0",
            "" if item.wstart is None else f"{item.wstart:.6f}",
            "" if item.wend is None else f"{item.wend:.6f}",
            "" if item.npts is None else str(item.npts))))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(combined)


def read(path: Path | str) -> Standard:
    """Read the file back, keeping the extinction curve it names. A file
    IRAF's standard wrote is read too."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.lstrip().startswith("["):
        return read_iraf(text)
    standard = Standard()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# extinction:"):
            standard.extinction_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("# caldir:"):
            standard.caldir = line.split(":", 1)[1].strip()
            continue
        if not line or line.startswith("#") or line.startswith("wavelength"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            standard.measurements.append(Measurement(
                wavelength=float(parts[0]), width=float(parts[1]),
                magnitude=float(parts[2]), counts=float(parts[4]),
                airmass=float(parts[5]),
                star=parts[6] if len(parts) > 6 and parts[6] != "-" else "",
                used=parts[7] != "0" if len(parts) > 7 else True,
                wstart=float(parts[8]) if len(parts) > 8 and parts[8] else None,
                wend=float(parts[9]) if len(parts) > 9 and parts[9] else None,
                npts=int(parts[10]) if len(parts) > 10 and parts[10] else None,
            ))
        except ValueError:
            continue
    standard.measurements = [item for item in standard.measurements if item.used]
    return standard


def read_iraf(text: str) -> Standard:
    """IRAF's std file: for each spectrum a line '[image] ap npts exptime
    airmass w1 w2 title', then 'wavelength flambda width counts' per band,
    the counts raw. They become counts per second per Angstrom of band and
    magnitudes back from flambda with IRAF's own constants."""
    import math

    from .flux import FNUZERO, LIGHT

    standard = Standard()
    exptime = airmass = 1.0
    w1 = w2 = None
    npts = None
    star = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            parts = line.split()
            npts = int(parts[2])
            exptime = float(parts[3]) if float(parts[3]) > 0 else 1.0
            airmass = float(parts[4])
            w1, w2 = float(parts[5]), float(parts[6])
            star = " ".join(parts[7:])
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        wave, flam, width, counts = (float(v) for v in parts[:4])
        magnitude = -2.5 * math.log10(flam * wave ** 2 / LIGHT / FNUZERO)
        standard.measurements.append(Measurement(
            wavelength=wave, width=width, magnitude=magnitude,
            counts=counts / (exptime * width), airmass=airmass, star=star,
            wstart=w1, wend=w2, npts=npts))
    return standard
