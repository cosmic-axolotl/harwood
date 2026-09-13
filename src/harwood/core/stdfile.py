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
           "star", "used")

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
            item.star or "-", "1" if item.used else "0")))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(combined)


def read(path: Path | str) -> Standard:
    """Read the file back, keeping the extinction curve it names."""
    path = Path(path)
    standard = Standard()
    for line in path.read_text(encoding="utf-8").splitlines():
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
            ))
        except ValueError:
            continue
    standard.measurements = [item for item in standard.measurements if item.used]
    return standard
