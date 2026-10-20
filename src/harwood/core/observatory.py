"""Where the telescope is, which the header usually only hints at.

An airmass is a property of the sky as seen *from somewhere*, so computing one
needs the site. Headers rarely carry the coordinates; at most they carry a name
-- `OBSERVAT = 'KPNO'` -- and the reduction package is expected to know the
rest. IRAF keeps that knowledge in `obsdb.dat`, which ships with harwood and is
read here as IRAF reads it; a reduction can add sites of its own in an
`obsdb.dat` of the same format beside its harwood.yaml.

**The sign of the longitude.** IRAF counts longitude positive *west*, and so
does this table, because that is what an IRAF user will recognise and what
`obsdb.dat` contains. Every modern library counts it positive east, so the
conversion happens once, at `Site.east`, and nowhere else. A longitude with the
wrong sign does not fail -- it puts the object on the other side of the meridian
and returns an airmass that looks entirely reasonable -- which is why it is
worth having exactly one place where the flip happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Site:
    """An observatory, with longitude positive *west*, as IRAF writes it."""

    code: str
    name: str
    longitude: float
    latitude: float
    altitude: float
    timezone: float = 0.0

    @property
    def east(self) -> float:
        """Longitude positive east, which is what every library wants, in
        -180..180 (obsdb writes the AAO as 210:56 west)."""
        return (-self.longitude + 180.0) % 360.0 - 180.0

    def describe(self) -> str:
        vertical = "S" if self.latitude < 0 else "N"
        sideways = "W" if self.longitude >= 0 else "E"
        return (f"{self.name}: {abs(self.latitude):.3f} {vertical}, "
                f"{abs(self.longitude):.3f} {sideways}, {self.altitude:.0f} m")


#: IRAF's own table of sites, shipped with harwood (data/iraf/obsdb.dat).
OBSDB = Path(__file__).resolve().parent.parent / "data" / "iraf" / "obsdb.dat"

#: A reduction's own sites, in the same format, beside its harwood.yaml (or in
#: the working directory). Read before IRAF's, so a site there can also correct
#: one of IRAF's. `harwood observatory set` writes it.
PROJECT_OBSDB = "obsdb.dat"

#: Sites harwood knows that IRAF's table does not.
EXTRAS = (
    Site("cerropachon", "Cerro Pachon", 70.7367, -30.240, 2737, 4),
)

#: Names harwood accepts for a site besides its code, lowercased.
ALIASES = {
    "kitt peak": "kpno", "kpno 4m": "kpno", "kpno 2.1m": "kpno",
    "kpno 0.9m": "kpno", "kpno coude feed": "kpno", "tololo": "ctio",
    "eso": "lasilla", "la silla": "lasilla", "vlt": "paranal",
    "orm": "lapalma", "la palma": "lapalma", "wht": "lapalma",
    "mmt": "mmto", "mauna kea": "keck", "gemini north": "gemini-north",
    "gemini south": "gemini-south", "opd": "lna", "pico dos dias": "lna",
    "siding spring": "aao",
}


def parse(text: str) -> list[Site]:
    """IRAF's obsdb format: `observatory = "code"`, then `key = value` lines.
    Angles may be sexagesimal (111:36.0). As IRAF's obsopen reads the file,
    the first entry for a code is the one that counts."""
    sites: list[Site] = []
    current: dict | None = None

    def close() -> None:
        if current and {"longitude", "latitude"} <= set(current):
            sites.append(Site(current["code"], current.get("name", current["code"]),
                              current["longitude"], current["latitude"],
                              current.get("altitude", 0.0), current.get("timezone", 0.0)))

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, _, value = (part.strip() for part in line.partition("="))
        value = value.strip().strip('"').strip("'")
        key = key.lower()
        if key == "observatory":
            close()
            current = {"code": value.lower()}
        elif current is not None:
            current[key] = value if key == "name" else _number(value)
    close()
    return sites


def _number(text: str) -> float:
    """112:36.5 or -22:32:04 or 45.58 -> degrees (or hours, for a timezone)."""
    text = text.strip()
    if ":" not in text:
        return float(text)
    sign = -1.0 if text.startswith("-") else 1.0
    parts = [float(p) for p in text.lstrip("+-").split(":")]
    return sign * sum(p / 60.0 ** k for k, p in enumerate(parts))


@lru_cache(maxsize=1)
def _iraf() -> dict[str, Site]:
    found: dict[str, Site] = {}
    for site in parse(OBSDB.read_text(encoding="latin-1")):
        found.setdefault(site.code, site)
    for site in EXTRAS:
        found.setdefault(site.code, site)
    return found


#: Every site in IRAF's table, and harwood's few extra.
SITES = _iraf()


def project_file(start: Path | None = None) -> Path:
    """Where this reduction keeps its own sites."""
    from .params import find_project

    project = find_project(start)
    return (project.parent if project else (start or Path.cwd())) / PROJECT_OBSDB


def sites(start: Path | None = None) -> dict[str, Site]:
    """The sites a task in this directory can use: the reduction's own first,
    then IRAF's."""
    own = project_file(start)
    merged = dict(_iraf())
    if own.is_file():
        for site in parse(own.read_text(encoding="latin-1")):
            merged[site.code] = site
    return merged


def find(name: str) -> Site | None:
    """A site by code, alias, or the start of its name."""
    if not name:
        return None
    known = sites()
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key in known:
        return known[key]
    matches = [site for site in known.values() if site.name.lower().startswith(key)]
    return matches[0] if len(matches) == 1 else None


def from_header(header) -> Site | None:
    """The site a header names, through OBSERVAT or TELESCOP."""
    for keyword in ("OBSERVAT", "OBSERVATORY", "TELESCOP", "SITE"):
        site = find(str(header.get(keyword, "")))
        if site is not None:
            return site
    return None


def default(start: Path | None = None) -> Site | None:
    """The reduction's own default site, set with `harwood observatory set`."""
    from .params import load_project

    data, _ = load_project(start)
    code = str(data.get("observatory") or "").strip()
    return find(code) if code else None


def listing() -> list[str]:
    """Every site, for the message that says which names are known."""
    return sorted(sites())


def add(site: Site, start: Path | None = None) -> Path:
    """Write a site into the reduction's own obsdb.dat, in IRAF's format,
    replacing an entry of the same code there."""
    path = project_file(start)
    kept = [s for s in (parse(path.read_text(encoding="latin-1")) if path.is_file()
                        else []) if s.code != site.code]
    entries = []
    for each in [*kept, site]:
        entries.append(f'observatory = "{each.code}"\n'
                       f'\tname = "{each.name}"\n'
                       f"\tlongitude = {each.longitude:.6f}\n"
                       f"\tlatitude = {each.latitude:.6f}\n"
                       f"\taltitude = {each.altitude:g}\n"
                       f"\ttimezone = {each.timezone:g}\n")
    path.write_text("# This reduction's own observatories, in IRAF's obsdb format.\n"
                    "# Read before IRAF's table; longitude is positive WEST.\n\n"
                    + "\n".join(entries), encoding="latin-1")
    return path
