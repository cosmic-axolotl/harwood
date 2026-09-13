"""Where the telescope is, which the header usually only hints at.

An airmass is a property of the sky as seen *from somewhere*, so computing one
needs the site. Headers rarely carry the coordinates; at most they carry a name
-- `OBSERVAT = 'KPNO'` -- and the reduction package is expected to know the
rest. IRAF keeps that knowledge in `obsdb.dat`; this is the same table for the
sites harwood is likely to meet, and `harwood observatories` prints it.

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
        """Longitude positive east, which is what every library wants."""
        return -self.longitude

    def describe(self) -> str:
        vertical = "S" if self.latitude < 0 else "N"
        sideways = "W" if self.longitude >= 0 else "E"
        return (f"{self.name}: {abs(self.latitude):.3f} {vertical}, "
                f"{abs(self.longitude):.3f} {sideways}, {self.altitude:.0f} m")


#: The sites from IRAF's obsdb.dat that harwood is likely to be pointed at.
#: Longitudes are west-positive; obsdb quotes most of them to the arcminute,
#: which is a few hundred metres on the ground and nothing at all to an
#: airmass.
SITES = {
    site.code: site for site in (
        Site("kpno", "Kitt Peak National Observatory", 111.600, 31.9633, 2120, 7),
        Site("ctio", "Cerro Tololo Interamerican Observatory",
             70.815, -30.16527, 2215, 4),
        Site("lasilla", "European Southern Observatory, La Silla",
             70.730, -29.2567, 2347, 4),
        Site("paranal", "European Southern Observatory, Paranal",
             70.4034, -24.625, 2635, 4),
        Site("lapalma", "Roque de los Muchachos, La Palma",
             17.880, 28.7583, 2327, 0),
        Site("mmto", "MMT Observatory", 110.885, 31.6883, 2600, 7),
        Site("lick", "Lick Observatory", 121.6367, 37.3433, 1290, 8),
        Site("keck", "W. M. Keck Observatory", 155.4717, 19.8267, 4160, 10),
        Site("palomar", "Palomar Observatory", 116.8633, 33.3567, 1706, 8),
        Site("apo", "Apache Point Observatory", 105.820, 32.780, 2798, 7),
        Site("cerropachon", "Cerro Pachon", 70.7367, -30.240, 2737, 4),
        Site("lna", "Laboratorio Nacional de Astrofisica, Brazil",
             45.5825, -22.5345, 1864, 3),
        Site("saao", "South African Astronomical Observatory",
             -20.8107, -32.3783, 1798, -2),
        Site("aao", "Anglo-Australian Observatory, Siding Spring",
             -149.0661, -31.2769, 1164, -10),
    )
}

#: Names harwood accepts for a site besides its code, lowercased.
ALIASES = {
    "kitt peak": "kpno", "kpno 4m": "kpno", "kpno 2.1m": "kpno",
    "kpno 0.9m": "kpno", "kpno coude feed": "kpno", "tololo": "ctio",
    "eso": "lasilla", "la silla": "lasilla", "vlt": "paranal",
    "orm": "lapalma", "la palma": "lapalma", "wht": "lapalma",
    "mmt": "mmto", "mauna kea": "keck", "gemini-north": "keck",
    "gemini-south": "cerropachon", "soar": "cerropachon",
    "gemini north": "keck", "gemini south": "cerropachon",
    "opd": "lna", "pico dos dias": "lna", "siding spring": "aao",
}


def find(name: str) -> Site | None:
    """A site by code, alias, or the start of its name."""
    if not name:
        return None
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key in SITES:
        return SITES[key]
    matches = [site for site in SITES.values()
               if site.name.lower().startswith(key)]
    return matches[0] if len(matches) == 1 else None


def from_header(header) -> Site | None:
    """The site a header names, through OBSERVAT or TELESCOP."""
    for keyword in ("OBSERVAT", "OBSERVATORY", "TELESCOP", "SITE"):
        site = find(str(header.get(keyword, "")))
        if site is not None:
            return site
    return None


def listing() -> list[str]:
    """Every site, for the message that says which names are known."""
    return sorted(SITES)
