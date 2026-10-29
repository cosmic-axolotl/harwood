"""mkconfig: a configuration file put together from its three sections.

From photcal/mkconfig.cl and mkconfig/t_config.x, without the prompting:
the catalog section comes from a format file (f<catalog>.dat, here or in
the standards directory), the observation section from the observations'
format file (f<observations>.dat, as mknobsfile and obsfile write it), the
transformation section from a template (t<transform>.dat); each may also be
named as the file itself. The three are copied one after the other, as
IRAF copies them. A template configuration is copied whole instead.

The standards directory is photcal$catalogs/, harwood's copy of IRAF's
catalogs, format files and transformation templates (Landolt, Elias,
UKIRT, NICMOS ...).

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

CATALOGS = Path(__file__).resolve().parent.parent.parent / "data" / "iraf" / "catalogs"
LOGICAL = "photcal$catalogs/"


def catdir(text: str) -> str:
    """The standards directory, photcal$catalogs/ meaning harwood's copy;
    as in the CL, file names are appended to it as they are."""
    if text.startswith("photcal$catalogs"):
        return str(CATALOGS) + "/" + text[len("photcal$catalogs"):].lstrip("/")
    return text


@dataclass
class Sources:
    catalog: str | None
    observations: str
    transform: str


def _find(name: str, candidates: list[str], where: str) -> str:
    for path in candidates:
        if Path(path).is_file():
            return path
    raise FileNotFoundError(f"Cannot find {name} in {where}")


def sources(catalog: str | None, observations: str, transform: str,
            standards: str = LOGICAL) -> Sources:
    """mkconfig.cl's search for each section's file."""
    cdir = catdir(standards)
    cat = None
    if catalog:
        head, tail = Path(catalog).parent, Path(catalog).name
        own = str(head / f"f{tail}.dat") if str(head) != "." else f"f{tail}.dat"
        cat = _find(catalog, [own, f"{cdir}f{catalog}.dat", catalog],
                    "current or standards directory")
    head, tail = Path(observations).parent, Path(observations).name
    own = str(head / f"f{tail}.dat") if str(head) != "." else f"f{tail}.dat"
    obs = _find(observations, [own, observations], "current directory")
    trans = _find(transform, [f"t{transform}.dat", f"{cdir}t{transform}.dat", transform],
                  "current or standards directory")
    return Sources(cat, obs, trans)


def mkconfig(config: str | Path, catalog: str | None = None, observations: str = "",
             transform: str = "", template: str = "",
             standards: str = LOGICAL) -> Sources | None:
    """Write the configuration. Returns the section files used, or None
    when a template was copied."""
    config = Path(config)
    if config.exists():
        raise FileExistsError(f"The configuration file {config} already exists")
    if template and Path(template).is_file():
        shutil.copyfile(template, config)
        return None
    found = sources(catalog, observations, transform, standards)
    parts = [found.catalog, found.observations, found.transform]
    config.write_bytes(b"".join(Path(p).read_bytes() for p in parts if p))
    return found
