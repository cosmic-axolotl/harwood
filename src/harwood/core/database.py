"""The `database/` directory: dispersion solutions, and how to keep them.

IRAF writes the result of `identify` into `database/id<image>`, and everything
downstream -- `reidentify`, `refspectra`, `dispcor` -- reads it from there. A
night's arcs are calibrated once and the solutions are reused, which is why the
directory matters more than it looks: it is where the work of a night lives.

Two formats:

* harwood's own, YAML, one file per solution, readable and diffable. It carries
  the line list used, every identified feature, the icfit parameters and the
  fit itself, so a solution can be understood and re-derived without the
  original session.
* IRAF's, read-only for now. Section 3 of the design contract asks for this by
  name: someone migrating has a `database/` full of solutions that took real
  effort, and harwood should be able to use them rather than make them again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import advice

DATABASE = "database"


@dataclass
class Solution:
    """One dispersion solution: pixels, wavelengths, and the fit between them."""

    image: str
    pixel: np.ndarray
    wavelength: np.ndarray
    function: str = "chebyshev"
    order: int = 3
    rms: float = float("nan")
    units: str = "Angstroms"
    coordlist: str = ""
    aperture: int = 1
    used: np.ndarray | None = None          #: which features the fit kept
    fitted: np.ndarray | None = None        #: the fit evaluated at `pixel`
    created: str = ""
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(np.size(self.pixel))

    def summary(self) -> str:
        kept = int(np.sum(self.used)) if self.used is not None else len(self)
        return (f"{self.image}: {kept} of {len(self)} features, "
                f"{self.function} order {self.order}, rms {self.rms:.4f} {self.units}")


def path_for(image: str, workdir: Path | str = ".", prefix: str = "id") -> Path:
    """Where a solution for one image lives, IRAF's naming kept."""
    name = Path(str(image)).name
    for suffix in (".fits", ".imh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return Path(workdir) / DATABASE / f"{prefix}{name}"


def write(solution: Solution, workdir: Path | str = ".",
          prefix: str = "id") -> Path:
    """Write a solution to `database/<prefix><image>`.

    The prefix is IRAF's way of keeping different kinds of solution in one
    directory without them colliding: `id` for a dispersion solution, `sens`
    for an instrument response, `ap` for a trace.
    """
    import yaml

    destination = path_for(solution.image, workdir, prefix)
    destination.parent.mkdir(parents=True, exist_ok=True)

    used = (np.ones(len(solution), dtype=bool) if solution.used is None
            else np.asarray(solution.used, dtype=bool))
    fitted = (np.full(len(solution), np.nan) if solution.fitted is None
              else np.asarray(solution.fitted, dtype=float))

    document = {
        "task": TASKS.get(prefix, "identify"),
        "image": solution.image,
        "aperture": solution.aperture,
        "units": solution.units,
        "coordlist": solution.coordlist,
        "created": solution.created or datetime.now(UTC).isoformat(
            timespec="seconds"),
        "fit": {"function": solution.function, "order": solution.order,
                "rms": round(float(solution.rms), 6)},
        **solution.extra,
    }
    lines = [
        "# harwood dispersion solution.",
        "#",
        "# One line per identified feature: the pixel it was measured at, the",
        "# wavelength it was matched to, what the fit gives there, and whether",
        "# the fit used it. Deleted or rejected features are kept on purpose --",
        "# knowing which lines were thrown out is part of knowing the solution.",
        "",
        yaml.safe_dump(document, sort_keys=False, allow_unicode=False).strip(),
        "",
        "features:",
        "#  pixel      wavelength    fitted       used",
    ]
    for index in range(len(solution)):
        lines.append(
            f"  - [{solution.pixel[index]:10.4f}, {solution.wavelength[index]:12.5f}, "
            f"{fitted[index]:12.5f}, {'true' if used[index] else 'false'}]"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


#: Which task writes each prefix, for the `task:` line and for the error
#: message that says how to make a missing solution.
TASKS = {"id": "identify", "sens": "sensfunc", "ap": "aptrace"}
#: What to do about each kind of missing solution -- the suggestion half of
#: the notice, so the message can stay a plain statement of what is absent.
MISSING = {
    "id": "run identify on the arc first; its solution goes in database/id<arc>.",
    "sens": "run standard on a flux standard and then sensfunc; the solution "
            "goes in database/sens<name>.",
    "ap": "run apall with tracing on; the trace goes in database/ap<image>.",
}

#: What each kind of solution is called, for the sentence that says it is gone.
KINDS = {"id": "dispersion solution", "sens": "sensitivity function",
         "ap": "trace"}


def read(image: str, workdir: Path | str = ".", prefix: str = "id") -> Solution:
    """Read a solution, harwood's format or IRAF's."""
    destination = path_for(image, workdir, prefix)
    if not destination.exists():
        advice.fail(
            f"no {KINDS.get(prefix, 'solution')} for '{image}'. Looked for "
            f"{destination}.",
            MISSING.get(prefix, MISSING["id"]), FileNotFoundError)
    text = destination.read_text(encoding="utf-8", errors="replace")
    if re.search(r"^begin\s", text, re.M):
        return read_iraf(destination)
    return _read_harwood(destination, text)


def _read_harwood(path: Path, text: str) -> Solution:
    import yaml

    document = yaml.safe_load(text) or {}
    features = document.get("features") or []
    if not features:
        advice.fail(f"{path} lists no features.",
                    "the file should end with a `features:` block, one line "
                    "per identified line. Run identify again to rewrite it.")
    rows = np.array([[row[0], row[1], row[2]] for row in features], dtype=float)
    used = np.array([bool(row[3]) if len(row) > 3 else True for row in features])
    fit = document.get("fit", {})
    return Solution(
        image=document.get("image", path.name),
        pixel=rows[:, 0], wavelength=rows[:, 1], fitted=rows[:, 2], used=used,
        function=fit.get("function", "chebyshev"), order=int(fit.get("order", 3)),
        rms=float(fit.get("rms", float("nan"))),
        units=document.get("units", "Angstroms"),
        coordlist=document.get("coordlist", ""),
        aperture=int(document.get("aperture", 1)),
        created=document.get("created", ""),
    )


def read_iraf(path: Path) -> Solution:
    """Read IRAF's own `database/id*` file.

    The format is a `begin` block with indented keywords, a `features` count
    followed by that many rows, and a trailing list of fit coefficients. Only
    the features and the fit description are taken: the coefficients describe
    IRAF's own normalised polynomial basis, and harwood refits from the
    features instead -- which gives the same answer and keeps one fitting code
    rather than two.

    **Unverified:** the feature columns are read as pixel, fitted, wavelength,
    which is the documented order, but this has not been checked against a file
    IRAF actually wrote -- none was to hand. If a solution read from IRAF comes
    back with the fitted and catalogue wavelengths swapped, that is where to
    look. The residuals would be identical in size and opposite in sign, which
    is exactly the kind of error that hides.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    image = path.name[2:] if path.name.startswith("id") else path.name
    header: dict[str, str] = {}
    features: list[tuple[float, float, float, bool]] = []
    reading = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if reading > 0:
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    pixel, fitted, wavelength = (float(parts[0]), float(parts[1]),
                                                 float(parts[2]))
                except ValueError:
                    reading = 0
                    continue
                used = True
                if len(parts) >= 5:
                    used = parts[-1] not in ("0", "no", "false")
                features.append((pixel, wavelength, fitted, used))
                reading -= 1
                continue
            reading = 0
        if stripped.startswith("begin"):
            parts = stripped.split()
            if len(parts) >= 3:
                image = parts[2]
            continue
        parts = stripped.split(None, 1)
        keyword = parts[0].lower()
        value = parts[1].strip() if len(parts) > 1 else ""
        if keyword == "features":
            try:
                reading = int(value.split()[0])
            except (ValueError, IndexError):
                reading = 0
            continue
        header[keyword] = value

    if not features:
        advice.fail(f"{path} is an IRAF database file with no features.",
                    "it may be a truncated or partly written solution; run "
                    "identify in harwood to make a new one.")

    return Solution(
        image=str(header.get("image", image)),
        pixel=np.array([f[0] for f in features], dtype=float),
        wavelength=np.array([f[1] for f in features], dtype=float),
        fitted=np.array([f[2] for f in features], dtype=float),
        used=np.array([f[3] for f in features], dtype=bool),
        function=(header.get("function", "chebyshev") or "chebyshev").lower(),
        order=int(float(header.get("order", 3) or 3)),
        units=header.get("units", "Angstroms") or "Angstroms",
        coordlist=header.get("coordlist", ""),
        extra={"from_iraf": str(path)},
    )


def listing(workdir: Path | str = ".") -> list[Path]:
    """Every solution in this reduction's database."""
    directory = Path(workdir) / DATABASE
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file())
