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

A file holds one record per aperture or image section, as IRAF's does: the
arc's apertures, or the lines of a long slit that reidentify traced. A
record is named by its image and section (IRAF's `identify arc - Ap 2`,
`identify arc[*,50]`); writing one replaces the record of that name and
keeps the others.
"""

from __future__ import annotations

import contextlib
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
    aperture: int | None = 1                #: None: a section of an image (INDEF)
    used: np.ndarray | None = None          #: which features the fit kept
    fitted: np.ndarray | None = None        #: the fit evaluated at `pixel`
    created: str = ""
    extra: dict = field(default_factory=dict)
    section: str = ""                       #: " - Ap 2", "[*,50]" or ""

    @property
    def record(self) -> str:
        """IRAF's name for the record: `<image><section>`."""
        return f"{_root(self.image)}{self.section}"

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


def _root(image: str) -> str:
    name = Path(str(image)).name
    for suffix in (".fits", ".imh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def write(solution: Solution, workdir: Path | str = ".",
          prefix: str = "id") -> Path:
    """Write a solution to `database/<prefix><image>`, replacing the record
    of the same image and section and keeping the file's others.

    The prefix is IRAF's way of keeping different kinds of solution in one
    directory without them colliding: `id` for a dispersion solution, `sens`
    for an instrument response, `ap` for a trace.
    """
    destination = path_for(solution.image, workdir, prefix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records: list[Solution] = []
    if destination.exists():
        # The file's other records stay -- an IRAF file's too, carried over
        # with IRAF's own curves, so writing one aperture never loses the
        # solutions of the others.
        text = destination.read_text(encoding="utf-8", errors="replace")
        existing = (read_iraf_all(destination) if re.search(r"^begin\s", text, re.M)
                    else _read_harwood_all(destination, text))
        records = [r for r in existing if r.record != solution.record]
    records.append(solution)
    blocks = [_document(record, prefix) for record in records]
    destination.write_text("\n---\n".join(blocks), encoding="utf-8")
    return destination


def _document(solution: Solution, prefix: str) -> str:
    import yaml

    used = (np.ones(len(solution), dtype=bool) if solution.used is None
            else np.asarray(solution.used, dtype=bool))
    fitted = (np.full(len(solution), np.nan) if solution.fitted is None
              else np.asarray(solution.fitted, dtype=float))

    document = {
        "task": TASKS.get(prefix, "identify"),
        "image": solution.image,
        **({"section": solution.section} if solution.section else {}),
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
        f"# harwood {KINDS.get(prefix, 'solution')}.",
        "#",
        *HEADINGS.get(prefix, HEADINGS["id"]),
        "",
        yaml.safe_dump(document, sort_keys=False, allow_unicode=False).strip(),
        "",
        "features:",
        "#  pixel      wavelength    fitted       used",
    ]
    for index in range(len(solution)):
        lines.append(
            f"  - [{solution.pixel[index]:12.6f}, {solution.wavelength[index]:16.8f}, "
            f"{fitted[index]:16.8f}, {'true' if used[index] else 'false'}]"
        )
    return "\n".join(lines) + "\n"


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

#: What each kind of solution is called -- in the file's own first line, and
#: in the sentence that says it is missing.
KINDS = {"id": "dispersion solution", "sens": "sensitivity function",
         "ap": "trace"}

#: What the rows of each kind mean. A file that does not say what its columns
#: are is a file that has to be decoded rather than read.
HEADINGS = {
    "id": [
        "# One line per identified feature: the pixel it was measured at, the",
        "# wavelength it was matched to, what the fit gives there, and whether",
        "# the fit used it. Deleted or rejected features are kept on purpose --",
        "# knowing which lines were thrown out is part of knowing the solution.",
    ],
    "ap": [
        "# One line per aperture: where its centre sits across the slit, how",
        "# wide it is, where the trace puts the centre, and whether it was",
        "# kept. `apertures` below carries the limits each one was given.",
    ],
    "sens": [
        "# One line per bandpass of the calibration stars: the wavelength, the",
        "# instrument's response there in magnitudes, what the fit gives, and",
        "# whether the fit used it. Rejected bands are kept so that a response",
        "# fitted over a gap can be seen to have been.",
    ],
}


def read(image: str, workdir: Path | str = ".", prefix: str = "id",
         aperture: int | None = None) -> Solution:
    """Read a solution, harwood's format or IRAF's: the first record, or
    the first of `aperture` -- and failing that, as dispcor falls back, the
    first one that belongs to no aperture."""
    records = read_all(image, workdir, prefix)
    if aperture is not None:
        for record in records:
            if record.aperture == aperture:
                return record
        for record in records:
            if record.aperture is None:
                return record
        advice.fail(
            f"database/{prefix}{_root(image)} has no {KINDS.get(prefix, 'solution')} "
            f"for aperture {aperture}; it holds "
            + ", ".join(str(r.aperture) for r in records) + ".",
            "run identify on that aperture of the arc (--section 'line N'), or "
            "reidentify to carry the solution to every aperture.", FileNotFoundError)
    return records[0]


def read_all(image: str, workdir: Path | str = ".", prefix: str = "id") -> list[Solution]:
    """Every record of `database/<prefix><image>`, in the file's order."""
    destination = path_for(image, workdir, prefix)
    if not destination.exists():
        advice.fail(
            f"no {KINDS.get(prefix, 'solution')} for '{image}'. Looked for "
            f"{destination}.",
            MISSING.get(prefix, MISSING["id"]), FileNotFoundError)
    text = destination.read_text(encoding="utf-8", errors="replace")
    if re.search(r"^begin\s", text, re.M):
        return read_iraf_all(destination)
    return _read_harwood_all(destination, text)


def _read_harwood_all(path: Path, text: str) -> list[Solution]:
    blocks = [block for block in re.split(r"^---\s*$", text, flags=re.M) if block.strip()]
    return [_read_harwood(path, block) for block in blocks]


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
        aperture=(None if document.get("aperture", 1) is None
                  else int(document.get("aperture", 1))),
        created=document.get("created", ""),
        extra={key: value for key, value in document.items() if key not in _OWN},
        section=str(document.get("section", "") or ""),
    )


#: The keys a harwood database file always has; anything else a task added.
_OWN = {"task", "image", "aperture", "units", "coordlist", "created", "fit", "features",
        "section"}


def read_iraf(path: Path) -> Solution:
    """The first record of IRAF's own `database/id*` file."""
    return read_iraf_all(path)[0]


def read_iraf_all(path: Path) -> list[Solution]:
    """Every record of IRAF's own `database/id*` file.

    Each is a `begin identify <image><section>` block with indented
    keywords, a `features` count followed by that many rows -- pixel, fit,
    user coordinate, width, type, used (checked against files IRAF 2.18
    wrote) -- and the curve's saved coefficients, which go to
    extra["curve"] so dispcor evaluates IRAF's own curve rather than a
    refit. `aperture` is a multispec record's; a section's has none.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[list[str]] = []
    for line in text.splitlines():
        if line.strip().startswith("begin"):
            blocks.append([line])
        elif blocks:
            blocks[-1].append(line)
    records = [_iraf_record(path, block) for block in blocks]
    records = [r for r in records if r is not None]
    if not records:
        advice.fail(f"{path} is an IRAF database file with no features.",
                    "it may be a truncated or partly written solution; run "
                    "identify in harwood to make a new one.")
    return records


def _iraf_record(path: Path, lines: list[str]) -> Solution | None:
    image = path.name[2:] if path.name.startswith("id") else path.name
    name = image
    header: dict[str, str] = {}
    features: list[tuple] = []
    reading = 0
    coefficients: list[float] = []
    ncoeff = 0

    def number(text):
        return float("nan") if text.upper() == "INDEF" else float(text)

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ncoeff > 0:
            coefficients.append(float(stripped.split()[0]))
            ncoeff -= 1
            continue
        if reading > 0:
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    pixel, fitted, wavelength = (number(parts[0]), number(parts[1]),
                                                 number(parts[2]))
                except ValueError:
                    reading = 0
                    continue
                used = True
                if len(parts) >= 6:
                    used = number(parts[5]) != 0
                elif len(parts) >= 5:
                    used = parts[-1] not in ("0", "no", "false")
                width = number(parts[3]) if len(parts) >= 4 else None
                features.append((pixel, wavelength, fitted, used, width))
                reading -= 1
                continue
            reading = 0
        if stripped.startswith("begin"):
            parts = stripped.split(None, 2)
            if len(parts) >= 3:
                name = parts[2]
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
        if keyword == "coefficients":
            coefficients = []
            ncoeff = int(value.split()[0]) if value else 0
            continue
        header[keyword] = value

    if not features:
        return None
    root = header.get("id", image)
    section = name[len(root):] if name.startswith(root) else ""
    aperture = header.get("aperture")
    try:
        aperture = int(aperture) if aperture not in (None, "", "INDEF") else None
    except ValueError:
        aperture = None
    extra: dict = {"from_iraf": str(path)}
    if coefficients:
        extra["curve"] = coefficients
    for key in ("aplow", "aphigh", "shift", "low_reject", "high_reject", "grow"):
        if key in header:
            with contextlib.suppress(ValueError):
                extra[key] = float(header[key])
    for key in ("niterate", "naverage"):
        if key in header:
            with contextlib.suppress(ValueError):
                extra[key] = int(float(header[key]))
    if "sample" in header:
        extra["sample"] = header["sample"]
    widths = [f[4] for f in features if f[4] is not None]
    if widths:
        extra["fwidth"] = widths[0]
    return Solution(
        image=str(header.get("id", image)),
        pixel=np.array([f[0] for f in features], dtype=float),
        wavelength=np.array([f[1] for f in features], dtype=float),
        fitted=np.array([f[2] for f in features], dtype=float),
        used=np.array([f[3] for f in features], dtype=bool),
        function=(header.get("function", "chebyshev") or "chebyshev").lower(),
        order=int(float(header.get("order", 3) or 3)),
        units=header.get("units", "Angstroms") or "Angstroms",
        coordlist=header.get("coordlist", ""),
        aperture=aperture, extra=extra, section=section,
    )


def listing(workdir: Path | str = ".") -> list[Path]:
    """Every solution in this reduction's database."""
    directory = Path(workdir) / DATABASE
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file())
