"""How long a task took before, and so how long it will probably take now.

The prediction is made from this computer's own record and from nothing
else: each successful run leaves a line saying how many images and pixels it
read and how many seconds it took (~/.local/state/harwood/timings.jsonl).
Speed is a property of the machine, not of the reduction, so the record is
kept per user and not in the night's directory -- a new night starts with
the estimates the last one left.

With runs of different sizes behind it, the estimate is a line through them
(a fixed cost plus a cost per pixel); with only one size, seconds per pixel
scaled to this run. A task that has never run here gets no estimate at all,
and says so: a made-up number would be worse than none, because it would be
believed.

Interactive runs are not recorded. Their time is mostly a person deciding,
which tells nothing about the next run.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

#: Fields that name the images a task reads. Outputs are not counted: they do
#: not exist yet, and the time goes into reading and computing.
INPUTS = ("input", "images", "image", "inputs", "references", "reference",
          "operand1", "operand2", "spectra")

#: Runs of one task looked at for an estimate: recent ones, since a faster
#: kernel or a new version should take over quickly.
RECENT = 30

#: Lines kept in the record before it is trimmed.
LIMIT = 5000


@dataclass(frozen=True)
class Size:
    images: int = 0
    pixels: int = 0


@dataclass(frozen=True)
class Estimate:
    seconds: float
    runs: int               #: how many earlier runs it is based on

    def basis(self) -> str:
        return (f"from {self.runs} earlier run{'s' if self.runs != 1 else ''} "
                "on this computer")


@dataclass
class Running:
    """What is running right now, for a window to show while it waits."""

    task: str
    started: float
    estimate: Estimate | None


#: Set while a task runs (by cli.main), read by the window's clock. A plain
#: assignment, so a reader in another thread sees either the old value or the
#: new one, never half of either.
current: Running | None = None


def path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "harwood" / "timings.jsonl"


def size_of(params: dict[str, Any]) -> Size:
    """The images a run reads and their pixels, from the headers alone."""
    from . import images

    count = pixels = 0
    for name in INPUTS:
        value = params.get(name)
        if not value:
            continue
        try:
            specs = images.expand(value if isinstance(value, list) else [str(value)])
        except Exception:
            continue
        for spec in specs:
            try:
                hdul, hdu = images.open_hdu(spec)
            except Exception:
                continue
            try:
                shape = [hdu.header.get(f"NAXIS{axis}", 0)
                         for axis in range(1, hdu.header.get("NAXIS", 0) + 1)]
            finally:
                hdul.close()
            count += 1
            pixels += _product(shape)
    return Size(count, pixels)


def record(task: str, size: Size, seconds: float, backend: str | None = None) -> None:
    """Keep one successful run. Silent when the record cannot be written: a
    full disk is not a reason for a reduction to stop."""
    line = json.dumps({"task": task, "images": size.images, "pixels": size.pixels,
                       "seconds": round(seconds, 4), "backend": backend,
                       "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
    try:
        path().parent.mkdir(parents=True, exist_ok=True)
        with path().open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        lines = path().read_text(encoding="utf-8").splitlines()
        if len(lines) > LIMIT * 5 // 4:
            path().write_text("\n".join(lines[-LIMIT:]) + "\n", encoding="utf-8")
    except OSError:
        pass


def load(task: str) -> list[dict[str, Any]]:
    try:
        text = path().read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    runs = []
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue                    # a line cut short by a crash
        if entry.get("task") == task and entry.get("seconds", 0) > 0:
            runs.append(entry)
    return runs[-RECENT:]


def estimate(task: str, size: Size) -> Estimate | None:
    """How long `task` should take on `size`, or None with nothing to go on."""
    runs = load(task)
    if not runs:
        return None
    sized = [run for run in runs if run.get("pixels", 0) > 0]
    if size.pixels and sized:
        distinct = {run["pixels"] for run in sized}
        if len(sized) >= 3 and len(distinct) >= 2:
            fixed, per_pixel = _line([run["pixels"] for run in sized],
                                     [run["seconds"] for run in sized])
            if per_pixel > 0 and fixed >= 0:
                return Estimate(fixed + per_pixel * size.pixels, len(sized))
        rate = median(run["seconds"] / run["pixels"] for run in sized)
        return Estimate(rate * size.pixels, len(sized))
    return Estimate(median(run["seconds"] for run in runs), len(runs))


def clock(seconds: float) -> str:
    """0:07, 2:41, 1:02:03 -- as a stopwatch writes it."""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def about(seconds: float) -> str:
    """An estimate said as one: 'under a second', 'about 40 s', 'about 3 min'."""
    if seconds < 1:
        return "under a second"
    if seconds < 90:
        return f"about {seconds:.0f} s"
    if seconds < 5400:
        return f"about {seconds / 60:.0f} min"
    return f"about {seconds / 3600:.1f} h"


def _product(shape: list[int]) -> int:
    total = 1
    for axis in shape:
        total *= int(axis)
    return total if shape else 0


def _line(x: list[float], y: list[float]) -> tuple[float, float]:
    """Least squares y = a + b x."""
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    spread = sum((a - mean_x) ** 2 for a in x)
    if spread == 0:
        return 0.0, 0.0
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True)) / spread
    return mean_y - slope * mean_x, slope
