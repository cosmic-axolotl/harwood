"""The running commentary: a plain text log you can watch a reduction in.

harwood already keeps two records. `harwood.log.jsonl` holds every execution
with its parameters and the command that repeats it, for machines and for
`harwood log`; the HISTORY cards in each header hold what was done to that
particular frame, and travel with it.

Neither is the thing IRAF's `logfile` is. That one is a plain text file that
grows a line at a time while a task runs, which you keep open in another window
and watch -- and read afterwards to see what happened to the night as a whole,
in order, without opening anything.

One line per frame per operation, oldest first::

    2026-09-13 11:04:22  ccdproc  m92010.fits
      overscan: legendre order 1 over [335:350,2:510], mean level 505.44
      trim: [1:318,2:510] -> 318x509
      zero: subtracted m92001.fits

It is append-only and never read back by harwood. That is the point: it is for
a person.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

DEFAULT = "harwood.log"


def resolve(logfile: str | Path | None, workdir: Path | str = ".") -> Path | None:
    """Where the log goes, or None when the task was told not to keep one.

    An empty string means no log, as it does in IRAF. A bare name is taken
    relative to the reduction directory, so a task run from a subdirectory
    still writes to the night's log rather than starting a second one.
    """
    if logfile is None:
        return None
    text = str(logfile).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else Path(workdir) / path


def write(logfile: str | Path | None, task: str, image: str, lines: list[str],
          workdir: Path | str = ".") -> Path | None:
    """Append one frame's worth of commentary."""
    path = resolve(logfile, workdir)
    if path is None or not lines:
        return None

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = [f"{stamp}  {task}  {image}"]
    body.extend(f"  {line}" for line in lines)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(body) + "\n")
    return path


def note(logfile: str | Path | None, text: str,
         workdir: Path | str = ".") -> Path | None:
    """A line that belongs to the run rather than to any one frame."""
    path = resolve(logfile, workdir)
    if path is None:
        return None
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}  {text}\n")
    return path
