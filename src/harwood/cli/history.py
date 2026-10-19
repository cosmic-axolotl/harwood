"""What was typed, kept from one session to the next.

Every line typed at hw> -- in the terminal session or in the window -- goes to
one file, so that the up arrow tomorrow finds what was typed in today's
window. The cl kept a history and so does every shell; losing it when the
window closes would be a step back from both.

It is not the log. The log (output/.../log, `harwood log`) records what a task
did, with every value and where it came from, and can replay it exactly. The
history is what a person typed, typos and all, so that it can be typed again.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Lines kept. Trimmed back to this only once it is a quarter over, so that
#: adding a line is an append and not a rewrite nearly every time.
LIMIT = 2000


def path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "harwood" / "history"


def load() -> list[str]:
    """Oldest first."""
    try:
        return path().read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def add(line: str) -> None:
    """Keep a typed line. The same line twice in a row is kept once, as
    bash's ignoredups does: twenty presses of the same `display` would push
    everything else out of reach of the arrow keys. A history that cannot be
    written (a read-only home, a full disk) is not a reason for a command
    not to run, so failures here are silent."""
    line = line.strip()
    if not line or "\n" in line:
        return
    lines = load()
    if lines and lines[-1] == line:
        return
    try:
        path().parent.mkdir(parents=True, exist_ok=True)
        if len(lines) + 1 > LIMIT * 5 // 4:
            path().write_text("\n".join([*lines, line][-LIMIT:]) + "\n", encoding="utf-8")
        else:
            with path().open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError:
        pass
