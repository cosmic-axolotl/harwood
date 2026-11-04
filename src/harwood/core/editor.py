"""IRAF's edit: a file handed to the person to change, and taken back when they are done.

photcal's scripts stop at the editor -- mkimsets to have the image list put in order,
then the sets named; mkcatalog and mkconfig to check what they wrote. In the window the
file opens in an editor of its own until it is saved or closed; at the terminal in
$VISUAL or $EDITOR, as IRAF's edit opens its editor, vi when neither is set. Run with
neither a window nor a terminal, there is no one to edit it, and the file stays as it
was written: `edit` says so by returning False.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


def edit(path: str | Path) -> bool:
    """The file edited, in the window or the terminal's editor; False when neither."""
    if "harwood.viewer._qt" in sys.modules and sys.modules["harwood.viewer._qt"].window():
        from ..viewer.edit_dialog import edit as in_window

        in_window(str(path))
        return True
    command = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if command is None:
        if sys.stdin is None or not sys.stdin.isatty():
            return False
        command = "vi"
    subprocess.call([*shlex.split(command), str(path)])
    return True
