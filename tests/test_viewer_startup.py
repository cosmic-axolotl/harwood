"""A panel opened from the terminal, in a process with no QApplication yet.

The entry points of viewer/ build their panel and hand it to dock.show, which made the
QApplication -- after the panel, a widget, had been made without one, and Qt aborts
there ("QWidget: Must construct a QApplication before a QWidget"): `harwood implot`,
`ccdproc --interactive` and every other task opening a window from the command line
died that way. Each entry point now makes the application first. Here, in a fresh
process, show is made to return at once, so that what is tested is the building of
the panel.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

CASES = {
    "implot": "from harwood.viewer import implot_panel\n"
              "implot_panel.run(np.arange(30.0).reshape(5, 6), 'x')",
    "icfit run": "from harwood.viewer import icfit_panel\n"
                 "icfit_panel.run(np.arange(10.0), np.arange(10.0) ** 2)",
    "icfit make": "from harwood.viewer import icfit_panel\n"
                  "icfit_panel.make(np.arange(10.0), np.arange(10.0) ** 2)",
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_panel_opens_where_no_application_was_made(case):
    script = ("import numpy as np\n"
              "import harwood.viewer.dock as dock\n"
              "dock.show = lambda panel, title, size=None: panel\n"
              f"{CASES[case]}\n"
              "print('built')\n")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "built" in done.stdout
    assert "Must construct a QApplication" not in done.stderr
