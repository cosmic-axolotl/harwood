"""Shared test set-up.

The interface preferences (theme, text size...) are saved to the user's
~/.config/harwood/interface.json whenever they change. A test must never
write there -- nor read what the person running the tests chose, which
would make a test pass or fail by somebody's taste -- so every test gets an
empty config directory of its own and the default look. The same for the
state harwood keeps (~/.local/state/harwood): a test typing at the prompt
must not end up in the up arrow of a real session, and a test's runs must
not become the timings a real estimate is made from.

The directory too. A task finds harwood.yaml by walking up from where it
runs, as git does, so a test run from anywhere under a home with a project
file in it read that instrument and those parameters, and failed or passed
by what somebody had last reduced. Every test runs in a directory of its
own, away from all of that, and what it writes stays out of the repository.
"""

import contextlib
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _away_from_any_project(tmp_path_factory):
    """Session-wide because a module's fixture runs before a test's: the
    reductions that several tests share have to be made out here as well."""
    from harwood.core import params

    started = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))
    assert params.find_project() is None, "the temporary directory is under a project"
    yield
    os.chdir(started)


@pytest.fixture(autouse=True)
def _own_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _own_interface_preferences(tmp_path_factory, monkeypatch):
    config = tmp_path_factory.mktemp("config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state")))
    try:
        from harwood.viewer import theme
    except ImportError:
        yield
        return
    saved = theme._theme
    theme._theme = theme.Theme(theme.Preferences())
    yield
    theme._theme = saved


#: Every window and panel the tests opened, for as long as the session lasts
_KEPT: list = []


@pytest.fixture(autouse=True)
def _no_window_is_destroyed_by_chance():
    """What a test opened is hidden after it, and kept. Let go, a window is destroyed whenever
    the garbage collector reaches it, from whichever thread it runs on, and Qt crashes taking a
    pyqtgraph scene down off the window's thread: in a test that had nothing to do with it."""
    yield
    import sys

    if "PySide6" not in sys.modules:
        return
    from PySide6 import QtWidgets

    if QtWidgets.QApplication.instance() is None:
        return
    from harwood.core import display
    from harwood.viewer import _qt

    display.reset()
    _qt.set_window(None)
    for widget in QtWidgets.QApplication.topLevelWidgets():
        if widget not in _KEPT:
            _KEPT.append(widget)
        with contextlib.suppress(RuntimeError):    # Qt has already destroyed it
            widget.hide()
