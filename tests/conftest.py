"""Shared test set-up.

The interface preferences (theme, text size...) are saved to the user's
~/.config/harwood/interface.json whenever they change. A test must never
write there -- nor read what the person running the tests chose, which
would make a test pass or fail by somebody's taste -- so every test gets an
empty config directory of its own and the default look. The same for the
state harwood keeps (~/.local/state/harwood): a test typing at the prompt
must not end up in the up arrow of a real session, and a test's runs must
not become the timings a real estimate is made from.
"""

import pytest


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
