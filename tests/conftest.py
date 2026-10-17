"""Shared test set-up.

The interface preferences (theme, text size...) are saved to the user's
~/.config/harwood/interface.json whenever they change. A test must never
write there -- nor read what the person running the tests chose, which
would make a test pass or fail by somebody's taste -- so every test gets an
empty config directory of its own and the default look.
"""

import pytest


@pytest.fixture(autouse=True)
def _own_interface_preferences(tmp_path_factory, monkeypatch):
    config = tmp_path_factory.mktemp("config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    try:
        from harwood.viewer import theme
    except ImportError:
        yield
        return
    saved = theme._theme
    theme._theme = theme.Theme(theme.Preferences())
    yield
    theme._theme = saved
