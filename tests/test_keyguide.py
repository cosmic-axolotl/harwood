"""The keys in view, and the DQ on the image (section 12, step 6).

Whatever is on the screen, the keys that work on it are listed beside it,
from the same lists that answer them; and an image's DQ is drawn over it,
named under the cursor and counted in a legend, so colour is never all
there is.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from harwood.core import dq, examine, photsession, tvmark  # noqa: E402


def _window(quality=None):
    from harwood.viewer._qt import application
    from harwood.viewer.window import HarwoodWindow

    application()
    window = HarwoodWindow()
    window.load(np.random.default_rng(0).normal(500, 20, (50, 60)), name="t",
                quality=quality)
    return window


def test_an_image_on_show_has_its_keys_beside_it():
    window = _window()
    listed = dict(window.key_guide.entries())
    for key in examine.KEYS:
        assert key in listed, f"imexamine's {key} works and must be listed"
    assert "Z" in listed and "D" in listed, "the window's own keys too"
    assert window.keys_dock.isVisibleTo(window), "always in view, not on demand"


def test_a_task_holding_the_keys_puts_its_own_in_the_panel():
    window = _window()
    window.set_task_keys("tvmark", tvmark.HELP)
    listed = dict(window.key_guide.entries())
    assert listed["c"] == tvmark.HELP["c"], "tvmark's c, not imexamine's"
    assert "tvmark" in window.key_guide.title
    window.set_task_keys("phot", photsession.HELP)
    assert dict(window.key_guide.entries())["space"].startswith("centre, sky")
    window.set_task_keys(None)
    assert dict(window.key_guide.entries())["c"] == examine.KEYS["c"], "and given back"


def test_the_dq_is_drawn_named_and_counted():
    quality = dq.empty((50, 60))
    quality[10, 20] = dq.SATURATED | dq.COSMIC_RAY
    quality[30, 5:9] = dq.BAD_PIXEL
    window = _window(quality)
    assert window.dq_item.image is not None, "the flagged pixels are drawn"
    legend = {name: detail for _, name, detail in window.key_guide.legend}
    assert legend == {"bad pixel": "4 px", "saturated": "1 px", "cosmic ray": "1 px"}

    from PySide6 import QtCore

    from harwood.core import pixels

    sx, sy = pixels.to_scene(21.0, 11.0)          # the centre of 1-based (21, 11)
    window._mouse_moved(window.view.mapViewToScene(QtCore.QPointF(sx, sy)))
    assert "DQ = saturated, cosmic ray" in window.status.text()

    window._key("D")
    assert window.dq_item.image is None, "D hides the layer"
    assert all("hidden" in detail for _, _, detail in window.key_guide.legend)
    window._key("D")
    assert window.dq_item.image is not None


def test_question_mark_logs_what_the_panel_shows():
    window = _window()
    window._cursor = (10.0, 10.0)
    before = window.session.toPlainText()
    window._key("?")
    logged = window.session.toPlainText()[len(before):]
    for key, what in window.key_guide.entries():
        assert f"[{key}] {what}" in logged


def test_every_panel_lists_every_key():
    """What used to be the first four or five on one line."""
    from harwood.viewer import (
        aperture_panel,
        icfit_panel,
        identify_panel,
        implot_panel,
        keyguide,
    )

    for module in (aperture_panel, icfit_panel, identify_panel, implot_panel):
        label = keyguide.strip(module.HELP)
        listed = dict(label.keys)
        for key, what in module.HELP:
            assert listed.get(key) == what, f"{module.__name__}: {key}"
