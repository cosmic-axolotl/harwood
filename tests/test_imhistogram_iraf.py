"""imhistogram against IRAF's own (listout = yes).

tests/data/iraf_imhistogram.json holds IRAF 2.18.1's lists for the field of
tests/iraf_field.py as a real image and rounded to 16-bit integers -- whose
limits IRAF rounds and whose bins autoscale makes a whole number of values
wide -- with given limits, a bin width, top_closed, the cumulative and the
difference histograms, and an image section. Every bin centre and count
must agree.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import field

from harwood.core.registry import get_task

CASES = json.loads((Path(__file__).parent / "data" / "iraf_imhistogram.json")
                   .read_text())["cases"]


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    where = tmp_path_factory.mktemp("imhistogram")
    fits.writeto(where / "field.fits", field())
    fits.writeto(where / "fieldi.fits", np.round(field()).astype(np.int16))
    return where


@pytest.mark.parametrize("case", list(CASES))
def test_imhistogram_lists_what_iraf_lists(images, case):
    params = dict(CASES[case])
    expected = params.pop("bins")
    image = str(images / params.pop("image"))
    params = {k: {"yes": True, "no": False}.get(v, v) for k, v in params.items()}
    rows = get_task("imhistogram").run(image=image, listout=True, **params).data["rows"]
    assert [row["count"] for row in rows] == [count for _, count in expected]
    assert [row["value"] for row in rows] == pytest.approx(
        [value for value, _ in expected], rel=5.01e-7)


def test_without_a_screen_the_plot_is_an_svg(tmp_path, monkeypatch):
    """IRAF plots by default; with no screen the plot is written instead of
    lost, with IRAF's title and the bins drawn as a line or as boxes."""
    from astropy.io import fits

    import harwood

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARWOOD_DISPLAY", "headless")
    data = (np.arange(400, dtype=np.float32).reshape(20, 20) % 37) + 100
    fits.PrimaryHDU(data, fits.Header({"OBJECT": "ramp"})).writeto("ramp.fits")
    result = harwood.run("imhistogram", image="ramp.fits", nbins=37)
    svg = result.outputs[0].read_text()
    assert result.outputs[0].name == "ramp.hist.svg"
    assert "Histogram of ramp.fits = ramp" in svg and "nbins=37" in svg
    assert svg.count("<polyline") == 1 and "rows" not in result.data
    boxes = harwood.run("imhistogram", image="ramp.fits", nbins=37, plot_type="box",
                        logy=False, device=str(tmp_path / "boxes.svg"))
    assert boxes.outputs == [tmp_path / "boxes.svg"]
    points = boxes.outputs[0].read_text().split('points="')[1].split('"')[0].split()
    assert len(points) == 2 * 37, "two corners per bin"
