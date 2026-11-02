"""imshift's sinc, lsinc and drizzle against IRAF 2.18.

drizzle is compared as it is: harwood writes what IRAF writes.

sinc and lsinc are compared twice. IRAF 2.18's t_imshift.x hands the
fractional shift to msisinit as a double where msisinit takes a real, so
its sinc table is built for the low four bytes of the shift read as a
float -- 4.2e-08 for a shift of 0.3, -1.1e+08 for 0.45 -- and a frame of
values near 130 comes out with values of 10^5 and more. With that one
mistake put back, harwood's sinc writes exactly what IRAF writes, which is
the proof the rest is IRAF's code; without it, harwood's sinc is the
interpolation IRAF's was meant to be.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harwood.core import imshift
from harwood.core.iraffmt import ctod

HERE = Path(__file__).parent / "data"
CASES = json.loads((HERE / "iraf_imshift_sinc_drizzle.json").read_text(encoding="utf-8"))
OUT = np.load(HERE / "iraf_imshift_sinc_drizzle.npz")


def frame() -> np.ndarray:
    yy, xx = np.indices((30, 40))
    return (100 + 3 * xx + 2 * yy + 50 * np.exp(-((xx - 20.3) ** 2 + (yy - 14.7) ** 2) / 6.0)
            + ((xx * 7 + yy * 13) % 11)).astype(np.float32)


def _iraf_shifts(case):
    """The shifts as IRAF read them: PyRAF hands them over as text."""
    return ctod(repr(case["xshift"])), ctod(repr(case["yshift"]))


@pytest.mark.parametrize("case", [c for c in CASES if c["interp"].startswith("drizzle")],
                         ids=lambda c: f"{c['interp']}-{c['xshift']},{c['yshift']}-"
                                       f"{c['boundary']}")
def test_drizzle_is_iraf_s(case):
    xs, ys = _iraf_shifts(case)
    got = imshift.shift(frame(), xs, ys, case["interp"], case["boundary"], 7.5)
    assert np.array_equal(got, OUT[case["key"]])


def _as_real(value: float) -> np.float32:
    """A double passed where a real is read: its first four bytes."""
    return np.frombuffer(np.float64(value).tobytes()[:4], dtype=np.float32)[0]


@pytest.mark.parametrize("case", [c for c in CASES if "sinc" in c["interp"]],
                         ids=lambda c: f"{c['interp']}-{c['xshift']},{c['yshift']}-"
                                       f"{c['boundary']}")
def test_sinc_with_iraf_s_mistake_is_iraf_s(case, monkeypatch):
    xs, ys = _iraf_shifts(case)
    fx = imshift._delta(abs(xs - int(xs)), xs)
    fy = imshift._delta(abs(ys - int(ys)), ys)
    table = imshift._sinc_table
    monkeypatch.setattr(imshift, "_sinc_table", lambda n, a, b: table(
        n, _as_real(fx - imshift._nint(fx)), _as_real(fy - imshift._nint(fy))))
    got = imshift.shift(frame(), xs, ys, case["interp"], case["boundary"], 7.5)
    assert np.array_equal(got, OUT[case["key"]])


def test_sinc_without_it_interpolates():
    """What IRAF's sinc was meant to do: a constant stays constant, a pixel
    shifted by a fraction lands between its neighbours, and a smooth image
    shifted and shifted back comes home."""
    flat = np.full((30, 40), 100.0, np.float32)
    assert np.allclose(imshift.shift(flat, 0.3, -0.45, "sinc"), 100.0, atol=1e-3)
    delta = np.zeros((41, 41), np.float32)
    delta[20, 20] = 1.0
    moved = imshift.shift(delta, 0.3, 0.0, "sinc", "constant")
    assert moved[20, 20] > moved[20, 21] > 0.2 and moved[20, 19] < 0
    assert moved.sum() == pytest.approx(1.0, abs=1e-4)
    yy, xx = np.indices((64, 64))
    smooth = (np.sin(xx / 5.0) + np.cos(yy / 7.0)).astype(np.float32)
    there = imshift.shift(smooth, 0.37, -0.21, "sinc21", "reflect")
    back = imshift.shift(there, -0.37, 0.21, "sinc21", "reflect")
    assert np.abs(back - smooth)[16:-16, 16:-16].max() < 2e-3


def test_interpolant_names_take_their_parameters():
    assert imshift.parse_interp("sinc") == ("sinc", 15, 1.0)
    assert imshift.parse_interp("sinc21") == ("sinc", 10, 1.0)
    assert imshift.parse_interp("lsinc11[20]") == ("lsinc", 5, 1.0)
    assert imshift.parse_interp("drizzle[0.5]") == ("drizzle", 0, 0.5)
    assert imshift.parse_interp("drizzle[2]") == ("drizzle", 0, 1.0), "out of 0..1: 1"
