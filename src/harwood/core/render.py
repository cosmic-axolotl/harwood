"""An image as it is shown: limits, stretch and colours, without Qt.

The window, the headless display and the `impicture` task all draw through
here, so a PNG saved from the window and one written by `harwood impicture`
from the same image with the same choices are the same pixels.

    limits(data, "zscale")              -> (low, high)
    stretch(data, low, high, "asinh")   -> 0..1
    lut("cividis")                      -> 256 x 3 colours
    rgb(data, ...)                      -> rows top down, ready for a file

Colour maps: grey, and the perceptually uniform viridis, cividis (made to
read the same with colour-vision deficiencies) and magma, which change
lightness steadily so that a brighter colour always means a brighter pixel;
plus harwood's earlier heat, cool and sepia. A map that runs through red and
green (a rainbow) is deliberately not offered.
"""

from __future__ import annotations

import numpy as np

SCALES = ("zscale", "minmax", "99.5%")
STRETCHES = ("linear", "sqrt", "log", "asinh")

#: Stops, evenly spaced from 0 to 1, linearly interpolated between.
#: viridis, cividis and magma are sampled from their published tables
#: (van der Walt and Smith; Nunez, Anderton and Renslow 2018), CC0.
COLORMAPS: dict[str, list[tuple[int, int, int]]] = {
    "grey": [(0, 0, 0), (255, 255, 255)],
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
                (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
                (180, 222, 44), (253, 231, 37)],
    "cividis": [(0, 32, 77), (0, 48, 111), (55, 66, 108), (87, 84, 109),
                (112, 103, 115), (138, 122, 118), (166, 143, 115), (194, 164, 105),
                (224, 187, 88), (253, 234, 69)],
    "magma": [(0, 0, 4), (24, 15, 62), (69, 16, 121), (114, 31, 129), (159, 47, 127),
              (205, 64, 113), (241, 96, 93), (254, 145, 104), (254, 196, 136),
              (252, 253, 191)],
    "heat": [(0, 0, 0), (178, 34, 34), (255, 170, 0), (255, 255, 224)],
    "cool": [(6, 12, 33), (32, 84, 140), (86, 176, 186), (238, 248, 230)],
    "sepia": [(12, 9, 6), (120, 78, 46), (214, 170, 116), (255, 246, 232)],
}
#: The maps that read the same for the common colour-vision deficiencies.
COLORBLIND_SAFE = ("grey", "viridis", "cividis", "magma")


def limits(data: np.ndarray, scale: str = "zscale") -> tuple[float, float]:
    """The display range: IRAF's zscale, the extremes, or the middle 99.5%."""
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    if scale == "minmax":
        return float(finite.min()), float(finite.max())
    if scale == "99.5%":
        low, high = np.percentile(finite, [0.25, 99.75])
        return float(low), float(high)
    if scale != "zscale":
        raise ValueError(f"scale '{scale}' is not one of {', '.join(SCALES)}")
    from astropy.visualization import ZScaleInterval

    low, high = ZScaleInterval().get_limits(finite)
    return float(low), float(high)


def stretch(data: np.ndarray, low: float, high: float, kind: str = "linear") -> np.ndarray:
    """Normalise to 0..1 through the chosen transfer function."""
    span = max(high - low, 1e-12)
    norm = np.clip((np.asarray(data, dtype=float) - low) / span, 0.0, 1.0)
    if kind == "sqrt":
        return np.sqrt(norm)
    if kind == "log":
        return np.log10(1 + 1000 * norm) / np.log10(1001)
    if kind == "asinh":
        return np.arcsinh(10 * norm) / np.arcsinh(10)
    if kind != "linear":
        raise ValueError(f"stretch '{kind}' is not one of {', '.join(STRETCHES)}")
    return norm


def lut(name: str, n: int = 256) -> np.ndarray:
    """n colours (uint8, n x 3) along the map."""
    if name not in COLORMAPS:
        raise ValueError(f"colour map '{name}' is not one of {', '.join(COLORMAPS)}")
    stops = np.asarray(COLORMAPS[name], dtype=float)
    where = np.linspace(0.0, 1.0, len(stops))
    at = np.linspace(0.0, 1.0, n)
    table = np.column_stack([np.interp(at, where, stops[:, k]) for k in range(3)])
    return np.clip(np.round(table), 0, 255).astype(np.uint8)


def rgb(data: np.ndarray, scale: str = "zscale", kind: str = "linear",
        colormap: str = "grey", window: tuple[float, float] | None = None) -> np.ndarray:
    """The image as shown, rows top down (FITS counts from the bottom),
    uint8 (ny, nx, 3). Blank (NaN) pixels take the map's first colour."""
    low, high = window if window is not None else limits(data, scale)
    shown = stretch(data, low, high, kind)
    table = lut(colormap)
    index = np.nan_to_num(shown, nan=0.0)
    index = np.clip((index * len(table)).astype(np.int64), 0, len(table) - 1)
    return table[index][::-1].copy()
