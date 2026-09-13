"""Writing a PNG with nothing but the standard library.

The headless display has to produce a picture on a machine where the GUI
extra was deliberately not installed -- a server, a batch job, a test. Pulling
in an imaging library to save a greyscale image would undo the point of
installing lean, so the file is written here instead. PNG is a small format
when all you need is one greyscale plane: a signature, three chunks, and
zlib, which is in the standard library.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def write_grey(path: Path | str, pixels) -> Path:
    """Write a 2-D array of 0..255 as an 8-bit greyscale PNG.

    Row order is the array's own, so the caller decides which way up the image
    goes -- FITS counts rows from the bottom and PNG from the top, and that
    flip belongs where somebody knows which convention they are in.
    """
    import numpy as np

    data = np.asarray(pixels)
    if data.ndim != 2:
        raise ValueError(f"a greyscale PNG needs a 2-D array, got {data.ndim}-D")
    data = np.clip(data, 0, 255).astype(np.uint8)
    height, width = data.shape

    # Each row is prefixed with its filter type; 0 means "no filtering", which
    # costs a little size and saves a lot of code.
    raw = b"".join(b"\x00" + row.tobytes() for row in data)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    body = (SIGNATURE
            + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(raw, 6))
            + _chunk(b"IEND", b""))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path
