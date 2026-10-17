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


def write_rgb(path: Path | str, pixels) -> Path:
    """Write an (ny, nx, 3) array of 0..255 as an 8-bit RGB PNG, rows as
    given (top first)."""
    import numpy as np

    data = np.asarray(pixels)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"an RGB PNG needs an (ny, nx, 3) array, got {data.shape}")
    data = np.clip(data, 0, 255).astype(np.uint8)
    height, width = data.shape[:2]
    raw = b"".join(b"\x00" + row.tobytes() for row in data)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    body = (SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(raw, 6))
            + _chunk(b"IEND", b""))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


#: What an image can be exported as: PNG with the standard library alone,
#: the rest through Qt (harwood[gui]).
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


def save_rgb(pixels, path: Path | str) -> Path:
    """An (ny, nx, 3) uint8 image to PNG, JPEG, TIFF or WebP, by suffix."""
    import numpy as np

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"cannot write '{suffix or path.name}': use one of "
                         + ", ".join(IMAGE_SUFFIXES))
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".png":
        return write_rgb(path, pixels)
    try:
        from PySide6 import QtGui
    except ImportError as error:
        raise ImportError(f"writing {suffix} needs Qt (pip install 'harwood[gui]'); "
                          "PNG needs nothing") from error
    data = np.ascontiguousarray(pixels, dtype=np.uint8)
    height, width = data.shape[:2]
    image = QtGui.QImage(data.data, width, height, 3 * width,
                         QtGui.QImage.Format.Format_RGB888).copy()
    if not image.save(str(path), quality=95):
        raise OSError(f"could not write {path}: is the folder writable?")
    return path
