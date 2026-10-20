"""IRAF's .pl pixel masks, read without IRAF.

A .pl file is what IRAF writes when a mask is an image (`imcopy x.fits
mask.pl`, ccdred's bad pixel masks, the output of objmasks). It is small
because each line is stored as a line list: run lengths of zeros and of a
current value, with instructions to change the value. The layout, from
sys/plio (plloadf.x, plload.x, pll2p.gx) and the PLF image kernel:

    3 MII ints        "SV" magic, title length, mask length in shorts
    title             the image header, one "KEYWORD = value" per line
    20 MII ints       PLEXTERN: magic 2014, naxes, axis lengths, list sizes
    index             the line-list offset of each line -- itself encoded
                      as a line list
    line lists        the shared buffer every offset points into

MII is big-endian, 32-bit for ints and 16-bit for shorts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SVMAGIC = 0o123126
PL_MAGICVAL = 2014
LEN_PLEXTERN = 20
PL_MAXDIM = 7
I_SHIFT = 0o10000


def read(path: Path | str) -> tuple[np.ndarray, dict]:
    """The mask, numpy-ordered (last axis is x), and the header that came
    with it (LTV, LTM and whatever else the image had)."""
    raw = Path(path).read_bytes()
    try:
        magic, titlelen, masklen = np.frombuffer(raw, ">i4", 3)
        if magic != SVMAGIC:
            raise ValueError
        title = raw[12:12 + titlelen].split(b"\0")[0].decode("latin-1")
        # The title is padded to a whole short before the mask begins.
        start = 12 + (titlelen + 1) // 2 * 2
        encoded = np.frombuffer(raw, ">i2", masklen, start).astype(np.int32)
    except ValueError:
        raise ValueError(f"{path} is not an IRAF pixel mask (.pl) file") from None
    return load(encoded), _header(title)


def load(buffer: np.ndarray) -> np.ndarray:
    """pl_load: an encoded mask (as shorts) to the pixel array."""
    ext = _mii32(buffer[:2 * LEN_PLEXTERN])
    if ext[0] != PL_MAGICVAL:
        raise ValueError("bad pixel mask: wrong magic number in the encoded mask")
    naxes = int(ext[1])
    axlen = [int(n) for n in ext[2:2 + PL_MAXDIM]][:naxes]
    lllen, nlpx = int(ext[10]), int(ext[12])
    start = 2 * LEN_PLEXTERN
    index = _signed16(buffer[start:start + nlpx])
    llbuf = _signed16(buffer[start + nlpx:start + nlpx + lllen])
    nlp = int(np.prod(axlen[1:])) if naxes > 1 else 1
    offsets = l2p(index, 1, nlp)
    out = np.zeros((nlp, axlen[0]), dtype=np.int32)
    for line, offset in enumerate(offsets):
        out[line] = l2p(llbuf[offset:], 1, axlen[0])
    return out.reshape(tuple(reversed(axlen)))


def l2p(ll: np.ndarray, xs: int, npix: int) -> np.ndarray:
    """pl_l2pi: decode one line list into npix pixels starting at xs."""
    out = np.zeros(npix, dtype=np.int64)
    if ll[2] > 0:                       # old format: a three-short header
        ll_len, ll_first = int(ll[2]), 4
    else:
        ll_len = int(ll[4]) * 32768 + int(ll[3])
        ll_first = int(ll[1]) + 1
    if npix <= 0 or ll_len <= 0:
        return out
    xe = xs + npix - 1
    op, x1, pv = 0, 1, 1
    ip = ll_first
    while ip <= ll_len:
        word = int(ll[ip - 1]) & 0xFFFF
        opcode, data = word // I_SHIFT, word & 0o7777
        if opcode in (0, 4, 5):          # ZN, HN, PN: a run of data pixels
            x2 = x1 + data - 1
            i1, i2 = max(x1, xs), min(x2, xe)
            n = i2 - i1 + 1
            if n > 0:
                if opcode == 4:
                    out[op:op + n] = pv
                elif opcode == 5 and i2 == x2:
                    out[op + n - 1] = pv
                op += n
            x1 = x2 + 1
        elif opcode == 1:                # SH: set the high value, two words
            pv = int(ll[ip]) * I_SHIFT + data
            ip += 1
        elif opcode == 2:                # IH
            pv += data
        elif opcode == 3:                # DH
            pv -= data
        else:                            # IS, DS: change, and one pixel of it
            pv = pv + data if opcode == 6 else pv - data
            if xs <= x1 <= xe:
                out[op] = pv
                op += 1
            x1 += 1
        if x1 > xe:
            break
        ip += 1
    return out


def _mii32(shorts: np.ndarray) -> np.ndarray:
    """Pairs of big-endian shorts back to the ints they were cut from."""
    pairs = (np.asarray(shorts, dtype=np.int64) & 0xFFFF).reshape(-1, 2)
    value = (pairs[:, 0] << 16) | pairs[:, 1]
    return np.where(value >= 2 ** 31, value - 2 ** 32, value)


def _signed16(shorts: np.ndarray) -> np.ndarray:
    value = np.asarray(shorts, dtype=np.int64) & 0xFFFF
    return np.where(value >= 32768, value - 65536, value)


def _header(title: str) -> dict:
    """The PLF header text: "KEY = value" lines, IRAF's $-keywords first."""
    header: dict = {}
    for line in title.splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip().lstrip("$"), value.split(" /")[0].strip()
        if value.startswith("'") or value.startswith('"'):
            header[key] = value.strip("'\"").rstrip()
        elif value in ("T", "F"):
            header[key] = value == "T"
        else:
            try:
                header[key] = int(value)
            except ValueError:
                try:
                    header[key] = float(value.replace("D", "E"))
                except ValueError:
                    header[key] = value
    return header
