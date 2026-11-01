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

import time
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


def fits_header(mask: np.ndarray, header: dict):
    """What read gives as the header of a FITS image: a mask's pixels are ints."""
    from astropy.io import fits

    out = fits.Header()
    out["BITPIX"] = 32
    out["NAXIS"] = mask.ndim
    for axis in range(1, mask.ndim + 1):
        out[f"NAXIS{axis}"] = mask.shape[mask.ndim - axis]
    title = header.get("TITLE")
    if title and "OBJECT" not in header:
        out["OBJECT"] = title
    for key, value in header.items():
        if key not in ("TITLE", "CTIME", "MTIME", "LIMTIME", "MINPIXVAL", "MAXPIXVAL"):
            out[key] = value
    return out


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


# ------------------------------------------------------------------ writing

M_ZN, M_SH, M_IH, M_DH = 0, I_SHIFT, 2 * I_SHIFT, 3 * I_SHIFT
M_HN, M_PN = 4 * I_SHIFT, 5 * I_SHIFT
M_MOVE = 4 * I_SHIFT
I_DATAMAX = I_SHIFT - 1
LL_CURHDRLEN = 7
LL_CURVERSION = -100
PL_MAXDEPTH = 27


def p2l(pixels) -> list[int]:
    """pl_p2li: a line of pixels as a line list: for each run of a value above 0,
    the zeros before it and the change of value to it. Below 0 is 0."""
    px = np.maximum(np.asarray(pixels, dtype=np.int64), 0)
    npix = px.size
    out = [0, LL_CURHDRLEN, LL_CURVERSION, 0, 0, 0, 0]
    edges = np.flatnonzero(px[1:] != px[:-1]) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [npix]))
    hi, iz = 1, 0
    for first, stop in zip(starts.tolist(), ends.tolist(), strict=True):
        value = int(px[first])
        if value == 0:
            continue
        run, zeros = stop - first, first - iz
        done = False
        change = value - hi
        if change != 0:
            hi = value
            if abs(change) > I_DATAMAX:
                out += [M_SH + (value & I_DATAMAX), value // I_SHIFT]
            else:
                out.append(M_DH - change if change < 0 else M_IH + change)
                if run == 1 and zeros == 0:
                    out[-1] |= M_MOVE
                    done = True
        if not done and zeros > 0:
            out += _zeros(zeros)
            if run == 1:
                out[-1] += M_PN + 1
                done = True
        if not done:
            while run > 0:
                out.append(M_HN + min(I_DATAMAX, run))
                run -= I_DATAMAX
        iz = stop
    if npix - iz > 0:
        out += _zeros(npix - iz)
    out[3], out[4] = len(out) % 32768, len(out) // 32768
    return out


def _zeros(n: int) -> list[int]:
    """I_DATAMAX - 1, so that a PN can still be made of the last."""
    words = []
    while n > 0:
        words.append(M_ZN + min(I_DATAMAX - 1, n))
        n -= I_DATAMAX - 1
    return words


class Mask:
    """A mask being written, line by line, as pl_update keeps it: a line the same
    as the empty line or the one before is only one more reference to it, a line
    written again takes the place of what it replaces if that fits and no other
    line uses it, and what lies in the file lies in the order it was first made."""

    def __init__(self, axlen: list[int]):
        self.axlen = list(axlen)
        self.nlp = int(np.prod(axlen[1:])) if len(axlen) > 1 else 1
        empty = p2l(np.zeros(axlen[0], dtype=np.int64))
        self.slots = [{"ll": empty, "nref": self.nlp, "blen": len(empty)}]
        self.lines = [0] * self.nlp

    def update(self, line: int, pixels) -> None:
        """pl_update of line (0-based) with these pixels."""
        ll = p2l(pixels)
        slots = self.slots
        old = slots[self.lines[line]]
        if ll[LL_CURHDRLEN:] == old["ll"][LL_CURHDRLEN:]:
            return
        new = None
        if ll[LL_CURHDRLEN:] == slots[0]["ll"][LL_CURHDRLEN:]:
            new = 0
        elif line > 0:
            before = slots[self.lines[line - 1]]
            if before["nref"] < 32767 and ll[LL_CURHDRLEN:] == before["ll"][LL_CURHDRLEN:]:
                new = self.lines[line - 1]
        if new is None:
            if old["nref"] <= 1 and len(ll) <= old["blen"] and self.lines[line] != 0:
                old["ll"], old["nref"] = ll, 1
                return
            slots.append({"ll": ll, "nref": 0, "blen": len(ll)})
            new = len(slots) - 1
        old["nref"] -= 1
        slots[new]["nref"] += 1
        self.lines[line] = new

    def encode(self) -> np.ndarray:
        """pl_save: the external form, as big-endian shorts."""
        offsets, buffer = {}, []
        for k, slot in enumerate(self.slots):
            if k == 0 or slot["nref"] > 0:
                offsets[k] = len(buffer)
                ll = list(slot["ll"])
                ll[0] = slot["nref"]
                ll[5], ll[6] = len(ll) % 32768, len(ll) // 32768
                buffer += ll
        index = p2l(np.array([offsets[k] for k in self.lines]))
        total = 2 * LEN_PLEXTERN + len(index) + len(buffer)
        axes = self.axlen + [1] * (PL_MAXDIM - len(self.axlen))
        ext = [PL_MAGICVAL, len(self.axlen), *axes, len(buffer), len(buffer), self.nlp,
               len(index), total, 0, self.axlen[0] * 3 + LL_CURHDRLEN,
               2 ** PL_MAXDEPTH - 1, 0, 0, 0]
        words = np.array(ext, dtype=">i4").tobytes()
        shorts = np.array(index + buffer, dtype=np.int64)
        return words + (shorts & 0xFFFF).astype(">u2").tobytes()


def from_array(mask: np.ndarray) -> Mask:
    """The Mask of an array (last axis x), written in order of the lines."""
    data = np.asarray(mask)
    out = Mask(list(reversed(data.shape)))
    for line, pixels in enumerate(data.reshape(-1, data.shape[-1])):
        out.update(line, pixels)
    return out


def write(path: Path | str, mask, title: str = "", header=None) -> Path:
    """pl_savef with the PLF kernel's text: an array, or a Mask already built, as
    a .pl file. IRAF's title is the OBJECT, or what is given; its header's cards
    go after it, one a line."""
    built = mask if isinstance(mask, Mask) else from_array(mask)
    path = Path(path)
    if path.suffix != ".pl":
        path = path.with_name(path.name + ".pl")
    now = int(time.time()) - 315532800      # IRAF's clock starts in 1980
    text = title or (str(header.get("OBJECT", "")).strip() if header is not None else "")
    lines = [f'$TITLE = "{text}"', f"$CTIME = {now}", f"$MTIME = {now}", "$LIMTIME = 0",
             "$MINPIXVAL = 0.", "$MAXPIXVAL = 0."]
    if header is not None:
        for card in header.cards:
            if card.keyword not in _STRUCTURAL:
                lines.append(card.image.rstrip())
    title_bytes = ("\n".join(lines) + "\n").encode("latin-1") + b"\0"
    encoded = built.encode()
    length = len(encoded) // 2
    head = np.array([SVMAGIC, len(title_bytes), length], dtype=">i4").tobytes()
    pad = b"\0" * (len(title_bytes) % 2)
    path.write_bytes(head + title_bytes + pad + encoded)
    return path


_STRUCTURAL = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4",
               "BSCALE", "BZERO", "BLANK", "END", "DATAMIN", "DATAMAX", "PCOUNT", "GCOUNT",
               "XTENSION", "ORIGIN"}
