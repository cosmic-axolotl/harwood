"""Reading images the way a user writes them.

IRAF let you name a piece of an image on the command line, and that syntax is
muscle memory worth keeping (section 3, rule 3)::

    obj.fits                 the first image extension
    obj.fits[SCI,1]          an extension by name and version
    obj.fits[1]              an extension by number
    obj.fits[SCI,1][100:400,*]   a section of one

Sections are 1-based and written (x, y); `core.pixels` converts. The quoting
rule from section 3.1 applies everywhere this syntax is documented: brackets
and `*` are shell wildcards, so the name goes in single quotes.

`@list.txt` file lists are handled here too, because every task that takes many
inputs takes them the same way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import pixels

_SPEC = re.compile(
    r"^(?P<path>[^\[\]]+?)"
    r"(?:\[(?P<ext>[^\[\]:]*?(?:,\s*\d+)?)\])?"
    r"(?:\[(?P<section>[^\[\]]*[:*][^\[\]]*)\])?$"
)


@dataclass(frozen=True)
class ImageSpec:
    """A parsed image reference."""

    path: Path
    extension: str | int | tuple[str, int] | None = None
    section: str | None = None

    def __str__(self) -> str:
        text = str(self.path)
        if self.extension is not None:
            text += f"[{_render_ext(self.extension)}]"
        if self.section:
            text += f"[{self.section.strip('[]')}]"
        return text


def _render_ext(extension) -> str:
    if isinstance(extension, tuple):
        return f"{extension[0]},{extension[1]}"
    return str(extension)


def parse(spec: str | Path) -> ImageSpec:
    """Parse ``'obj.fits[SCI,1][100:400,*]'`` into its parts."""
    text = str(spec).strip()
    match = _SPEC.match(text)
    if match is None:
        raise ValueError(f"cannot read image specification '{spec}'")

    extension: Any = match.group("ext")
    if extension is not None:
        extension = extension.strip()
        if "," in extension:
            name, _, version = extension.partition(",")
            extension = (name.strip(), int(version))
        elif extension.isdigit():
            extension = int(extension)
    section = match.group("section")
    return ImageSpec(Path(match.group("path")), extension, section)


def split_brackets(text: str) -> tuple[str, str]:
    """Separate a name from its trailing extension and section brackets.

    Square brackets mean two different things and IRAF lets both through::

        sp0[1-9].fits            a glob character class
        obj.fits[SCI,1]          an extension
        obj.fits[100:400,*]      a section

    The rule, which is IRAF's in practice: **the brackets that belong to the
    image are the trailing ones**, and anything earlier is part of the file
    name pattern. So the name is globbed and the trailing brackets are kept
    and applied to every file the glob matched.
    """
    depth = 0
    cut = len(text)
    for index in range(len(text) - 1, -1, -1):
        char = text[index]
        if char == "]":
            if depth == 0:
                cut = min(cut, index + 1)
            depth += 1
        elif char == "[":
            depth -= 1
            if depth == 0:
                cut = index
        elif depth == 0 and char not in " \t":
            break
    return text[:cut], text[cut:]


def split_list(text: str) -> list[str]:
    """Split an IRAF image list on commas, ignoring commas inside brackets.

    ``sp18.fits,sp25.fits`` is two images; ``obj.fits[SCI,1]`` is one.
    """
    pieces, current, depth = [], [], 0
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            pieces.append("".join(current))
            current = []
        else:
            current.append(char)
    pieces.append("".join(current))
    return [piece.strip() for piece in pieces if piece.strip()]


def expand(items: list[str] | str) -> list[ImageSpec]:
    """Expand inputs the way IRAF's image templates work.

    Handles ``@file`` lists, comma-separated lists, and shell globs including
    character classes -- and keeps any trailing extension or section, applying
    it to every file a pattern matched.

    harwood expands globs itself rather than relying on the shell, because
    section 3.1 tells users to quote anything containing brackets or ``*``, and
    a quoted pattern never reaches the shell to be expanded.
    """
    import glob as globbing

    if isinstance(items, str):
        items = [items]

    out: list[ImageSpec] = []
    for item in items:
        for piece in split_list(str(item)):
            if piece.startswith("@"):
                listing = Path(piece[1:])
                if not listing.is_file():
                    raise FileNotFoundError(f"list file '{listing}' does not exist")
                for line in listing.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        out.extend(expand(line))
                continue

            name, brackets = split_brackets(piece)
            if any(char in name for char in "*?[") and not Path(name).exists():
                matches = sorted(globbing.glob(name))
                if not matches:
                    raise FileNotFoundError(f"no file matches '{name}'")
                out.extend(parse(found + brackets) for found in matches)
            else:
                out.append(parse(piece))
    return out


def open_hdu(spec: ImageSpec | str, mode: str = "readonly"):
    """Open the HDU a spec names. The caller closes the file."""
    from astropy.io import fits

    spec = parse(spec) if not isinstance(spec, ImageSpec) else spec
    hdul = fits.open(str(spec.path), mode=mode, memmap=False)
    try:
        index = _resolve_extension(hdul, spec.extension)
    except Exception:
        hdul.close()
        raise
    return hdul, hdul[index]


def _resolve_extension(hdul, extension):
    if extension is None:
        # The first HDU that actually holds pixels -- MEF files put nothing in
        # the primary HDU, and asking the user to know that is the kind of
        # detail section 2 is trying to remove.
        for index, hdu in enumerate(hdul):
            if getattr(hdu, "data", None) is not None:
                return index
        return 0
    if isinstance(extension, tuple):
        return hdul.index_of(extension)
    return hdul.index_of(extension) if isinstance(extension, str) else extension


def read(spec: ImageSpec | str) -> tuple[Any, Any]:
    """Read pixels and header, applying the section if one was given.

    Data comes back as float64: measurements are never made on the integers a
    detector happens to write, and a silent integer overflow in a sum of bias
    frames is exactly the kind of quiet wrong answer harwood exists to avoid.
    """
    import numpy as np

    spec = parse(spec) if not isinstance(spec, ImageSpec) else spec
    hdul, hdu = open_hdu(spec)
    try:
        data = np.asarray(hdu.data, dtype=np.float64)
        header = hdu.header.copy()
    finally:
        hdul.close()
    if spec.section:
        data = data[pixels.parse_section(spec.section, data.shape)]
        header["HRWDSECT"] = (spec.section, "section read by harwood, 1-based (x,y)")
    return data, header


def read_stack(specs: list[ImageSpec | str]):
    """Read many images into one (n, ...) array, checking they match."""
    import numpy as np

    frames, headers = [], []
    shape = None
    for spec in specs:
        data, header = read(spec)
        if shape is None:
            shape = data.shape
        elif data.shape != shape:
            raise ValueError(
                f"'{spec}' is {data.shape} but the first image is {shape}; "
                "combining needs images of one shape"
            )
        frames.append(data)
        headers.append(header)
    if not frames:
        raise ValueError("no input images")
    return np.stack(frames), headers
