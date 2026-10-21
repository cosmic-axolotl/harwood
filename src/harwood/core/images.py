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
    r"(?:\[(?P<ext>[^\[\]:*]*?(?:,\s*\d+)?)\])?"
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
    path = Path(match.group("path"))
    # IRAF names an image without its extension: 'arc' is arc.fits.
    named = path.with_name(path.name + ".fits")
    if not path.exists() and path.suffix != ".fits" and named.exists():
        path = named
    return ImageSpec(path, extension, section)


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


def substitute(name: str | Path, template: str) -> Path:
    """Build an output name from an input name, IRAF's `%old%new%` template.

    ``%m%tr%`` turns ``m92001.fits`` into ``tr92001.fits`` -- the first
    occurrence of `old` in the name is replaced by `new`. It is how IRAF names
    a whole night's worth of outputs in one parameter, and it beats a
    directory when the point is to keep the files side by side and see at a
    glance which have been through which step.

    A template without `%` is treated as a directory to write into, which is
    the other thing people reach for.
    """
    source = Path(str(name))
    if "%" not in template:
        return Path(template) / source.name

    parts = template.split("%")
    if len(parts) < 4:
        raise ValueError(
            f"'{template}' is not a substitution template. Write %old%new%, "
            "as in %m%tr% to turn m92001.fits into tr92001.fits."
        )
    before, old, new = parts[0], parts[1], parts[2]
    rest = "%".join(parts[3:])
    if old not in source.name:
        raise ValueError(
            f"'{source.name}' does not contain '{old}', so the template "
            f"'{template}' has nothing to replace."
        )
    replaced = source.name.replace(old, new, 1)
    return source.parent / (before + replaced if not rest else before + replaced)


def outputs_for(specs: list, template: str) -> list[Path]:
    """Work out one output name per input, from whatever the user wrote.

    Four forms, all of which people reach for and IRAF accepts:

        one name              a single input, a single output
        %old%new%             substitution, keeping the files side by side
        a,b,c  or  @list      a list, matched one to one with the inputs
        a directory           the inputs' own names, written there

    Getting this wrong quietly -- writing every input to the same output, say
    -- destroys a night's work, so a count mismatch is an error rather than a
    guess.
    """
    template = str(template).strip()
    if not template:
        # Nowhere named: this run's own directory, with the frames keeping
        # their names. See core/output.py for why that is the default.
        from . import output

        folder = output.directory()
        return [folder / Path(str(s.path if hasattr(s, "path") else s)).name
                for s in specs]
    if "%" in template:
        return [substitute(spec.path if hasattr(spec, "path") else spec, template)
                for spec in specs]

    if template.startswith("@") or "," in template:
        names = [str(s.path) if hasattr(s, "path") else str(s)
                 for s in expand(template)] if template.startswith("@") \
            else [piece.strip() for piece in template.split(",") if piece.strip()]
        if len(names) != len(specs):
            raise ValueError(
                f"{len(specs)} inputs but {len(names)} outputs. Give one output "
                "per input, a directory, or a template like '%m%tr%'."
            )
        return [Path(name) for name in names]

    if len(specs) == 1:
        return [Path(template)]

    directory = Path(template)
    directory.mkdir(parents=True, exist_ok=True)
    return [directory / Path(str(s.path if hasattr(s, "path") else s)).name
            for s in specs]


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
        header = inherited(hdul, hdu)
    finally:
        hdul.close()
    if spec.section:
        slices = pixels.parse_section(spec.section, data.shape)
        pixels.section_header(header, pixels.section_geometry(slices, data.shape))
        data = data[slices]
        header["HRWDSECT"] = (spec.section, "section read by harwood, 1-based (x,y)")
    return data, header


def inherited(hdul, hdu):
    """The extension's header, with the primary's underneath when it says
    INHERIT = T -- as harwood's SCI/VAR/DQ files do (core/mef.py), so a task
    reading SCI still sees OBJECT, EXPTIME and FILTER. The extension wins."""
    from . import mef

    header = hdu.header.copy()
    if hdu is hdul[0] or not hdu.header.get("INHERIT", False):
        return header
    out = mef.merged(hdul[0].header, header)
    for key in ("BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTNAME", "EXTVER"):
        if key in header:
            out[key] = header[key]
    return out


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
