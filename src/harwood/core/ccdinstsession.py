"""IRAF's ccdinstrument: the translations an instrument file gives an image, and the
command loop that edits them. No Qt and no terminal: lines come from `read`, text goes
to `write` and `ewrite`.

From iraf-community 2.18: noao/imred/ccdred/src/{t_ccdinst.x,hdrmap.x,ccdtypes.x,
ccdsubsets.x,ccdflag.x}. See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from . import images, instrument

CMDS = [
    "quit",
    "?",
    "help",
    "show",
    "instrument",
    "imheader",
    "read",
    "write",
    "newimage",
    "translate",
    "imagetyp",
    "subset",
    "exptime",
    "darktime",
    "fixfile",
    "biassec",
    "ccdsec",
    "datasec",
    "trimsec",
    "darkcor",
    "fixpix",
    "flatcor",
    "fringcor",
    "illumcor",
    "overscan",
    "readcor",
    "scancor",
    "trim",
    "zerocor",
    "ccdmean",
    "fringscl",
    "illumflt",
    "mkfringe",
    "mkillum",
    "skyflat",
    "ncombine",
    "date-obs",
    "dec",
    "ra",
    "title",
    "next",
    "nscanrow",
]
CCDTYPES = ("object", "zero", "dark", "flat", "illum", "fringe", "other", "comp")
LEVELS = ("basic", "common", "all")
FLAGS = {
    "fixpix", "overscan", "trim", "zerocor", "darkcor", "flatcor", "illumcor", "fringcor",
    "readcor", "scancor", "illumflt", "mkfringe", "mkillum", "skyflat",
}  # fmt: skip
PROMPTS = {
    "imagetyp": "image type",
    "subset": "subset parameter",
    "exptime": "exposure time",
    "darktime": "dark time",
    "fixfile": "bad pixel file",
    "biassec": "bias section",
    "ccdsec": "original CCD section",
    "datasec": "data section",
    "trimsec": "trim section",
    "darkcor": "dark count flag",
    "fixpix": "bad pixel flag",
    "flatcor": "flat field flag",
    "fringcor": "fringe flag",
    "illumcor": "illumination flag",
    "overscan": "overscan flag",
    "readcor": "read correction flag",
    "scancor": "scan mode flag",
    "nscanrow": "scan mode rows",
    "trim": "trim flag",
    "zerocor": "zero level flag",
    "ccdmean": "mean value",
    "fringscl": "fringe scale",
    "illumflt": "illumination flat image",
    "mkfringe": "fringe image",
    "mkillum": "illumination image",
    "skyflat": "sky flat image",
    "ncombine": "number of images combined",
    "date-obs": "date of observation",
    "dec": "declination",
    "ra": "ra",
    "title": "title",
}
HELP = Path(__file__).parent.parent / "data" / "iraf" / "ccdred"


def strdic(word: str, entries) -> tuple[int, str]:
    """strdic: the place in the dictionary from 1 and the whole word; 0 if it is in
    none, or begins two. An empty word begins the first and every other."""
    word = word.lstrip(" \t")
    match, found = 0, ""
    for entry, text in enumerate(entries, 1):
        if text.startswith(word):
            if text == word:
                return entry, text
            if match != 0:
                return 0, found
            match, found = entry, text
    return match, found


def words(line: str) -> list[str]:
    """gargwrd, one after the other: blank-separated, or a quoted string."""
    out, i, n = [], 0, len(line)
    while True:
        while i < n and line[i] in " \t":
            i += 1
        if i >= n:
            return out
        if line[i] in "\"'":
            end = line.find(line[i], i + 1)
            end = n if end < 0 else end
            out.append(line[i + 1 : end])
            i = end + 1
        else:
            start = i
            while i < n and line[i] not in " \t":
                i += 1
            out.append(line[start:i])


class Map:
    """hdrmap: the symbols of a translation file, the last entered found first and
    written first."""

    def __init__(self) -> None:
        self.table: list[list[str]] | None = None

    def open(self, name: str, warn: Callable[[str], None]) -> None:
        """hdmopen. A file that is not there leaves a table with nothing in it."""
        if not name:
            self.table = None
            return
        self.table = []
        try:
            path = instrument.resolve(name)
        except (FileNotFoundError, ValueError):
            warn(f"Warning: Cannot open file ({name})\n")
            return
        for line in path.read_text(encoding="latin-1").splitlines():
            fields = words(line)
            if not fields or fields[0].startswith("#"):
                continue
            self.table.insert(0, [fields[0], *(fields[1:3] + ["", ""])[:2]])

    def _find(self, parameter: str):
        for symbol in self.table or []:
            if symbol[0] == parameter:
                return symbol
        return None

    def name(self, parameter: str) -> str:
        symbol = self._find(parameter)
        return symbol[1] if symbol else parameter

    def default(self, parameter: str) -> str:
        symbol = self._find(parameter)
        return symbol[2] if symbol else ""

    def put_name(self, parameter: str, text: str) -> None:
        if self.table is None:
            return
        symbol = self._find(parameter)
        if symbol is None:
            symbol = [parameter, "", ""]
            self.table.insert(0, symbol)
        symbol[1] = text

    def put_default(self, parameter: str, text: str) -> None:
        if self.table is None:
            return
        symbol = self._find(parameter)
        if symbol is None:
            symbol = [parameter, parameter, ""]
            self.table.insert(0, symbol)
        symbol[2] = text

    def get(self, header, image: str, parameter: str) -> str:
        """hdmgstr: the image's value, or else the default, or else nothing."""
        from ..tasks.imutil import image_parameter

        key = self.name(parameter)
        value = image_parameter(header, image, key)
        if value == "" and key.lower().removeprefix("i_") == "title" and "OBJECT" in header:
            return " " * 8  # the FITS kernel's title for an OBJECT with nothing in it
        return self.default(parameter) if value is None else value

    def has(self, header, image: str, parameter: str) -> bool:
        """hdmaccf."""
        from ..tasks.imutil import image_parameter

        return image_parameter(header, image, self.name(parameter)) is not None

    def text(self) -> str:
        """hdmwrite: what the file is written with."""
        out = []
        for parameter, name, default in self.table or []:
            line = (f"'{parameter}'" if _blanks(parameter) else parameter).ljust(29)
            line += (f" '{name}'" if _blanks(name) else f" {name}").ljust(9)
            if default:
                line += f" '{default}'" if _blanks(default) else f" {default}"
            out.append(line + "\n")
        return "".join(out)


def _blanks(text: str) -> bool:
    return " " in text or "\t" in text


def _write(path: str, text: str) -> None:
    file = Path(path)
    if file.exists():
        file.unlink()
    file.write_text(text, encoding="latin-1")


class Session:
    """t_ccdinst's state: the map, the image and whether anything is to be saved."""

    def __init__(self, inst: str, ssfile: str, level: str, read, write, ewrite) -> None:
        self.inst, self.ssfile = inst, ssfile
        self.level = LEVELS.index(level) + 1
        self.read, self.write, self.ewrite = read, write, ewrite
        self.map = Map()
        self.map.open(inst, ewrite)
        self.update = False
        self.ncmd = "next"
        self.image, self.header = "", None
        self._fields: list[str] = []
        self._nscan = 0

    # ------------------------------------------------------------ scan

    def _scan(self) -> bool:
        line = self.read()
        if line is None:
            return False
        self._fields = words(line)
        self._nscan = 0
        return True

    def _garg(self) -> str:
        """gargwrd: the next word and nscan one more, or nothing and nscan the same."""
        if self._nscan < len(self._fields):
            self._nscan += 1
            return self._fields[self._nscan - 1]
        return ""

    def _ask(self, prompt: str, default: str) -> str:
        """ccdinst_g."""
        self.write(f"{prompt} ({default}): ")
        answer = ""
        if self._scan():
            answer = self._garg()
        return default if self._nscan == 0 else answer

    # ------------------------------------------------------------ what is shown

    def types(self) -> str:
        """ccdtypes."""
        text = self.map.get(self.header, self.image, "imagetyp")
        if not text:
            return "none"
        name = self.map.name(text) or text
        place, whole = strdic(name, CCDTYPES)
        return whole if place else "unknown"

    def subset(self) -> str:
        """ccdsubset: the record file is read and added to."""
        text = self.map.get(self.header, self.image, "subset")
        fields = words(text)
        subset1 = fields[0] if fields else ""
        if text:
            base, count = subset1, 1
            file = Path(self.ssfile)
            records = []
            if file.is_file():
                records = [words(line) for line in file.read_text().splitlines()]
            i = 0
            while i < len(records):
                row = (records[i] + ["", ""])[:2]
                i += 1
                if row[0] == text:
                    subset1, count = row[1], 0
                    break
                if row[1] == subset1:
                    subset1 = f"{base}{count}"
                    count += 1
                    i = 0
            if count > 0:
                with open(file, "a") as out:
                    out.write(f"'{text}'\t{subset1}\n")
        return "".join(c if c.isalnum() or c == "." else "_" for c in subset1)

    def _p(self, name: str) -> None:
        """ccdinst_p: a line of the table."""
        key = self.map.name(name)
        default = self.map.default(name)
        value = self.map.get(self.header, self.image, name)
        if not value:
            value = "?"
        has = self.map.has(self.header, self.image, name)
        head = f"{name:<8}  {key:<8}  {default:<8}"
        if name == "imagetyp":
            self.write(f"{head}  {self.types():<8}  {value[:39]}\n")
        elif name == "subset":
            self.write(f"{head}  {self.subset():<8}  {value[:39]}\n")
        elif name in FLAGS:
            flag = self.map.get(self.header, self.image, name) != default
            shown = value if has else "?"
            self.write(f"{head}  {'yes' if flag else 'no':<8}  {shown[:39]}\n")
        else:
            self.write(f"{head}  {value:<8}  {(value if has else '?')[:39]}\n")

    def show(self) -> None:
        """ccdinst_hdr."""
        self.write(f"Image: {self.image}\nInstrument file: {self.inst}\n")
        self.write(f"Subset file: {self.ssfile}\n")
        titles = ("CCDRED", "IMAGE", "DEFAULT", "CCDRED", "IMAGE")
        self.write("\n" + "  ".join(f"{t:<8}" for t in titles) + "\n")
        titles = ("PARAM", "KEYWORD", "VALUE", "VALUE", "VALUE")
        self.write("  ".join(f"{t:<8}" for t in titles) + "\n")
        self.write("-" * 78 + "\n")
        for name in ("imagetyp", "subset", "exptime", "darktime"):
            self._p(name)
        if self.level > 1:
            self.write("\n")
            self._p("biassec")
            self._p("trimsec")
            self.write("\n")
            for name in ("fixpix", "overscan", "trim", "zerocor", "darkcor", "flatcor"):
                self._p(name)
        if self.level > 2:
            for name in ("datasec", "ccdsec", "fixfile"):
                self._p(name)
            self.write("\n")
            for name in ("illumcor", "fringcor", "readcor", "scancor", "nscanrow"):
                self._p(name)
            self.write("\n")
            for name in ("illumflt", "mkfringe", "mkillum", "skyflat"):
                self._p(name)
            self.write("\n")
            for name in ("ccdmean", "fringscl", "ncombine", "date-obs", "dec", "ra", "title"):
                self._p(name)
        self.write("\n")

    def imheader(self) -> None:
        """ccdinst_i: OBJECT and the user area, as the FITS kernel leaves it."""
        title = str(self.header.get("OBJECT", "")).rstrip()
        self.write(f"OBJECT  = '{title}'\n")
        for image in user_area(self.image):
            self.write(image.rstrip()[:80] + "\n")
        self.write("\n")

    # ------------------------------------------------------------ editing

    def _edit(self, name: str) -> None:
        """ccdinst_e: the default changes only on the line of the command."""
        key, default = self._garg(), self._garg()
        oldkey, olddef = self.map.name(name), self.map.default(name)
        count = self._nscan
        if count < 2:
            self.write(f"Image keyword for {PROMPTS[name]} ({oldkey}): ")
            if self._scan():
                key = self._garg()
            if self._nscan == 0:
                key = oldkey
        if count < 3:
            default = olddef
        if key != oldkey:
            self.map.put_name(name, key)
            self.update = True
        if default != olddef:
            self.map.put_default(name, default)
            self.update = True
        self._p(name)

    def _translate(self) -> None:
        current = self.types()
        text = self.map.get(self.header, self.image, "imagetyp")
        wanted = self._garg()
        if self._nscan < 2:
            self.write(f"CCDRED image type for '{text}' ({current}): ")
            if self._scan():
                wanted = self._garg()
            if self._nscan == 0:
                wanted = current
        place, whole = strdic(wanted, CCDTYPES)
        if place == 0:
            self.write("Unknown CCDRED image type\n")
            whole = current
        if whole != current and not text:
            # stenter's null key, which ends the task in IRAF and loses every change
            self.ewrite("This image has no image type to translate\n")
        elif whole != current:
            self.map.put_name(text, whole)
            self._p("imagetyp")
            self.update = True

    def _read_file(self) -> None:
        name = self._garg()
        if self._nscan < 2:
            name = self._ask("Instrument file", self.inst)
        if self.update:
            self.write("WARNING: Previous changes lost\n")
        same = name == self.inst
        self.update = False
        if same:
            # IRAF closes the table and leaves nothing in its place
            self.map.open(self.inst, self.ewrite)
            return
        self.map.open(name, self.ewrite)
        self.show()
        self.update = True

    def _write_file(self) -> None:
        name = self._garg()
        if self._nscan < 2:
            name = self._ask("Instrument file", self.inst)
        if self.map.table is None:
            self.update = False
            return
        try:
            _write(name, self.map.text())
            self.update = False
        except OSError:
            self.ewrite(f"Warning: Cannot open file ({name})\n")

    def _new_image(self) -> None:
        name = self._garg()
        if self._nscan < 2:
            name = self._ask("New image name", self.image)
        if name == self.image:
            return
        header = open_image(name)
        if header is None:
            self.ewrite(f"Warning: Cannot open image ({name})\n")
            return
        self.image, self.header = name, header
        self.show()

    def edit(self, image: str, header) -> None:
        """ccdinst_edit: the loop, from an implied show to quit, next or the end."""
        self.image, self.header = image, header
        self._fields, self._nscan = ["show"], 0
        while True:
            command = self._garg()
            place, whole = strdic(command, CMDS)
            self.ncmd = whole if place else ""
            if place and whole in ("next", "quit"):
                return
            if place == 0:
                if self._nscan > 0:
                    self.ewrite("Unrecognized or ambiguous command\x07\n")
            elif whole in ("?", "help"):
                key = HELP / f"ccdinst{self.level}.key"
                self.write(key.read_text())
            elif whole == "show":
                self.show()
            elif whole == "instrument":
                self.write(self.map.text() + "\n")
            elif whole == "imheader":
                self.imheader()
            elif whole == "read":
                self._read_file()
            elif whole == "write":
                self._write_file()
            elif whole == "newimage":
                self._new_image()
            elif whole == "translate":
                self._translate()
            else:
                self._edit(whole)
            self.write("ccdinstrument> ")
            if not self._scan():
                return

    def finish(self) -> None:
        """The end of t_ccdinst: asked once, if anything changed."""
        if not self.update:
            return
        self.write(f"Update instrument file {self.inst} (yes)? ")
        if self._scan():
            answer = self._garg().lower()
            if answer.startswith(("n", "f")):
                self.update = False
        if self.update:
            try:
                _write(self.inst, self.map.text())
            except OSError:
                self.ewrite(f"Warning: Cannot open file ({self.inst})\n")


def user_area(name: str) -> list[str]:
    """The cards the FITS kernel leaves in the user area: of a primary the ones it does
    not make its own; of an extension that inherits, the primary's first, without
    comments and what the extension also has."""
    import re

    hdul, hdu = images.open_hdu(images.parse(name))
    try:
        cards = []
        if hdu is not hdul[0] and hdu.header.get("INHERIT", False):
            theirs = {card.keyword for card in hdu.header.cards}
            cards = [
                card
                for card in hdul[0].header.cards
                if card.keyword not in ("COMMENT", "HISTORY", "")
                and card.keyword not in theirs
                and not re.fullmatch(r"SIMPLE|BITPIX|NAXIS\d*|EXTEND", card.keyword)
            ]
        drop = r"SIMPLE|BITPIX|NAXIS\d*|XTENSION"
        cards += [card for card in hdu.header.cards if not re.fullmatch(drop, card.keyword)]
        return [card.image for card in cards]
    finally:
        hdul.close()


def open_image(name: str):
    """immap: the header, or None."""
    try:
        return images.read_header(images.parse(name))
    except (OSError, ValueError, KeyError, IndexError):
        return None
