"""daofind's interactive mode: the keys and colon commands, without a screen.

From apphot/daofind (ap_fdfind, ap_fdcolon, ap_fdstars, ap_find) and aplib
(ap_apcolon, ap_nscolon, ap_nshow). A person tries a threshold, sees what it
finds, and writes the list when it looks right. A key is a method call that
collects what IRAF prints, so the window and a cursor file (`--icommands`) are
the same session, and the session can be held to IRAF's output without one.

The quirks kept, because they decide the numbers. f does not write and the space
bar does. A parameter that changes the kernel (fwhmpsf, sigma, nsigma...) makes
the next find a new one; one that only changes the selection (threshold, sharplo...)
makes it a refit, as does finding again after an f. A refit gets every star's
sharpness wrong: ap_find marks the kernel's centre as skipped each time it runs
and counts the kernel again, so the second time it divides by one pixel fewer.
That is IRAF's number, and a star can pass or fail sharplo by it, so `refit` says
when it happened and the task warns. After the space bar nothing is found again
until a parameter changes, and each list written goes to a file of its own: the
output name, then name.2, name.3. (IRAF's third is named for the image, as
field.fits.coo.1, by way of a pattern match meant for something else.)

Not ported: i, v and w, which mark and prompt for parameters and save the psets
(r over a star gives the FWHM; `:fwhmpsf` sets it; run.yaml keeps what was used),
and the colon commands for what harwood's daofind has no use for, as `:datamin`.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import daofind, iraffmt
from .photcal.fitsession import strdic

f32 = np.float32

HELP = [("space", "find the stars and write them"),
        ("f", "find the stars, without writing"),
        ("d", "the radial profile of the star at the cursor"),
        ("?", "these keys"),
        (":", "fwhmpsf sigma threshold nsigma ratio theta sharplo sharphi roundlo roundhi "
              "epadu readnoise scale show"),
        ("q", "finish")]

#: A word, what it changes, and its units. "kernel": the next find is a new one.
_REALS = {"fwhmpsf": ("kernel", "scaleunit"),
          "sigma": ("kernel", "counts"),
          "epadu": ("kernel", "e-/adu"),
          "readnoise": ("kernel", "e-"),
          "nsigma": ("kernel", "sigma"),
          "ratio": ("kernel", "number"),
          "theta": ("kernel", "degrees"),
          "threshold": ("fit", "sigma"),
          "sharplo": ("fit", "number"),
          "sharphi": ("fit", "number"),
          "roundlo": ("fit", "number"),
          "roundhi": ("fit", "number")}
#: ap_fdcolon looks a word up in each of these in turn, so an abbreviation only
#: has to be unambiguous within one: :sh is show, though it also begins sharplo.
_APPHOT = ("fwhmpsf", "emission", "exposure", "itime", "datamin", "datamax", "image", "coords",
           "output", "scale", "airmass", "xairmass", "filter", "ifilter", "obstime", "otime")
_NOISE = ("noise", "sigma", "epadu", "gain", "ccdread", "readnoise")
_FIND = ("nsigma", "ratio", "sharplo", "sharphi", "roundlo", "roundhi", "mkdetections",
         "theta", "threshold")


def _g(value) -> str:
    return iraffmt.g(float(f32(value)))


@dataclass
class Session:
    data: np.ndarray
    image: str
    output: str
    params: dict
    said: list[str] = field(default_factory=list)   #: what IRAF prints, line by line
    written: list[tuple[str, dict, bool]] = field(default_factory=list)
    found: dict | None = None                       #: the last find, for the display
    wx: float = 0.0
    wy: float = 0.0
    done: bool = False
    _newden: bool = True
    _newfit: bool = True

    def __post_init__(self) -> None:
        self.params = {name: float(value) for name, value in self.params.items()}
        self.outname = self.output

    def key(self, key: str, x: float = 0.0, y: float = 0.0) -> None:
        if key == "q":
            self.done = True
            return
        if key == "?":
            self.said += [f"{k}\t{what}" for k, what in HELP]
        elif key in ("f", " "):
            self._find(key == " ")
        elif key in ("i", "v", "w"):
            self.said.append(f"{key} is not ported: r over a star gives the FWHM, :fwhmpsf "
                             "sets it, and run.yaml keeps what was used.")
        elif key != "d":
            self.said.append("Unknown or ambiguous keystroke command")
        self.wx, self.wy = x, y

    def _find(self, write: bool) -> None:
        """The switch of ap_fdfind, on what has changed since the last time."""
        if self._newden:
            self._newden = False
            refit = False
        elif self._newfit:
            refit = True
        else:
            self.said.append("Detection parameters have not changed")
            return
        if write:
            self._newfit = False
            if self.written:
                self.outname = f"{self.output}.{len(self.written) + 1}"
        self._run(write, refit)

    def _run(self, write: bool, refit: bool) -> None:
        """ap_fdstars and what ap_find prints."""
        p = self.params
        names = ("threshold", "nsigma", "ratio", "theta", "sharplo", "sharphi", "roundlo",
                 "roundhi", "epadu", "readnoise")
        found = daofind.find(self.data, p["fwhmpsf"], p["sigma"], scale=p["scale"],
                             refit=refit, **{name: p[name] for name in names})
        self.said += ["", f"Image: {self.image}  fwhmpsf: {_g(p['fwhmpsf'])}  "
                          f"ratio:  {_g(p['ratio'])}  theta:  {_g(p['theta'])}  "
                          f"nsigma: {_g(p['nsigma'])}", ""]
        for n in range(found["x"].size):
            self.said.append(
                f"{found['x'][n]:8.2f} {found['y'][n]:8.2f} {found['mag'][n]:8.3f} "
                f"{found['sharpness'][n]:7.3f} {found['sround'][n]:7.3f} "
                f"{found['ground'][n]:7.3f} {n + 1:5d}")
        self.said += ["", f"threshold: {_g(f32(p['threshold']) * f32(p['sigma']))} "
                          f"relerr: {found['relerr']:5.3f}  {_g(p['sharplo'])} <= sharp <= "
                          f"{_g(p['sharphi'])}  {_g(p['roundlo'])} <= round <= "
                          f"{_g(p['roundhi'])}  ", "", f"Output file: {self.outname}", ""]
        self.found = found
        if write:
            self.written.append((self.outname, found, refit))

    # ------------------------------------------------------- colon commands

    def colon(self, text: str) -> None:
        """ap_fdcolon."""
        words = text.lstrip(":").split()
        if not words:
            return
        for words_in in (_APPHOT, _NOISE, _FIND, ("show",)):
            found = strdic(words[0], words_in)
            if found:
                break
        else:
            self.said.append("Unknown or ambiguous colon command")
            return
        command = words_in[found - 1]
        if command == "show":
            self._show(words[1] if len(words) > 1 else "")
        elif command == "scale" or command in _REALS:
            self._real(command, words[1] if len(words) > 1 else None)
        else:
            self.said.append(f":{command} is not ported: harwood's daofind has no use for it.")

    def _real(self, command: str, value: str | None) -> None:
        what, units = _REALS.get(command, ("kernel", "units"))
        number = None
        if value is not None:
            try:
                number = float(f32(value))
            except ValueError:
                number = None
        if number is None:
            current = self.params[command]
            if command == "scale":
                current = 1.0 / current
            self.said.append(f"{command} = {_g(current)} {units}")
            return
        self.params[command] = 1.0 / number if command == "scale" else number
        self._newfit = True
        if what == "kernel":
            self._newden = True

    def _show(self, what: str) -> None:
        which = strdic(what, ("data", "find")) if what else 0
        self.said.append("")
        if which == 1:
            self.said += self._show_data()
        elif which == 2:
            self.said += self._show_find()
        else:
            self.said += [*self._show_data(), "", *self._show_find()]
        self.said.append("")

    def _show_data(self) -> list[str]:
        """ap_nshow."""
        p = self.params
        return ["", "Data Characteristics",
                f"    image: {self.image} ({self.wx:0.2f}, {self.wy:0.2f})   "
                f"scale: {_g(1.0 / p['scale'])}",
                f"    output: {self.outname}    coords: ",
                f"    fwhmpsf = {_g(p['fwhmpsf'])} scaleunit    emission = yes",
                "    datamin = INDEF counts    datamax = INDEF counts",
                "    exposure =     itime = 1. timeunit",
                "    filter =     ifilter = INDEF",
                "    airmass =     xairmass = INDEF",
                "    obstime =     otime = INDEF",
                "",
                "Noise Model",
                f"    noise = poisson  model    sigma = {_g(p['sigma'])} counts",
                f"    gain =     epadu = {_g(p['epadu'])} e-/adu",
                f"    ccdread =     readnoise = {_g(p['readnoise'])} e-"]

    def _show_find(self) -> list[str]:
        """ap_fshow."""
        p = self.params
        return ["Kernel Parameters",
                f"    fwhmpsf = {_g(p['fwhmpsf'])} scaleunit    emission = yes",
                f"    nsigma = {_g(p['nsigma'])} sigma    ratio = {_g(p['ratio'])}   "
                f"theta = {_g(p['theta'])} degrees",
                "",
                "Detection Parameters",
                f"    threshold = {_g(p['threshold'])} sigma",
                f"    sharplo = {_g(p['sharplo'])}    sharphi = {_g(p['sharphi'])}",
                f"    roundlo = {_g(p['roundlo'])}    roundhi = {_g(p['roundhi'])}"]
