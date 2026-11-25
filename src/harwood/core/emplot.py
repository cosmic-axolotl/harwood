"""emplot -- RVSAO's graph of emsao's result and its cursor, as RVSAO 2.8.5 draws it.

The spectrum (smoothed as for the search, or with its continuum subtracted), each
line of the emission list marked where the velocity puts it -- a solid line over a
line found and fit, a dashed one where none was found or the cursor dropped it --
and the absorption lines below; unzoomed, the graph takes the left two thirds and a
summary the right: the date, the velocity correction, VELOCITY, the correlation's
and the emission lines' velocities, each template and each line found. Zoomed, the
graph takes the screen, titled with the velocity plotted.

The cursor zooms (z z, u, .), toggles the heading (h), the full screen (/), the
continuum (k) and the labels (o), plots at another velocity (v), sets the velocity
from a line named or typed (e, a), refits (b r g l m i c + - f), sets VELOCITY by
hand (s) and its quality (y n j), edits the spectrum (d d, ,) and moves along a
multispec (( )). From Emsao/{emplot,emeplot,emiplot}.x; what the keys change in the
fit is core/emsao.py's. Kept, because they decide what happens:

* q is gt_gcur's end of file: it leaves the cursor loop with the refits the keys
  before it asked for still asked for; Q is read as q and runs the q of emplot's
  code (leave without refitting), which q itself never reaches.
* The questions of o, i, s and v read the next cursor key (clgcur), not gt_gcur: q
  there is a key like the others.
* ? and the space print the keys and leave the loop, like f.
* Whatever was edited is written when the loop ends, by ] too; only x forgets it.
* An unzoomed graph's y limits keep the dy of the graph before when its minimum is
  not above zero (emeplot's dy is read before it is set); a label's height keeps the
  last one found when no pixel brackets its wavelength.

Not kept: c closes the graph and never opens it again, so the next graph drawn is
an illegal file descriptor (emsao stops); here c asks for the order and the graph
stays. d's second press calls fillpix without its last argument and ends in a
segmentation violation, always; here it fills, as its code means to. # (debug) is
not ported: the key is read and does nothing.

Translated from RVSAO, Copyright (c) 1991-2022 Smithsonian Astrophysical
Observatory, whose notice is in LICENSES/RVSAO-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import iraffmt
from .rvsao import C0, INDEFD, f32

VCORREL, VEMISS, VCOMB, VSEARCH, VGUESS, ZGUESS, VCORTEMP = 1, 2, 3, 4, 5, 6, 7
FULL = (0.08, 0.66, 0.10, 0.90)            # gsview of the unzoomed graph
ZOOMED = (0.1125, 0.8875, 0.147, 0.9412)   # glabax's own on stdgraph, gsview 0 1 0 1
SUMMARY = (0.67, 1.0, 0.0, 1.0)
XLABEL = "Wavelength in angstroms"

KEYS = (
    "*** Emission line display cursor commands ***\n"
    "a  Set redshift guess from absorption line\n"
    "b  Set blue limit of line search\n"
    "c  Change continuum parameters\n"
    "d  Delete data between 1st and 2nd positions\n"
    "e  Set redshift guess from emission line\n"
    "f  Refit redshift\n"
    "g  Number of times to smooth plotted spectrum\n"
    "h  Toggle print of heading with filename and redshift\n"
    "i  Change initial velocity for search\n"
    "j  Conditional velocity\n"
    "k  Plot continuum-subtracted spectrum\n"
    "l  Line search parameters\n"
    "m  Number of times to smooth fit spectrum\n"
    "n  Disapprove velocity\n"
    "o  Turn line labelling on and off\n"
    "p  Replot current graph\n"
    "q  Leave plot, writing out edited spectrum\n"
    "r  Set red limit of line search\n"
    "s  Set VELOCITY to a specific value\n"
    "t  Switch template for correlation velocity\n"
    "u  Unzoom\n"
    "v  Plot at e>mission x>correlation c>ombined velocity\n"
    "w  Show rest and observed wavelengths\n"
    "x  Plot correlation if available or exit\n"
    "y  Approve velocity\n"
    "z  Zoom between 1st and 2nd positions\n"
    ".  Cancel delete or zoom\n"
    ",  Set a single pixel value to the cursor location\n"
    "#  Turn debug on or off temporarily\n"
    "/  Toggle plot between full screen and lines\n"
    "+  Add emission line to fit\n"
    "-  Subtract emission line from fit\n"
    "(  Plot previous aperture or order\n"
    ")  Plot next aperture or order\n"
    "]  Leave plot without writing spectrum\n"
    "@  Make hard copy of screen\n"
    "?  Display this menu\n")


def _f(spec: str, value) -> str:
    """IRAF's printf of one value, INDEF as IRAF prints it."""
    return iraffmt.Format(spec)(value)


@dataclass
class Text:
    """A text drawn: where (in the graph's coordinates, or the screen's 0-1 when
    `ndc`), what, its size (1 = the device's), and whether it reads upwards."""
    x: float
    y: float
    text: str
    size: float = 1.0
    ndc: bool = False
    up: bool = False


@dataclass
class Frame:
    """One graph emeplot draws: the window and viewport, the title and the axis'
    label, the spectrum, the lines' marks (segments x1 y1 x2 y2), the labels, and --
    unzoomed -- the summary's texts on the screen."""
    window: tuple[float, float, float, float]
    viewport: tuple[float, float, float, float]
    title: str
    xlabel: str
    x: np.ndarray
    y: np.ndarray
    marks: list[tuple[float, float, float, float]] = field(default_factory=list)
    labels: list[Text] = field(default_factory=list)
    summary: list[Text] = field(default_factory=list)

    def ndc(self, x: float, y: float) -> tuple[float, float]:
        x1, x2, y1, y2 = self.window
        v1, v2, v3, v4 = self.viewport
        return (v1 + (x - x1) / (x2 - x1) * (v2 - v1),
                v3 + (y - y1) / (y2 - y1) * (v4 - v3))

    def svg(self, width: int = 1000, height: int = 760) -> str:
        """The frame as a picture (a hard copy): its lines and texts, an axis box."""
        def px(x, y, ndc=False):
            nx, ny = (x, y) if ndc else self.ndc(x, y)
            return nx * width, (1.0 - ny) * height

        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
               f'height="{height}" viewBox="0 0 {width} {height}">',
               f'<rect width="{width}" height="{height}" fill="white"/>']
        v1, v2, v3, v4 = self.viewport
        out.append(f'<rect x="{v1 * width:.1f}" y="{(1 - v4) * height:.1f}" '
                   f'width="{(v2 - v1) * width:.1f}" height="{(v4 - v3) * height:.1f}" '
                   'fill="none" stroke="black"/>')
        x1, x2, y1, y2 = self.window
        keep = (self.y >= y1) & (self.y <= y2)
        pts = " ".join("{:.1f},{:.1f}".format(*px(a, b))
                       for a, b, k in zip(self.x, self.y, keep, strict=True) if k)
        out.append(f'<polyline points="{pts}" fill="none" stroke="black"/>')
        for a, b, c, d in self.marks:
            (p, q), (r, s) = px(a, b), px(c, d)
            out.append(f'<line x1="{p:.1f}" y1="{q:.1f}" x2="{r:.1f}" y2="{s:.1f}" '
                       'stroke="black"/>')

        def esc(s):
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        for t in self.labels + self.summary:
            p, q = px(t.x, t.y, t.ndc)
            size = 14 * t.size
            anchor = "middle" if not t.ndc else "start"
            turn = f' transform="rotate(-90 {p:.1f} {q:.1f})"' if t.up else ""
            out.append(f'<text x="{p:.1f}" y="{q:.1f}" font-size="{size:.1f}" '
                       f'font-family="monospace" text-anchor="{anchor}"{turn}>'
                       f'{esc(t.text)}</text>')
        if self.title:
            out.append(f'<text x="{width / 2:.1f}" y="{0.04 * height:.1f}" font-size="14" '
                       f'text-anchor="middle" font-family="monospace">{esc(self.title)}'
                       '</text>')
        out.append(f'<text x="{(v1 + v2) / 2 * width:.1f}" y="{0.98 * height:.1f}" '
                   f'font-size="14" text-anchor="middle">{esc(self.xlabel)}</text>')
        out.append("</svg>\n")
        return "\n".join(out)


class Graph:
    """emeplot and emiplot: what is drawn from emsao's state. Its dy and the height
    of the last label are emeplot's locals, which keep their values from one graph to
    the next (SPP's locals are static)."""

    def __init__(self) -> None:
        self.dy = f32(0)
        self.y1 = f32(0)

    def draw(self, em, fit, specid: str, mspec: int, pix, wpix, npix: int) -> Frame:
        """emeplot: npix < 0 is a zoomed (or full-screen) graph."""
        alldata = npix >= 0
        npix = abs(npix)
        pix = np.array(pix[:npix], f32)           # copies: the keys edit the spectrum
        wpix = np.array(wpix[:npix], f32)
        r = fit.result
        dispvert = bool(em.dispem and not em.dispabs)
        if em.vplot == VEMISS:
            velplot = r.vel
        elif em.vplot == VCORREL:
            velplot = r.xcvel
        elif em.vplot == VCOMB:
            velplot = r.combined
        elif em.vplot == VGUESS:
            velplot = em.gvel
        else:
            em.vplot = VCOMB
            velplot = r.combined
        if velplot == INDEFD:
            velplot = 0.0
        text = ""
        if not alldata:
            if em.vplot == VEMISS:
                text = (f"{specid}  em vel= {_f('%.2f', velplot)} ({_f('%.2f', r.err)}) "
                        "km/sec")
            elif em.vplot == VCORREL:
                text = (f"{specid}  corr vel= {_f('%.2f', velplot)} "
                        f"({_f('%.2f', r.xcerr)}) km/sec R= {_f('%.2f', r.xcr)}")
            elif em.vplot == VCOMB:
                text = (f"{specid}  comb vel= {_f('%.2f', velplot)} "
                        f"({_f('%.2f', r.combined_err)}) km/sec R= {_f('%.2f', r.xcr)}")
            elif em.vplot == VGUESS:
                if velplot < 1.0:
                    velplot = velplot * C0
                text = f"{specid} vel= {_f('%8.2f', velplot)} km/sec"
        bcz = 1.0 + (r.spectrum.hcv / C0)
        em.shift = (1.0 + (velplot / C0)) / bcz
        shift = em.shift
        xmin, xmax = wpix[0], wpix[npix - 1]
        if xmin > xmax:
            xmin, xmax = wpix[npix - 1], wpix[0]
        ymin = ymax = pix[0]
        for v in pix[1:]:
            if v < ymin:
                ymin = v
            if v > ymax:
                ymax = v
        dy = self.dy
        if ymin > 0:
            ymin = f32(0)
            dy = f32(ymax / f32(100))
            ymax = f32(ymax + dy * f32(30 if dispvert else 10))
        elif ymin == ymax:
            dy = f32(0.01)
            ymin = f32(ymin - dy * f32(100))
            ymax = f32(ymax + dy * f32(100))
        else:
            ymin = f32(ymin - dy)
            dy = f32(f32(ymax - ymin) / f32(100))
            ymax = f32(ymax + dy * f32(30 if dispvert else 10))
        self.dy = dy
        dy4 = f32(dy * f32(0.25))
        frame = Frame((float(xmin), float(xmax), float(ymin), float(ymax)),
                      FULL if alldata else ZOOMED,
                      "" if (mspec < 0 or alldata) else text, XLABEL, wpix, pix)
        size = 0.6 if dispvert else 0.8
        npix1 = npix - 1
        lines = em.lines
        found = r.lines

        def bracket(wx: float) -> int | None:
            j_last = None
            for j in range(npix1):
                a, b = float(wpix[j]), float(wpix[j + 1])
                if (a <= wx < b) or (b < wx <= a):
                    j_last = j
            return j_last

        if (em.dispem or em.plotem) and lines.wlref:
            lfound = fit.lfound
            for i, wl in enumerate(lines.wlref):
                ifound = lfound[i]
                wx = (found[ifound - 1].wlrest if ifound > 0 else wl) * shift
                x = f32(wx)
                if not (x > xmin and x < xmax):
                    continue
                j = bracket(wx)
                if j is not None:
                    jm1, jp1 = max(j - 1, 0), min(j + 1, npix - 1)
                    y1 = pix[j]
                    if y1 < pix[jm1]:
                        y1 = pix[jm1]
                    if y1 < pix[jp1]:
                        y1 = pix[jp1]
                    self.y1 = f32(y1 + f32(2.5) * dy)
                y1 = self.y1
                if dispvert:
                    y2 = f32(y1 + f32(10) * dy)
                    if y2 > f32(ymax - f32(30) * dy):
                        y2 = f32(ymax - f32(25) * dy)
                else:
                    y2 = f32(y1 + f32(30) * dy)
                    if y2 > f32(ymax - f32(5) * dy):
                        y2 = f32(ymax - f32(5) * dy)
                if y2 < y1:
                    y2 = f32(y1 + dy)
                if not y1 < ymax:
                    continue
                if ifound > 0 and (found[ifound - 1].params[0, 9] > 0
                                   or fit.override[ifound - 1] >= 0):
                    frame.marks.append((float(x), float(y1), float(x), float(y2)))
                else:
                    ny = int(1.0 + f32(f32(y2 - y1) / dy))
                    y = y1
                    for _ in range(ny):
                        frame.marks.append((float(x), float(f32(y - dy4)), float(x),
                                            float(f32(y + dy4))))
                        y = f32(y + dy)
                label = lines.nmref[i]
                if dispvert:
                    label += " " + _f("%8.3f", wl)
                frame.labels.append(Text(float(x), float(f32(y2 + dy)), label, size,
                                         up=dispvert))
        if em.dispabs and lines.absorption:
            for wl, name in lines.absorption:
                wx = wl * shift
                x = f32(wx)
                if not (x > xmin and x < xmax):
                    continue
                j = bracket(wx)
                if j is not None:
                    jm1, jp1 = max(j - 1, 0), min(j + 1, npix - 1)
                    y1 = pix[j]
                    if y1 > pix[jm1]:
                        y1 = pix[jm1]
                    if y1 > pix[jp1]:
                        y1 = pix[jp1]
                    self.y1 = f32(y1 - f32(2.5) * dy)
                y1 = self.y1
                y2 = f32(y1 - f32(30) * dy)
                if y2 < f32(ymin + f32(5) * dy):
                    y2 = f32(ymin + f32(5) * dy)
                if y2 > y1:
                    y2 = f32(y1 - dy)
                ny = int(1.0 + f32(f32(y1 - y2) / dy))
                y = y1
                for _ in range(ny):
                    frame.marks.append((float(x), float(f32(y + dy4)), float(x),
                                        float(f32(y - dy4))))
                    y = f32(y - dy)
                label = name
                if dispvert:
                    label += " " + _f("%8.3f", wl)
                frame.labels.append(Text(float(x), float(f32(y2 - f32(3) * dy)), label,
                                         size, up=dispvert))
        if alldata:
            frame.summary = self.summary(em, fit, specid)
        return frame

    @staticmethod
    def summary(em, fit, specid: str) -> list[Text]:
        """emiplot: the summary page, on the screen's coordinates."""
        from .rvsao import imgdate, juldate, logtime

        r = fit.result
        spec = r.spectrum
        head = spec.head
        out = [Text(0.65, 0.01, f"rvsao.emsao 2.8.5 {logtime()}", 0.6, ndc=True)]
        ut, dj, _hdj, _date = juldate(head)
        text = f"File: {specid}"
        if dj != 0.0:
            text += f"  JulDate: {_f('%.5f', dj)}"
        out.append(Text(0.025, 0.96, text, 0.8, ndc=True))
        ra = head.d("RA", INDEFD)
        if ra != INDEFD and ":" not in (head.string("RA") or ""):
            ra = ra / 15.0
        dec = head.d("DEC", INDEFD)
        epoch = head.d("EPOCH", 1950.0)
        epoch = head.d("EQUINOX", epoch)
        if ra != INDEFD and dec != INDEFD:
            text = (f"Object:{spec.object[:10]:<10} RA: {_f('%011.2h', ra)} "
                    f"DEC: {_f('%011.2h', dec)} {_f('%.1f', epoch)}")
        else:
            text = f"Object: {spec.object}  (No position)"
        out.append(Text(0.025, 0.91, text, 0.8, ndc=True))

        x0 = SUMMARY[0]            # WCS 3: its window's 0 is the screen's 0.67
        lineno = 0.96
        mm = dd = yyyy = None
        for key in ("DATE-OBS", "DATE"):
            if head.has(key):
                mm, dd, yyyy = imgdate(head, key)
                break
        if None not in (mm, dd, yyyy):
            month = "JanFebMarAprMayJunJulAugSepOctNovDec"[3 * (mm - 1):3 * mm] \
                if 1 <= mm <= 12 else ""
            text = f"{yyyy:04d}-{month:3.3s}-{dd:02d} {_f('%011.2h', ut)}"
        else:
            text = "(No observation date)"
        out.append(Text(x0, lineno, text, 1.0, ndc=True))
        names = {3: "Object HCV: ", 4: "Object BCV: ", 5: "Object file HCV: ",
                 2: "Object file BCV: " if spec.vb else "Object file HCV: "}
        if em.svcor in names:
            text = names[em.svcor] + _f("%7.3f", spec.hcv)
        else:
            text = "Object BCV:    0.000"
        lineno -= 0.04
        out.append(Text(x0, lineno, text, 0.6, ndc=True))
        if r.combined != INDEFD:
            q = {1: "?", 2: "?", 3: "X", 4: "Q"}.get(fit.quality, "_")
            text = (f" VELOCITY = {_f('%.2f', r.combined)} +- "
                    f"{_f('%.2f', r.combined_err)} km/sec {q}")
            if em.vplot in (VCOMB, VGUESS):
                text = "*" + text[1:]
            lineno -= 0.04
            out.append(Text(x0, lineno, text, 0.6, ndc=True))
        if r.xcvel != INDEFD:
            text = (f" Corr vel = {_f('%.2f', r.xcvel)} +- {_f('%.2f', r.xcerr)} km/sec "
                    f"R= {_f('%.2f', r.xcr)}")
            if em.vplot == VCORREL:
                text = "*" + text[1:]
            lineno -= 0.04
            out.append(Text(x0, lineno, text, 0.6, ndc=True))
        text = (f" Emis vel = {_f('%.2f', r.vel)} +- {_f('%.2f', r.err)} km/sec "
                f"{r.nfit}/{len(r.lines)} lines")
        if em.vplot == VEMISS:
            text = "*" + text[1:]
        lineno -= 0.04
        out.append(Text(x0, lineno, text, 0.6, ndc=True))
        temps = fit.templates
        if temps and len(temps) + len(r.lines) < 17:
            lineno -= 0.06
            out.append(Text(x0, lineno, "Template   CZ   error    R", 0.7, ndc=True))
            for k in fit.template_order():
                t = temps[k]
                text = (f"{t.id[:7]:<7} {_f('%9.3f', t.zvel)} {_f('%7.3f', t.czerr)} "
                        f"{_f('%6.2f', t.czr)}")
                if t.schop:
                    text += " -e"
                text += {1: " -f", 2: " +h", 3: " -f +h"}.get(t.filt, "")
                lineno -= 0.04
                out.append(Text(x0, lineno, text, 0.6, ndc=True))
        if r.lines:
            lineno -= 0.06
            out.append(Text(x0, lineno, "Line  Rest   Obs.      CZ     error", 0.6,
                            ndc=True))
            for k, f in enumerate(r.lines):
                if f.params[0, 9] > 0:
                    over = fit.override[k]
                    wch = "-" if over < 0 else "+" if over > 0 else " "
                else:
                    wch = "X"
                text = (f"{f.name[:4]:<4} {_f('%7.2f', f.wlrest)} {_f('%7.2f', f.wlobs)} "
                        f"{_f('%8.2f', (C0 * f.params[0, 8]) + spec.hcv)} "
                        f"{_f('%5.2f', C0 * f.params[1, 8])} {wch}")
                if f.drop > 0:
                    text += str(f.drop)
                lineno -= 0.04
                out.append(Text(x0, lineno, text, 0.6, ndc=True))
        else:
            lineno -= 0.04
            out.append(Text(x0, lineno, "No emission lines found\n", 0.6, ndc=True))
        velset = head.string("VELSET")
        if velset is not None:
            lineno -= 0.04
            out.append(Text(x0, lineno, velset.rstrip(), 0.6, ndc=True))
        return out


# ------------------------------------------------------------- the cursor


class Plot:
    """emplot: the graph of one spectrum and the cursor's keys on it. `read_key`
    gives the next cursor read (x, y, wcs, key) or None at the end; `read_line` the
    next line typed (getline: its text without the newline, or None at the end).
    What emplot prints is said through `say`. After `run`, `refit` holds what the fit
    is asked to do again ("search", "lines", "velocity" or None), `mspec` the
    aperture to go to, and `frames` every graph drawn."""

    def __init__(self, em, fit, mspec: int, nmax: int, read_key, read_line, say,
                 curmode: bool = True) -> None:
        self.em, self.fit = em, fit
        self.mspec0 = mspec          # emplot's argument: the aperture to go on with
        self.mspec = mspec           # the heading's sign
        self.nmax = nmax
        self.read_key, self.read_line, self.say = read_key, read_line, say
        self.curmode = curmode
        self.frames: list[Frame] = []
        self.hardcopies: list[Frame] = []
        r = fit.result
        self.ipix01, self.ipix02 = fit.pix1, fit.pix2
        self.ipix1, self.ipix2 = fit.pix1, fit.pix2
        self.npix0 = fit.pix2 - fit.pix1 + 1
        self.npix = self.npix0
        self.splvel = INDEFD
        self.vcomb, self.vcerr = r.combined, r.combined_err
        self.dfirst = self.zfirst = True
        self.ipx1 = self.ipx2 = 0
        self.edited = False

    # -------------------------------------------------------------- drawing

    def _wcs(self):
        return self.fit.result.spectrum.wcs

    def draw(self) -> Frame:
        fit = self.fit
        if self.dispmode == 3 or (0 < self.npix < self.npix0):
            self.npix = -self.npix
        pix = fit.smspec if self.contsub else fit.smplot
        frame = self.em.graph.draw(self.em, fit, fit.result.spectrum.name, self.mspec,
                                   pix[self.ipix1 - 1:], fit.wlspec[self.ipix1 - 1:],
                                   self.npix)
        self.npix = abs(self.npix)
        self.frames.append(frame)
        return frame

    # -------------------------------------------------------------- the loop

    def run(self) -> None:
        """emplot: the graph, then the keys from read_key until q, the end, or a key
        that leaves."""
        if not self.start():
            return
        while True:
            read = self.read_key()
            if read is None or self.press(read[0], read[1], read[3]):
                break
        self.finish()

    def start(self) -> bool:
        """What emplot does before the cursor: the velocity the graph is drawn at,
        the hard copy, and the graph. False when there is no cursor to read."""
        em, fit = self.em, self.fit
        r = fit.result
        wcs = self._wcs()
        fit.specref, fit.waverest = 0, 0.0
        if em.plotem:
            em.dispem = True
        if r.spectrum.head.has("VELSET"):
            em.dispem = em.dispabs = True
        wblue = wcs.p2w(float(self.ipix01))
        wred = wcs.p2w(float(self.ipix02))
        self.reversed = False
        if wblue > wred:
            wred, wblue = wcs.p2w(float(self.ipix01)), wcs.p2w(float(self.ipix02))
            self.reversed = True
        self.wblue, self.wred = wblue, wred
        velp = {VEMISS: r.vel, VCORREL: r.xcvel, VGUESS: em.gvel}.get(em.vplot, r.combined)
        if velp == INDEFD:
            velp = 0.0
        self.bcz = 1.0 + (r.spectrum.hcv / C0)
        em.shift = (1.0 + (velp / C0)) / self.bcz
        fit.lfit = fit.vfit = fit.sfit = False
        dispmode = int(em.p.get("dispmode", 2))
        self.contsub = dispmode in (4, 5)
        self.dispmode = {4: 2, 5: 3}.get(dispmode, dispmode)
        self.it0 = 1
        if bool(em.p.get("hardcopy", False)):
            self.hardcopies.append(self.draw())
            self.frames.pop()
        if not bool(em.p.get("displot", True)):
            return False
        self.draw()
        return self.curmode

    def press(self, x: float, y: float, key: str) -> bool:
        """One cursor read: True when emplot leaves its loop (q, gt_gcur's end of
        file, among them)."""
        if key == "q":
            return True
        if "A" <= key <= "Z":
            key = key.lower()
        stop = self.key(key, x, y)
        if self.newgraph:
            self.draw()
        return stop

    def finish(self) -> None:
        """After the loop: what was edited is to be written."""
        if self.edited:
            self.fit.edited = True

    def _ix(self, x: float) -> tuple[float, int]:
        px = self._wcs().w2p(x)
        ix = int(px)
        if px - float(ix) > 0.5:
            ix += 1
        return px, ix

    def key(self, key: str, x: float, y: float) -> bool:
        """One key: True to leave the loop."""
        em, fit = self.em, self.fit
        r = fit.result
        wcs = self._wcs()
        self.newgraph = False
        self.x, self.y = x, y
        wx = float(f32(x))                      # clgcur's x is real
        wr = wx / em.shift
        px, ix = self._ix(wx)
        say = self.say
        if key == "p":
            self.newgraph = True
        elif key == "/":
            self.dispmode = 2 if self.dispmode == 3 else 3
            self.newgraph = True
        elif key == "h":
            if self.mspec > 0:
                self.mspec = -self.mspec
            elif self.mspec == 0:
                self.mspec = -1
            else:
                self.mspec = -self.mspec
            self.newgraph = True
        elif key in (")", "("):
            if key == ")":
                if self.mspec0 > 0:
                    self.mspec0 = self.mspec0 + 1 if self.mspec0 < self.nmax else 1
            elif self.mspec0 > 1:
                self.mspec0 -= 1
            else:
                self.mspec0 = self.nmax
            fit.lfit = fit.vfit = fit.sfit = False
            return True
        elif key == "@":
            say(f"Making hard copy on {em.p.get('plotter', 'stdplot')}\n")
            frame = self.draw()
            self.frames.pop()
            self.hardcopies.append(frame)
        elif key in ("+", "-"):
            iw, diff = 0, 1.0e2
            for i, f in enumerate(r.lines, 1):
                if abs(wr - f.wlrest) < diff:
                    diff, iw = abs(wr - f.wlrest), i
            if iw > 0:
                fit.override[iw - 1] = 1 if key == "+" else -1
                f = r.lines[iw - 1]
                say(f"{'Adding' if key == '+' else 'Dropping'} {f.name} line at "
                    f"{_f('%d', f.wlrest)} angstroms\n")
                fit.vfit = True
        elif key in ("y", "n", "j"):
            fit.quality = {"y": 4, "n": 3, "j": 1}[key]
            say({"y": "Setting quality flag to 4=good\n",
                 "n": "Setting quality flag to 3=bad\n",
                 "j": "Setting quality flag to 1=questionable\n"}[key])
            fit.qplot = True
        elif key in ("a", "e"):
            self._line(key, wx)
        elif key == "d":
            if self.dfirst:
                self.ipx1 = ix
                say(f"d again to delete data from pixel {self.ipx1:.7f}\n")
                self.dfirst = False
            else:
                self.dfirst = True
                self.ipx2 = ix
                say(f"filling from pixel {self.ipx1:.7f} to pixel {self.ipx2:.7f}\n")
                from .rvsao import fillpix
                a, b = self.ipx1, self.ipx2
                fillpix(fit.specwcs, a, b)
                fillpix(fit.smplot, a, b)
                self.edited = True
                self.newgraph = True
        elif key == ",":
            ip = min(max(ix, 1), fit.specwcs.size)
            say(f"replacing pixel {ix} with {float(f32(y)):.7f}\n")
            fit.specwcs[ip - 1] = f32(y)
            fit.smplot[ip - 1] = f32(y)
            self.edited = True
            self.newgraph = True
        elif key == "o":
            say("Label a>bsorption e>mission b>oth lines else none: ")
            sub = self._subkey()
            say(f"{sub}\n")
            if sub == "a":
                say("label absorption lines\n")
                em.dispabs, em.dispem, em.plotem = True, False, False
            elif sub == "e":
                say("label emission lines\n")
                em.dispem, em.dispabs = True, False
            elif sub == "b":
                say("label absorption and emission lines\n")
                em.dispem = em.dispabs = True
            else:
                say("do not label absorption and emission lines\n")
                em.dispem = em.dispabs = False
            self.newgraph = True
        elif key == "i":
            say("s>earch x>cor e>mission f>ile g>uess velocity: ")
            sub = self._subkey()
            say(f"{sub}\n")
            if sub == "e":
                say("restart from file emission velocity\n")
                em.vinit, fit.sfit = VEMISS, True
            elif sub == "x":
                say("restart from correlation velocity\n")
                em.vinit, fit.sfit = VCORREL, True
            elif sub == "c":
                say("restart from combination velocity\n")
                em.vinit, fit.sfit = VCOMB, True
            elif sub == "g":
                say("\nRestart using this velocity: ")
                line = self.read_line()
                if line:
                    value = _ctod(line)
                    r.combined = value if value is not None else r.combined
                    em.vinit, fit.sfit = VCOMB, True
            else:
                say("restart and search for velocity\n")
                em.vinit, fit.sfit = VSEARCH, True
        elif key == "k":
            self.contsub = not self.contsub
            say("Plot continuum-subtracted spectrum\n" if self.contsub
                else "Plot spectrum including continuum\n")
            self.newgraph = True
        elif key == "l":
            say("Line search parameters\n")
            say(f"Number of sigma above continuum ({_f('%4.2f', fit.zsig)}): ")
            line = self.read_line()
            if line:
                value = _ctod(line)
                fit.zsig = value if value is not None else fit.zsig
                fit.lfit = True
            say(f"Wavelength to search around redshifted line center "
                f"({_f('%4.1f', fit.wspan)}):")
            line = self.read_line()
            if line:
                value = _ctod(line)
                fit.wspan = value if value is not None else fit.wspan
                fit.sfit = True
            say(f"Number of pixels to fit around redshifted line center ({fit.npfit}):")
            line = self.read_line()
            if line:
                value = _ctoi(line)
                fit.npfit = value if value is not None else fit.npfit
                fit.lfit = True
        elif key == "m":
            say(f"Number of times to smooth fit spectrum ({fit.esmooth}): ")
            line = self.read_line()
            if line:
                value = _ctoi(line)
                fit.esmooth = value if value is not None else fit.esmooth
                fit.lfit = True
        elif key in ("b", "r"):
            blue = key == "b"
            if blue:
                say(f"Blue wavelength limit in angstroms ({_f('%8.1f', self.wblue)}): ")
            else:
                say(f"Red wavelength limit in angstroms ({_f('%8.1f', self.wred)}): ")
            line = self.read_line()
            if line:
                value = _ctod(line)
                if blue:
                    self.wblue = value if value is not None else self.wblue
                    p = int(wcs.w2p(self.wblue) + 0.99)
                    if self.reversed:
                        self.ipix2 = p
                    else:
                        self.ipix1 = p
                else:
                    self.wred = value if value is not None else self.wred
                    p = int(wcs.w2p(self.wred) - 0.99)
                    if self.reversed:
                        self.ipix1 = p
                    else:
                        self.ipix2 = p
                self.npix = self.ipix2 - self.ipix1 + 1
                fit.sfit = True
                self.newgraph = True
        elif key == "g":
            say(f"Number of times to smooth spectrum ({fit.nsmooth}): ")
            line = self.read_line()
            if line:
                value = _ctoi(line)
                fit.nsmooth = value if value is not None else fit.nsmooth
                say(f"Smoothing spectrum {fit.nsmooth} times\n")
                from .rvsao import smooth
                n = fit.specwcs.size
                fit.smplot[:] = fit.specwcs
                smooth(fit.smplot, n, fit.nsmooth)
                fit.smspec[:] = fit.sfspec
                smooth(fit.smspec, n, fit.nsmooth)
                fit.sfit = True
                self.newgraph = True
        elif key == "s":
            self._set_velocity()
        elif key == "v":
            say("e>m x>corr c>ombined g>uessed velocity: ")
            sub = self._subkey()
            say(f"{sub}\n")
            if sub == "e":
                say("emission velocity redshift used\n")
                em.vplot = VEMISS
            elif sub == "x":
                say("correlation velocity redshift used\n")
                em.vplot = VCORREL
            elif sub == "c":
                say("combination velocity redshift used\n")
                em.vplot = VCOMB
            else:
                say("last guessed velocity redshift used\n")
                em.vplot = VGUESS
            self.newgraph = True
            fit.lfit = True
        elif key == "c":
            say("Continuum order: ")
            line = self.read_line()
            if line:
                value = _ctoi(line)
                if value is not None:
                    em.order = value
                fit.lfit = True
        elif key == "t":
            if fit.templates:
                say("template for redshift = ")
                line = self.read_line()
                it = self.it0
                if line:
                    value = _ctoi(line)
                    it = value if value is not None and 1 <= value <= len(fit.templates) \
                        else self.it0
                order = fit.template_order()
                t = fit.templates[order[it - 1]]
                fit.best = order[it - 1] + 1
                self.it0 = it
                r.xcvel, r.xcerr, r.xcr = t.zvel, t.czerr, t.czr
                em.vplot = VCORREL
                self.newgraph = True
        elif key == ".":
            if not self.zfirst:
                say("zoom cancelled\n")
                self.ipix1, self.ipix2 = self.ipix01, self.ipix02
                self.npix = self.ipix2 - self.ipix1 + 1
                self.zfirst = True
            if not self.dfirst:
                say("delete cancelled\n")
                self.dfirst = True
        elif key == "z":
            if self.zfirst:
                self.ipx1 = self.ipix1 if ix < 1 else ix
                self.zfirst = False
                say(f"z again to zoom from {_f('%8.2f', float(fit.wlspec[self.ipx1 - 1]))}"
                    " angstroms\n")
            else:
                self.ipx2 = self.ipix2 if ix < 1 else ix
                self.zfirst = True
                self.ipix1, self.ipix2 = sorted((self.ipx1, self.ipx2))
                self.npix = self.ipix2 - self.ipix1 + 1
                say("zooming from " + self._range() + "\n")
                self.newgraph = True
        elif key == "u":
            self.ipix1, self.ipix2 = self.ipix01, self.ipix02
            self.npix = self.npix0
            say("unzoom to " + self._range() + "\n")
            self.newgraph = True
        elif key in (" ", "?"):
            say(KEYS)
            return True
        elif key == "w":
            say(f"Wavelength: Rest: {_f('%9.3f', wr)} -> Observed: {_f('%9.3f', wx)} "
                f"Pixel {_f('%9.3f', px)} = {_f('%8.2f', float(f32(y)))}\n")
        elif key == "f":
            if fit.edited or self.edited:
                fit.savespec = fit.qplot = True
            return True
        elif key == "]":
            fit.lfit = fit.vfit = fit.sfit = False
            fit.savespec = False
            return True
        elif key == "q":                      # Q: emplot's own q
            fit.lfit = fit.vfit = fit.sfit = False
            if self.edited or fit.edited:
                fit.savespec = fit.qplot = True
            return True
        elif key == "x":
            fit.lfit = fit.vfit = fit.sfit = False
            self.edited = fit.edited = False
            fit.savespec = False
            fit.plotcorr = True
            fit.qplot = False
            return True
        return False

    def _range(self) -> str:
        w = self.fit.wlspec
        a, b = float(w[self.ipix1 - 1]), float(w[self.ipix2 - 1])
        if a > b:
            a, b = b, a
        return (f"{_f('%8.2f', a)} to {_f('%8.2f', b)} angstroms (pixels {self.ipix1} - "
                f"{self.ipix2})")

    def _subkey(self) -> str:
        read = self.read_key()
        if read is None:
            return "\x04"           # EOF: clgcur leaves the key as it was; none known
        return read[3]

    def _line(self, kind: str, wx: float) -> None:
        """e and a: the velocity from a line named, or its rest wavelength typed."""
        em, fit = self.em, self.fit
        r = fit.result
        lines = em.lines
        if kind == "a":
            names = [n for _, n in lines.absorption]
            waves = [w for w, _ in lines.absorption]
            self.say("Absorption line (" + "".join(f" {n}" for n in names)
                     + " or angstroms): ")
        else:
            names, waves = lines.nmref, lines.wlref
            self.say("Emission line (" + "".join(f" {n}" for n in names)
                     + " or angstroms): ")
        line = self.read_line()
        if line is None or line == "":
            return
        lname = list(line)
        for i in range(len(lname)):
            c = lname[i]
            if i == 0 and ord(c) > 96:
                lname[i] = chr(ord(c) - 32)
            if kind == "a" and i == 2 and ord(c) > 96:
                lname[i] = chr(ord(c) - 32)
            if kind == "e" and i > 0 and c in "iv":
                lname[i] = chr(ord(c) - 32)
        lname = "".join(lname)
        specref, waverest = 0, 0.0
        for i, name in enumerate(names, 1):
            if name.startswith(lname) and len(name) == len(lname):
                specref, waverest = i, waves[i - 1]
        if specref == 0:
            value = _ctod(lname)
            waverest = value if value is not None else 0.0
        if waverest > 0.0:
            cvel = wx / waverest
            fit.cvel = cvel
            self.splvel = (cvel - 1.0) * C0
            r.combined = self.splvel
            word = "absorption" if kind == "a" else "emission"
            if specref > 0:
                self.say(f"{names[specref - 1]} {word}: Observed {_f('%7.2f', wx)}A -> ")
                if kind == "a":
                    specref = -specref
            else:
                self.say(f"{word.capitalize()}: Observed {_f('%7.2f', wx)}A -> ")
            self.say(f"Rest {_f('%7.2f', waverest)}A vel= {_f('%7.1f', r.combined)}  "
                     f"z= {_f('%6.4f', cvel - 1.0)}\n")
            em.shift = (1.0 + (r.combined / C0)) / self.bcz
            em.gvel = r.combined
            fit.lfit = True
            em.vplot = VGUESS
            em.dispem = em.dispabs = True
            self.newgraph = True
        else:
            self.say(f"* No {'absorption' if kind == 'a' else 'emission'} line found: "
                     f"{lname}\n\n")
        fit.specref, fit.waverest = specref, waverest

    def _set_velocity(self) -> None:
        """s: VELOCITY from the emission lines, the correlation, their combination,
        the line set with e or a, the last guess (with an error typed) or typed."""
        em, fit = self.em, self.fit
        r = fit.result
        say = self.say
        say("set VELOCITY to e>m x>corr c>omb l>ine g>uess s>et: ")
        sub = self._subkey()
        say(f"{sub}\n")
        if sub == "c":
            say("combined velocity redshift used\n")
            r.combined, r.combined_err = self.vcomb, self.vcerr
            fit.specref, fit.waverest = 0, 0.0
        elif sub == "e":
            say("emission velocity redshift used\n")
            r.combined, r.combined_err = r.vel, r.err
            fit.specref, fit.waverest = 0, 0.0
        elif sub == "x":
            say("correlation velocity redshift used\n")
            r.combined, r.combined_err = r.xcvel, r.xcerr
            fit.specref, fit.waverest = 0, 0.0
        elif sub == "l":
            lines = em.lines
            if fit.specref > 0:
                say(f"velocity {_f('%.3f', self.splvel)} km/sec from "
                    f"{lines.nmref[fit.specref - 1]} line used\n")
                r.combined, r.combined_err = self.splvel, 300.0
            elif fit.specref < 0:
                say(f"velocity {_f('%.3f', self.splvel)} km/sec from "
                    f"{lines.absorption[-fit.specref - 1][1]} line used\n")
                r.combined, r.combined_err = self.splvel, 300.0
            elif fit.waverest > 0:
                say(f"velocity {_f('%.3f', self.splvel)} km/sec from "
                    f"{_f('%.3f', fit.waverest)}A line used\n")
                r.combined, r.combined_err = self.splvel, 300.0
            else:
                say("no lines have been identified\n")
        elif sub == "g":
            say(f"last guessed velocity, {_f('%.3f', r.combined)} km/sec, used\n")
            say("Error in km/sec = ")
            line = self.read_line()
            if line:
                value = _ctod(line)
                r.combined_err = value if value is not None else r.combined_err
            fit.specref, fit.waverest = 0, 0.0
        else:
            say("Velocity in km/sec = ")
            line = self.read_line()
            if line:
                value = _ctod(line)
                r.combined = value if value is not None else r.combined
            say("Error in km/sec = ")
            line = self.read_line()
            if line:
                value = _ctod(line)
                r.combined_err = value if value is not None else r.combined_err
        fit.lfit = fit.vfit = fit.sfit = False
        em.vplot = VCOMB
        self.newgraph = True


def _ctod(text: str) -> float | None:
    """ctod: the number at the start of the text, as IRAF reads it (its digits
    summed as IRAF sums them), None when there is none."""
    from .rvsao import Scan

    return Scan(text).d()


def _ctoi(text: str) -> int | None:
    from .rvsao import Scan

    value = Scan(text).d()
    return None if value is None else int(value)


def nmax(spec) -> int:
    """The apertures ( and ) go round: IM_LEN(im, 2), or the third axis' length when
    the second is 1; 1 for a one-dimensional image."""
    head = spec.head
    ndim = head.i("NAXIS", 1)
    if ndim < 2:
        return 1
    if ndim < 3:
        return head.i("NAXIS2", 1)
    n2 = head.i("NAXIS2", 1)
    return n2 if n2 > 1 else head.i("NAXIS3", 1)
