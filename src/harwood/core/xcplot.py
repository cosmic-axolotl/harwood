"""xcplot -- RVSAO's graph of xcsao's result and its cursor, as RVSAO 2.8.5 draws it.

Above, the spectrum (smoothed nsmooth times), with the range correlated marked by
dashed lines; below, the correlation with the best template (or the one t chose)
around its peak, the points at -pkfrac of its height marked when pkfrac is
negative; at the right the summary: the date, the correction, the range, the grid
and the filter, each template's velocity and peak, and each one's velocity, error
and R. dispmode 0 shows only the range correlated, a negative dispmode the
spectrum from zero.

The keys: r redraws, t takes another template's velocity, g smooths the spectrum
drawn, m changes the display mode, c the peak's fit and the filter, v the
velocities a peak may have, f correlates again; y, n and j set the quality; @ makes
a picture; x leaves without writing the header; ? lists the keys. From
Xcsao/{xcplot,xcsplot,xcorplot,xciplot}.x and Util/plotutil.x's rvscale. Kept,
because they decide what is drawn and said:

* q is gt_gcur's end of file, and any key xcplot does not know leaves too.
* Without a date of observation, and so no Julian date, the summary says "No
  observation date" twice: the line for the Julian date is the one before.
* m reads its answer with getline: an empty line is read as the display mode m
  read last (ctoi leaves its number as it was).
* g smooths the spectrum but does not draw it again: r shows it.

Not kept: c's peak fraction and filter and v's velocity limits are read, a new
correlation asked for, and then lost -- xcfit reads pkfrac, minvel and maxvel from
the parameters again for each template, and the filter from its own copies -- so
that only c's peak-fitting mode reaches the correlation. Here all of them do: the
correlation after c or v is the one those parameters give from the start.

Not ported yet: e (edit the spectrum: plotspec), p (choose the peak by hand), d
(debug), and the display modes and keys that go to emsao's graph (2, 3, l, m to 2 or
3): harwood says so and stays on this graph.

Translated from RVSAO, Copyright (c) 1991-2022 Smithsonian Astrophysical
Observatory, whose notice is in LICENSES/RVSAO-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import iraffmt
from .emplot import Text
from .rvsao import INDEFD, f32
from .xcsao import COR_PIX, COR_WAV, ZEROCONT

SPECTRUM_VIEW = (0.06, 0.66, 0.60, 0.90)
CORRELATION_VIEW = (0.06, 0.66, 0.15, 0.45)
SUMMARY_X = 0.67

KEYS = ("XCSAO commands:\n"
        " e  edit spectrum\n"
        " c  change correlation peak fit parameters\n"
        " d  toggle debug flag\n"
        " f  rerun cross-correlation\n"
        " j  conditional velocity\n"
        " l  plot spectrum with absorption lines\n"
        " m  change display mode to -1, 0, 1, or 2\n"
        " n  disapprove velocity\n"
        " p  refit correlation peak interactively\n"
        " q  exit from this spectrum\n"
        " r  replot summary\n"
        " t  change template to use for xcor velocity\n"
        " v  change velocity limits for result\n"
        " x  Exit without updating header\n"
        " y  approve velocity\n"
        " @  make hard copy\n")


def _f(spec: str, value) -> str:
    return iraffmt.Format(spec)(value)


@dataclass
class Panel:
    """One graph: its window and viewport, title and label, the curve, the dashed
    marks (segments x1 y1 x2 y2) and the circles' centres."""
    window: tuple[float, float, float, float]
    viewport: tuple[float, float, float, float]
    title: str
    xlabel: str
    x: np.ndarray
    y: np.ndarray
    marks: list[tuple[float, float, float, float]] = field(default_factory=list)
    circles: list[tuple[float, float]] = field(default_factory=list)

    def ndc(self, x: float, y: float) -> tuple[float, float]:
        x1, x2, y1, y2 = self.window
        v1, v2, v3, v4 = self.viewport
        return (v1 + (x - x1) / (x2 - x1) * (v2 - v1),
                v3 + (y - y1) / (y2 - y1) * (v4 - v3))


@dataclass
class Frame:
    """What one redraw puts on the screen: the spectrum's graph, the correlation's
    (None when dvel is 0), and the summary's texts (on the screen's 0-1)."""
    spectrum: Panel
    correlation: Panel | None
    texts: list[Text]

    def svg(self, width: int = 1000, height: int = 760) -> str:
        """The frame as a picture (a hard copy): each graph's box, curve, marks,
        title and label, and the summary."""
        def esc(t):
            return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
               f'height="{height}" viewBox="0 0 {width} {height}">',
               f'<rect width="{width}" height="{height}" fill="white"/>']
        for panel in (self.spectrum, self.correlation):
            if panel is None:
                continue
            v1, v2, v3, v4 = panel.viewport
            out.append(f'<rect x="{v1 * width:.1f}" y="{(1 - v4) * height:.1f}" '
                       f'width="{(v2 - v1) * width:.1f}" '
                       f'height="{(v4 - v3) * height:.1f}" fill="none" stroke="black"/>')
            x1, x2, y1, y2 = panel.window
            lo, hi = min(x1, x2), max(x1, x2)
            pts = []
            for a, b in zip(panel.x, panel.y, strict=True):
                if lo <= a <= hi and y1 <= b <= y2:
                    nx, ny = panel.ndc(float(a), float(b))
                    pts.append(f"{nx * width:.1f},{(1 - ny) * height:.1f}")
            out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="black"/>')
            for a, b, c, d in panel.marks:
                (p, q), (r, t) = panel.ndc(a, b), panel.ndc(c, d)
                out.append(f'<line x1="{p * width:.1f}" y1="{(1 - q) * height:.1f}" '
                           f'x2="{r * width:.1f}" y2="{(1 - t) * height:.1f}" '
                           'stroke="black"/>')
            for text, y in ((panel.title, v4 + 0.02), (panel.xlabel, v3 - 0.06)):
                if text:
                    out.append(f'<text x="{(v1 + v2) / 2 * width:.1f}" '
                               f'y="{(1 - y) * height:.1f}" font-size="13" '
                               f'text-anchor="middle">{esc(text)}</text>')
        for t in self.texts:
            out.append(f'<text x="{t.x * width:.1f}" y="{(1 - t.y) * height:.1f}" '
                       f'font-size="{13 * t.size:.1f}" font-family="monospace">'
                       f'{esc(t.text)}</text>')
        out.append("</svg>\n")
        return "\n".join(out)


def rvscale(v, axis: int) -> tuple[float, float]:
    """rvscale: the y limits of the values, 10% of their range beyond (axis 2), or
    from zero (axis -2)."""
    v = np.asarray(v, f32)
    minval = maxval = v[0]
    for pixval in v[1:]:
        if pixval < minval:
            minval = pixval
        elif pixval > maxval:
            maxval = pixval
    diff = f32(f32(maxval - minval) * f32(0.1))
    if axis == -2:
        return 0.0, float(f32(maxval + diff))
    return float(f32(minval - diff)), float(f32(maxval + diff))


def xcsplot(xc, r, wlspec, smspec, minwav: float, maxwav: float,
            dispmode: int) -> Panel:
    """xcsplot: the spectrum, and the range correlated."""
    npoints = wlspec.size
    ipw, npw = 1, npoints
    if dispmode == 0:
        if minwav != maxwav:
            for i in range(1, npoints + 1):
                if wlspec[i - 1] < minwav:
                    ipw = i + 1
                if wlspec[i - 1] > maxwav:
                    npw = i - 1
                    break
        npw = npw - ipw + 1
    if dispmode == 0:
        x1, x2 = float(f32(minwav)), float(f32(maxwav))
    else:
        dwl = f32(f32(wlspec[-1] - wlspec[0]) / f32(npoints - 1))
        x1, x2 = float(f32(wlspec[0] - dwl)), float(f32(wlspec[-1] + dwl))
    y1, y2 = 0.0, 1.0
    if npw > 0:
        y1, y2 = rvscale(smspec[ipw - 1:ipw - 1 + npw], -2 if dispmode < 0 else 2)
    label = "Pixels" if xc.cor == COR_PIX else xc.xlabel(r.spectrum)
    panel = Panel((x1, x2, y1, y2), SPECTRUM_VIEW, "", label,
                  np.array(wlspec, f32), np.array(smspec, f32))
    if dispmode != 0 and minwav != maxwav:
        dy = f32(f32(f32(y2) - f32(y1)) / f32(50.0))
        dy4 = f32(dy * f32(0.25))
        for wx in (float(f32(minwav)), float(f32(maxwav))):
            wy = f32(y1)
            for _ in range(50):
                panel.marks.append((wx, float(f32(wy - dy4)), wx, float(f32(wy + dy4))))
                wy = f32(wy + dy)
    return panel


def xcorplot(xc, r, it: int, order: list[int], itemp: int) -> Panel | None:
    """xcorplot: the correlation with template itemp (xcplot's), around the peak of
    the it-th by R -- whose name the title has: the two are the same template
    unless their R are equal (all 0 when no peak is within the velocity limits)."""
    k = order[it - 1]
    t = r.temps[k]
    data = r.temps[itemp]
    title = f"{it} Corr. Template: {t.template.spec.object}"
    cvel, dvel = xc.p.get("cvel"), xc.p.get("dvel")
    xcr = f32(t.zvel if cvel is None else cvel)
    if dvel is None:
        dxcr = f32(f32(20.0) * f32(t.vwidth))
        if dxcr < 1.0:
            dxcr = f32(300.0)
    else:
        dxcr = f32(dvel)
    if dxcr == 0:
        return None
    xpts = np.asarray(data.xvel if data.xvel is not None else [], f32)
    ypts = np.asarray(data.xcor if data.xcor is not None else [], f32)
    n = xpts.size
    y1, y2 = rvscale(ypts if n > 0 else np.array([0.0, 1.0], f32), 2)
    label = {COR_PIX: "Pixel Shift", COR_WAV: "Wavelength Shift [Angstroms]"}.get(
        xc.cor, "Velocity [km/s]")
    panel = Panel((float(f32(xcr - dxcr)), float(f32(xcr + dxcr)), y1, y2),
                  CORRELATION_VIEW, title, label, xpts, ypts)
    pkfrac = float(xc.p["pkfrac"])
    if n > 0 and pkfrac < 0.0:
        from .xcsao import pkwidth

        fracpeak = -pkfrac
        ph, _pw, _il, _ir, _imax = pkwidth(ypts, xpts, fracpeak, xc.pkindx, None, None)
        npts4 = r.npts // 4
        level = float(f32(fracpeak * ph))
        for i in range(npts4):
            if 4 * i + 1 < n:
                panel.circles.append((float(xpts[4 * i + 1]), level))
    return panel


def xciplot(xc, r, minwav: float, maxwav: float, order: list[int]) -> list[Text]:
    """xciplot: the summary, on the screen's coordinates."""
    from .rvsao import BCV, FBCV, FHCV, HCV, imgdate, juldate, logtime

    spec = r.spectrum
    head = spec.head
    out = [Text(0.01, 0.01, f"rvsao.xcsao 2.8.5 {logtime()}", 0.6, ndc=True)]
    best = r.temps[r.best - 1]
    fracpeak = abs(float(xc.p["pkfrac"]))
    if fracpeak < 1.0:
        text = f"{_f('%4.2f', fracpeak)}-ht. peak fit, {r.npmax} pts."
    else:
        text = f"Peak fit with {r.npmax} pts."
    out.append(Text(0.35, 0.01, text, 1.0, ndc=True))
    out.append(Text(0.025, 0.96, f"File: {spec.name}", 0.8, ndc=True))
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
    x0 = SUMMARY_X
    lineno = 0.96
    ut, dj, hdj, _date = juldate(head)
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
        text = "No observation date"
    out.append(Text(x0, lineno, text, 1.0, ndc=True))
    if hdj != 0.0:
        text = f"HJD: {_f('%.5f', hdj)}"
    elif dj != 0.0:
        text = f"JulDate: {_f('%.5f', dj)}"
    lineno -= 0.04
    out.append(Text(x0, lineno, text, 0.75, ndc=True))
    if xc.svcor == HCV:
        text = f"Object HCV: {_f('%7.3f', spec.hcv)}"
    elif xc.svcor in (FBCV, FHCV):
        text = (f"Object file {'BCV' if spec.vb else 'HCV'}: "
                f"{_f('%7.3f', spec.hcv)}")
    elif xc.svcor == BCV:
        text = f"Object BCV: {_f('%7.3f', spec.hcv)}"
    else:
        text = "Object BCV:    0.000"
    lineno -= 0.04
    out.append(Text(x0, lineno, text, 0.75, ndc=True))
    lineno -= 0.04
    out.append(Text(x0, lineno, f"lambda: {_f('%.1f', minwav)} {_f('%.1f', maxwav)}",
                    0.75, ndc=True))
    text = f"nbins: {r.npts}"
    if r.zpad:
        text += "x2"
    if r.tshift != 0.0:
        text += f"  Tsh: {_f('%6.2f', r.tshift)}"
    if best.schop:
        text += " -em"
    if best.achop:
        text += " -ab"
    if not best.scont:
        text += " +tc"
    if best.conproc == ZEROCONT:
        text += " divc"
    lineno -= 0.04
    out.append(Text(x0, lineno, text, 0.75, ndc=True))
    lo, toplo, topnrn, nrun = r.filter
    lineno -= 0.04
    out.append(Text(x0, lineno, f"Filter: {lo} {toplo} {topnrn} {nrun}", 0.75, ndc=True))
    lineno -= 0.04
    out.append(Text(x0, lineno, f"Frac. endmask: {_f('%.2f', float(f32(r.han)))}", 0.75,
                    ndc=True))
    temps = r.temps
    if len(temps) < 6:
        lineno -= 0.04
        out.append(Text(x0, lineno, "Template   vel.    hcv   peak", 0.65, ndc=True))
        for k in order:
            t = temps[k]
            tmpvel = t.template.vel + t.template.shift + t.vshift
            text = (f"{t.template.id[:7]:<7} {_f('%7.3f', tmpvel)} "
                    f"{_f('%7.3f', t.template.hcv)} {_f('%6.2f', t.cz)}")
            lineno -= 0.04
            out.append(Text(x0, lineno, text, 0.65, ndc=True))
    lineno -= 0.06
    head_text = "Template Shift  error    R" if xc.cor in (COR_PIX, COR_WAV) else \
        "Template  CZ    error    R"
    out.append(Text(x0, lineno, head_text, 0.7, ndc=True))
    for k in order[:10]:
        t = temps[k]
        text = (f"{t.template.id[:7]:<7} {_f('%7.3f', t.zvel)} {_f('%7.3f', t.czerr)} "
                f"{_f('%5.2f', t.czr)} ")
        if t.overlap:
            text += "+w"
        if t.schop:
            text += "-e"
        if t.achop:
            text += "-a"
        text += {1: "-f", 2: "+h", 3: "-f+h"}.get(t.template.filt, "")
        if not t.scont:
            text += "+tc"
        if t.conproc == ZEROCONT:
            text += "/c"
        lineno -= 0.04
        out.append(Text(x0, lineno, text, 0.65, ndc=True))
    return out


class Plot:
    """xcplot: the graph of one spectrum's correlations and the cursor's keys on it.
    `read_key` gives the next cursor read (x, y, wcs, key) or None at the end;
    `read_line` the next line typed. After `run`, `newfit` says whether to
    correlate again, and `frames` holds every graph drawn."""

    def __init__(self, xc, r, read_key, read_line, say, curmode: bool = True) -> None:
        from .rvsao import smooth

        self.xc, self.r = xc, r
        self.read_key, self.read_line, self.say = read_key, read_line, say
        self.curmode = curmode
        self.frames: list[Frame] = []
        self.hardcopies: list[Frame] = []
        spec = r.spectrum
        np1, np2 = r.span
        wcs = spec.wcs
        self.wlspec = np.array([wcs.p2w(float(i)) for i in range(np1, np2 + 1)], f32)
        self.specwcs = np.asarray(spec.data[np1 - 1:np2], f32)
        self.smspec = self.specwcs.copy()
        smooth(self.smspec, self.smspec.size, xc.nsmooth)
        self.order = sorted(range(len(r.temps)), key=lambda k: -r.temps[k].czr)
        self.itemp = r.best - 1
        self.it0 = 1
        best = r.temps[self.itemp]
        self.strtwav, self.finwav = best.wl1, best.wl2
        self.newfit = False
        self.qplot = False
        self.mode = int(xc.p.get("dispmode", 1))

    def draw(self) -> Frame:
        xc, r = self.xc, self.r
        frame = Frame(xcsplot(xc, r, self.wlspec, self.smspec, self.strtwav, self.finwav,
                              self.mode),
                      xcorplot(xc, r, self.it0, self.order, self.itemp),
                      xciplot(xc, r, self.strtwav, self.finwav, self.order))
        self.frames.append(frame)
        return frame

    def start(self) -> bool:
        p = self.xc.p
        if bool(p.get("hardcopy", False)):
            self.hardcopies.append(self.draw())
            self.frames.pop()
        if not bool(p.get("displot", True)):
            return False
        if self.mode > 1:
            self.say("XCSAO: emsao's graph (dispmode 2 and 3) is not ported yet\n")
            self.mode = 1
        self.draw()
        return self.curmode

    def run(self) -> None:
        if not self.start():
            return
        while True:
            read = self.read_key()
            if read is None or self.press(read[0], read[1], read[3]):
                break

    def press(self, x: float, y: float, key: str) -> bool:
        """One cursor read: True when xcplot leaves its loop."""
        if key == "q":                          # gt_gcur: q is the end of file
            return True
        return self.key(key)

    def key(self, key: str) -> bool:
        xc, r, say = self.xc, self.r, self.say
        p = xc.p
        if key == "g":
            from .rvsao import smooth

            say(f"Number of times to smooth spectrum ({xc.nsmooth}): ")
            line = self.read_line()
            if line:
                value = _ctoi(line)
                if value is not None:
                    xc.nsmooth = value
                self.smspec = self.specwcs.copy()
                smooth(self.smspec, self.smspec.size, xc.nsmooth)
        elif key == "m":
            before = self.mode
            say(f"Display mode (0-3) ({self.mode}): ")
            line = self.read_line()
            if line is not None:
                value = _ctoi(line)
                if value is not None:
                    xc.jtemp = value
                if -4 < xc.jtemp < 4:
                    self.mode = xc.jtemp
            say(f"Display mode {before} -> {self.mode} ({xc.jtemp})\n")
            if self.mode != before:
                if self.mode > 1:
                    say("XCSAO: emsao's graph (dispmode 2 and 3) is not ported yet\n")
                    self.mode = before
                else:
                    self.draw()
        elif key == "c":
            say("Correlation peak fitting parameters\n")
            say("Peak-fitting mode (1-parabola 2-quartic 3-cosx/1+x^2) "
                f"({int(p['pkmode'])}) = ")
            line = self.read_line()
            if line:
                value = _ctoi(line)
                if value is not None:
                    p["pkmode"] = value
                self.newfit = True
            say("Fraction of peak or number of points for peak fitting "
                f"({_f('%7.3f', float(p['pkfrac']))}) = ")
            line = self.read_line()
            if line:
                value = _ctod(line)
                if value is not None:
                    p["pkfrac"] = value
                self.newfit = True
            lo, toplo, topnrn, nrun = r.filter
            say(f"Filter parameters for transform ({lo} {toplo} {nrun} {topnrn}) = ")
            line = self.read_line()
            if line:
                words = line.split()
                names = ("low_bin", "top_low", "nrun", "top_nrun")
                for name, word in zip(names, words, strict=False):
                    value = _ctoi(word)
                    if value is None:
                        break
                    p[name] = value
                self.newfit = True
            if self.newfit:
                return True
        elif key == "f":
            say("Rerunning correlations\n")
            self.newfit = True
            return True
        elif key == "t":
            if r.temps:
                say("template for redshift = ")
                line = self.read_line()
                it = self.it0
                if line:
                    value = _ctoi(line)
                    it = value if value is not None and 1 <= value <= len(r.temps) \
                        else self.it0
                self.itemp = self.order[it - 1]
                r.best = self.itemp + 1
                self.it0 = it
                t = r.temps[self.itemp]
                r.xcvel, r.xcerr, r.xcr = t.zvel, t.czerr, t.czr
                self.strtwav, self.finwav = t.wl1, t.wl2
                self.draw()
        elif key == "v":
            minvel, maxvel = p.get("minvel"), p.get("maxvel")
            say("Minimum allowable velocity in km/sec "
                f"({_f('%8.1f', -5000.0 if minvel is None else minvel)}) = ")
            line = self.read_line()
            if line:
                value = _ctod(line)
                if value is not None:
                    p["minvel"] = value
            say("Maximum allowable velocity in km/sec "
                f"({_f('%8.1f', 100000.0 if maxvel is None else maxvel)}) = ")
            line = self.read_line()
            if line:
                value = _ctod(line)
                if value is not None:
                    p["maxvel"] = value
            self.newfit = True
            return True
        elif key in ("y", "n", "j"):
            r.quality = {"y": 4, "n": 3, "j": 1}[key]
            say({"y": "Setting quality flag to 4=good\n",
                 "n": "Setting quality flag to 3=bad\n",
                 "j": "Setting quality flag to 1=questionable\n"}[key])
            self.qplot = True
        elif key == "@":
            say(f"Making hard copy on {p.get('plotter', 'stdplot')}\n")
            self.hardcopies.append(self.draw())
            self.frames.pop()
        elif key == "r":
            say("Replot XCSAO summary\n")
            self.draw()
        elif key == "x":
            self.qplot = False
            return True
        elif key in (" ", "h", "?"):
            say(KEYS)
        elif key in ("e", "p", "d", "l"):
            say({"e": "XCSAO: e (edit the spectrum) is not ported yet\n",
                 "p": "XCSAO: p (choose the peak by hand) is not ported yet\n",
                 "d": "XCSAO: d (debug) is not ported\n",
                 "l": "XCSAO: l (emsao's graph) is not ported yet\n"}[key])
        else:
            return True
        return False


def _ctod(text: str) -> float | None:
    from .rvsao import Scan

    return Scan(text).d()


def _ctoi(text: str) -> int | None:
    from .rvsao import Scan

    value = Scan(text).d()
    return None if value is None else int(value)
