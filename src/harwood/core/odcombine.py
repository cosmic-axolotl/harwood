"""odcombine: spectra put on one grid of wavelengths and combined by imcombine's code.

From noao/onedspec/odcombine/t_odcombine.x (odc_gspec, odc_rebin, odc_bpm, odc_output,
odc_default, odc_imtgetim) and its src/ -- a copy of imcombine's icombine between the
old one of scombine and that of images.immatch: its rejections as ccdred's (the
`odcombine` flavor of core/combine.py), its scaling, grow and minmax as images', and
its log, iclog.x, with the task's name and a column for the spectrum each line is.

Each spectrum is resampled (shdr_rebin, onedspec's interp) onto the output's pixels and
kept only where it reaches: the first and last pixels it covers by more than 1% of
one. icombine then places them by these offsets ("physical"), so a pixel a spectrum
does not reach is not in its stack, and the output runs only from the first pixel any
reaches to the last.

IRAF's ways kept, its defects among them:
  * every spectrum is written to its temporary image as a copy of the output's
    template, made from the *first* spectrum's image: the header values imcombine reads
    (EXPTIME through expname, a !keyword scale, zero or weight, a keyword's rdnoise,
    gain or snoise, NCOMBINE) are the first's for all. scale=exposure scales nothing.
  * a bad pixel mask (BPM in the header) goes only to the last spectrum taken of each
    output so far, as each image is read: with group all or images, only the image's
    last aperture has its mask; with group apertures, an image without an aperture
    gives its mask to the previous image's spectrum of that aperture.
  * the mask, as 0 and 1000, is resampled like the spectrum, and is bad where that is
    10 or more; one empty on the output is no mask, though the log still names it.
  * the log's `statsec =` line prints the date, not the section.

The odcombine installed with IRAF 2.18.1 never writes an output (its template for the
temporary spectra matches nothing); the gabarito is its source with that template
corrected, tests/data/iraf_odcombine.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from . import combine as engine
from . import iraffmt, sarith, scombine

f32 = np.float32
MAX_REAL = float(scombine.MAX_REAL)
MASKTYPES = ("none", "goodvalue", "badvalue", "goodbits", "badbits")
MASKFORMATS = ("bpmpixel", "bpmspectrum")


@dataclass
class Member:
    """A spectrum taken for an output: as shdr holds it, and its bad pixel mask."""
    sh: sarith.Sh
    line: int                       #: LINDEX: its line in its image
    header: object                  #: its image's header
    aperture: object                #: harwood's aperture it came from
    mask: sarith.Sh | None = None
    mask_label: str = ""            #: ICBPM: the mask, as the log names it

    @property
    def label(self) -> str:
        """ICFNAME: the image, its section and the aperture."""
        return f"{self.sh.image}{self.sh.section}({self.sh.ap})"


@dataclass
class Placed:
    """A spectrum on the output's pixels: the part it reaches, from pixel p1."""
    member: Member
    p1: int
    data: np.ndarray
    mask: np.ndarray | None = None  #: 1 where bad, as odc_bpm leaves it


@dataclass
class Result:
    combined: np.ndarray
    sigma: np.ndarray
    used: np.ndarray
    rejected: np.ndarray            #: [image, pixel]
    keep: np.ndarray                #: [image, pixel]: what went into each pixel
    first: int                      #: the output's first pixel, on the grid
    scaling: object
    nout: int
    log: list[str] = field(default_factory=list)


def add_spectra(groups: list[list[Member]], members: list[Member], group: str) -> None:
    """odc_gspec's grouping: by aperture number, or all into one."""
    for member in members:
        j = 0
        if group == "apertures":
            j = next((k for k, g in enumerate(groups) if g[0].sh.ap == member.sh.ap),
                     len(groups))
        if j == len(groups):
            groups.append([])
        groups[j].append(member)


def add_masks(groups: list[list[Member]], name: str, masks, mformat: str,
              section_of) -> None:
    """odc_gspec's masks, from the image named `name` with its mask apertures `masks`:
    to the last member of each output, IRAF's way. `section_of(nlines, line)` is the
    IMSEC of a line of the mask."""
    for g in groups:
        member = g[-1]
        if mformat == "bpmpixel":
            k = member.line
            if k > len(masks):
                continue
            values = np.asarray(masks[k - 1].flux, dtype=f32)
            sh = copy.deepcopy(member.sh)
            n = min(sh.n, values.size)
            sh.data = sh.data.copy()
            sh.data[:n] = values[:n]
            member.mask = sh
            member.mask_label = f"{name}{section_of(len(masks), k)}"
        else:
            line = next((i for i, m in enumerate(masks, 1) if m.number == member.sh.ap),
                        None)
            if line is None:
                continue
            sh = sarith.open_aperture(masks[line - 1], name, section_of(len(masks), line),
                                      "flux")
            member.mask = sh
            member.mask_label = f"{name}{sh.section}({sh.ap})"


def grid(members: list[Member], first: bool, w1, w2, dw, nw, log: bool):
    """odc_output and odc_default: the output's (dtype, wavelengths, z) -- the first
    spectrum's own with `first`, a nonlinear one made linear, or the inputs' range in
    the smallest of their steps, as given."""
    a1 = members[0].aperture
    if first:
        dc = members[0].sh.dc
        if dc == 2:
            from . import spectrum

            start, step = spectrum.start_and_step(a1.wavelength, members[0].sh.n)
            return 0, scombine.grid_wavelengths(0, start, step, members[0].sh.n), 0.0
        wave = None if a1.wavelength is None else np.asarray(a1.wavelength, float)
        return dc, wave, float(a1.redshift or 0.0)
    specs = [SimpleNamespace(sh=m.sh, dc=m.sh.dc if m.aperture.wavelength is not None
                             else -1) for m in members]
    dtype, w1, dw, nw = scombine.default_grid(specs, w1, w2, dw, nw, log)
    return dtype, scombine.grid_wavelengths(dtype, w1, dw, nw), 0.0


def _bpm(sh: sarith.Sh, out: sarith.Sh, mtype: str, mvalue: int, interp: str) -> None:
    """odc_bpm: the mask as 0 (good) and 1000, resampled, and bad where 10 or more."""
    values = np.array([sarith._nint(float(v)) for v in sh.data], dtype=np.int64)
    if mtype == "goodvalue":
        bad = values != mvalue
    elif mtype == "badvalue":
        bad = values == mvalue
    elif mtype == "goodbits":
        bad = (values & mvalue) == 0
    else:
        bad = (values & mvalue) != 0
    sh.data = np.where(bad, f32(1000), f32(0)).astype(f32)
    sarith.rebin(sh, out, interp)
    sh.data = np.array([0 if sarith._nint(float(v)) < 10 else 1 for v in sh.data],
                       dtype=f32)


def place(member: Member, out: sarith.Sh, mtype: str, mvalue: int,
          interp: str) -> Placed | None:
    """odc_rebin: the spectrum, and its mask, resampled onto the output and cut to the
    pixels it reaches; None where it reaches none."""
    sh = member.sh
    c1 = out.wl(sh.lw(0.5))
    c2 = out.wl(sh.lw(sh.n + 0.5))
    p1 = max(1, sarith._nint(min(c1, c2) + 0.01))
    p2 = min(out.n, sarith._nint(max(c1, c2) - 0.01))
    if p2 - p1 + 1 < 1:
        return None
    sarith.rebin(sh, out, interp)
    mask = None
    if member.mask is not None:
        _bpm(member.mask, out, mtype, mvalue, interp)
        mask = member.mask.data[p1 - 1:p2].copy()
    return Placed(member, p1, sh.data[p1 - 1:p2].copy(), mask)


def _scale_names(text: str) -> str:
    return "".join(text.split())


def combine(placed: list[Placed], params: dict, header) -> Result:
    """icombine on the placed spectra: offsets, masks, scaling, rejection and the log.

    `params` holds imcombine's: combine, reject, scale, zero, weight, statsec, expname,
    lthreshold, hthreshold, nlow, nhigh, nkeep, mclip, lsigma, hsigma, rdnoise, gain,
    snoise, sigscale, pclip, grow, blank, and `output` and the names of the other outputs
    for the log. `header` is the template's: every image's, as IRAF reads it."""
    from ..tasks.imcombine import _noise_values, _restore, _sections

    n = len(placed)
    offsets = np.array([p.p1 - 1 for p in placed])
    lengths = np.array([p.data.size for p in placed])
    aligned = bool(np.all(offsets == offsets[0]) and
                   np.all(offsets + lengths == offsets[0] + lengths[0]))
    low = int(offsets.min())
    offsets = offsets - low
    npts = int((offsets + lengths).max())

    # icmask: a mask empty on the output is no mask; with none left, masks are off.
    has = [p.mask is not None for p in placed]
    marked = [p.mask is not None and bool(np.any(p.mask != 0)) for p in placed]
    masks_on = params["smasktype"] != "none" and any(marked)
    bad = [np.asarray(p.mask != 0) if masks_on and p.mask is not None
           else np.zeros(p.data.size, dtype=bool) for p in placed]

    stack = np.zeros((n, 1, npts))
    covered = np.zeros(stack.shape, dtype=bool)
    for i, p in enumerate(placed):
        span = slice(int(offsets[i]), int(offsets[i]) + p.data.size)
        stack[i, 0, span] = p.data.astype(np.float64)
        covered[i, 0, span] = ~bad[i]
    valid = covered.copy()
    lt, ht = params["lthreshold"], params["hthreshold"]
    if lt is not None or ht is not None:
        lo = -MAX_REAL if lt is None else float(f32(lt))
        hi = MAX_REAL if ht is None else float(f32(ht))
        valid &= (stack >= lo) & (stack <= hi)

    headers = [header] * n
    frames = [p.data.astype(np.float64) for p in placed]
    geometry = SimpleNamespace(offsets=[(int(o),) for o in offsets])
    sections = _sections(params["statsec"], frames, geometry, (npts,))
    scale, zero, weight = (_scale_names(params[k]) for k in ("scale", "zero", "weight"))
    scaling = engine.ic_scale(
        frames, headers, scale=scale, zero=zero, weight=weight,
        exposure_key=params["expname"], section=sections, lthresh=lt, hthresh=ht,
        sigscale=params["sigscale"], flavor="odcombine", masks=bad)

    reject = params["reject"]
    combined, sigma, used, rejected = engine.combine(
        stack, valid, scaling, combine=params["combine"], reject=reject,
        mclip=params["mclip"], nlow=params["nlow"], nhigh=params["nhigh"],
        nkeep=params["nkeep"], lsigma=params["lsigma"], hsigma=params["hsigma"],
        pclip=params["pclip"], grow=params["grow"], blank=params["blank"],
        rdnoise=_noise_values(params["rdnoise"], headers),
        gain=_noise_values(params["gain"], headers),
        snoise=_noise_values(params["snoise"], headers),
        flavor="odcombine", marked=np.array([masks_on and any(marked)]))
    combined, sigma, used = combined.reshape(npts), sigma.reshape(npts), used.reshape(npts)
    rejected = rejected.reshape(n, npts)
    keep = valid.reshape(n, npts) & ~rejected
    if not np.array_equal(keep.sum(axis=0), np.maximum(used, 0)):
        keep = _restore(keep, valid.reshape(n, npts), used, stack.reshape(n, npts),
                        scaling)
    ncombine = int(header.get("NCOMBINE", 1)) if _is_int(header.get("NCOMBINE")) else 1
    result = Result(combined, sigma, used, rejected, keep, low, scaling, ncombine * n)
    result.log = log_lines(placed, params, scaling, offsets, aligned, masks_on, has,
                           header, ncombine, result.nout)
    return result


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _r(value, fmt: str, decpl=None, width=None) -> str:
    return scombine._r(value, fmt, decpl, width)


def log_lines(placed, params, scaling, offsets, aligned, masks_on, has, header,
              ncombine, nout) -> list[str]:
    """ic_log, as odcombine's src has it."""
    n = len(placed)
    yes = lambda b: "yes" if b else "no"  # noqa: E731
    g = lambda v: _r(v, "g")  # noqa: E731
    out = ["", f"{iraffmt.cnvdate()}: ODCOMBINE"]
    sname, zname, wname = (_scale_names(params[k]) for k in ("scale", "zero", "weight"))
    out.append(f"  combine = {params['combine']}, scale = {sname}, zero = {zname}, "
               f"weight = {wname}")
    reject = params["reject"]
    flow, fhigh = f32(params["nlow"]), f32(params["nhigh"])
    if reject == "minmax":
        if flow >= 1:
            flow = f32(flow / f32(n))
        if fhigh >= 1:
            fhigh = f32(fhigh / f32(n))
        if int(f32(flow * f32(n))) + int(f32(fhigh * f32(n))) == 0:
            reject = "none"
    nkeep = params["nkeep"]
    if reject == "minmax":
        out.append(f"  reject = minmax, nlow = {sarith._nint(float(f32(flow * f32(n))))}, "
                   f"nhigh = {sarith._nint(float(f32(fhigh * f32(n))))}")
    elif reject == "ccdclip":
        out.append(f"  reject = ccdclip, mclip = {yes(params['mclip'])}, nkeep = {nkeep}")
        out.append(f"  rdnoise = {params['rdnoise']}, gain = {params['gain']}, "
                   f"snoise = {params['snoise']}, sigma = {g(params['lsigma'])}, "
                   f"hsigma = {g(params['hsigma'])}")
    elif reject == "crreject":
        out.append(f"  reject = crreject, mclip = {yes(params['mclip'])}, nkeep = {nkeep}")
        out.append(f"  rdnoise = {params['rdnoise']}, gain = {params['gain']}, "
                   f"snoise = {params['snoise']}, hsigma = {g(params['hsigma'])}")
    elif reject == "pclip":
        half = int(n / 2.0)
        p = f32(params["pclip"])
        if abs(p) < 1:
            p = f32(p * f32(half))
        p = min(-1, max(-half, int(p))) if p < 0 else max(1, min(half, int(p)))
        out.append(f"  reject = pclip, nkeep = {nkeep}")
        out.append(f"  pclip = {g(p)}, lsigma = {g(params['lsigma'])}, "
                   f"hsigma = {g(params['hsigma'])}")
    elif reject in ("sigclip", "avsigclip"):
        out.append(f"  reject = {reject}, mclip = {yes(params['mclip'])}, nkeep = {nkeep}")
        out.append(f"  lsigma = {g(params['lsigma'])}, hsigma = {g(params['hsigma'])}")
    if reject != "none" and params["grow"] >= 1:
        out.append(f"  grow = {g(params['grow'])}")
    lt, ht = params["lthreshold"], params["hthreshold"]
    if lt is not None and ht is not None:
        out.append(f"  lthreshold = {g(lt)}, hthreshold = {g(ht)}")
    elif lt is not None:
        out.append(f"  lthreshold = {g(lt)}")
    elif ht is not None:
        out.append(f"  hthreshold = {g(ht)}")
    out.append(f"  blank = {g(params['blank'])}")
    if params["statsec"].strip():
        # iclog.x prints the buffer that holds the date, not the section.
        out.append(f"  statsec = {iraffmt.cnvdate()}")
    if masks_on:
        out.append("  masktype = goodval, maskval = 0")

    # The statistics as ic_scale keeps them: the values of the scale, zero or weight
    # that came from them, before they are made relative.
    modes, medians, means = ([None] * n for _ in range(3))
    for kind in (sname, zname, wname):
        if kind in ("mode", "median", "mean"):
            values = [scaling.statistics[i][kind] for i in range(n)]
            {"mode": modes, "median": medians, "mean": means}[kind][:] = values
    differs = lambda v: any(v[i] != v[0] for i in range(n))  # noqa: E731
    ccd = reject in ("ccdclip", "crreject")
    prrdn = ccd and scombine._leading_number(params["rdnoise"]) is None
    prgain = ccd and scombine._leading_number(params["gain"]) is None
    prsn = ccd and scombine._leading_number(params["snoise"]) is None
    prmask = masks_on and any(has)
    prmode, prmedian, prmean = differs(modes), differs(medians), differs(means)
    head = f"  {'Images':>20} "
    for show, width, label in ((prmode, 7, "Mode"), (prmedian, 7, "Median"),
                               (prmean, 7, "Mean"), (prrdn, 7, "Rdnoise"),
                               (prgain, 6, "Gain"), (prsn, 6, "Snoise"),
                               (scaling.doscale, 6, "Scale"), (scaling.dozero, 7, "Zero"),
                               (scaling.dowts, 6, "Weight"), (not aligned, 9, "Offsets")):
        if show:
            head += f" {label:>{width}}"
    if prmask:
        head += " Maskfile"
    out.append(head)
    for i, p in enumerate(placed):
        row = f" {p.member.label:>21}"
        if prmode:
            row += " " + _r(modes[i], "g", 5, 7)
        if prmedian:
            row += " " + _r(medians[i], "g", 5, 7)
        if prmean:
            row += " " + _r(means[i], "g", 5, 7)
        for show, key, width in ((prrdn, "rdnoise", 7), (prgain, "gain", 6),
                                 (prsn, "snoise", 6)):
            if show:
                row += " " + _r(float(header[params[key].strip()]), "g", None, width)
        if scaling.doscale:
            row += " " + _r(f32(f32(1) / f32(scaling.scales[i])), "f", 3, 6)
        if scaling.dozero:
            row += " " + _r(-f32(scaling.zeros[i]), "g", 5, 7)
        if scaling.dowts:
            row += " " + _r(scaling.weights[i], "f", 3, 6)
        if not aligned:
            row += f" {int(offsets[i]):9d}"
        if prmask and p.member.mask is not None:
            row += f" {p.member.mask_label}"
        out.append(row)
    out += ["", f"  Output image = {params['output']}, ncombine = {nout}"]
    for key, label in (("bpmasks", "Bad pixel mask"), ("rejmasks", "Rejection mask"),
                       ("nrejmasks", "Number rejected mask"), ("expmasks", "Exposure mask"),
                       ("sigmas", "Sigma image")):
        if params.get(key):
            out.append(f"  {label} = {params[key]}")
    return out
