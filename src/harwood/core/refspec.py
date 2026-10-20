"""Which arc belongs to which exposure.

A dispersion solution is not a property of the spectrograph, it is a property
of the spectrograph *at a moment*. The instrument flexes as the telescope
tracks, and the wavelength that lands on a given pixel moves with it -- which
is why a night has an arc at the start and another at the end rather than one.

So every object exposure has to be told which arc to use, and the useful
answer is often "both". Between two arcs taken three hours apart, the flexure
is smooth, and an exposure in the middle is better served by a weighted
average of the two solutions than by whichever happens to be nearer. That is
IRAF's `interp`, and it is what the tutorials use.

The selection modes are IRAF's:

``nearest``     the closest arc in time, either side
``preceding``   the closest arc before the exposure
``following``   the closest arc after it
``interp``      the arcs either side, weighted by how close each one is
``average``     the arcs either side, weighted equally
``match``       an arc named explicitly

Times come from `core.obstime`, at the middle of the exposure and wrapped
across midnight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import advice

MODES = ("nearest", "preceding", "following", "interp", "average", "match")
MAX_REAL = 3.4028234663852886e38


@dataclass
class Frame:
    """A spectrum as refspectra sees it: its sort value, aperture and group."""

    name: str
    value: float | None = None
    ap: int = 1
    group: str = ""


@dataclass
class Assignment:
    """The arc or arcs one exposure should be calibrated with."""

    image: str
    time: float | None = None
    references: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    reason: str = ""
    #: what IRAF prints on stderr when nothing could be assigned
    messages: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.references:
            return f"{self.image}: none"
        if len(self.references) == 1:
            return f"{self.image}: {self.references[0]}"
        return f"{self.image}: " + " + ".join(
            f"{name} ({weight:.3f})"
            for name, weight in zip(self.references, self.weights, strict=False))


def _frame(item) -> Frame:
    return item if isinstance(item, Frame) else Frame(item[0], item[1])


def assign(images, references, select: str = "interp", ignoreaps: bool = True,
           group: str = "", notes: list | None = None) -> list[Assignment]:
    """Work out which reference goes with each image, as IRAF's refinterp,
    refnearest, refprecede, reffollow, refaverage and refmatch do.

    Both lists hold Frames, or (name, value) pairs; a value of None means
    the header did not say, and such an exposure gets no assignment rather
    than a guess. A reference must be in the same group as the image and,
    unless ignoreaps, have its aperture. `group` is the keyword's name, for
    the messages."""
    if select not in MODES:
        advice.fail(
            f"no selection mode called '{select}'.",
            f"choose one of {', '.join(MODES)}. 'interp' is the one that uses "
            "both arcs of the night.")
    images = [_frame(item) for item in images]
    references = [_frame(item) for item in references]
    notes = notes if notes is not None else []

    if select == "average":
        return _average(images, references, notes)
    if select == "match":
        return [Assignment(image.name, image.value, [ref.name], [1.0],
                           "matched in order with the reference list")
                for image, ref in zip(images, references, strict=False)]

    usable = [ref for ref in references if ref.value is not None]
    if not usable:
        advice.fail(
            "none of the reference spectra has a usable time, so they cannot "
            "be put in order.",
            "run setjd (or setairmass) on the arcs, or check they carry "
            "DATE-OBS, UT and EXPTIME.")

    out = []
    for image in images:
        if image.value is None:
            out.append(Assignment(image.name, None,
                                  reason="the header does not say when this was taken"))
            continue
        out.append(_one(image, usable, select, ignoreaps, group))
    return out


def _f32(x: float) -> float:
    import numpy as np

    return float(np.float32(x))


def _one(image: Frame, refs: list[Frame], select: str, ignoreaps: bool,
         group: str) -> Assignment:
    i1 = i2 = None
    d1, d2 = MAX_REAL, -MAX_REAL
    for i, ref in enumerate(refs):
        if ref.group != image.group or (not ignoreaps and ref.ap != image.ap):
            continue
        d = image.value - ref.value
        if select == "nearest":
            if abs(d) < d1:
                i1, d1 = i, abs(d)
        elif select == "interp":
            if 0.0 <= d < d1:
                i1, d1 = i, d
            elif d2 < d <= 0.0:
                i2, d2 = i, d
        elif select == "preceding":
            if 0.0 <= d < d1:
                i1, d1 = i, d
            # refprecede.x tests d < d2 here, with d2 = -MAX_REAL: nothing is
            # ever below it, so IRAF never falls back on a following arc,
            # whatever its manual says.
        elif select == "following":
            if 0.0 < d < d1:
                i1, d1 = i, d
            if d2 < d <= 0.0:
                i2, d2 = i, d

    def one(index: int, why: str) -> Assignment:
        return Assignment(image.name, image.value, [refs[index].name], [1.0], why)

    if select == "nearest" and i1 is not None:
        return one(i1, "the nearest arc")
    if select == "preceding" and i1 is not None:
        return one(i1, "the arc before")
    if select == "following":
        if i2 is not None:
            return one(i2, "the arc after")
        if i1 is not None:
            return one(i1, "no arc after it, so the one before")
    if select == "interp":
        if i1 is not None and i2 is not None:
            if d1 - d2 == 0.0:
                w1 = w2 = 0.5
            else:
                w1, w2 = _f32(-d2 / (d1 - d2)), _f32(d1 / (d1 - d2))
            return Assignment(image.name, image.value, [refs[i1].name, refs[i2].name],
                              [w1, w2], "the arcs either side, by nearness")
        if i1 is not None or i2 is not None:
            return one(i1 if i1 is not None else i2,
                       "outside the arcs, so the nearest one rather than an "
                       "interpolation")

    messages = [f"[{image.name}] No reference spectrum found"]
    for ref in refs:
        if ref.group != image.group:
            messages.append(f"Input [{image.name}] {group} = {image.group} : "
                            f"Ref [{ref.name}] {group} = {ref.group}")
        elif not ignoreaps and ref.ap != image.ap:
            messages.append(f"Input [{image.name}] ap = {image.ap} : "
                            f"Ref [{ref.name}] ap = {ref.ap}")
    why = {"preceding": "no arc was taken before this exposure"}.get(
        select, "no arc in its group and aperture")
    return Assignment(image.name, image.value, reason=why, messages=messages)


def _average(images: list[Frame], refs: list[Frame], notes: list) -> list[Assignment]:
    """refaverage: the first two references, equally, whatever their time,
    aperture or group."""
    if not refs:
        advice.fail("no reference spectra specified", "name the arcs with --references")
    if len(refs) == 1:
        chosen, weights = [refs[0].name], [1.0]
    else:
        if len(refs) > 2:
            notes.append("WARNING: Averaging only first two reference spectra")
        chosen, weights = [refs[0].name, refs[1].name], [0.5, 0.5]
    return [Assignment(image.name, image.value, list(chosen), list(weights),
                       "the first two arcs, weighted equally") for image in images]


def read_keywords(header) -> tuple[list[str], list[float]]:
    """The assignment already in a header, as REFSPEC1 and REFSPEC2.

    IRAF writes `REFSPEC1 = 'sp0018.ms 0.63'` -- the name and its weight in one
    string -- and harwood reads that form as well as its own, so a header
    written by IRAF is usable here.
    """
    names, weights = [], []
    for index in (1, 2):
        value = header.get(f"REFSPEC{index}")
        if not value:
            continue
        parts = str(value).split()
        names.append(parts[0])
        weights.append(float(parts[1]) if len(parts) > 1 else 1.0)
    return names, weights


def _weight(value: float) -> str:
    """%.8g of a single-precision weight, as IRAF prints it."""
    from .iraffmt import printf

    return printf(_f32(value), "g", decpl=8)


def keyword_values(assignment: Assignment) -> tuple[str, str | None]:
    """REFSPEC1 and REFSPEC2 as refspectra writes them: names with their
    weights when there are two, the bare name when the first weighs all."""
    names, weights = assignment.references, assignment.weights
    if len(names) > 1 and weights[0] < 0.9999995:
        return f"{names[0]} {_weight(weights[0])}", f"{names[1]} {_weight(weights[1])}"
    return names[0], None


def log_lines(assignment: Assignment) -> list[str]:
    """What refspectra prints for an assignment: [image] refspec1='...'."""
    names, weights = assignment.references, assignment.weights
    if len(names) > 1 and weights[0] < 0.995:
        return [f"[{assignment.image}] refspec1='{names[0]} {_weight(weights[0])}'",
                f"[{assignment.image}] refspec2='{names[1]} {_weight(weights[1])}'"]
    return [f"[{assignment.image}] refspec1='{names[0]}'"]


def write_keywords(header, assignment: Assignment) -> None:
    """Put an assignment into a header, in the form IRAF writes."""
    first, second = keyword_values(assignment)
    header["REFSPEC1"] = (first, "arc used to calibrate this spectrum")
    if second is None:
        header.remove("REFSPEC2", ignore_missing=True)
    else:
        header["REFSPEC2"] = (second, "arc used to calibrate this spectrum")
