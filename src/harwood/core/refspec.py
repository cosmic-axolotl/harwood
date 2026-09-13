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

MODES = ("nearest", "preceding", "following", "interp", "average", "match")


@dataclass
class Assignment:
    """The arc or arcs one exposure should be calibrated with."""

    image: str
    time: float | None = None
    references: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    reason: str = ""

    def describe(self) -> str:
        if not self.references:
            return f"{self.image}: none"
        if len(self.references) == 1:
            return f"{self.image}: {self.references[0]}"
        return f"{self.image}: " + " + ".join(
            f"{name} ({weight:.3f})"
            for name, weight in zip(self.references, self.weights, strict=False))


def assign(images: list[tuple[str, float | None]],
           references: list[tuple[str, float | None]],
           select: str = "interp") -> list[Assignment]:
    """Work out which reference goes with each image.

    Both lists are (name, time) pairs; a time of None means the header did not
    say, and such an exposure gets no assignment rather than a guess.
    """
    if select not in MODES:
        raise ValueError(
            f"no selection mode called '{select}'. Choose one of "
            f"{', '.join(MODES)}."
        )

    usable = sorted(((name, when) for name, when in references if when is not None),
                    key=lambda item: item[1])
    if not usable:
        raise ValueError(
            "none of the reference spectra has a usable time, so they cannot be "
            "put in order. Check that the headers carry UT and EXPTIME."
        )

    assignments = []
    for name, when in images:
        if when is None:
            assignments.append(Assignment(
                name, None, reason="the header does not say when this was taken"))
            continue
        assignments.append(_one(name, when, usable, select))
    return assignments


def _one(name: str, when: float, references, select: str) -> Assignment:
    before = [item for item in references if item[1] <= when]
    after = [item for item in references if item[1] > when]

    if select == "preceding":
        chosen = before[-1] if before else None
        if chosen is None:
            return Assignment(name, when,
                              reason="no arc was taken before this exposure")
        return Assignment(name, when, [chosen[0]], [1.0],
                          f"the arc {(when - chosen[1]) * 60:.0f} minutes before")

    if select == "following":
        chosen = after[0] if after else None
        if chosen is None:
            return Assignment(name, when,
                              reason="no arc was taken after this exposure")
        return Assignment(name, when, [chosen[0]], [1.0],
                          f"the arc {(chosen[1] - when) * 60:.0f} minutes after")

    if select == "nearest":
        chosen = min(references, key=lambda item: abs(item[1] - when))
        return Assignment(name, when, [chosen[0]], [1.0],
                          f"the nearest arc, {abs(chosen[1] - when) * 60:.0f} "
                          "minutes away")

    if not before or not after:
        # Outside the arcs: interpolation would be extrapolation, and the
        # nearest arc is the honest answer -- said out loud, because a
        # spectrum calibrated with an arc from three hours away is worse than
        # the rms of that arc suggests.
        chosen = (before or after)[-1 if before else 0]
        gap = abs(chosen[1] - when) * 60
        return Assignment(name, when, [chosen[0]], [1.0],
                          f"outside the arcs, so the nearest one, {gap:.0f} "
                          "minutes away, rather than an interpolation")

    first, second = before[-1], after[0]
    if select == "average":
        return Assignment(name, when, [first[0], second[0]], [0.5, 0.5],
                          "the arcs either side, weighted equally")

    span = second[1] - first[1]
    if span <= 0:
        return Assignment(name, when, [first[0]], [1.0], "the arcs coincide")
    share = (second[1] - when) / span
    return Assignment(
        name, when, [first[0], second[0]], [share, 1.0 - share],
        f"between the arcs {(when - first[1]) * 60:.0f} minutes after one and "
        f"{(second[1] - when) * 60:.0f} before the other")


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


def write_keywords(header, assignment: Assignment) -> None:
    """Put an assignment into a header, in the form IRAF writes."""
    for index in (1, 2):
        header.remove(f"REFSPEC{index}", ignore_missing=True)
    for index, (name, weight) in enumerate(
            zip(assignment.references, assignment.weights, strict=False), start=1):
        header[f"REFSPEC{index}"] = (
            f"{name} {weight:.5f}" if len(assignment.references) > 1 else name,
            "arc used to calibrate this spectrum")
