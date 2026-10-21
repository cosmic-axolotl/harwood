"""The DQ plane: why a pixel is not to be trusted, one bit per reason.

The bits are DRAGONS' (Gemini's pipeline), so a harwood file and a DRAGONS
file mean the same thing by the same number; design contract, section 12.1.
A good pixel is 0, and reasons add up: 12 is a saturated pixel a cosmic ray
also hit. Bits from 128 up are left for what section 11 brings -- a CMOS's
telegraph noise, an infrared array's persistence -- when real data needs
them.
"""

from __future__ import annotations

import numpy as np

DTYPE = np.uint16

GOOD = 0
BAD_PIXEL = 1
NON_LINEAR = 2
SATURATED = 4
COSMIC_RAY = 8
NO_DATA = 16
OVERLAP = 32
UNILLUMINATED = 64

#: Every bit, in order, with the name the header and the viewer use.
NAMES = {BAD_PIXEL: "bad pixel", NON_LINEAR: "non linear", SATURATED: "saturated",
         COSMIC_RAY: "cosmic ray", NO_DATA: "no data", OVERLAP: "overlap",
         UNILLUMINATED: "unilluminated"}

#: Written in the DQ extension's header, so a reader knows what the bits mean.
CONVENTION = "DRAGONS"


def empty(shape) -> np.ndarray:
    """A DQ plane with every pixel good."""
    return np.zeros(shape, dtype=DTYPE)


def describe(value: int) -> str:
    """The reasons a DQ value holds, in words: 12 -> 'saturated, cosmic ray'."""
    value = int(value)
    if value == GOOD:
        return "good"
    reasons = [name for bit, name in NAMES.items() if value & bit]
    rest = value & ~sum(NAMES)
    if rest:
        reasons.append(f"other ({rest})")
    return ", ".join(reasons)


def header_cards(header) -> None:
    """The cards a DQ extension carries: the convention and each bit's name,
    so the file explains itself without this module."""
    header["DQCONV"] = (CONVENTION, "meaning of the DQ bits")
    for bit, name in NAMES.items():
        header[f"DQBIT{bit.bit_length() - 1}"] = (name, f"DQ bit value {bit}")
