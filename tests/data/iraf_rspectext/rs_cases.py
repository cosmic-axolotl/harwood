"""The cases of IRAF's rspectext gabarito: the text each reads, and its parameters.
`header` reads what IRAF's wspectext wrote, with its header, of `linear`'s output."""

import numpy as np

N = 60


def text() -> str:
    """Wavelengths that are not linear and fluxes of nine figures, with the lines
    t_rstext skips and one of three numbers."""
    pix = np.arange(N, dtype=float)
    wave = 4000.0 + 2.0 * pix + 0.003 * pix ** 2
    flux = 1234.56789 + 50.123456 * np.sin(pix / 6.0)
    lines = ["# a comment, not two numbers", "17"]
    lines += [f"{w:.6f} {f:.6f}" for w, f in zip(wave, flux, strict=True)]
    lines[10] += " 99"
    return "\n".join(lines) + "\n"


CASES = {
    "linear": ("spec.txt", dict(dtype="linear", crval1=4000.0, cdelt1=2.5, title="a star")),
    "log": ("spec.txt", dict(dtype="log", crval1=3.6, cdelt1=0.0003)),
    "none": ("spec.txt", dict(dtype="none")),
    "nonlinear": ("spec.txt", dict(dtype="nonlinear", title="table")),
    "interp": ("spec.txt", dict(dtype="interp")),
    "flux": ("spec.txt", dict(dtype="linear", flux=True)),
    "header": ("hdr.txt", dict(dtype="none", title="not this")),
}
