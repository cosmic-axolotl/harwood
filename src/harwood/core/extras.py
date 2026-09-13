"""Turning "No module named 'photutils'" into something you can act on.

harwood installs lean on purpose (section 4): the base is numpy, astropy and a
command line, and everything heavy is an extra. The cost of that is a class of
error message that is accurate and useless -- a task stops because photutils is
not there, and nothing says that `pip install harwood[phot]` is the answer.

So the extras are written down once, here, and the command line turns any
missing import into the line that fixes it.
"""

from __future__ import annotations

#: Which extra provides which module.
PROVIDED_BY = {
    "photutils": "phot",
    "scipy": "phot",
    "ccdproc": "ccd",
    "astroscrappy": "ccd",
    "PySide6": "gui",
    "pyqtgraph": "gui",
    "numba": "fast",
}

#: What each extra is for, in a few words.
PURPOSE = {
    "phot": "detection, photometry and curve fitting",
    "ccd": "CCD reduction and cosmic ray removal",
    "gui": "the viewer and the interactive panels",
    "fast": "the compiled numeric kernels",
}


def hint(error: BaseException) -> str | None:
    """The install line for a missing optional module, or None.

    Returns None for an ImportError that is not about an extra, because
    inventing an install command for a real bug would send someone the wrong
    way entirely.
    """
    if not isinstance(error, ImportError):
        return None
    name = (error.name or "").split(".")[0]
    extra = PROVIDED_BY.get(name)
    if extra is None:
        return None
    return (f"{name} is part of harwood[{extra}], which harwood does not install "
            f"by default -- it is for {PURPOSE[extra]}. "
            f"Run: pip install harwood[{extra}]")
